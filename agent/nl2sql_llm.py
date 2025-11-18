"""Schema-aware NL2SQL generation via direct LLM prompting (no FAISS)."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd
from openai import OpenAI
from .config import load_config
from .types import GraphState, add_thinking_step
from .utils.cleaners import format_sql_result_as_markdown
from .summary_utils import extract_focus_word

_LOGGER = logging.getLogger(__name__)
NL2SQL_LOGGER = logging.getLogger('hope.agent')

PRIMARY_LLM = "gpt-4o-mini"
FALLBACK_LLM = "gpt-4o"

_GREETING_TOKENS = {"hi", "hello", "hey", "thanks", "thank you"}
_GREETING_RESPONSE = "Hi there! Ask me about occupancy, revenue, or reviews :)"

SCHEMA_BLOCK = """Tables:
1. listings_cleaned(listings_id, host_id, host_name, neighbourhood, neighbourhood_group, latitude, longitude, property_type, room_type, accommodates, bathrooms, bedrooms, beds, price_in_usd, review_scores_rating, review_scores_accuracy, review_scores_cleanliness, review_scores_checkin, review_scores_communication, review_scores_location, review_scores_value, occupancy_rate_30, occupancy_rate_60, occupancy_rate_90, occupancy_rate_365, host_listings_count, estimated_revenue_30, estimated_revenue_60, estimated_revenue_90, estimated_revenue_365)
    - Note: `neighbourhood_group` contains top-level boroughs like 'Brooklyn', 'Manhattan', etc.
    - Note: `neighbourhood` contains subareas (e.g., 'Midtown', 'Williamsburg').
    - Note: All financial values are stored and expressed in **USD ($)**
    - Note: occupancy_rate_30, occupancy_rate_60, occupancy_rate_90, occupancy_rate_365 represent **projected occupancy rates for the next 30/60/90/365 days**, not historical data.
    - Note: Do not use 'is_highbury' in WHERE clauses when querying this table — it does not have that column.

2. highbury_listings(listings_id, host_id, host_name, neighbourhood, neighbourhood_group, latitude, longitude, property_type, room_type, accommodates, bathrooms, bedrooms, beds, price_in_usd, review_scores_rating, review_scores_accuracy, review_scores_cleanliness, review_scores_checkin, review_scores_communication, review_scores_location, review_scores_value, occupancy_rate_30, occupancy_rate_60, occupancy_rate_90, occupancy_rate_365, host_listings_count, estimated_revenue_30, estimated_revenue_60, estimated_revenue_90, estimated_revenue_365)
    - Note: `neighbourhood_group` contains top-level boroughs like 'Brooklyn', 'Manhattan', etc.
    - Note: `neighbourhood` contains subareas (e.g., 'Midtown', 'Williamsburg').
    - Note: All financial values are stored and expressed in **USD ($)**.
    - Note: occupancy_rate_30, occupancy_rate_60, occupancy_rate_90, occupancy_rate_365 represent **projected occupancy rates for the next 30/60/90/365 days**, not historical data.
    - Note: Do not use 'is_highbury' in WHERE clauses when querying 'highbury_listings' — that table already represents the Highbury portfolio.

3. amenities_norm(listing_id, amenity_id, amenity_canonical, category, subtype)
    - Note: When filtering amenities, use ILIKE or LIKE with wildcards (e.g. lower(a.amenity_canonical) LIKE '%parking%').
    - Note: This table does not contain an 'is_highbury' column.
    - Note: Amenities are stored in a separate table: amenities_norm (columns: listing_id, amenity_canonical, category, subtype, etc.).
    - Note: To find common amenities, aggregate on amenity_canonical or category and count distinct listing_id.
    - Note: Join on listing_id if comparing to listings_cleaned or highbury_listings.
"""

def is_greeting(text: str) -> bool:
    """Return True when the incoming query is a trivial greeting."""
    return str(text or "").strip().lower() in _GREETING_TOKENS


def build_prompt(user_query: str, filters: Optional[Dict[str, Any]] = None) -> str:
    """Construct the instruction prompt used for direct SQL generation."""
    filter_hint = ""
    if filters:
        clean = {k: v for k, v in filters.items() if v not in (None, "", [], {})}
        if clean:
            filter_hint = "Filter context: " + ", ".join(f"{k}={v}" for k, v in clean.items()) + "\n\n"

    prompt = (
        "You are a senior data analyst generating valid DuckDB SQL queries for an Airbnb analytics dataset.\n\n"
        "Schema:\n"
        f"{SCHEMA_BLOCK}\n\n"
        "Guidelines:\n"
        "- Generate only one SQL query.\n"
        "- Use DuckDB SQL syntax only.\n"
        "- Always start with a SELECT statement.\n"
        "- Use explicit JOINs where necessary between these tables.\n"
        "- Prefer lower() for string comparisons (e.g., lower(neighbourhood_group)='manhattan').\n"
        "- For “Highbury” portfolio queries, use the table highbury_listings.\n"
        "- For market-wide or general queries, use listings_cleaned.\n"
        "- For amenities questions, use amenities_norm (joined via listing_id).\n"
        "- For descriptive text or highbury flags, use text_extract.\n"
        "- Round numeric aggregates where appropriate (e.g., ROUND(AVG(price_in_usd), 2)).\n"
        "- Return only SQL, without markdown or commentary.\n\n"
        f"{filter_hint}"
        f'User question:\n"{user_query}"'
    )
    return prompt


_CLIENT: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Instantiate and cache the OpenAI client."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    cfg = load_config()
    api_key = cfg.openai_api_key
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured; cannot execute NL2SQL.")
    _CLIENT = OpenAI(api_key=api_key)
    return _CLIENT


def _stream_sql(prompt: str, stream_handler: Optional[Any] = None) -> Tuple[str, str]:
    """Generate SQL text via OpenAI chat completions with streaming."""
    client = _get_client()
    last_error: Optional[Exception] = None
    for model in (PRIMARY_LLM, FALLBACK_LLM):
        try:
            stream = client.chat.completions.create(
                model=model,
                temperature=0.5,
                stream=True,
                messages=[
                    {"role": "system", "content": prompt},
                ],
            )
            sql_chunks: List[str] = []
            for chunk in stream:
                delta = getattr(chunk.choices[0], "delta", None)
                token = getattr(delta, "content", None) if delta else None
                if token:
                    sql_chunks.append(token)
                    if stream_handler:
                        try:
                            stream_handler(token)
                        except Exception as handler_exc:  # pragma: no cover
                            _LOGGER.debug("Stream handler error: %s", handler_exc)
            sql_text = "".join(sql_chunks).strip()
            if sql_text:
                return sql_text, model
        except Exception as exc:  # pragma: no cover - network failure
            last_error = exc
            _LOGGER.error("Streaming SQL generation failed with %s: %s", model, exc)
    raise RuntimeError(f"LLM failed to generate SQL: {last_error}")


def _rows_preview(df: pd.DataFrame, limit: int = 20) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    return df.head(limit).to_dict(orient="records")


def _summarize_result(df: pd.DataFrame, columns: List[str]) -> Tuple[str, str]:
    """Generate a human-friendly summary and Markdown preview from the result DataFrame."""
    if df.empty:
        message = "No listings found for this query."
        return message, message

    preview = _rows_preview(df)
    markdown_rows = [[row.get(col) for col in columns] for row in preview]
    markdown = format_sql_result_as_markdown(markdown_rows, columns)

    # Single scalar (1×1)
    if df.shape == (1, 1):
        value = df.iat[0, 0]
        summary = f"The **{columns[0].replace('_', ' ')}** is **{value}**."
        return summary, markdown

    # Context-aware multi-row summaries
    lower_cols = [c.lower() for c in columns]
    n_rows = len(df)

    if "amenity_canonical" in lower_cols:
        summary = f"Here are the top {min(10, n_rows)} most common amenities."
    elif "room_type" in lower_cols and "average_price" in lower_cols:
        summary = "Average price by room type:"
    elif "neighbourhood_group" in lower_cols and "average_review_score" in lower_cols:
        summary = "Average review score by neighbourhood group:"
    elif "portfolio" in lower_cols and "average_price" in lower_cols:
        summary = "Comparison of average prices between portfolios:"
    elif any("occupancy" in c for c in lower_cols):
        summary = "Average occupancy rates by group:"
    elif any("revenue" in c for c in lower_cols):
        summary = "Estimated revenue results:"
    else:
        summary = "Here are the requested results."

    return summary, markdown


def clean_sql_before_execute(sql: str) -> str:
    """Remove hallucinated or invalid conditions before DuckDB execution."""
    # Drop invalid is_highbury clauses if querying highbury_listings
    if "FROM highbury_listings" in sql and "is_highbury" in sql:
        sql = re.sub(r"AND\s+h\.is_highbury\s*=\s*TRUE\s*;?", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"AND\s+\w+\.is_highbury\s*=\s*TRUE\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"WHERE\s+\w+\.is_highbury\s*=\s*TRUE\s*(?=(AND|OR|GROUP|ORDER|LIMIT|;|$))", "WHERE 1=1 ", sql, flags=re.IGNORECASE)
        sql = re.sub(r"AND\s+is_highbury\s*=\s*TRUE\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"WHERE\s+is_highbury\s*=\s*TRUE\s*(?=(AND|OR|GROUP|ORDER|LIMIT|;|$))", "WHERE 1=1 ", sql, flags=re.IGNORECASE)
        sql = re.sub(r"\s+WHERE 1=1\s+(GROUP|ORDER|LIMIT|;)", r" \1", sql, flags=re.IGNORECASE)
        sql = sql.replace("WHERE 1=1", "WHERE TRUE")
    # Remove duplicated semicolons
    sql = re.sub(r";{2,}", ";", sql)
    return sql


def format_sql_summary(query: str, result_bundle: Dict[str, Any]) -> str:
    """Generate adaptive natural-language summary for SQL answers."""
    summary = (result_bundle.get("summary") or "").strip()
    lower_query = (query or "").lower()
    columns = result_bundle.get("columns") or []

    if not summary:
        if "avg_price" in columns or "price_in_usd" in lower_query:
            summary = "Average price insight retrieved successfully."
        elif "occupancy_rate" in lower_query:
            summary = "Occupancy rate metrics computed."
        elif "revenue" in lower_query:
            summary = "Revenue or pricing summary generated."
        else:
            summary = "Numerical insight retrieved."

    focus_match = re.search(r"(price|occupancy|revenue|rate|availability|bedroom|bathroom|amenity)", lower_query)
    focus_word = focus_match.group(1) if focus_match else extract_focus_word(query)

    if "compare" in lower_query:
        header = f"Comparison results based on {focus_word}"
    elif any(word in lower_query for word in ["average", "mean", "avg"]):
        header = f"Average {focus_word} insights"
    else:
        header = f"{focus_word.capitalize()} overview"

    return f"{header}\n\n{summary.strip()}"


def _strip_sql_blocks(text: str) -> str:
    """Remove SQL statements from free-form summaries."""
    if not text:
        return text

    cleaned = re.sub(r"(?is)```sql.*?```", "", text)
    block_pattern = re.compile(
        r"(?is)^\s*(SELECT|WITH|INSERT|UPDATE|DELETE)\b.*?(?:;|\Z)", re.MULTILINE
    )
    cleaned = block_pattern.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def summarise_table_overview(summary: str, result_bundle: Dict[str, Any]) -> str:
    """Strip markdown tables from NL2SQL summaries and fallback to a concise preview."""
    if not summary:
        return summary

    cleaned = _strip_sql_blocks(summary)
    table_pattern = re.compile(r"(?:^\s*\|.*\|\s*\n?){2,}", re.MULTILINE)
    cleaned = table_pattern.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    if cleaned:
        return cleaned

    columns = result_bundle.get("columns") or []
    rows = result_bundle.get("rows") or []
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        example = rows[0]
        preview_pairs: List[str] = []
        for col in columns[: min(4, len(columns))]:
            value = example.get(col)
            if value is None:
                continue
            preview_pairs.append(f"{col}={value}")
        if preview_pairs:
            return "Key metrics: " + ", ".join(preview_pairs)
    return "Structured metrics available in the table below."


def execute_duckdb(sql: str) -> Dict[str, Any]:
    """Execute SQL against DuckDB and normalise results for downstream consumers."""
    global pd  # ensure we use the global pandas import
    cfg = load_config()
    conn = duckdb.connect(cfg.duckdb_path_str)
    df: Optional[pd.DataFrame] = None

    try:
        sql = clean_sql_before_execute(sql)
        cursor = conn.execute(sql)
        try:
            df = cursor.fetchdf()
        except Exception as fetch_exc:
            _LOGGER.debug("DuckDB fetchdf failed (%s); retrying with manual fetch.", fetch_exc)
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            df = pd.DataFrame(rows, columns=columns)
    except Exception as exc:
        _LOGGER.error("DuckDB execution failed: %s", exc, exc_info=True)
        return {
            "df": pd.DataFrame(),
            "rows": [],
            "columns": [],
            "summary": "SQL execution failed. Please refine the question.",
            "markdown_table": f"Execution error: {exc}",
            "error": str(exc),
        }
    finally:
        conn.close()

    row_count = len(df) if isinstance(df, pd.DataFrame) else 0
    NL2SQL_LOGGER.info("[NL2SQL] Retrieved %d rows for query.", row_count)

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame()

    if row_count == 0:
        summary = "No records found for this query."
        result = {
            "df": df,
            "rows": [],
            "columns": list(df.columns),
            "summary": summary,
            "markdown_table": summary,
        }
        return result

    columns = list(df.columns)
    summary, markdown = _summarize_result(df, columns)
    result = {
        "df": df,
        "rows": _rows_preview(df),
        "columns": columns,
        "summary": summary,
        "markdown_table": markdown,
    }
    # --- Ensure SQL data survives LangGraph serialization ---
    try:
        if isinstance(df, pd.DataFrame):
            rows_preview = df.head(20).to_dict(orient="records")
            columns_serializable = list(df.columns)
            markdown_preview = df.head(10).to_markdown(index=False)
        else:
            rows_preview, columns_serializable, markdown_preview = [], [], ""
        result["rows"] = rows_preview
        result["columns"] = columns_serializable
        result["markdown_table"] = markdown_preview
        result["df"] = None
        _LOGGER.info(
            f"[DUCKDB_PATCH] Injected {len(rows_preview)} serializable rows and {len(columns_serializable)} columns for fusion."
        )
    except Exception as duck_patch_exc:
        _LOGGER.warning(f"[DUCKDB_PATCH] Failed to build serializable rows: {duck_patch_exc}")
    return result


def generate_sql_from_question(question: str, *, filters: Optional[Dict[str, Any]] = None) -> str:
    """Generate SQL text for an arbitrary natural language question without execution."""
    if not question or not question.strip():
        raise ValueError("Question is required to generate SQL.")

    if is_greeting(question):
        return "SELECT 'Hi there! Ask me about occupancy, revenue, or reviews :)' AS message"

    prompt = build_prompt(question, filters=filters)
    sql_text, model_used = _stream_sql(prompt)
    sql_text = sql_text.strip()
    if not sql_text.lower().startswith("select"):
        raise RuntimeError("Generated SQL must start with SELECT.")

    NL2SQL_LOGGER.info("[NL2SQL] Generated SQL via %s", model_used)
    return sql_text



def plan_to_sql_llm(state: GraphState) -> GraphState:
    """Generate and execute schema-aware SQL for the provided state."""
    user_query = state.query or state.raw_input.get("query") or ""
    state.telemetry = state.telemetry or {}
    state.extras = state.extras or {}
    state.raw_input = state.raw_input or {}
    state.plan = state.plan or {}
    state.sql = state.sql or {}

    if (state.plan or {}).get("mode") == "chat":
        state.sql = {
            "text": "",
            "df": None,
            "rows": [],
            "columns": [],
            "summary": "",
            "markdown_table": "",
        }
        return state

    if not user_query.strip():
        state.sql = {
            "text": "",
            "df": pd.DataFrame(),
            "rows": [],
            "columns": [],
            "summary": "No query provided.",
            "markdown_table": "",
        }
        return state

    if is_greeting(user_query):
        state.sql = {
            "text": "",
            "df": pd.DataFrame(),
            "rows": [],
            "columns": [],
            "summary": _GREETING_RESPONSE,
            "markdown_table": _GREETING_RESPONSE,
        }
        state.telemetry["nl2sql_greeting"] = True
        return state

    prompt = build_prompt(user_query, filters=state.filters)
    stream_handler = (state.raw_input or {}).get("stream_handler")

    try:
        sql_text, model_used = _stream_sql(prompt, stream_handler=stream_handler)
    except Exception as exc:
        _LOGGER.error("NL2SQL generation failed: %s", exc)
        state.sql = {
            "text": "",
            "df": pd.DataFrame(),
            "rows": [],
            "columns": [],
            "summary": "Unable to generate SQL for this question.",
            "markdown_table": f"Generation error: {exc}",
            "error": str(exc),
        }
        state.telemetry["nl2sql_error"] = str(exc)
        return state

    sql_text = sql_text.strip()
    if not sql_text.lower().startswith("select"):
        message = "Sorry, that doesn't look like a valid SQL query."
        state.sql = {
            "text": "",
            "df": pd.DataFrame(),
            "rows": [],
            "columns": [],
            "summary": message,
            "markdown_table": message,
            "error": "non_select_sql",
        }
        state.telemetry["nl2sql_error"] = "non_select_sql"
        return state

    state.model_used = model_used
    state.telemetry["nl2sql_model"] = model_used
    state.telemetry["nl2sql_engine"] = "direct_llm"

    try:
        _LOGGER.info("[NL2SQL] Executing SQL:\n%s", sql_text)
        sql_result = execute_duckdb(sql_text)
    except Exception as exc:
        _LOGGER.error("DuckDB execution failed: %s", exc)
        sql_result = {
            "df": pd.DataFrame(),
            "rows": [],
            "columns": [],
            "summary": "SQL execution failed. Please refine the question.",
            "markdown_table": f"Execution error: {exc}",
            "error": str(exc),
        }

    bundle = {
        "text": sql_text,
        "query": sql_text,
        "df": sql_result.get("df", pd.DataFrame()),
        "rows": sql_result.get("rows", []),
        "columns": sql_result.get("columns", []),
        "summary": sql_result.get("summary"),
        "markdown_table": sql_result.get("markdown_table"),
        "table": state.plan.get("sql_table") if state.plan else "listings_cleaned",
        "params": [],
        "explain": None,
    }
    if "error" in sql_result:
        bundle["error"] = sql_result["error"]
    else:
        summary_text = format_sql_summary(user_query, bundle)
        summary_text = _strip_sql_blocks(summary_text)
        policy_label = str(
            (state.plan or {}).get("policy")
            or (state.telemetry or {}).get("policy")
            or ""
        ).upper()
        has_structured_rows = bool(sql_result.get("rows"))
        if not has_structured_rows:
            tables = sql_result.get("tables")
            if isinstance(tables, list) and tables:
                has_structured_rows = True
        if policy_label.startswith("SQL_"):
            summary_text = summarise_table_overview(summary_text, bundle)
            if has_structured_rows and bundle.get("markdown_table"):
                bundle["markdown_table"] = ""
        bundle["summary"] = summary_text

    bundle.setdefault("rows", sql_result.get("rows", []))
    bundle.setdefault("columns", sql_result.get("columns", []))
    bundle.setdefault("markdown_table", sql_result.get("markdown_table", ""))

    state.sql = bundle
    _LOGGER.info(
        "[SQL_STATE_PATCH] Preserved %d rows and %d columns in graph state.",
        len(state.sql.get("rows", [])),
        len(state.sql.get("columns", [])),
    )

    state.extras.setdefault("applied_filters", dict(state.filters))
    state.telemetry.setdefault("policy", (state.plan or {}).get("policy", "SQL_AUTO"))

    summary_text = (state.sql.get("summary") or "")
    markdown_table_text = (state.sql.get("markdown_table") or "")
    sql_text = (state.sql.get("text") or "")
    if "select" in summary_text.lower():
        _LOGGER.warning("[LEAK_DETECTOR][NL2SQL] SQL found inside summary:\n%s", summary_text[:200])
    elif "select" in markdown_table_text.lower():
        _LOGGER.warning("[LEAK_DETECTOR][NL2SQL] SQL found inside markdown_table:\n%s", markdown_table_text[:200])
    elif "select" in sql_text.lower():
        _LOGGER.info("[LEAK_DETECTOR][NL2SQL] SQL present only in state.sql.text (expected).")

    add_thinking_step(
        state,
        phase="sql_query",
        title="Analysed structured metrics",
        detail=(summary_text or "Generated SQL preview for the current question."),
        meta={
            "rows": len(state.sql.get("rows", []) or []),
            "columns": len(state.sql.get("columns", []) or []),
            "table": state.sql.get("table"),
            "model": state.telemetry.get("nl2sql_model"),
        },
    )

    return state


__all__ = [
    "plan_to_sql_llm",
    "generate_sql_from_question",
    "build_prompt",
    "is_greeting",
    "_get_client",
    "_stream_sql",
    "execute_duckdb",
    "_rows_preview",
    "_summarize_result",
    "clean_sql_before_execute",
    "format_sql_summary",
]

# [LEAK_DETECTOR]: Added logging to monitor SQL text leakage.
