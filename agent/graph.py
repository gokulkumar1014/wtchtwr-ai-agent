"""Stateful LangGraph orchestration for the HOPE Agent."""
from __future__ import annotations

# Ensure LangGraph is up-to-date if compatible
try:  # pragma: no cover - best-effort upgrade for older deployments
    import langgraph  # type: ignore
    import subprocess

    if not hasattr(langgraph.StateGraph, "add_memory"):
        subprocess.run(["pip", "install", "-U", "langgraph"], check=False)
except Exception:
    pass

import asyncio
import logging
import re
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict, List, Optional

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

try:
    from langgraph.checkpoint.memory import MemorySaver
except Exception:  # pragma: no cover - fallback when LangGraph is unavailable
    class MemorySaver:  # type: ignore[override]
        def __init__(self) -> None:
            self._store: Dict[str, Dict[str, Any]] = {}

        def get(self, thread_id: str) -> Dict[str, Any]:
            return dict(self._store.get(thread_id, {}))

        def put(self, thread_id: str, memory: Dict[str, Any]) -> None:
            self._store[thread_id] = dict(memory or {})

from langgraph.graph import END, StateGraph

from .compose import (
    ComposeError,
    build_composer_input,
    compose_answer,
    critic_guard,
    fallback_text,
    _build_sentiment_summary_block,
    _render_expansion_sources,
)
from .config import load_config
from .guards import egress as _legacy_egress
from .guards import guardrails as _legacy_guardrails
from .guards import ingress as _legacy_ingress
from .intents import classify_intent as _legacy_classify_intent
from .nl2sql_llm import plan_to_sql_llm
from .policy import plan_steps as _legacy_plan_steps
from .policy import resolve_entities as _legacy_resolve_entities
from .policy import choose_path
from .expansion_scout import exec_expansion_scout
from .portfolio_triage import run_portfolio_triage
from .types import GraphState, State, add_thinking_step, graph_to_state, state_to_graph
from .utils.cleaners import normalize_rag_text
from .vector_qdrant import exec_rag as _legacy_exec_rag
from .vector_qdrant import summarize_hits as _legacy_summarize_hits

_LOGGER = logging.getLogger(__name__)
try:
    if getattr(load_config(), "debug", False):
        _LOGGER.setLevel(logging.DEBUG)
    else:
        _LOGGER.setLevel(logging.WARNING)
except Exception:
    _LOGGER.setLevel(logging.WARNING)

_memory_store = MemorySaver()

_CONVERSATION_TRACE_LIMIT = 12


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

_NEGATIVE_REVIEW_CLAIMS = (
    "no reviews",
    "no available reviews",
    "no relevant reviews",
    "no review data",
    "no review snippets",
    "no guest reviews",
    "no guest feedback",
)

_SQL_HINTS = (
    "price",
    "count",
    "average",
    "list",
    "show",
    "sql",
    "neighbourhood",
    "neighborhood",
)
_RAG_HINTS = ("review", "feedback", "comments", "guest", "opinion", "mention")

def classify_intent(user_query: str) -> str:
    """Return a coarse routing hint for NL2SQL vs chat handling."""
    if not user_query:
        return "chat"
    query = str(user_query).strip().lower()
    if not query:
        return "chat"
    has_sql = any(token in query for token in _SQL_HINTS)
    has_rag = any(token in query for token in _RAG_HINTS)
    if has_sql and has_rag:
        return "hybrid"
    if has_sql:
        return "nl2sql"
    if has_rag:
        return "rag"
    return "chat"


def _needs_rag_fallback(answer: str, rag_snippets: List[Dict[str, Any]]) -> bool:
    """Detect when the LLM claims no reviews despite having RAG hits."""
    if not rag_snippets:
        return False
    text = (answer or "").strip().lower()
    if not text:
        return True
    return any(phrase in text for phrase in _NEGATIVE_REVIEW_CLAIMS)


def _sanitize_memory(memory: Dict[str, Any]) -> Dict[str, Any]:
    """Convert DataFrames and other unserializable values before persistence."""
    safe: Dict[str, Any] = {}
    for key, value in (memory or {}).items():
        if isinstance(value, pd.DataFrame):
            try:
                safe[key] = value.to_markdown(index=False)
            except Exception:
                safe[key] = value.to_string()
        elif isinstance(value, (dict, list, str, int, float, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


def _format_rag_snippet(hit: Dict[str, Any]) -> str:
    """Render one RAG hit into a readable fallback string."""
    listing_id = hit.get("listing_id")
    listing_label = f"Listing {listing_id}" if listing_id else "Listing"
    borough = hit.get("borough") or hit.get("neighbourhood_group") or ""
    month = hit.get("month") or ""
    year = hit.get("year") or ""
    snippet = (hit.get("snippet") or "").strip()

    parts = []
    if borough:
        parts.append(str(borough))
    if month or year:
        parts.append(" ".join([str(m) for m in (month, year) if m]))
    loc = " | ".join(parts)
    return f"- {listing_label} ({loc}): {snippet}" if loc else f"- {listing_label}: {snippet}"


def _rag_fallback_answer(rag_snippets: List[Dict[str, Any]], summary: Optional[str]) -> str:
    """Compose deterministic fallback answer using RAG hits."""
    if not rag_snippets:
        return ""
    lines: List[str] = []
    if summary:
        lines.append(summary.strip())
    lines.append("Representative review feedback:")
    for hit in rag_snippets[:5]:
        lines.append(_format_rag_snippet(hit))
    if len(rag_snippets) > 5:
        lines.append(f"...and {len(rag_snippets) - 5} more snippet(s).")
    return "\n\n".join(lines).strip()


def _merge_states(original: GraphState, updated: GraphState) -> GraphState:
    """Merge two GraphState instances, preserving both SQL and RAG data."""
    merged = GraphState(
        query=updated.query or original.query,
        tenant=updated.tenant or original.tenant,
        intent=updated.intent or original.intent,
        scope=updated.scope or original.scope,
        filters=updated.filters or original.filters,
        plan=updated.plan or original.plan,
        sql=updated.sql or original.sql or {},
        rag=updated.rag or original.rag or {},
        result_bundle=updated.result_bundle or original.result_bundle,
        telemetry={**(original.telemetry or {}), **(updated.telemetry or {})},
        memory={**(original.memory or {}), **(updated.memory or {})},
        history=updated.history or original.history,
        model_used=updated.model_used or original.model_used,
        timestamp=updated.timestamp or original.timestamp,
        start_time=updated.start_time or original.start_time,
        raw_input={**(original.raw_input or {}), **(updated.raw_input or {})},
        guardrail_blocked=updated.guardrail_blocked or original.guardrail_blocked,
        extras={**(original.extras or {}), **(updated.extras or {})},
    )
    merged.debug_thinking = updated.debug_thinking or original.debug_thinking
    merged_thinking: List[Any] = []
    if original.thinking:
        merged_thinking.extend(original.thinking)
    if updated.thinking:
        merged_thinking.extend(updated.thinking)
    merged.thinking = merged_thinking

    sql_rows_original = len((original.sql or {}).get("rows", [])) if isinstance(original.sql, dict) else 0
    sql_rows_updated = len((updated.sql or {}).get("rows", [])) if isinstance(updated.sql, dict) else 0
    fused_rows = len((merged.sql or {}).get("rows", [])) if isinstance(merged.sql, dict) else 0
    if sql_rows_original and not fused_rows:
        merged.sql["rows"] = (original.sql or {}).get("rows", [])
        merged.sql["columns"] = (original.sql or {}).get("columns", [])
        merged.sql["markdown_table"] = (original.sql or {}).get("markdown_table", "")
        _LOGGER.warning("[MERGE_PATCH] Preserved SQL rows from original state")
    elif sql_rows_updated and not fused_rows:
        merged.sql["rows"] = (updated.sql or {}).get("rows", [])
        merged.sql["columns"] = (updated.sql or {}).get("columns", [])
        merged.sql["markdown_table"] = (updated.sql or {}).get("markdown_table", "")
        _LOGGER.warning("[MERGE_PATCH] Preserved SQL rows from updated state")
    try:
        sql_rows = len(merged.sql.get("rows", [])) if merged.sql else 0
        rag_hits = len(merged.rag.get("hits", [])) if merged.rag else 0
        _LOGGER.info(f"[HYBRID] merged {sql_rows} SQL rows and {rag_hits} review hits.")
    except Exception:
        pass
    return merged


def _format_metric_value(value: Any, key_hint: str) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, int):
        formatted = f"{value:,}"
    elif isinstance(value, float):
        formatted = f"{value:,.2f}".rstrip("0").rstrip(".")
    else:
        return str(value)

    lowered = key_hint.lower()
    if any(token in lowered for token in ("price", "revenue", "income", "adr")):
        return f"${formatted}"
    if any(token in lowered for token in ("rate", "percent", "percentage", "share")):
        return f"{formatted}%"
    return formatted


def _summarize_sql_rows_for_hybrid(rows: List[Dict[str, Any]]) -> Optional[str]:
    if not rows:
        return None

    sample = next((row for row in rows if isinstance(row, dict) and row), None)
    if not sample:
        return None

    string_keys = [k for k, v in sample.items() if isinstance(v, str) and v]
    numeric_keys = [k for k, v in sample.items() if isinstance(v, (int, float))]
    if not numeric_keys:
        return None

    def _score_label(key: str) -> tuple[int, str]:
        lowered = key.lower()
        if any(term in lowered for term in ("neighbour", "neighborhood", "borough", "room", "type", "category")):
            return (0, lowered)
        return (1, lowered)

    def _score_numeric(key: str) -> tuple[int, str]:
        lowered = key.lower()
        priority_terms = ("average", "avg", "price", "rate", "revenue", "value", "score", "count")
        if any(term in lowered for term in priority_terms):
            return (0, lowered)
        return (1, lowered)

    string_keys.sort(key=_score_label)
    numeric_keys.sort(key=_score_numeric)

    label_key = string_keys[0] if string_keys else None
    metric_key = numeric_keys[0]

    if not label_key:
        # Fall back to any non-numeric field
        label_key = next((k for k in sample.keys() if k != metric_key), None)

    highlights: List[str] = []
    for row in rows[: min(3, len(rows))]:
        if not isinstance(row, dict):
            continue
        label = str(row.get(label_key, "")).strip() if label_key else ""
        metric_val = row.get(metric_key)
        if label and isinstance(metric_val, (int, float)):
            highlights.append(f"{label} ({_format_metric_value(metric_val, metric_key)})")
    if not highlights:
        return None

    label_readable = label_key.replace("_", " ") if label_key else "entries"
    metric_readable = metric_key.replace("_", " ")
    prefix = f"Top {len(highlights)} {label_readable} by {metric_readable}:"
    return f"{prefix} " + ", ".join(highlights)


def _apply_legacy(state: GraphState, func) -> GraphState:
    """Run a legacy dict-based node and convert back to GraphState."""
    legacy_state = graph_to_state(state)
    updated_dict = func(legacy_state)
    updated_graph = state_to_graph(updated_dict)
    return _merge_states(state, updated_graph)


# ---------------------------------------------------------------------------
# Legacy helper logic reused for composer output
# ---------------------------------------------------------------------------

def _collect_rows(df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    """Extract first 20 rows for UI display."""
    if df is None or getattr(df, "empty", True):
        return []
    return df.head(20).to_dict(orient="records")


def _collect_aggregates(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregates when single numeric row exists."""
    if len(rows) != 1:
        return {}
    return {k: v for k, v in rows[0].items() if isinstance(v, (int, float))}


def _prepare_bundle(legacy_state: State) -> State:
    """Bundle SQL + RAG outputs for composer/UI."""
    plan = legacy_state.get("plan", {}) or {}
    telemetry = legacy_state.setdefault("telemetry", {})
    sql_state = legacy_state.get("sql", {}) if isinstance(legacy_state.get("sql"), dict) else {}
    df: Optional[pd.DataFrame] = sql_state.get("df")

    rows = _collect_rows(df)
    aggregates = _collect_aggregates(rows)
    rag_snippets = list(legacy_state.get("rag_snippets") or [])
    rag_meta = legacy_state.get("rag", {}) or {}

    markdown_table = sql_state.get("markdown_table")

    bundle = {
        "policy": telemetry.get("policy", plan.get("policy", "").upper()),
        "scope": legacy_state.get("scope"),
        "filters": legacy_state.get("filters", {}),
        "applied_filters": legacy_state.get("applied_filters", {}),
        "sql": sql_state.get("text"),
        "sql_table": sql_state.get("table"),
        "sql_params": sql_state.get("params", []),
        "sql_explain": sql_state.get("explain"),
        "columns": list(df.columns) if df is not None else [],
        "rows": rows,
        "aggregates": aggregates,
        "rag_snippets": rag_snippets,
        "summary": rag_meta.get("summary") or sql_state.get("summary"),
        "markdown_table": markdown_table,
    }

    if markdown_table:
        _LOGGER.info("[NL2SQL] Rendered single markdown table for SQL output")

    if not rows and not rag_snippets:
        bundle["rag_hint"] = "No results matched the current filters. Try broadening month/year or location."

    telemetry.update({
        "scope": legacy_state.get("scope"),
        "mode": plan.get("mode", "sql"),
        "rag_hit_count": len(rag_snippets),
    })

    legacy_state["result_bundle"] = bundle
    legacy_state["rows_count"] = len(rows)
    return legacy_state


def _hybrid_compose_fusion(sql_bundle: Dict[str, Any], rag_bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine SQL + RAG bundles into one hybrid bundle for composer.
    Keeps markdown_table, aggregates, and rag_snippets with summary.
    """
    sql_bundle = sql_bundle or {}
    rag_bundle = rag_bundle or {}

    fused = dict(sql_bundle)

    # --- SQL Data Retention for Hybrid Fusion ---
    if sql_bundle:
        for key in ["df", "table", "markdown_table", "rows", "columns", "aggregates"]:
            if key in sql_bundle and key not in fused:
                fused[key] = sql_bundle[key]
        try:
            import pandas as pd
            if not fused.get("rows") and isinstance(sql_bundle.get("df"), pd.DataFrame):
                df = sql_bundle["df"]
                fused["columns"] = list(df.columns)
                fused["rows"] = df.head(20).to_dict(orient="records")
                _LOGGER.warning(f"[HYBRID_DIAG] Injected {len(fused['rows'])} SQL rows from DataFrame.")
        except Exception as inj_exc:
            _LOGGER.warning(f"[HYBRID_DIAG] Failed to inject SQL preview: {inj_exc}")

    rag_snippets = (
        rag_bundle.get("rag_snippets")
        or rag_bundle.get("snippets")
        or rag_bundle.get("hits")
        or fused.get("rag_snippets")
        or []
    )
    fused["rag_snippets"] = rag_snippets

    sql_summary = (fused.get("summary") or sql_bundle.get("summary") or "").strip()
    rag_summary = (rag_bundle.get("summary") or "").strip()

    if sql_summary and rag_summary:
        summary = f"{sql_summary}\n\nReview insights: {rag_summary}".strip()
    elif rag_summary:
        summary = rag_summary
    else:
        summary = sql_summary

    fused.setdefault("markdown_table", sql_bundle.get("markdown_table"))
    fused.setdefault("aggregates", sql_bundle.get("aggregates"))
    fused.setdefault("sql_table", sql_bundle.get("sql_table"))
    fused["summary"] = summary
    fused["policy"] = "SQL_RAG_FUSED"
    return fused


def _compose_legacy(legacy_state: State) -> State:
    """Generate final conversational answer using legacy composer logic."""
    cfg = load_config()
    bundle = legacy_state.get("result_bundle", {})
    query = legacy_state.get("query", "")
    history = legacy_state.get("history", [])
    rag_snippets = bundle.get("rag_snippets", [])
    rag_summary = bundle.get("summary")
    sql_text = bundle.get("sql")
    rows = bundle.get("rows", [])
    aggregates = bundle.get("aggregates", {})
    policy = bundle.get("policy", legacy_state.get("telemetry", {}).get("policy", ""))
    filters = bundle.get("filters", {})
    expansion_report = legacy_state.get("expansion_report") or bundle.get("expansion_report")
    expansion_sources = legacy_state.get("expansion_sources") or bundle.get("expansion_sources") or []
    if expansion_report:
        bundle.setdefault("expansion_report", expansion_report)
    if expansion_sources:
        bundle.setdefault("expansion_sources", expansion_sources)

    input_meta = legacy_state.get("_input", {}) or {}
    stream_handler = input_meta.get("stream_handler")
    composer_enabled = input_meta.get("composer_enabled")
    if composer_enabled is None:
        composer_enabled = cfg.stream_composer and bool(cfg.openai_api_key)

    answer_text = ""
    usage: Dict[str, Any] = {}
    start = time.perf_counter()
    try:
        if composer_enabled:
            try:
                sql_context = bundle.get("sql")
                if isinstance(sql_context, str) and sql_context.strip().lower().startswith(("select", "with", "insert", "update", "delete")):
                    _LOGGER.info("[CLEANUP] Hiding SQL text from composer prompt.")
                    sql_context = ""

                triage_context = bundle.get("portfolio_triage") or legacy_state.get("extras", {}).get("portfolio_triage")
                messages = build_composer_input(
                    history,
                    query,
                    policy,
                    sql_context,
                    rows,
                    aggregates,
                    rag_snippets,
                    filters,
                    legacy_state.get("intent"),
                    portfolio_triage=triage_context,
                    expansion_report=expansion_report,
                    expansion_sources=expansion_sources,
                )
                answer_text, usage = compose_answer(
                    messages, cfg.openai_model, stream_handler=stream_handler
                )
            except ComposeError as exc:
                _LOGGER.warning("Composer unavailable: %s", exc)
                composer_enabled = False

        if not composer_enabled:
            answer_text = fallback_text(bundle)
            if stream_handler:
                stream_handler(answer_text + "\n")

        if _needs_rag_fallback(answer_text, rag_snippets):
            _LOGGER.info("LLM claimed no reviews despite RAG hits → overriding with fallback answer.")
            answer_text = _rag_fallback_answer(rag_snippets, rag_summary)
            usage = {}
            if stream_handler:
                stream_handler(answer_text + "\n")

        if bundle.get("rag_snippets") and answer_text:
            answer_text = normalize_rag_text(answer_text)

        sentiment_block = _build_sentiment_summary_block(rag_snippets)
        if sentiment_block:
            if answer_text:
                answer_text = f"{answer_text.rstrip()}\n\n{sentiment_block}"
            else:
                answer_text = sentiment_block

        if expansion_sources:
            sources_block = _render_expansion_sources(expansion_sources)
            if sources_block and "Web Sources Used" not in (answer_text or ""):
                answer_text = f"{answer_text.rstrip()}\n\n{sources_block}".strip()

        if isinstance(answer_text, str) and re.search(r"(?i)\bselect\b", answer_text):
            _LOGGER.warning(
                "[LEAK_DETECTOR][GRAPH_COMPOSE] SQL text present inside composed final output:\n%s",
                answer_text[:200],
            )

        warning = critic_guard(answer_text, legacy_state.get("rows_count", 0), aggregates)

        legacy_state["answer_text"] = answer_text
        legacy_state["answer_usage"] = usage
        if warning:
            legacy_state["answer_warning"] = warning

        telemetry = legacy_state.setdefault("telemetry", {})
        telemetry["model"] = cfg.openai_model
        if usage:
            telemetry["tokens"] = usage

        return legacy_state
    finally:
        legacy_state.setdefault("telemetry", {})["compose_latency_s"] = round(
            time.perf_counter() - start, 2
        )


# ---------------------------------------------------------------------------
# LangGraph node implementations
# ---------------------------------------------------------------------------

def _ingress_node(state: GraphState) -> GraphState:
    if len((state.query or "").split()) > 100:
        state.guardrail_blocked = True
        state.extras = dict(state.extras or {})
        state.extras["guard_reason"] = "query too long"
        return state
    raw = {**state.raw_input}
    if "query" not in raw:
        raw["query"] = state.query
    if "tenant" not in raw and state.tenant is not None:
        raw["tenant"] = state.tenant
    if "debug_thinking" not in raw:
        raw["debug_thinking"] = state.debug_thinking
    history = raw.get("history") or state.history or []
    user_filters = raw.get("user_filters")

    legacy_state = _legacy_ingress(raw.get("query", ""), raw.get("tenant"), user_filters)
    legacy_state["_input"] = raw
    legacy_state["history"] = history
    legacy_state["memory"] = {**legacy_state.get("memory", {}), **state.memory}
    legacy_state["telemetry"] = {**legacy_state.get("telemetry", {}), **state.telemetry}
    legacy_state["debug_thinking"] = bool(raw.get("debug_thinking"))
    legacy_state.setdefault("thinking", [])

    graph_state = state_to_graph(legacy_state)
    graph_state.history = history
    graph_state.raw_input = raw
    graph_state.tenant = raw.get("tenant")
    graph_state.start_time = legacy_state.get("_start_time", time.perf_counter())
    graph_state.debug_thinking = bool(raw.get("debug_thinking"))
    return _merge_states(state, graph_state)


def _guardrails_node(state: GraphState) -> GraphState:
    next_state = _apply_legacy(state, _legacy_guardrails)
    next_state.guardrail_blocked = bool(next_state.guardrail_blocked)
    return next_state


def _classify_intent_node(state: GraphState) -> GraphState:
    next_state = _apply_legacy(state, _legacy_classify_intent)
    hint = classify_intent(next_state.query or state.query or "")
    next_state.telemetry = next_state.telemetry or {}
    next_state.telemetry["intent_hint"] = hint
    if not next_state.intent and hint == "nl2sql":
        next_state.intent = "FACT_SQL"
    if next_state.intent:
        next_state.memory["last_intent"] = next_state.intent
    if next_state.scope:
        next_state.memory["last_scope"] = next_state.scope
    return next_state


def _resolve_entities_node(state: GraphState) -> GraphState:
    next_state = _apply_legacy(state, _legacy_resolve_entities)
    next_state.memory["last_filters"] = next_state.filters
    return next_state


def _plan_steps_node(state: GraphState) -> GraphState:
    next_state = _apply_legacy(state, _legacy_plan_steps)
    next_state.memory["last_plan"] = next_state.plan
    return next_state


def _plan_to_sql_node(state: GraphState) -> GraphState:
    next_state = plan_to_sql_llm(state)
    next_state.memory["last_sql"] = {
        "query": next_state.sql.get("text") or next_state.sql.get("query"),
        "summary": next_state.sql.get("summary"),
    }
    return next_state


def _exec_rag_node(state: GraphState) -> GraphState:
    legacy_state = graph_to_state(state)
    legacy_state["rag_needed"] = True
    after_exec = _legacy_exec_rag(legacy_state)
    summarized = _legacy_summarize_hits(after_exec)
    next_state = state_to_graph(summarized)
    rag_hits = next_state.extras.get("rag_snippets") or summarized.get("rag_snippets") or []
    next_state.memory["last_rag"] = {
        "summary": next_state.rag.get("summary"),
        "hits": rag_hits[:5],
    }
    return _merge_states(state, next_state)


def _expansion_scout_node(state: GraphState) -> GraphState:
    legacy_state = graph_to_state(state)
    updated = exec_expansion_scout(legacy_state)
    next_state = state_to_graph(updated)
    return _merge_states(state, next_state)


def _hybrid_fusion_node(state: GraphState) -> GraphState:
    plan_mode = str((state.plan or {}).get("mode") or "").lower()
    intent_upper = str(state.intent or "").upper()
    hybrid_requested = plan_mode == "hybrid" or intent_upper in {"FACT_SQL_RAG", "HYBRID"}

    if not hybrid_requested:
        if plan_mode == "rag" or intent_upper == "REVIEWS_RAG":
            rag_input = state_to_graph(graph_to_state(state))
            rag_state = _exec_rag_node(rag_input)
            return _merge_states(state, rag_state)

        sql_input = state_to_graph(graph_to_state(state))
        sql_state = _plan_to_sql_node(sql_input)
        return _merge_states(state, sql_state)

    total_start = time.perf_counter()

    def _run_sql() -> tuple[GraphState, float]:
        sql_input = state_to_graph(graph_to_state(state))
        start = time.perf_counter()
        result = _plan_to_sql_node(sql_input)
        return result, time.perf_counter() - start

    def _run_rag() -> tuple[GraphState, float]:
        rag_input = state_to_graph(graph_to_state(state))
        start = time.perf_counter()
        result = _exec_rag_node(rag_input)
        return result, time.perf_counter() - start

    with ThreadPoolExecutor(max_workers=2) as executor:
        sql_future = executor.submit(_run_sql)
        rag_future = executor.submit(_run_rag)
        sql_state, sql_latency = sql_future.result()
        rag_state, rag_latency = rag_future.result()

    merged = _merge_states(state, sql_state)
    merged = _merge_states(merged, rag_state)

    merged_sql = merged.sql or {}
    merged_rag = merged.rag or {}

    sql_df = merged_sql.get("df")
    sql_rows_count = 0
    if hasattr(sql_df, "shape"):
        try:
            sql_rows_count = int(getattr(sql_df, "shape")[0])
        except Exception:
            sql_rows_count = 0
    if not sql_rows_count:
        sql_rows = merged_sql.get("rows")
        if isinstance(sql_rows, list):
            sql_rows_count = len(sql_rows)

    rag_hits = merged.extras.get("rag_snippets") if merged.extras else None
    if not rag_hits:
        rag_hits = merged_rag.get("rag_snippets") or merged_rag.get("hits") or merged_rag.get("snippets")
    if not rag_hits:
        rag_hits = graph_to_state(rag_state).get("rag_snippets")
    if not isinstance(rag_hits, list):
        rag_hits = list(rag_hits) if rag_hits else []

    total_elapsed = time.perf_counter() - total_start

    telemetry = merged.telemetry or {}
    merged.telemetry = telemetry
    telemetry["hybrid_sql_latency_s"] = round(sql_latency, 2)
    telemetry["hybrid_rag_latency_s"] = round(rag_latency, 2)
    telemetry["hybrid_fused"] = True
    telemetry["fusion_stage"] = "sql_rag_fused"
    telemetry["fusion_sql_rows"] = sql_rows_count
    telemetry["fusion_rag_hits"] = len(rag_hits)

    result_bundle = dict(merged.result_bundle or {})
    sql_markdown = merged_sql.get("markdown_table")
    if sql_markdown:
        result_bundle["markdown_table"] = sql_markdown
    sql_summary = merged_sql.get("summary")
    if sql_summary and not result_bundle.get("summary"):
        result_bundle["summary"] = sql_summary
    if sql_summary:
        result_bundle.setdefault("sql_summary", sql_summary)
    if rag_hits:
        result_bundle["rag_snippets"] = rag_hits
    rag_summary = merged_rag.get("summary")
    if rag_summary:
        result_bundle["rag_summary"] = rag_summary
        result_bundle.setdefault("summary", rag_summary)

    bundle_rows = result_bundle.get("rows")
    if isinstance(bundle_rows, list) and not sql_rows_count:
        sql_rows_count = len(bundle_rows)
        telemetry["fusion_sql_rows"] = sql_rows_count

    merged.result_bundle = result_bundle

    _LOGGER.info(
        "[HYBRID] Merged SQL (%d rows) with RAG (%d hits) in %.2fs",
        sql_rows_count,
        len(rag_hits),
        total_elapsed,
    )

    return merged


def _portfolio_triage_node(state: GraphState) -> GraphState:
    next_state = run_portfolio_triage(state)
    return next_state


def _compose_node(state: GraphState) -> GraphState:
    legacy_state = graph_to_state(state)
    plan_mode = str((legacy_state.get("plan") or {}).get("mode") or "").lower()
    intent_upper = str(legacy_state.get("intent") or "").upper()

    triage_context = (
        (state.result_bundle or {}).get("portfolio_triage")
        or (state.extras or {}).get("portfolio_triage")
        or legacy_state.get("portfolio_triage")
    )

    legacy_state = _prepare_bundle(legacy_state)
    if triage_context:
        bundle_ref = legacy_state.setdefault("result_bundle", {})
        bundle_ref["portfolio_triage"] = triage_context
        extras_ref = legacy_state.setdefault("extras", {})
        extras_ref["portfolio_triage"] = triage_context

    conversational_intents = {"GREETING", "THANKS", "SMALLTALK"}
    if intent_upper in conversational_intents or plan_mode == "chat":
        conversational_responses = {
            "GREETING": "👋 Hi there! I’m wtchtwr - your Airbnb data companion. You can ask me about prices, occupancy, revenue, or guest reviews!",
            "THANKS": "😊 Always happy to help! Anything else you’d like to explore?",
            "SMALLTALK": "I’m wtchtwr - an AI analytics companion built to uncover insights from Airbnb data. Try asking something like 'Average price in Brooklyn' or 'Reviews about cleanliness in Manhattan'.",
        }
        answer_text = conversational_responses.get(
            intent_upper,
            "I’m wtchtwr - your Airbnb analytics companion. Let me know what metrics or reviews you’d like to explore!",
        )
        plan_ref = legacy_state.setdefault("plan", {})
        plan_ref.setdefault("mode", "chat")
        plan_ref.setdefault("policy", "CONVERSATION")
        legacy_state["answer_text"] = answer_text
        legacy_state["answer_usage"] = {}
        legacy_state["result_bundle"] = {
            "policy": "CONVERSATION",
            "scope": "General",
            "filters": legacy_state.get("filters", {}),
            "rag_snippets": [],
            "rows": [],
            "summary": answer_text,
        }
        telemetry = legacy_state.setdefault("telemetry", {})
        telemetry["mode"] = "chat"
        telemetry.setdefault("policy", "CONVERSATION")

        next_state = state_to_graph(legacy_state)
        if answer_text:
            next_state.memory["last_answer"] = answer_text
            next_state.memory["answer_summary"] = answer_text[:280]
            next_state.memory["last_summary"] = answer_text
        return _merge_states(state, next_state)

    # --- HYBRID SQL INJECTION PATCH (DataFrame Recovery) ---
    try:
        sql_data = legacy_state.get("sql", {}) or {}
        bundle = legacy_state.setdefault("result_bundle", {})
        _LOGGER.warning(f"[HYBRID_DIAG] sql_data keys={list(sql_data.keys())}")

        # Attempt to extract dataframe rows if explicit rows are missing
        df = sql_data.get("df")
        rows_from_df, columns_from_df = [], []
        if df is not None:
            try:
                import pandas as pd, io
                if isinstance(df, pd.DataFrame):
                    columns_from_df = list(df.columns)
                    rows_from_df = df.head(20).to_dict(orient="records")
                    _LOGGER.warning(f"[HYBRID_DIAG] Extracted {len(rows_from_df)} rows and {len(columns_from_df)} columns from DataFrame.")
                elif isinstance(df, str):
                    # Attempt to parse markdown/CSV text back into a DataFrame
                    try:
                        if "|" in df:  # Markdown-style table
                            df_clean = "\n".join([line for line in df.splitlines() if line.strip()])
                            parsed_df = pd.read_csv(io.StringIO(df_clean), sep="|", engine="python")
                            parsed_df = parsed_df.loc[:, ~parsed_df.columns.str.contains("^Unnamed")]
                            columns_from_df = list(parsed_df.columns)
                            rows_from_df = parsed_df.head(20).to_dict(orient="records")
                            _LOGGER.warning(f"[HYBRID_DIAG] Parsed markdown/pipe table with {len(rows_from_df)} rows.")
                        else:  # Generic CSV or JSON text
                            parsed_df = pd.read_json(io.StringIO(df)) if df.strip().startswith("[") else pd.read_csv(io.StringIO(df))
                            columns_from_df = list(parsed_df.columns)
                            rows_from_df = parsed_df.head(20).to_dict(orient="records")
                            _LOGGER.warning(f"[HYBRID_DIAG] Parsed text-based DataFrame with {len(rows_from_df)} rows.")
                    except Exception as parse_exc:
                        _LOGGER.warning(f"[HYBRID_DIAG] Could not parse stringified df: {parse_exc}")
            except Exception as df_exc:
                _LOGGER.warning(f"[HYBRID] Failed to interpret SQL df: {df_exc}")

        # Inject rows/columns/aggregates if missing
        if "rows" not in bundle or not bundle.get("rows"):
            bundle["rows"] = sql_data.get("rows") or rows_from_df
        if "columns" not in bundle or not bundle.get("columns"):
            bundle["columns"] = sql_data.get("columns") or columns_from_df
        if "aggregates" not in bundle or not bundle.get("aggregates"):
            bundle["aggregates"] = sql_data.get("aggregates") or {}
        if "markdown_table" not in bundle or not bundle.get("markdown_table"):
            bundle["markdown_table"] = sql_data.get("table") or sql_data.get("markdown_table") or ""
        legacy_state["result_bundle"] = bundle

        if bundle.get("rows"):
            _LOGGER.warning(f"[HYBRID_DIAG] ✅ Injected {len(bundle['rows'])} SQL rows into result_bundle.")
        else:
            _LOGGER.warning("[HYBRID_DIAG] ⚠️ SQL injection patch failed — no rows recovered.")

    except Exception as inj_exc:
        _LOGGER.warning(f"[HYBRID] SQL injection patch failed: {inj_exc}")

    # --- HYBRID DIAGNOSTIC: SQL TYPE & CONTENT INSPECTION ---
    try:
        sql_obj = legacy_state.get("sql", {})
        df_obj = sql_obj.get("df")
        df_type = type(df_obj).__name__
        df_shape = getattr(df_obj, "shape", None)
        rag_obj = legacy_state.get("rag", {})
        rag_keys = list(rag_obj.keys()) if isinstance(rag_obj, dict) else type(rag_obj).__name__
        _LOGGER.warning(f"[DEBUG_SQL_INSPECT] sql.keys={list(sql_obj.keys())}")
        _LOGGER.warning(f"[DEBUG_SQL_INSPECT] df.type={df_type} df.shape={df_shape}")
        _LOGGER.warning(f"[DEBUG_SQL_INSPECT] rag.keys={rag_keys}")
    except Exception as dbg_exc:
        _LOGGER.error(f"[DEBUG_SQL_INSPECT] ❌ Failed to inspect SQL payload: {dbg_exc}")

    # --- HYBRID DIAGNOSTIC LOG ---
    try:
        bundle = legacy_state.get("result_bundle", {})
        sql_rows = len(bundle.get("rows") or [])
        rag_hits = len(bundle.get("rag_snippets") or [])
        summary = bundle.get("summary", "") or ""
        _LOGGER.warning(
            f"[HYBRID_DIAG] SQL rows={sql_rows} | RAG hits={rag_hits} | "
            f"summary_snippet_present={bool(summary)}"
        )
        if sql_rows and rag_hits:
            _LOGGER.warning("[HYBRID_DIAG] ✅ Both SQL and RAG detected pre-compose fusion.")
        elif sql_rows and not rag_hits:
            _LOGGER.warning("[HYBRID_DIAG] ⚠️ SQL only — RAG missing.")
        elif rag_hits and not sql_rows:
            _LOGGER.warning("[HYBRID_DIAG] ⚠️ RAG only — SQL missing.")
        else:
            _LOGGER.warning("[HYBRID_DIAG] ❌ Neither SQL nor RAG content detected.")
    except Exception as diag_exc:
        _LOGGER.warning(f"[HYBRID_DIAG] Diagnostic block failed: {diag_exc}")

        if plan_mode == "hybrid" or intent_upper == "FACT_SQL_RAG":
            existing_bundle = legacy_state.get("result_bundle", {}) or {}
            sql_payload = dict(legacy_state.get("sql", {}) or {})
            rag_payload = dict(legacy_state.get("rag", {}) or {})

        if not rag_payload.get("rag_snippets"):
            rag_payload["rag_snippets"] = existing_bundle.get("rag_snippets") or []
        if not sql_payload.get("markdown_table"):
            sql_payload["markdown_table"] = existing_bundle.get("markdown_table")
        if not sql_payload.get("aggregates"):
            sql_payload["aggregates"] = existing_bundle.get("aggregates")
        if not sql_payload.get("summary"):
            sql_payload["summary"] = existing_bundle.get("summary")

        fused_bundle = _hybrid_compose_fusion(sql_payload, rag_payload)
        fused_bundle = {**existing_bundle, **fused_bundle}

        sql_highlights = _summarize_sql_rows_for_hybrid(fused_bundle.get("rows") or [])
        if sql_highlights:
            existing_summary = (fused_bundle.get("summary") or "").strip()
            fused_bundle["summary"] = (
                f"{sql_highlights}\n\n{existing_summary}".strip()
                if existing_summary
                else sql_highlights
            )
            fused_bundle.setdefault("sql_highlights", sql_highlights)

        legacy_state["result_bundle"] = fused_bundle
        telemetry = legacy_state.setdefault("telemetry", {})
        telemetry["fusion_stage"] = "sql_rag_fused"
        _LOGGER.info("[HYBRID] Composer received fused bundle with SQL+RAG")

    # ✅ Ensure markdown table is always present before compose
    import pandas as pd
    bundle = legacy_state.get("result_bundle", {})
    rows = bundle.get("rows", [])
    if len(rows) > 1 and not bundle.get("markdown_table"):
        try:
            df = pd.DataFrame(rows)
            bundle["markdown_table"] = df.to_markdown(index=False)
            _LOGGER.warning("[COMPOSE_PATCH] ✅ Injected markdown table for compose context")
        except Exception as md_exc:
            _LOGGER.warning(f"[COMPOSE_PATCH] Failed to build markdown: {md_exc}")
    legacy_state["result_bundle"] = bundle

    # ✅ Now hand off to composer
    legacy_state = _compose_legacy(legacy_state)
    next_state = state_to_graph(legacy_state)

    answer_text = legacy_state.get("answer_text", "")
    if answer_text:
        next_state.memory["last_answer"] = answer_text
        next_state.memory["answer_summary"] = answer_text[:280]
        next_state.memory["last_summary"] = answer_text
        preview_line = answer_text.splitlines()[0][:160] if isinstance(answer_text, str) else ""
        add_thinking_step(
            next_state,
            phase="compose",
            title="Composed final answer",
            detail=preview_line,
            meta={"mode": plan_mode or intent_upper},
        )

    return _merge_states(state, next_state)



def _egress_node(state: GraphState) -> Dict[str, Any]:
    legacy_state = graph_to_state(state)
    response = _legacy_egress(legacy_state)
    response.setdefault("telemetry", legacy_state.get("telemetry", {}))
    response["state_snapshot"] = legacy_state
    history = legacy_state.get("history") or []
    if history:
        trace: List[Dict[str, str]] = []
        for turn in history[-_CONVERSATION_TRACE_LIMIT:]:
            if not isinstance(turn, dict):
                continue
            role = turn.get("role")
            if role not in {"user", "assistant"}:
                continue
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            trace.append({"role": role, "content": content})
        if trace:
            memory_block = legacy_state.setdefault("memory", {}) or {}
            memory_block["conversation_trace"] = trace
    response["memory"] = _sanitize_memory(legacy_state.get("memory", {}) or {})
    response["sql"] = legacy_state.get("sql", {})
    response["rag"] = legacy_state.get("rag", {})
    response["summary"] = (
        legacy_state.get("sql", {}).get("summary")
        or legacy_state.get("rag", {}).get("summary")
    )
    return response


def _guardrail_route(state: GraphState) -> str:
    return "blocked" if state.guardrail_blocked else "continue"


def _intent_route(state: GraphState) -> str:
    intent = str(state.intent or "").upper()
    return "expansion" if intent == "EXPANSION_SCOUT" else "continue"


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class GraphRunResult:
    """Wrapper for LangGraph outputs with convenient accessors."""

    raw: Dict[str, Any]
    latency_s: float

    @property
    def policy(self) -> Optional[str]:
        return self.raw.get("policy")

    @property
    def telemetry(self) -> Dict[str, Any]:
        return self.raw.get("telemetry") or {}

    @property
    def result_bundle(self) -> Dict[str, Any]:
        return self.raw.get("result_bundle") or {}

    @property
    def sql(self) -> Dict[str, Any]:
        bundle = self.result_bundle
        sql_state = self.raw.get("sql") or {}
        sql_text = sql_state.get("text") or self.raw.get("sql_text") or bundle.get("sql")
        return {
            "text": sql_text,
            "summary": sql_state.get("summary") or bundle.get("summary"),
            "markdown_table": sql_state.get("markdown_table") or bundle.get("markdown_table"),
            "rows": sql_state.get("rows") or bundle.get("rows"),
            "columns": sql_state.get("columns") or bundle.get("columns"),
            "aggregates": sql_state.get("aggregates") or bundle.get("aggregates"),
        }

    @property
    def final_answer(self) -> Optional[str]:
        return self.raw.get("answer_text")

    @property
    def markdown(self) -> Optional[str]:
        return self.sql.get("markdown_table")

    @property
    def state(self) -> Dict[str, Any]:
        return self.raw.get("state_snapshot") or {}

    @property
    def memory(self) -> Dict[str, Any]:
        return self.raw.get("memory") or self.state.get("memory") or {}

    def dict(self) -> Dict[str, Any]:
        """Return a telemetry-enriched copy of the raw response."""
        payload = dict(self.raw)
        telemetry = payload.setdefault("telemetry", {})
        telemetry.setdefault("total_latency_s", self.latency_s)
        telemetry.setdefault("latency_ms", self.latency_s * 1000)
        payload.setdefault("latency", round(self.latency_s, 2))
        return payload


@lru_cache(maxsize=1)
def build_graph() -> StateGraph:
    """Build and compile the LangGraph pipeline."""
    builder = StateGraph(GraphState)
    # Memory initialisation disabled; operate the graph in a stateless mode.

    builder.add_node("ingress", _ingress_node)
    builder.add_node("guardrails", _guardrails_node)
    builder.add_node("classify_intent", _classify_intent_node)
    builder.add_node("expansion_scout", _expansion_scout_node)
    builder.add_node("resolve_entities", _resolve_entities_node)
    builder.add_node("plan_steps", _plan_steps_node)
    builder.add_node("plan_to_sql", _plan_to_sql_node)
    builder.add_node("exec_rag", _exec_rag_node)
    builder.add_node("hybrid_fusion", _hybrid_fusion_node)
    builder.add_node("portfolio_triage", _portfolio_triage_node)
    builder.add_node("compose", _compose_node)
    builder.add_node("egress", _egress_node)

    builder.set_entry_point("ingress")
    builder.add_edge("ingress", "guardrails")
    builder.add_conditional_edges("guardrails", _guardrail_route, {
        "continue": "classify_intent",
        "blocked": "egress",
    })
    builder.add_conditional_edges("classify_intent", _intent_route, {
        "expansion": "expansion_scout",
        "__default__": "resolve_entities",
        "continue": "resolve_entities",
    })
    builder.add_edge("resolve_entities", "plan_steps")
    builder.add_conditional_edges("plan_steps", choose_path, {
        "nl2sql": "plan_to_sql",
        "rag": "exec_rag",
        "hybrid": "hybrid_fusion",
        "portfolio_triage": "portfolio_triage",
        "expansion_scout": "expansion_scout",
        "__default__": "plan_to_sql",
    })
    builder.add_edge("plan_to_sql", "compose")
    builder.add_edge("exec_rag", "compose")
    builder.add_edge("hybrid_fusion", "compose")
    builder.add_edge("portfolio_triage", "compose")
    builder.add_edge("expansion_scout", "compose")
    builder.add_edge("compose", "egress")
    builder.add_edge("egress", END)

    return builder.compile()


def _compiled_graph() -> StateGraph:
    """Backwards-compatible accessor for the cached compiled graph."""
    return build_graph()


def _default_thread_id(raw_input: Dict[str, Any]) -> str:
    return (
        raw_input.get("thread_id")
        or raw_input.get("conversation_id")
        or raw_input.get("tenant")
        or "default"
    )


def get_memory_context(thread_id: str) -> Dict[str, Any]:
    """Return the persisted memory context for a thread."""
    if not thread_id:
        return {}
    try:
        return dict(_memory_store.get(thread_id) or {})
    except Exception:  # pragma: no cover - store failure fallback
        return {}


def save_memory_context(thread_id: str, memory: Dict[str, Any]) -> None:
    """Persist memory context for a thread when available."""
    if not thread_id or memory is None:
        return
    try:
        _memory_store.put(thread_id, dict(memory))
    except Exception:  # pragma: no cover - store failure fallback
        pass


def run_graph(initial_state: GraphState, thread_id: Optional[str] = None) -> GraphRunResult:
    """Execute the LangGraph pipeline starting from the provided state."""
    raw_input = dict(initial_state.raw_input or {})
    raw_input.setdefault("query", initial_state.query)
    if initial_state.tenant is not None:
        raw_input.setdefault("tenant", initial_state.tenant)
    raw_input.setdefault("user_filters", initial_state.filters or {})
    raw_input.setdefault("history", initial_state.history or [])
    raw_input.setdefault("debug_thinking", initial_state.debug_thinking)
    initial_state.raw_input = raw_input
    initial_state.telemetry = initial_state.telemetry or {}
    thread_key = thread_id or _default_thread_id(raw_input)

    existing_memory: Dict[str, Any] = {}
    try:
        loaded_memory = get_memory_context(thread_key)
        if loaded_memory:
            existing_memory.update(loaded_memory)
    except Exception as exc:  # pragma: no cover - defensive logging
        _LOGGER.warning("Unable to load memory context for %s: %s", thread_key, exc)
    if initial_state.memory:
        existing_memory.update(_sanitize_memory(dict(initial_state.memory)))
    existing_memory["last_query"] = initial_state.query
    initial_state.memory = existing_memory

    compiled = build_graph()

    _LOGGER.info(
        "[LangGraph] 🚀 Starting HOPE Agent Graph | tenant=%s | thread=%s",
        initial_state.tenant,
        thread_key,
    )
    start = time.time()
    response = compiled.invoke(
        initial_state,
        config={"configurable": {"thread_id": thread_key}},
    )
    elapsed = time.time() - start
    _LOGGER.info("[LangGraph] ✅ Completed in %.2fs", elapsed)

    if not isinstance(response, dict):
        response = {"answer_text": str(response)}

    telemetry = response.setdefault("telemetry", {})
    telemetry.setdefault("total_latency_s", elapsed)
    telemetry.setdefault("latency_ms", elapsed * 1000)
    response.setdefault("latency", round(elapsed, 2))
    final_memory = response.get("memory") or {}
    final_memory.setdefault("last_query", initial_state.query)
    final_memory = _sanitize_memory(final_memory)

    try:
        save_memory_context(thread_key, final_memory)
    except Exception as exc:  # pragma: no cover - defensive logging
        _LOGGER.warning("Unable to persist memory context for %s: %s", thread_key, exc)

    return GraphRunResult(raw=response, latency_s=elapsed)


def run(
    query: str,
    tenant: Optional[str] = None,
    user_filters: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, str]]] = None,
    stream_handler: Optional[Any] = None,
    composer_enabled: Optional[bool] = None,
    debug_thinking: Optional[bool] = None,
    thread_id: Optional[str] = None,
    stream_scope: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the HOPE Agent LangGraph pipeline."""
    cfg = load_config()
    history = history or []
    if len(history) > cfg.chat_max_turns:
        history = history[-cfg.chat_max_turns:]

    if composer_enabled is None:
        composer_enabled = cfg.stream_composer and bool(cfg.openai_api_key)

    raw_input: Dict[str, Any] = {
        "query": query,
        "tenant": tenant,
        "user_filters": user_filters,
        "history": history,
        "stream_handler": stream_handler,
        "composer_enabled": composer_enabled,
        "thread_id": thread_id,
        "stream_scope": stream_scope,
        "debug_thinking": debug_thinking,
    }

    initial_state = GraphState(
        query=query,
        tenant=tenant,
        filters=user_filters or {},
        history=history,
        telemetry={},
        memory={},
        raw_input=raw_input,
        debug_thinking=bool(debug_thinking),
    )

    result = run_graph(initial_state, thread_id=thread_id)
    return result.dict()


async def stream(
    query: str,
    tenant: Optional[str] = None,
    user_filters: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, str]]] = None,
    composer_enabled: Optional[bool] = None,
    debug_thinking: Optional[bool] = None,
    thread_id: Optional[str] = None,
    stream_scope: Optional[str] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Stream LangGraph tokens followed by the final result."""

    loop = asyncio.get_running_loop()
    token_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    def handler(text: str) -> None:
        if text:
            loop.call_soon_threadsafe(token_queue.put_nowait, ("token", text))

    def worker() -> None:
        try:
            result = run(
                query,
                tenant=tenant,
                user_filters=user_filters,
                history=history,
                stream_handler=handler,
                composer_enabled=composer_enabled,
                debug_thinking=debug_thinking,
                thread_id=thread_id,
                stream_scope=stream_scope,
            )
            loop.call_soon_threadsafe(token_queue.put_nowait, ("final", result))
        except Exception as exc:  # pragma: no cover - propagate to caller
            loop.call_soon_threadsafe(token_queue.put_nowait, ("error", exc))
        finally:
            loop.call_soon_threadsafe(token_queue.put_nowait, ("done", None))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    try:
        while True:
            kind, value = await token_queue.get()
            if kind == "token":
                yield {"type": "token", "payload": value}
            elif kind == "final":
                yield {"type": "final", "payload": value}
            elif kind == "error":
                raise value
            elif kind == "done":
                break
    finally:
        thread.join()


__all__ = [
    "build_graph",
    "_compiled_graph",
    "get_memory_context",
    "save_memory_context",
    "run_graph",
    "GraphRunResult",
    "run",
    "stream",
]

# [LEAK_DETECTOR]: Added logging to monitor SQL text leakage.
# [SQL_LEAK_FIX] Sanitized SQL text from composed assistant response.
