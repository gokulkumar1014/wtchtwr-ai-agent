"""FastAPI backend wiring for the HOPE application.

This service exposes conversation management endpoints for the React client,
bridges chat requests into the LangGraph pipeline, and surfaces lightweight
DuckDB analytics for dashboard/data-explorer usage.
"""
from __future__ import annotations

import csv
import html
import io
import json
import re
import threading
import logging
import math
import time
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
from uuid import uuid4

import duckdb
import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, EmailStr, Field, validator
from .dashboard_router import router as dashboard_router
from backend.data_explorer import EXPLORER, DataExplorerError
from backend.emailer import send_email, is_configured as email_is_configured
from .gdrive import drive_uploader
from . import exporter
from agent.config import load_config
from agent.graph import get_memory_context, run as run_agent, stream as stream_agent
from agent.slack.bot import create_slack_app
from agent.utils.db_utils import get_db_path

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


# ---------------------------------------------------------------------------
# FastAPI init + CORS
# ---------------------------------------------------------------------------

app = FastAPI(title="HOPE Backend", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(dashboard_router, prefix="/api/dashboard", tags=["dashboard"])

logger = logging.getLogger("hope.backend")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

SLACK_PORT = 3000
SLACK_THREAD: Optional[threading.Thread] = None
SLACK_RUNNING = False

RAG_TABLE_NAME = "RAG Sources"
RAG_TABLE_SOURCE = "rag"
RAG_EXPORT_COLUMNS = [
    "snippet",
    "listing_id",
    "neighbourhood",
    "neighbourhood_group",
    "month",
    "year",
    "comment_id",
    "sentiment_label",
    "compound",
    "positive",
    "neutral",
    "negative",
    "is_highbury",
]

SUMMARY_TRACE_LIMIT = 12
SUMMARY_PAIR_LIMIT = 6
CONCISE_SUMMARY_WORD_LIMIT = 200


@app.middleware("http")
async def telemetry_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{duration:.2f}s"
    logger.info("[HTTP] %s %s took %.2fs", request.method, request.url.path, duration)
    return response


# ---------------------------------------------------------------------------
# Pydantic payloads
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    thread_id: Optional[str] = Field(default="default", max_length=200)
    debug_thinking: Optional[bool] = False


class ConversationCreateRequest(BaseModel):
    title: Optional[str] = Field(default=None, max_length=200)


class MessageCreateRequest(BaseModel):
    role: str = Field(..., pattern=r"^(user|assistant)$")
    content: Optional[str] = Field(default=None, max_length=4000)

    @validator("content")
    def _validate_content(cls, value: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        if values.get("role") == "user" and (value is None or not value.strip()):
            raise ValueError("User messages require content.")
        return value


class ConversationSummaryResponse(BaseModel):
    conversation_id: str
    concise: str
    detailed: str
    concise_sections: Optional[List[Dict[str, Any]]] = None
    detailed_topics: Optional[List[Dict[str, Any]]] = None


class ConversationSummaryEmailRequest(BaseModel):
    email: EmailStr
    variant: str = Field("concise", pattern=r"^(concise|detailed)$")


class ExportMessageRequest(BaseModel):
    table_index: int = Field(0, ge=0)
    delivery: str = Field("download", pattern=r"^(download|email)$")
    email: Optional[EmailStr] = None
    email_mode: Optional[str] = Field(default="csv", pattern=r"^(sql|csv|both)$")


class ExportMetadata(BaseModel):
    token: str
    format: str
    rows: int
    expires_at: str
    filename: str
    session_only: bool = False


class ExportActionResponse(BaseModel):
    delivery: str = Field(..., pattern=r"^(download|email)$")
    metadata: Optional[ExportMetadata] = None
    detail: Optional[str] = None


@dataclass
class FileDelivery:
    mode: str
    attachments: List[Tuple[str, bytes, str]]
    link: Optional[str]
    csv_filename: str
    zip_filename: Optional[str] = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

EXPORT_RETENTION = timedelta(hours=1)
MAX_HISTORY_TURNS = 20
CONVERSATION_TTL_DAYS = 30
_EMAIL_ATTACHMENT_LIMIT = 25 * 1024 * 1024  # 25 MB

DB_PATH = get_db_path()
SCHEMA_LOCK = threading.Lock()
_CONNECTION_MODE_LOCK = threading.Lock()
_CONNECTION_MODE: Optional[bool] = None


def utcnow() -> datetime:
    return datetime.now(tz=UTC)


def isoformat(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create conversation-related tables when missing."""
    with SCHEMA_LOCK:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id VARCHAR PRIMARY KEY,
                title VARCHAR,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id VARCHAR PRIMARY KEY,
                conversation_id VARCHAR NOT NULL,
                role VARCHAR NOT NULL,
                content TEXT,
                nl_summary TEXT,
                payload JSON,
                timestamp TIMESTAMP NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS exports_cache (
                token VARCHAR PRIMARY KEY,
                created_at TIMESTAMP,
                filename VARCHAR,
                mime_type VARCHAR,
                payload BLOB
            )
            """
        )
        ensure_thread_map(conn)


def ensure_thread_map(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS thread_map (
            thread_id VARCHAR PRIMARY KEY,
            conversation_id VARCHAR
        )
        """
    )


def open_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection; caller must close.

    DuckDB rejects mixing read/write modes on the same database file. We pin the
    connection mode to the first caller (defaulting to read/write) so subsequent
    connections reuse the same configuration.
    """
    global _CONNECTION_MODE
    with _CONNECTION_MODE_LOCK:
        if _CONNECTION_MODE is None:
            _CONNECTION_MODE = read_only
        elif not read_only and _CONNECTION_MODE:
            _CONNECTION_MODE = False
        read_only = _CONNECTION_MODE or False

    conn = duckdb.connect(str(DB_PATH), read_only=read_only)
    if not read_only:
        ensure_schema(conn)
    return conn


def to_primitive(value: Any) -> Any:
    """Convert DuckDB/numpy/Decimal objects into JSON-serialisable primitives."""
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return isoformat(value)
    try:
        import numpy as np  # type: ignore

        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            numeric = float(value)
            return None if not math.isfinite(numeric) else numeric
    except Exception:
        pass
    return str(value)


def _sanitize_nan(value: Any) -> Any:
    """Replace NaN/Inf with None for JSON-safe responses."""
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _sanitize_nan(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_nan(v) for v in value]
    return value


def _parse_markdown_table(markdown: str) -> Optional[Tuple[List[str], List[Dict[str, Any]]]]:
    """Parse a simple GitHub-flavoured markdown table into columns and rows."""
    if not markdown:
        return None
    lines = [line.strip() for line in markdown.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    header_line = lines[0]
    separator_line = lines[1]
    if "|" not in header_line or "|" not in separator_line:
        return None
    columns = [cell.strip() for cell in header_line.strip("|").split("|")]
    if not columns:
        return None
    rows: List[Dict[str, Any]] = []
    for line in lines[2:]:
        if "|" not in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(columns):
            continue
        row = {columns[idx]: cells[idx] for idx in range(len(columns))}
        rows.append(row)
    if not rows:
        return None
    return columns, rows


def _strip_markdown_tables(text: str) -> str:
    """Remove markdown table blocks from text."""
    if not text:
        return text
    table_pattern = re.compile(r"(?:^\s*\|.*\|\s*\n?){2,}", re.MULTILINE)
    cleaned = table_pattern.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _strip_sql_statements(text: str) -> str:
    """Remove raw SQL statements from LLM summaries."""
    if not text:
        return text

    cleaned = re.sub(r"(?is)```sql.*?```", "", text)
    block_pattern = re.compile(
        r"(?is)^\s*(SELECT|WITH|INSERT|UPDATE|DELETE)\b.*?(?:;|\Z)", re.MULTILINE
    )
    cleaned = block_pattern.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _summarize_sentiment(snippets: Sequence[Any]) -> Optional[Dict[str, Any]]:
    """Aggregate sentiment counts and compound averages from RAG snippets."""
    if not snippets:
        return None

    counts = {"positive": 0, "neutral": 0, "negative": 0}
    compounds: List[float] = []

    for entry in snippets:
        if isinstance(entry, dict):
            label = str(entry.get("sentiment_label") or "").strip().lower()
            if label in counts:
                counts[label] += 1
            try:
                comp = float(entry.get("compound"))
                if math.isfinite(comp):
                    compounds.append(comp)
            except (TypeError, ValueError):
                continue

    total = sum(counts.values())
    if total == 0:
        return None

    dominant_label = max(counts, key=lambda k: counts[k])
    dominant_count = counts[dominant_label]
    tie_counts = [value for value in counts.values() if value == dominant_count]
    if dominant_count == 0 or len(tie_counts) > 1:
        overall = "Mixed"
        dominant_display = None
    else:
        dominant_display = dominant_label.capitalize()
        overall = dominant_display

    avg_strength = sum(compounds) / len(compounds) if compounds else None

    return {
        "positive": counts["positive"],
        "neutral": counts["neutral"],
        "negative": counts["negative"],
        "total_reviews": total,
        "dominant_label": dominant_label if dominant_count else None,
        "dominant_label_display": dominant_display or ("Mixed" if overall == "Mixed" else None),
        "overall_sentiment": overall,
        "average_sentiment_strength": round(avg_strength, 3) if avg_strength is not None else None,
    }


def normalise_payload(data: Any) -> Any:
    """Recursively normalise payload structures to plain Python primitives with concise logging."""
    dict_count = 0
    list_count = 0
    primitive_count = 0

    def _normalise(obj: Any, depth: int) -> Any:
        nonlocal dict_count, list_count, primitive_count
        if isinstance(obj, dict):
            dict_count += 1
            logger.debug("[PERSIST_TRACE][NORMALISE] dict keys=%s", list(obj.keys()))
            payload_obj = obj.get("payload")
            preserve_payload = isinstance(payload_obj, dict)
            if preserve_payload:
                    logger.debug(
                        "[PERSIST_TRACE][NORMALISE] ✅ nested payload keys=%s",
                        list(payload_obj.keys()),
                    )
                    logger.debug(
                        "[PERSIST_FIX][NORMALISE] Preserved nested payload keys=%s",
                        list(payload_obj.keys()),
                    )
            elif depth == 0:
                logger.debug("[PERSIST_TRACE][NORMALISE] ⚠️ No nested payload during normalization.")
            normalized: Dict[str, Any] = {}
            for key, value in obj.items():
                if key == "payload" and preserve_payload:
                    normalized[key] = payload_obj
                else:
                    normalized[key] = _normalise(value, depth + 1)
            return _sanitize_nan(normalized)
        if isinstance(obj, list):
            list_count += 1
            logger.debug("[PERSIST_TRACE][NORMALISE] list len=%d", len(obj))
            normalized_list = [_normalise(item, depth + 1) for item in obj]
            return _sanitize_nan(normalized_list)
        primitive_count += 1
        return _sanitize_nan(to_primitive(obj))

    result = _normalise(data, depth=0)
    logger.debug(
        "[PERSIST_SUMMARY][NORMALISE] %d dicts, %d lists processed, %d primitives skipped",
        dict_count,
        list_count,
        primitive_count,
    )
    return result


def safe_json(data):
    """Convert numpy/pandas objects safely for JSON serialization."""

    def convert(obj):
        if isinstance(obj, (np.generic,)):
            return obj.item()
        if isinstance(obj, (int, float)):
            if isinstance(obj, float) and not math.isfinite(obj):
                return None
            return obj
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, pd.Series):
            return obj.to_list()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    return _sanitize_nan(convert(data))


def _ensure_email_configured() -> None:
    if email_is_configured():
        return
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Email delivery is not configured. Please contact an administrator.",
    )


def _email_http_exception(exc: RuntimeError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Failed to send email: {exc}",
    )


def _sections_to_plaintext(sections: Sequence[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for section in sections:
        title = section.get("title")
        items = section.get("items") or []
        if not title or not items:
            continue
        lines.append(f"{title}:")
        for item in items:
            if item:
                lines.append(f"- {item}")
        lines.append("")
    return lines


def _topics_to_plaintext(topics: Sequence[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for topic in topics:
        title = topic.get("title")
        question_items = topic.get("question_items") or []
        answer_items = topic.get("answer_items") or []
        tables = topic.get("answer_tables") or []
        fallback_table = topic.get("answer_table")
        if not tables and isinstance(fallback_table, dict):
            tables = [fallback_table]
        if title:
            lines.append(str(title))
        question_lines = _structured_plaintext_list(question_items, allow_headings=False)
        if question_lines:
            lines.append("Question:")
            lines.extend(question_lines)
        answer_lines = _structured_plaintext_list(answer_items, allow_headings=True)
        if answer_lines:
            lines.append("Answer:")
            lines.extend(answer_lines)
        for table in tables:
            headers = table.get("headers") or []
            rows = table.get("rows") or []
            if not headers or not rows:
                continue
            header_line = " | ".join(headers)
            lines.append(header_line)
            lines.append("-" * len(header_line))
            for row in rows:
                lines.append(" | ".join(row))
        lines.append("")
    return lines


def _render_sections_html(sections: Sequence[Dict[str, Any]]) -> str:
    cards: List[str] = []
    for section in sections:
        title = section.get("title")
        items = section.get("items") or []
        clean_items = [html.escape(str(item)) for item in items if item]
        if not title or not clean_items:
            continue
        title_html = html.escape(str(title))
        list_html = "".join(f"<li>{item}</li>" for item in clean_items)
        cards.append(
            "<div style='border:1px solid #e2e8f0;border-radius:12px;padding:12px;margin-top:8px;'>"
            f"<div style='font-size:12px;text-transform:uppercase;color:#64748b;letter-spacing:0.08em;'>{title_html}</div>"
            f"<ul style='margin:8px 0 0 18px;padding:0;list-style:disc;color:#0f172a;'>{list_html}</ul>"
            "</div>"
        )
    return "".join(cards)


def _render_topics_html(topics: Sequence[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for topic in topics:
        title = html.escape(topic.get("title") or "")
        question_items = [str(item) for item in topic.get("question_items") or [] if item]
        answer_items = [str(item) for item in topic.get("answer_items") or [] if item]
        answer_tables = topic.get("answer_tables") or []
        fallback_table = topic.get("answer_table")
        if not answer_tables and isinstance(fallback_table, dict):
            answer_tables = [fallback_table]
        table_html = ""
        if answer_tables:
            table_chunks: List[str] = []
            for idx, answer_table in enumerate(answer_tables, start=1):
                headers = [html.escape(str(h)) for h in answer_table.get("headers") or [] if h]
                rows = answer_table.get("rows") or []
                if not headers or not rows:
                    continue
                header_html = "".join(
                    f"<th style='text-align:left;padding:6px 10px;border-bottom:2px solid #cbd5f5;font-size:12px;color:#475569;'>{header}</th>"
                    for header in headers
                )
                row_html = "".join(
                    "<tr>" + "".join(
                        f"<td style='padding:6px 10px;border-bottom:1px solid #e2e8f0;'>{html.escape(str(cell))}</td>"
                        for cell in row
                    ) + "</tr>"
                    for row in rows
                )
                caption_html = (
                    f"<div style='font-size:11px;font-weight:600;color:#475569;margin-bottom:4px;'>Table {idx}</div>"
                    if len(answer_tables) > 1 else ""
                )
                table_chunks.append(
                    f"<div style='overflow-x:auto;margin-top:10px;'>{caption_html}"
                    "<table style='width:100%;border-collapse:collapse;font-size:13px;color:#0f172a;'>"
                    f"<thead><tr>{header_html}</tr></thead><tbody>{row_html}</tbody></table></div>"
                )
            table_html = "".join(table_chunks)
        question_list_html = _structured_html_list(question_items, allow_headings=False)
        question_html = ""
        if question_list_html:
            question_html = (
                "<div style='margin-top:6px;'>"
                "<div style='font-size:11px;text-transform:uppercase;color:#94a3b8;letter-spacing:0.08em;'>Question</div>"
                f"{question_list_html}"
                "</div>"
            )
        answer_list_html = _structured_html_list(answer_items, allow_headings=True)
        answer_html = ""
        if answer_list_html:
            answer_html = (
                "<div style='margin-top:10px;'>"
                "<div style='font-size:11px;text-transform:uppercase;color:#94a3b8;letter-spacing:0.08em;'>Answer</div>"
                f"{answer_list_html}"
                "</div>"
            )
        blocks.append(
            "<div style='border:1px solid #e2e8f0;border-radius:14px;padding:14px;margin-top:12px;'>"
            f"<div style='font-weight:600;color:#0f172a;'>{title}</div>"
            f"{question_html}{answer_html}{table_html}"
            "</div>"
        )
    return "".join(blocks)


def _build_email_content(
    summary_type: str,
    conversation_id: str,
    intro: str,
    summary_text: str,
    *,
    concise_sections: Optional[Sequence[Dict[str, Any]]] = None,
    detailed_topics: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[str, str]:
    generated_on = datetime.now().strftime("%b %d, %Y – %I:%M %p")
    conv_display = conversation_id or "N/A"
    safe_summary = (summary_text or "").strip()

    body_lines = [intro, ""]
    if concise_sections:
        body_lines.extend(_sections_to_plaintext(concise_sections))
    elif detailed_topics:
        body_lines.extend(_topics_to_plaintext(detailed_topics))
    else:
        body_lines.append(safe_summary)
    body_lines += [
        "",
        "- wtchtwr",
        "Airbnb Property Performance & Market Insights Agent",
        "",
        f"Summary Type: {summary_type}",
        f"Generated On: {generated_on}",
        f"Conversation ID: {conv_display}",
    ]
    body = "\n".join(line for line in body_lines if line is not None).strip()

    meta_block = (
        f"<p style='font-size:13px;color:#666;'>"
        f"Summary Type: <b>{html.escape(summary_type)}</b><br>"
        f"Generated On: {generated_on}<br>"
        f"Conversation ID: {html.escape(conv_display)}"
        f"</p>"
    )

    intro_html = html.escape(intro)
    if concise_sections:
        structured_html = _render_sections_html(concise_sections)
    elif detailed_topics:
        structured_html = _render_topics_html(detailed_topics)
    else:
        summary_html = html.escape(safe_summary)
        structured_html = (
            "<div style='background:#f9f9f9;padding:12px;border-radius:8px;'>"
            f"<pre style='white-space:pre-wrap;font-family:inherit;margin:0;'>{summary_html}</pre>"
            "</div>"
        )

    html_body = f"""
<html><body style='font-family:Arial, sans-serif; line-height:1.6; color:#222;'>
  <p>{intro_html}</p>
  {structured_html}
  <p style='margin-top:1em;font-size:12px;color:#555;'>
    - wtchtwr <br>
    <span style='color:#888;'>Airbnb Property Performance & Market Insights Agent</span>
  </p>
  {meta_block}
</body></html>
""".strip()

    return body, html_body


def slugify(text: str, fallback: str = "export") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", text.strip()).strip("-").lower()
    return cleaned or fallback


def _safe_filename(label: Optional[str]) -> str:
    if not label:
        return "export"
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", label.strip()).strip("-")
    return cleaned or "export"


def _zip_bytes(filename: str, content: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(filename, content)
    return buffer.getvalue()


def _prepare_csv_delivery(label: str, csv_bytes: bytes, *, recipient: Optional[str]) -> FileDelivery:
    base_name = _safe_filename(label)
    csv_filename = f"{base_name}.csv"
    attachments: List[Tuple[str, bytes, str]] = []
    link: Optional[str] = None

    if len(csv_bytes) <= _EMAIL_ATTACHMENT_LIMIT:
        attachments.append((csv_filename, csv_bytes, "text/csv"))
        return FileDelivery("attachment", attachments, None, csv_filename, None)

    zipped = _zip_bytes(csv_filename, csv_bytes)
    zip_filename = f"{base_name}.zip"
    if len(zipped) <= _EMAIL_ATTACHMENT_LIMIT:
        attachments.append((zip_filename, zipped, "application/zip"))
        return FileDelivery("zip", attachments, None, csv_filename, zip_filename)

    if not drive_uploader.enabled():
        raise RuntimeError(
            "Export is larger than 25 MB. Configure Google Drive upload (GDRIVE_* environment variables) "
            "to share oversized files."
        )
    link = drive_uploader.upload_bytes(
        name=zip_filename,
        content=zipped,
        mimetype="application/zip",
        share_with=recipient,
    )
    logger.info("Large export uploaded to Drive for %s: %s", recipient or "recipient", link)
    return FileDelivery("drive", [], link, csv_filename, zip_filename)


def trim_history(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Prepare LangGraph history payload from stored messages."""
    history: List[Dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        if role == "user":
            content = (message.get("content") or "").strip()
        else:
            content = (
                message.get("content")
                or message.get("nl_summary")
                or message.get("payload", {}).get("summary")
                or ""
            ).strip()
        if not content:
            continue
        history.append({"role": role, "content": content})
    if len(history) > MAX_HISTORY_TURNS:
        history = history[-MAX_HISTORY_TURNS:]
    return history


def build_assistant_payload(result: Dict[str, Any], question: str) -> Dict[str, Any]:
    bundle = result.get("result_bundle") or {}
    telemetry = result.get("telemetry") or {}
    state_snapshot = result.get("state_snapshot") or {}
    extras = state_snapshot.get("extras") if isinstance(state_snapshot.get("extras"), dict) else {}
    intent = str(state_snapshot.get("intent") or result.get("intent") or "").upper()
    expansion_report = (
        bundle.get("expansion_report")
        or result.get("expansion_report")
        or extras.get("expansion_report")
    )
    expansion_sources = (
        bundle.get("expansion_sources")
        or result.get("expansion_sources")
        or extras.get("expansion_sources")
        or []
    )
    state_sql = state_snapshot.get("sql") or {}
    sql_text = (
        result.get("sql_text")
        or (result.get("sql") or {}).get("text")
        or bundle.get("sql")
        or state_sql.get("text")
        or state_sql.get("query")
    )
    sql_params = result.get("sql_params") or []
    rag_snippets = bundle.get("rag_snippets") or []
    columns = bundle.get("columns") or []
    rows = bundle.get("rows") or []
    markdown_table = bundle.get("markdown_table")

    tables: List[Dict[str, Any]] = []
    if columns and rows:
        table_name = bundle.get("sql_table") or ("RAG Snippets" if rag_snippets else "Results")
        tables.append(
            {
                "name": table_name,
                "columns": columns,
                "data": [normalise_payload(row) for row in rows],
                "row_count": len(rows),
                "source": "sql",
                "sql": sql_text,
            }
        )

    if not tables and markdown_table:
        parsed = _parse_markdown_table(markdown_table)
        if parsed:
            md_columns, md_rows = parsed
            table_name = bundle.get("sql_table") or ("RAG Snippets" if rag_snippets else "Results")
            tables.append(
                {
                    "name": table_name,
                    "columns": md_columns,
                    "data": [normalise_payload(row) for row in md_rows],
                    "row_count": len(md_rows),
                    "source": "sql",
                    "sql": sql_text,
                }
            )

    if rag_snippets:
        def _coerce_text(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, (int, float, Decimal)):
                return str(value)
            return str(value).strip()

        rag_rows: List[Dict[str, Any]] = []
        for entry in rag_snippets:
            if isinstance(entry, str):
                snippet_text = entry.strip()
                if snippet_text:
                    rag_rows.append({"snippet": snippet_text})
                continue
            if not isinstance(entry, dict):
                continue
            snippet_text = (
                _coerce_text(entry.get("snippet"))
                or _coerce_text(entry.get("text"))
            ).strip()
            if not snippet_text:
                continue
            listing_id = entry.get("listing_id") or entry.get("listingId")
            comment_identifier = (
                entry.get("comment_id")
                or entry.get("commentId")
                or entry.get("commentID")
                or entry.get("review_id")
                or entry.get("reviewId")
            )
            if not comment_identifier:
                citation = entry.get("citation")
                if isinstance(citation, str) and ":" in citation:
                    comment_identifier = citation.split(":", 1)[1]
            row = {
                "snippet": snippet_text,
                "listing_id": _coerce_text(listing_id),
                "neighbourhood": _coerce_text(entry.get("neighbourhood") or entry.get("neighborhood")),
                "neighbourhood_group": _coerce_text(entry.get("neighbourhood_group") or entry.get("neighborhood_group")),
                "month": _coerce_text(entry.get("month")),
                "year": _coerce_text(entry.get("year")),
                "comment_id": _coerce_text(comment_identifier),
                "sentiment_label": _coerce_text(entry.get("sentiment_label")),
                "compound": _coerce_text(entry.get("compound")),
                "positive": _coerce_text(entry.get("positive")),
                "neutral": _coerce_text(entry.get("neutral")),
                "negative": _coerce_text(entry.get("negative")),
                "is_highbury": _coerce_text(entry.get("is_highbury")),
            }
            rag_rows.append(row)

        if rag_rows:
            normalised_rows = [
                normalise_payload({column: row.get(column, "") for column in RAG_EXPORT_COLUMNS})
                for row in rag_rows
            ]
            tables.append(
                {
                    "name": RAG_TABLE_NAME,
                    "columns": list(RAG_EXPORT_COLUMNS),
                    "data": normalised_rows,
                    "row_count": len(normalised_rows),
                    "source": RAG_TABLE_SOURCE,
                }
            )

    if sql_text and rag_snippets:
        response_type = "hybrid"
    elif intent == "EXPANSION_SCOUT" or expansion_report:
        response_type = "expansion"
    elif sql_text:
        response_type = "sql"
    elif rag_snippets:
        response_type = "rag"
    else:
        response_type = "text"

    payload: Dict[str, Any] = {
        "sql": sql_text,
        "params": sql_params,
        "tables": tables,
        "summary": bundle.get("summary") or result.get("answer_text"),
        "question": question,
        "row_count": len(rows) if rows else None,
        "duration_ms": telemetry.get("latency_ms"),
        "response_type": response_type,
        "rag_snippets": rag_snippets,
        "aggregates": bundle.get("aggregates"),
        "policy": bundle.get("policy") or telemetry.get("policy"),
        "markdown_table": markdown_table,
    }
    if intent:
        payload["intent"] = intent
    if expansion_report:
        payload["expansion_report"] = expansion_report
    if expansion_sources:
        payload["expansion_sources"] = expansion_sources
    sentiment_summary = _summarize_sentiment(rag_snippets)
    if sentiment_summary:
        payload["sentiment_analytics"] = sentiment_summary
    bundle_summary = bundle.get("summary") or result.get("answer_text")
    if bundle_summary in (None, "", "nan"):
        bundle_summary = "No valid records found for this query."
    payload["summary"] = bundle_summary
    triage_context = bundle.get("portfolio_triage") or result.get("portfolio_triage") or extras.get("portfolio_triage")
    if triage_context:
        payload["portfolio_triage"] = triage_context
    thinking_trace = result.get("thinking_trace") or state_snapshot.get("thinking")
    if thinking_trace:
        payload["thinking_trace"] = thinking_trace
    payload = {k: v for k, v in payload.items() if v is not None}
    payload = _sanitize_nan(payload)
    logger.info("[NL2SQL] Final normalized assistant payload prepared.")
    try:
        logger.debug('[build_assistant_payload] keys: %s', list(payload.keys()))
    except Exception:
        pass
    return normalise_payload(payload)


def generate_conversation_title(messages: Sequence[Dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user" and message.get("content"):
            content = re.sub(r"\s+", " ", message["content"]).strip()
            if content:
                return content[:80]
    return "New conversation"


def conversation_from_rows(convo_row: Tuple, message_rows: Iterable[Tuple]) -> Dict[str, Any]:
    conversation = {
        "id": convo_row[0],
        "title": convo_row[1] or "",
        "created_at": isoformat(convo_row[2]),
        "updated_at": isoformat(convo_row[3]),
        "messages": [],
    }
    messages: List[Dict[str, Any]] = []
    for row in message_rows:
        payload = row[5]
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode("utf-8", errors="ignore")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = None
        normalized_payload = None
        if payload is not None:
            normalized_payload = normalise_payload(payload)
        message = {
            "id": row[0],
            "role": row[2],
            "content": row[3],
            "nl_summary": row[4],
            "payload": normalized_payload,
            "timestamp": isoformat(row[6]),
        }
        current_payload = message.get("payload")
        if isinstance(current_payload, str):
            logger.debug(
                "[PERSIST_ALERT][DESERIALIZE] payload is a string of length %d: %s...",
                len(current_payload),
                current_payload[:200],
            )
            try:
                message["payload"] = json.loads(current_payload)
                logger.debug(
                    "[PERSIST_FIX][DESERIALIZE] Decoded payload JSON with keys=%s",
                    list(message["payload"].keys()) if isinstance(message["payload"], dict) else None,
                )
            except Exception as exc:
                logger.error("[PERSIST_FIX][DESERIALIZE] ❌ Failed to decode payload: %s", exc)
        else:
            logger.debug(
                "[PERSIST_ALERT][DESERIALIZE] payload type=%s keys=%s",
                type(current_payload),
                list(current_payload.keys()) if isinstance(current_payload, dict) else None,
            )
        logger.debug(
            "[PERSIST_TRACE][SERIALIZE] message role=%s keys=%s",
            message.get("role"),
            list(message.keys()),
        )
        if message.get("payload"):
            logger.debug(
                "[PERSIST_TRACE][SERIALIZE] ✅ payload keys=%s",
                list(message["payload"].keys()) if isinstance(message["payload"], dict) else type(message["payload"]),
            )
        else:
            logger.debug("[PERSIST_TRACE][SERIALIZE] ❌ No payload in this message.")
        messages.append(message)
    conversation["messages"] = messages
    return conversation


def fetch_conversation(conn: duckdb.DuckDBPyConnection, conversation_id: str) -> Dict[str, Any]:
    convo = conn.execute(
        "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
        [conversation_id],
    ).fetchone()
    if not convo:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    messages = conn.execute(
        """
        SELECT id, conversation_id, role, content, nl_summary, payload, timestamp
        FROM messages
        WHERE conversation_id = ?
        ORDER BY timestamp ASC
        """,
        [conversation_id],
    ).fetchall()
    return conversation_from_rows(convo, messages)


def list_conversations(conn: duckdb.DuckDBPyConnection) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT c.id, c.title, c.created_at, c.updated_at
        FROM conversations c
        WHERE c.updated_at >= ?
        ORDER BY c.updated_at DESC
        """,
        [utcnow() - timedelta(days=CONVERSATION_TTL_DAYS)],
    ).fetchall()
    conversations: List[Dict[str, Any]] = []
    for convo in rows:
        messages = conn.execute(
            """
            SELECT id, conversation_id, role, content, nl_summary, payload, timestamp
            FROM messages
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
            """,
            [convo[0]],
        ).fetchall()
        conversations.append(conversation_from_rows(convo, messages))
    return conversations


def upsert_export_blob(token: str, filename: str, mime_type: str, payload: bytes) -> None:
    now = utcnow()
    with open_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO exports_cache (token, created_at, filename, mime_type, payload)
            VALUES (?, ?, ?, ?, ?)
            """,
            [token, now, filename, mime_type, payload],
        )


def get_export_blob(token: str) -> Optional[Tuple[str, str, bytes]]:
    with open_connection(read_only=True) as conn:
        row = conn.execute(
            """
            SELECT filename, mime_type, payload
            FROM exports_cache
            WHERE token = ?
            """,
            [token],
        ).fetchone()
    if not row:
        return None
    return row[0], row[1], row[2]


def purge_expired_exports() -> None:
    cutoff = utcnow() - EXPORT_RETENTION
    with open_connection() as conn:
        conn.execute("DELETE FROM exports_cache WHERE created_at < ?", [cutoff])


# ---------------------------------------------------------------------------
# Slack bot lifecycle helpers
# ---------------------------------------------------------------------------


def _start_slack_bot_thread() -> None:
    """Initialize and launch the Slack bot in a background thread."""
    global SLACK_THREAD, SLACK_RUNNING

    if SLACK_THREAD and SLACK_THREAD.is_alive():
        SLACK_RUNNING = True
        return

    try:
        slack_app = create_slack_app()
    except Exception as exc:
        SLACK_RUNNING = False
        logger.warning("[SlackBot] Failed to start: %s", exc)
        return

    SLACK_THREAD = threading.Thread(
        target=slack_app.start,
        kwargs={"port": SLACK_PORT},
        daemon=True,
        name="SlackBotThread",
    )
    SLACK_THREAD.start()
    SLACK_RUNNING = True
    logger.info("[SlackBot] Initialized successfully on port %s.", SLACK_PORT)


# ---------------------------------------------------------------------------
# API endpoints – LangGraph wrappers
# ---------------------------------------------------------------------------


@app.post("/api/run_query")
async def run_query(req: QueryRequest):
    try:
        logger.info('[LangGraph] Query received: %s', req.query)
        result = run_agent(req.query, thread_id=req.thread_id, debug_thinking=req.debug_thinking)
        try:
            logger.debug('[run_query] result keys: %s', list(result.keys()))
            if isinstance(result.get("sql"), dict):
                logger.debug('[run_query] sql keys: %s', list(result['sql'].keys()) if isinstance(result.get('sql'), dict) else None)
        except Exception:
            pass
        safe_response = safe_json(normalise_payload(result))
        try:
            state_snapshot = result.get("state_snapshot") or result.get("state") or {}
            intent = str(state_snapshot.get("intent") or result.get("intent") or "").upper()
            if intent == "EXPANSION_SCOUT" and isinstance(safe_response, dict):
                bundle = result.get("result_bundle") or {}
                report = (
                    state_snapshot.get("expansion_report")
                    or bundle.get("expansion_report")
                    or result.get("expansion_report")
                    or safe_response.get("answer_text")
                )
                sources = (
                    state_snapshot.get("expansion_sources")
                    or bundle.get("expansion_sources")
                    or result.get("expansion_sources")
                    or []
                )
                safe_response.setdefault("expansion_report", report)
                safe_response.setdefault("expansion_sources", sources)
        except Exception:
            logger.warning("[run_query] Unable to inject expansion payload", exc_info=True)
        try:
            logger.debug('[run_query] Final payload keys: %s', list(safe_response.keys()))
        except Exception:
            pass
        return JSONResponse(content=safe_response)
    except HTTPException:
        raise
    except Exception as exc:
        import traceback

        print("❌ Message Error:", traceback.format_exc())
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@app.post("/api/rag")
async def run_rag(req: QueryRequest):
    try:
        query = req.query
        if "review" not in query.lower():
            query = f"reviews {query}"
        logger.info("[LangGraph] Query received (RAG): %s", query)
        result = run_agent(
            query,
            thread_id=req.thread_id,
            user_filters={"reviews": True},
            debug_thinking=req.debug_thinking,
        )
        result["policy"] = result.get("policy") or "RAG_MARKET"
        state_payload = result.get("state") or {}
        bundle = state_payload.get("result_bundle") or {}
        bundle["policy"] = bundle.get("policy") or result["policy"]
        state_payload["result_bundle"] = bundle
        result["state"] = state_payload
        safe_response = safe_json(normalise_payload(result))
        return JSONResponse(content=safe_response)
    except HTTPException:
        raise
    except Exception as exc:
        import traceback

        print("❌ Message Error:", traceback.format_exc())
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@app.get("/api/insights")
async def insights():
    try:
        db_path: Path = get_db_path()
        try:
            conn = duckdb.connect(str(db_path), read_only=True)
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=f"Unable to open DuckDB: {exc}") from exc
        with conn:
            tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}

            def _execute(sql: str) -> Optional[Any]:
                try:
                    value = conn.execute(sql).fetchone()
                except Exception:
                    return None
                if not value:
                    return None
                return to_primitive(value[0])

            data = {
                "avg_price_market": _execute("SELECT ROUND(AVG(price_in_usd), 2) FROM listings_cleaned"),
                "avg_occupancy_highbury": _execute(
                    "SELECT ROUND(AVG(occupancy_rate_30), 2) FROM listings_cleaned WHERE lower(host_name)='highbury'"
                ),
                "total_listings": _execute("SELECT COUNT(*) FROM listings_cleaned"),
                "recent_reviews": 0,
            }
            if "reviews" in tables:
                data["recent_reviews"] = (
                    _execute("SELECT COUNT(*) FROM reviews WHERE review_date >= DATE '2025-01-01'") or 0
                )
        safe_response = safe_json(normalise_payload(data))
        return JSONResponse(content=safe_response)
    except HTTPException:
        raise
    except Exception as exc:
        import traceback

        print("❌ Message Error:", traceback.format_exc())
        return JSONResponse(content={"error": str(exc)}, status_code=500)


# ---------------------------------------------------------------------------
# API endpoints – Conversations
# ---------------------------------------------------------------------------


@app.get("/api/conversations")
async def api_list_conversations():
    try:
        with open_connection(read_only=False) as conn:
            data = normalise_payload(list_conversations(conn))
        safe_response = safe_json(data)
        return JSONResponse(content=safe_response)
    except HTTPException:
        raise
    except Exception as exc:
        import traceback

        print("❌ Message Error:", traceback.format_exc())
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@app.post("/api/conversations", status_code=status.HTTP_201_CREATED)
async def api_create_conversation(req: ConversationCreateRequest = Body(default_factory=ConversationCreateRequest)):
    try:
        conv_id = str(uuid4())
        now = utcnow()
        with open_connection() as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                [conv_id, req.title or "", now, now],
            )
            conversation = fetch_conversation(conn, conv_id)
        safe_response = safe_json(normalise_payload(conversation))
        return JSONResponse(content=safe_response, status_code=status.HTTP_201_CREATED)
    except HTTPException:
        raise
    except Exception as exc:
        import traceback

        print("❌ Message Error:", traceback.format_exc())
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@app.get("/api/conversations/{conversation_id}")
async def api_get_conversation(conversation_id: str):
    try:
        with open_connection(read_only=True) as conn:
            conversation = fetch_conversation(conn, conversation_id)
        messages = conversation.get("messages") or []
        logger.warning("[PERSIST_VERIFY][RESPONSE] Sending %d messages", len(messages))
        for msg in messages[-3:]:
            logger.warning(
                "[PERSIST_VERIFY][RESPONSE] msg.role=%s keys=%s",
                msg.get("role"),
                list(msg.keys()),
            )
        if messages:
            logger.warning(
                "[PERSIST_VERIFY][FINAL_RESPONSE] Sending assistant message keys=%s",
                list(messages[-1].keys()),
            )
        payload_keys = (
            list(conversation.get("payload", {}).keys())
            if isinstance(conversation.get("payload"), dict)
            else None
        )
        logger.warning(
            "[PERSIST_SUMMARY] Final assistant payload keys=%s | total_messages=%d",
            payload_keys,
            len(messages),
        )
        safe_convo = normalise_payload(conversation)
        if isinstance(safe_convo, dict) and "messages" in safe_convo:
            for msg in safe_convo["messages"]:
                if isinstance(msg, dict) and "payload" in msg:
                    payload_keys = list(msg["payload"].keys()) if isinstance(msg["payload"], dict) else msg["payload"]
                    logger.warning(
                        "[PERSIST_VERIFY][FINAL_RESPONSE_MESSAGE] ✅ Payload in final message keys=%s",
                        payload_keys,
                    )
        safe_response = safe_json(safe_convo)
        if isinstance(safe_response, dict):
            logger.warning(
                "[PERSIST_VERIFY][FINAL_RESPONSE_CONVERSATION] ✅ Final conversation keys=%s",
                list(safe_response.keys()),
            )
        return JSONResponse(content=safe_response)
    except HTTPException:
        raise
    except Exception as exc:
        import traceback

        print("❌ Message Error:", traceback.format_exc())
        return JSONResponse(content={"error": str(exc)}, status_code=500)


def insert_message(
    conn: duckdb.DuckDBPyConnection,
    conversation_id: str,
    role: str,
    content: Optional[str],
    nl_summary: Optional[str],
    payload: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    message_id = str(uuid4())
    now = utcnow()
    conn.execute(
        """
        INSERT INTO messages (id, conversation_id, role, content, nl_summary, payload, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            message_id,
            conversation_id,
            role,
            content,
            nl_summary,
            json.dumps(payload) if payload is not None else None,
            now,
        ],
    )
    conn.execute(
        "UPDATE conversations SET updated_at = ?, title = COALESCE(NULLIF(title, ''), ?) WHERE id = ?",
        [now, generate_conversation_title([{"role": role, "content": content or ""}]), conversation_id],
    )
    return {
        "id": message_id,
        "role": role,
        "content": content,
        "nl_summary": nl_summary,
        "payload": payload,
        "timestamp": isoformat(now),
    }


def map_thread_to_conversation(conn: duckdb.DuckDBPyConnection, thread_id: str, conversation_id: str) -> None:
    conn.execute("INSERT OR REPLACE INTO thread_map VALUES (?, ?)", [thread_id, conversation_id])


def _record_user_message(conversation_id: str, content: str) -> List[Dict[str, str]]:
    with open_connection() as conn:
        conversation = fetch_conversation(conn, conversation_id)
        user_entry = insert_message(conn, conversation_id, "user", content, None, None)
        map_thread_to_conversation(conn, conversation_id, conversation_id)
        history = trim_history(conversation["messages"] + [user_entry])
    return history


def _execute_langgraph(
    query: str,
    tenant: Optional[str],
    user_filters: Optional[Dict[str, Any]],
    history: List[Dict[str, str]],
    *,
    thread_id: Optional[str] = None,
    stream_handler: Optional[Any] = None,
    composer_enabled: Optional[bool] = None,
    debug_thinking: Optional[bool] = None,
) -> tuple[Dict[str, Any], float]:
    start = time.perf_counter()
    result = run_agent(
        query,
        tenant=tenant,
        user_filters=user_filters,
        history=history,
        stream_handler=stream_handler,
        composer_enabled=composer_enabled,
        debug_thinking=debug_thinking,
        thread_id=thread_id,
    )
    elapsed = time.perf_counter() - start
    logger.info("[LangGraph] %s... processed in %.2fs", query[:80], elapsed)
    normalized = normalise_payload(result)
    telemetry = normalized.setdefault("telemetry", {})
    telemetry["total_latency_s"] = elapsed
    telemetry.setdefault("latency_ms", elapsed * 1000)
    normalized["latency"] = round(elapsed, 2)
    return normalized, elapsed


def _finalize_assistant_message(
    conversation_id: str,
    query: str,
    result: Dict[str, Any],
    latency_s: Optional[float],
) -> Dict[str, Any]:
    import logging

    debug_logger = logging.getLogger("hope.debug")
    debug_logger.warning("[DEBUG_DUMP] result keys: %s", list(result.keys()))
    try:
        debug_logger.warning(
            "[DEBUG_DUMP] result snippet: %s",
            json.dumps(result, indent=2)[:600],
        )
    except Exception:
        debug_logger.warning("[DEBUG_DUMP] result (repr): %r", result)

    bundle = result.get("result_bundle", {}) or {}
    rag_snippets = bundle.get("rag_snippets") or []
    markdown_table = bundle.get("markdown_table")

    raw_content = (result.get("content") or "").strip()
    bundle_summary = (bundle.get("summary") or "").strip()
    answer_text = (result.get("answer_text") or "").strip()

    pipeline_label = str(
        result.get("policy")
        or bundle.get("policy")
        or result.get("pipeline")
        or ""
    ).upper()
    has_structured_rows = bool(bundle.get("rows"))
    if not has_structured_rows:
        tables = bundle.get("tables")
        if isinstance(tables, list) and tables:
            has_structured_rows = True

    final_content = raw_content or bundle_summary or answer_text
    if not final_content and has_structured_rows:
        final_content = bundle_summary or answer_text

    final_content = _strip_sql_statements(final_content)

    # Preserve markdown tables so previously streamed NL2SQL answers keep their full context
    # when the conversation reloads or the user revisits the page.

    if not final_content or not final_content.strip():
        final_content = bundle_summary or answer_text or "No insight available from the model. Please rephrase your question or try again."

    if isinstance(final_content, str) and "select" in final_content.lower():
        logging.warning(
            "[LEAK_DETECTOR][BACKEND_RESPONSE] SQL text visible in backend response:\n%s",
            final_content[:200],
        )

    debug_logger.warning("[DEBUG_DUMP] final content preview: %s", final_content[:280])

    result["content"] = final_content
    result["answer_text"] = final_content
    result["render_markdown"] = True
    bundle["summary"] = final_content
    result["result_bundle"] = bundle

    logger.debug("[Conversations] Final answer_text decided (%d chars)", len(final_content))
    if result.get("content") or result.get("answer_text"):
        logger.info("✅ Received LLM markdown content for assistant reply.")
    else:
        logger.warning("⚠️ No LLM markdown found — using fallback.")

    if latency_s is not None:
        result.setdefault("telemetry", {})["latency_ms"] = latency_s * 1000

    assistant_payload = build_assistant_payload(result, query)
    if latency_s is not None:
        assistant_payload["duration_ms"] = latency_s * 1000
    assistant_payload["summary"] = final_content
    if markdown_table:
        assistant_payload["markdown_table"] = markdown_table
    if rag_snippets:
        assistant_payload["rag_snippets"] = rag_snippets

    with open_connection() as conn:
        insert_message(conn, conversation_id, "assistant", final_content, final_content, assistant_payload)
        updated = fetch_conversation(conn, conversation_id)

    if latency_s is not None:
        updated["latency"] = round(latency_s, 2)
    updated = _sanitize_nan(updated)

    if "messages" in updated and updated["messages"]:
        last_msg = updated["messages"][-1]
        if last_msg.get("role") == "assistant":
            last_msg["content"] = final_content
            last_msg["render_markdown"] = True
            logger.warning(
                "[PERSIST_TRACE][FINALIZE] assistant message keys=%s",
                list(last_msg.keys()),
            )
            if "payload" in last_msg and isinstance(last_msg["payload"], dict):
                logger.warning(
                    "[PERSIST_TRACE][FINALIZE] ✅ payload keys: %s",
                    list(last_msg["payload"].keys()),
                )
                updated["payload"] = last_msg["payload"]
            else:
                logger.warning("[PERSIST_TRACE][FINALIZE] ❌ No payload in assistant message!")
                updated.pop("payload", None)
        else:
            logger.warning("[PERSIST_TRACE][FINALIZE] Last message is not assistant; role=%s", last_msg.get("role"))
            updated.pop("payload", None)
    else:
        logger.warning("[PERSIST_TRACE][FINALIZE] Updated conversation missing messages array.")
        updated.pop("payload", None)

    debug_logger.warning("[DEBUG_DUMP] persisted assistant message: %s", final_content[:280])

    return updated


def _insert_greeting_response(conversation_id: str, query: Optional[str]) -> Dict[str, Any]:
    greeting = "👋 Hi! I'm wtchtwr — your property insights assistant. Ask me about occupancy, revenue, amenities, or reviews!"
    payload = {
        "summary": greeting,
        "tables": [],
        "question": query or "",
        "response_type": "text",
        "duration_ms": 0.0,
    }
    with open_connection() as conn:
        insert_message(conn, conversation_id, "assistant", greeting, greeting, payload)
        updated = fetch_conversation(conn, conversation_id)
    updated["latency"] = 0.0
    updated["payload"] = payload
    return updated


def _insert_error_response(conversation_id: str, query: str, detail: str) -> Dict[str, Any]:
    message = "Unable to fetch answer right now—please try again shortly."
    payload = {
        "tables": [],
        "summary": message,
        "question": query,
        "response_type": "error",
        "error_detail": detail,
    }
    with open_connection() as conn:
        insert_message(conn, conversation_id, "assistant", message, message, payload)
        updated = fetch_conversation(conn, conversation_id)
    updated.setdefault("latency", None)
    updated["payload"] = payload
    return updated


def _message_text(message: Dict[str, Any]) -> str:
    if not isinstance(message, dict):
        return ""
    content = (message.get("content") or message.get("nl_summary") or "").strip()
    if content:
        return content
    payload = message.get("payload")
    if isinstance(payload, dict):
        fallback = (
            payload.get("summary")
            or payload.get("answer_text")
            or payload.get("markdown_table")
            or ""
        )
        return str(fallback).strip()
    return ""


def _clean_summary_line(value: Optional[str]) -> str:
    if not value:
        return ""
    line = str(value).strip()
    if not line:
        return ""
    line = re.sub(r"^>+\s*", "", line)
    line = re.sub(r"^[-*•]+\s*", "", line)
    line = re.sub(r"^#{1,6}\s*", "", line)
    line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
    line = re.sub(r"__(.+?)__", r"\1", line)
    line = re.sub(r"`(.+?)`", r"\1", line)
    line = line.replace("**", "").replace("__", "").replace("`", "")
    return line.strip()


def _format_bullets(items: Sequence[Optional[str]]) -> List[str]:
    formatted: List[str] = []
    for item in items:
        cleaned = _clean_summary_line(item)
        if cleaned:
            formatted.append(cleaned)
    return formatted


_SUMMARY_HEADING_KEYWORDS = [
    "key insights",
    "overview",
    "summary",
    "focus area",
    "focus areas",
    "focus",
    "areas for improvement",
    "improvement needed",
    "positive feedback",
    "concerns noted",
    "recommendations",
    "next steps",
    "strategy",
    "strategies",
    "action plan",
    "evaluation",
    "insights",
]


def _normalize_heading_label(text: str) -> str:
    return re.sub(r"[:：]\s*$", "", text or "").strip()


def _structured_summary_entries(
    items: Sequence[str],
    *,
    allow_headings: bool = True,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for raw in items:
        text = _clean_summary_line(raw)
        if not text:
            continue
        lowered = text.lower()
        numeric_match = re.match(r"^(\d+(?:\.\d+)*)", text)
        depth = 0
        if numeric_match:
            depth = min(len(numeric_match.group(1).split(".")), 2)
        heading_candidate = False
        if allow_headings:
            heading_candidate = any(
                lowered == keyword or lowered.startswith(f"{keyword}:")
                for keyword in _SUMMARY_HEADING_KEYWORDS
            )
            heading_candidate = heading_candidate or bool(re.search(r"[:：]\s*$", text))
            heading_candidate = heading_candidate or bool(re.match(r"^\d+(?:\.\d+)*[\.\)]", text))
        if heading_candidate:
            entries.append({"text": _normalize_heading_label(text), "kind": "heading", "depth": depth})
        else:
            entries.append({"text": text, "kind": "bullet", "depth": depth})
    return entries


def _structured_plaintext_list(
    items: Sequence[str],
    *,
    allow_headings: bool = True,
) -> List[str]:
    entries = _structured_summary_entries(items, allow_headings=allow_headings)
    lines: List[str] = []
    for entry in entries:
        depth = entry.get("depth", 0) or 0
        indent = "  " * min(depth, 3)
        text = entry.get("text") or ""
        if entry.get("kind") == "heading":
            if depth <= 0:
                lines.append(text.upper())
            else:
                lines.append(f"{indent}{text}")
        else:
            lines.append(f"{indent}- {text}")
    return lines


def _structured_html_list(
    items: Sequence[str],
    *,
    allow_headings: bool = True,
) -> str:
    entries = _structured_summary_entries(items, allow_headings=allow_headings)
    if not entries:
        return ""
    chunks: List[str] = []
    for idx, entry in enumerate(entries):
        depth = entry.get("depth", 0) or 0
        text = html.escape(entry.get("text") or "")
        if entry.get("kind") == "heading":
            if depth <= 0:
                style = (
                    "font-size:11px;text-transform:uppercase;letter-spacing:0.08em;"
                    "color:#94a3b8;margin-top:10px;"
                )
            else:
                margin = 12 * min(depth, 3)
                style = (
                    f"font-size:13px;font-weight:600;color:#0f172a;margin-top:10px;margin-left:{margin}px;"
                )
            chunks.append(f"<div style='{style}'>{text}</div>")
        else:
            margin = 12 * min(depth, 3)
            bullet = (
                f"<div style='display:flex;gap:8px;align-items:flex-start;margin-top:8px;"
                f"margin-left:{margin}px;'>"
                "<span style='margin-top:6px;width:5px;height:5px;border-radius:999px;background:#94a3b8;"
                "flex:none;'></span>"
                f"<span style='font-size:13px;color:#0f172a;line-height:1.5;'>{text}</span>"
                "</div>"
            )
            chunks.append(bullet)
    return "".join(chunks)


def _build_concise_sections(
    focus: Optional[Union[str, Sequence[str]]],
    insights: Optional[Sequence[str]],
    next_steps: Optional[Sequence[str]],
) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    if focus is None:
        focus_items: List[str] = []
    elif isinstance(focus, str):
        focus_items = _format_bullets([focus])
    else:
        focus_items = _format_bullets(list(focus))
    if focus_items:
        sections.append({"title": "Focus", "items": focus_items})
    insight_items = _format_bullets(insights or [])
    if insight_items:
        sections.append({"title": "Insights", "items": insight_items})
    next_items = _format_bullets(next_steps or [])
    if next_items:
        sections.append({"title": "Next Step", "items": next_items})
    return sections


def _extract_markdown_table_block(text: str) -> Tuple[List[Dict[str, Any]], str]:
    """Extract one or more markdown tables (optionally prefixed by bullet markers)."""
    if not text or "|" not in text:
        return [], text

    table_pattern = re.compile(r"(?:^\s*(?:[-*•·‣▪‒–—]\s*)?\|.*\|\s*$\n?){2,}", re.MULTILINE)
    tables: List[Dict[str, Any]] = []
    removed_spans: List[Tuple[int, int]] = []

    for match in table_pattern.finditer(text):
        block = match.group(0)
        normalized_lines: List[str] = []
        for raw_line in block.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            stripped = re.sub(r"^\s*(?:[-*•·‣▪‒–—]\s*)?", "", stripped)
            if stripped and stripped.startswith("|"):
                normalized_lines.append(stripped)
        if not normalized_lines:
            continue
        normalized_block = "\n".join(normalized_lines)
        parsed = _parse_markdown_table(normalized_block)
        if not parsed:
            continue
        raw_columns, rows = parsed
        display_headers = [_clean_summary_line(col) or col for col in raw_columns]
        table_rows: List[List[str]] = []
        for row in rows:
            table_rows.append([
                _clean_summary_line(str(row.get(col, ""))) or str(row.get(col, "")).strip()
                for col in raw_columns
            ])
        tables.append({"headers": display_headers, "rows": table_rows})
        removed_spans.append(match.span())

    if not tables:
        return [], text

    remaining_parts: List[str] = []
    last_idx = 0
    for start, end in removed_spans:
        remaining_parts.append(text[last_idx:start])
        last_idx = end
    remaining_parts.append(text[last_idx:])
    remaining = re.sub(r"\n{3,}", "\n\n", "".join(remaining_parts)).strip()
    return tables, remaining


def _sections_to_markdown(sections: Sequence[Dict[str, Any]]) -> str:
    chunks: List[str] = []
    for section in sections:
        title = section.get("title")
        items = section.get("items") or []
        if not title or not items:
            continue
        lines = [f"**{title}**"] + [f"- {item}" for item in items]
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks).strip()


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        candidate = re.sub(r"^(\w+)?", "", candidate, count=1).strip()
    try:
        return json.loads(candidate)
    except Exception:
        pass
    try:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
    except Exception:
        return None
    return None


def _prepare_trace(messages: Sequence[Dict[str, Any]], limit: Optional[int] = SUMMARY_TRACE_LIMIT) -> List[Dict[str, str]]:
    trace: List[Dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        text = _message_text(message)
        if not text:
            continue
        trace.append({"role": str(role), "content": text})
    if limit is not None and len(trace) > limit:
        trace = trace[-limit:]
    return trace


def _resolve_thread_candidates(conn: duckdb.DuckDBPyConnection, conversation_id: str) -> List[str]:
    candidates: List[str] = []
    seen: Set[str] = set()
    if conversation_id:
        candidates.append(conversation_id)
        seen.add(conversation_id)
    try:
        rows = conn.execute(
            "SELECT thread_id FROM thread_map WHERE conversation_id = ?",
            [conversation_id],
        ).fetchall()
        for row in rows:
            thread_id = row[0]
            if thread_id and thread_id not in seen:
                candidates.append(thread_id)
                seen.add(thread_id)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("[Summary] Unable to resolve thread map for %s: %s", conversation_id, exc)
    return candidates


def _load_trace_from_memory(thread_candidates: Sequence[str], *, limit: Optional[int] = None) -> List[Dict[str, str]]:
    for thread_id in thread_candidates:
        if not thread_id:
            continue
        try:
            memory = get_memory_context(thread_id) or {}
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("[Summary] Memory fetch failed for %s: %s", thread_id, exc)
            continue
        raw_trace = memory.get("conversation_trace") or memory.get("history")
        if isinstance(raw_trace, list):
            trace = _prepare_trace(raw_trace, limit)
            if trace:
                return trace
    return []


def _load_memory_payload(thread_candidates: Sequence[str]) -> Dict[str, Any]:
    for thread_id in thread_candidates:
        if not thread_id:
            continue
        try:
            memory = get_memory_context(thread_id) or {}
            if memory:
                return memory
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("[Summary] Memory fetch failed for %s: %s", thread_id, exc)
    return {}


def _merge_traces(primary: Sequence[Dict[str, str]], secondary: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    """Merge two traces, preferring the full conversation order from primary."""
    merged: List[Dict[str, str]] = []
    seen: Set[Tuple[str, str]] = set()

    def _append(turn: Dict[str, str]) -> None:
        role = str(turn.get("role") or "")
        content = str(turn.get("content") or "")
        key = (role, content)
        if not content or key in seen:
            return
        merged.append({"role": role, "content": content})
        seen.add(key)

    for turn in primary:
        _append(turn)
    for turn in secondary:
        _append(turn)
    return merged


def _trace_to_pairs(trace: Sequence[Dict[str, str]]) -> List[Tuple[str, str]]:
    """Convert a role/content trace into user→assistant pairs."""
    pairs: List[Tuple[str, str]] = []
    pending_question: Optional[str] = None
    for turn in trace:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            pending_question = content
        elif role == "assistant" and pending_question:
            pairs.append((pending_question, content))
            pending_question = None
    return pairs


def _trace_to_transcript(trace: Sequence[Dict[str, str]]) -> str:
    lines: List[str] = []
    for turn in trace:
        role = "User" if turn.get("role") == "user" else "HOPE"
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _summarize_trace_with_llm(trace: Sequence[Dict[str, str]]) -> Optional[List[Dict[str, Any]]]:
    if OpenAI is None:
        return None
    cfg = load_config()
    if not cfg.openai_api_key:
        return None
    transcript = _trace_to_transcript(trace)
    if not transcript:
        return None
    system_prompt = (
        "You are an analytics note-taker summarising a conversation between a business stakeholder and the HOPE agent. "
        "Return JSON with this exact shape: "
        '{"focus": "<one sentence>", "insights": ["<point>", "..."], "next_steps": ["<action>"]}. '
        "Limit to 110 words overall and avoid referencing SQL, pipelines, or tool names."
    )
    user_prompt = (
        "Conversation transcript:\n"
        f"{transcript}\n\n"
        "Summarise the conversation per the required JSON schema."
    )
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model=cfg.openai_model,
            temperature=0.2,
            max_tokens=220,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        choice = response.choices[0].message if response.choices else None
        summary = (choice.content or "").strip() if choice else ""
        data = _extract_json_block(summary) if summary else None
        if isinstance(data, dict):
            focus = data.get("focus")
            insights = data.get("insights") if isinstance(data.get("insights"), list) else []
            next_steps = data.get("next_steps")
            if isinstance(next_steps, str):
                next_steps = [next_steps]
            elif not isinstance(next_steps, list):
                next_steps = []
            sections = _build_concise_sections(focus, insights, next_steps or [])
            if sections:
                return sections
        return None
    except Exception as exc:  # pragma: no cover - best-effort logging
        logger.warning("[Summary] Concise LLM rewrite failed: %s", exc)
        return None


def _trim_highlight(text: str, limit: int = 200) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit - 3].rstrip()}..."


def _table_highlights(tables: Sequence[Dict[str, Any]], limit: int = 3) -> List[str]:
    highlights: List[str] = []
    for table in tables:
        headers = table.get("headers") or []
        rows = table.get("rows") or []
        if len(headers) < 2 or not rows:
            continue
        label_idx, value_idx = 0, 1
        for row in rows:
            if len(row) <= max(label_idx, value_idx):
                continue
            label = str(row[label_idx]).strip()
            value = str(row[value_idx]).strip()
            if not label or not value:
                continue
            highlights.append(_trim_highlight(f"{label}: {value}"))
            if len(highlights) >= limit:
                return highlights
    return highlights


def _text_highlights(text: str, limit: int = 4) -> List[str]:
    highlights: List[str] = []
    if not text:
        return highlights
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullet_prefix = re.compile(r"^(\d+[\).]|[-*•·‣▪‒–—])\s+")
    for line in lines:
        lowered = line.lower()
        candidate = None
        if lowered.startswith("theme:"):
            candidate = line.split(":", 1)[1].strip() or line
        elif lowered.startswith("operator read"):
            candidate = line.split(":", 1)[1].strip() or line
        elif bullet_prefix.match(line):
            candidate = bullet_prefix.sub("", line).strip() or line
        if candidate and "|" not in candidate:
            highlights.append(_trim_highlight(_clean_summary_line(candidate) or candidate))
        if len(highlights) >= limit:
            return highlights
    if highlights:
        return highlights
    sentences = re.split(r"(?<=[\.!?])\s+", text.strip())
    for sentence in sentences:
        cleaned = _clean_summary_line(sentence)
        if not cleaned or "|" in cleaned:
            continue
        highlights.append(_trim_highlight(cleaned))
        if len(highlights) >= limit:
            break
    return highlights


def _extract_insight_highlights(text: Optional[str], limit: int = 6) -> List[str]:
    if not text:
        return []
    tables, remainder = _extract_markdown_table_block(text)
    highlights = _table_highlights(tables, limit=limit)
    remaining_slots = max(limit - len(highlights), 0)
    if remaining_slots:
        highlights.extend(_text_highlights(remainder, limit=remaining_slots))
    return highlights


def _focus_items_from_pairs(pairs: Sequence[Tuple[str, str]], limit: int = 5) -> List[str]:
    items: List[str] = []
    seen: Set[str] = set()
    for question, _ in pairs:
        cleaned = _clean_summary_line(question)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        items.append(_trim_highlight(cleaned))
        if len(items) >= limit:
            break
    return items


def _insight_items_from_pairs(
    pairs: Sequence[Tuple[str, str]],
    *,
    extra_insights: Optional[Sequence[str]] = None,
    limit: int = 6,
) -> List[str]:
    items: List[str] = []
    per_pair: List[Tuple[str, List[str]]] = []
    for question, answer in pairs:
        question_label = _trim_highlight(_clean_summary_line(question) or question or "Conversation insight")
        highlights = _extract_insight_highlights(answer, limit=4)
        if not highlights:
            fallback = _clean_summary_line(answer) or answer
            if fallback:
                highlights = [_trim_highlight(fallback)]
        if highlights:
            per_pair.append((question_label, highlights))

    for label, highlights in per_pair:
        if not highlights:
            continue
        summary = "; ".join(highlights[:3])
        items.append(f"{label}: {summary}")
        if len(items) >= limit:
            return items

    if extra_insights:
        for insight in extra_insights:
            snippet_highlights = _extract_insight_highlights(insight)
            if not snippet_highlights:
                continue
            summary = "; ".join(snippet_highlights[:3])
            if summary and summary not in items:
                items.append(summary)
            if len(items) >= limit:
                return items
    return items


def _derive_next_steps(
    focus_items: Sequence[str],
    insight_items: Sequence[str],
    memory_payload: Optional[Dict[str, Any]] = None,
) -> List[str]:
    text = " ".join(list(focus_items) + list(insight_items)).lower()
    steps: List[str] = []
    if not text and memory_payload:
        text = " ".join(str(v).lower() for v in memory_payload.values() if isinstance(v, str))

    price_keywords = ("price", "adr", "rate", "cost", "market")
    review_keywords = ("review", "feedback", "sentiment", "communication", "clean", "noise")
    amenity_keywords = ("amenit", "feature", "facility", "furnish", "bed", "kitchen")

    last_intent = ""
    if memory_payload:
        last_intent = str(memory_payload.get("last_intent") or "").upper()

    def _has(tokens: Sequence[str]) -> bool:
        return any(token in text for token in tokens)

    if _has(price_keywords) or last_intent in {"FACT_SQL", "FACT_SQL_RAG"}:
        steps.append("Compare pricing trends across boroughs or benchmark Highbury ADR against market averages.")
    if _has(review_keywords) or last_intent in {"REVIEWS_RAG", "RAG"}:
        steps.append("Dig into recent guest reviews to prioritize fixes (cleaning, responsiveness, access).")
    if _has(amenity_keywords):
        steps.append("Audit amenity coverage versus competitors and flag gaps that impact booking decisions.")

    if not steps:
        steps.append("Request deeper breakdowns or compare another borough, timeframe, or segment.")
    return steps[:2]


def _fallback_concise_from_trace(
    trace: Sequence[Dict[str, str]],
    *,
    extra_insights: Optional[Sequence[str]] = None,
    memory_payload: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    pairs = _trace_to_pairs(trace)
    focus_items = _focus_items_from_pairs(pairs)
    insight_items = _insight_items_from_pairs(pairs, extra_insights=extra_insights)

    next_steps = _derive_next_steps(focus_items, insight_items, memory_payload)

    sections = _build_concise_sections(focus_items, insight_items, next_steps)
    if sections:
        return sections
    fallback = _build_concise_sections(None, [trace[-1].get("content")] if trace else [], [])
    if fallback:
        return fallback
    return [{"title": "Summary", "items": ["Not enough conversation history to summarise."]}]


def _build_concise_summary(
    thread_candidates: Sequence[str],
    fallback_trace: Sequence[Dict[str, str]],
) -> Tuple[str, List[Dict[str, Any]]]:
    memory_trace = _load_trace_from_memory(thread_candidates, limit=None)
    memory_payload = _load_memory_payload(thread_candidates)
    memory_insights: List[str] = []
    for key in ("answer_summary", "last_summary", "last_answer"):
        val = memory_payload.get(key)
        if isinstance(val, str) and val.strip():
            memory_insights.append(val.strip())
    if not memory_insights:
        recent_bundle_summary = (memory_payload.get("last_sql") or {}).get("summary")
        if isinstance(recent_bundle_summary, str) and recent_bundle_summary.strip():
            memory_insights.append(recent_bundle_summary.strip())

    trace = _merge_traces(fallback_trace, memory_trace) if fallback_trace or memory_trace else []
    if not trace:
        trace = list(fallback_trace)
    if not trace:
        default_sections = [{"title": "Summary", "items": ["Not enough conversation history to summarise."]}]
        return _sections_to_markdown(default_sections), default_sections
    sections = _summarize_trace_with_llm(trace)
    if not sections:
        sections = _fallback_concise_from_trace(
            trace,
            extra_insights=memory_insights,
            memory_payload=memory_payload,
        )
    text = _sections_to_markdown(sections)
    if not text:
        text = "Not enough conversation history to summarise."
    return text, sections


def _pair_turns(messages: Sequence[Dict[str, Any]]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    pending_question: Optional[str] = None
    for message in messages:
        role = message.get("role")
        text = _message_text(message)
        if not text:
            continue
        if role == "user":
            pending_question = text
            continue
        if role == "assistant" and pending_question:
            pairs.append((pending_question, text))
            pending_question = None
    return pairs


def _build_detailed_summary(messages: Sequence[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    pairs = _pair_turns(messages)
    if not pairs:
        return "", []
    selected = pairs[-SUMMARY_PAIR_LIMIT:]
    start_index = len(pairs) - len(selected) + 1
    topics: List[Dict[str, Any]] = []
    markdown_blocks: List[str] = []
    for offset, (question, answer) in enumerate(selected):
        idx = start_index + offset
        question_text = str(question or "").strip()
        answer_text = str(answer or "")
        table_payloads, stripped_answer = _extract_markdown_table_block(answer_text)
        answer_lines = stripped_answer.splitlines() if stripped_answer else []
        question_items = _format_bullets([question_text])
        answer_items = _format_bullets(answer_lines) or _format_bullets([stripped_answer])
        topic = {
            "title": f"Topic {idx}",
            "question": question_text,
            "question_items": question_items,
            "answer": answer_text,
            "answer_items": answer_items,
        }
        if table_payloads:
            topic["answer_table"] = table_payloads[0]
            topic["answer_tables"] = table_payloads
        topics.append(topic)
        block_lines: List[str] = [f"**Topic {idx}**"]
        if question_items:
            block_lines.append("**Question**")
            block_lines.extend(f"- {item}" for item in question_items)
        if answer_items:
            block_lines.append("**Answer**")
            block_lines.extend(f"- {item}" for item in answer_items)
        if table_payloads:
            for payload in table_payloads:
                header_line = " | ".join(payload["headers"])
                block_lines.append(header_line)
                for row in payload["rows"]:
                    block_lines.append(" | ".join(row))
        markdown_blocks.append("\n".join(block_lines))
    return "\n\n".join(markdown_blocks), topics


def _stream_response(
    conversation_id: str,
    query: str,
    tenant: Optional[str],
    user_filters: Optional[Dict[str, Any]],
    history: List[Dict[str, str]],
    composer_enabled: Optional[bool],
    debug_thinking: Optional[bool],
):
    start_time = time.perf_counter()
    last_final_result: Optional[Dict[str, Any]] = None
    stream_tokens: List[str] = []
    logger.info("[STREAM] Started SSE for %s", conversation_id)

    # ------------------------------------------------------------------
    # Helper: detect and suppress SQL fragments while streaming tokens.
    # ------------------------------------------------------------------
    sql_stream_active = False  # True while we are discarding an SQL block
    sql_start_keywords = {"SELECT", "WITH", "INSERT", "UPDATE", "DELETE"}
    sql_inline_pattern = re.compile(r"\bSELECT\b.+\bFROM\b", re.S)

    def _sanitize_stream_token(text: str) -> tuple[str, bool]:
        """Return (sanitized_token, suppressed?) for the outgoing stream."""
        nonlocal sql_stream_active
        token = text or ""
        stripped = token.lstrip()
        suppressed = False

        def _consume_sql_chunk() -> tuple[str, bool]:
            nonlocal sql_stream_active
            suppressed_token = True
            if ";" in token:
                before, remainder = token.split(";", 1)
                # ensure we discard everything up to the first semicolon
                sql_stream_active = False
                return remainder.lstrip(), suppressed_token
            sql_stream_active = True
            return "", suppressed_token

        if sql_stream_active:
            remainder, _ = _consume_sql_chunk()
            return remainder, True

        if stripped:
            leading_word_match = re.match(r"([A-Za-z_]+)", stripped)
            leading_word = leading_word_match.group(1) if leading_word_match else ""
            if leading_word and leading_word == leading_word.upper() and leading_word in sql_start_keywords:
                remainder, _ = _consume_sql_chunk()
                return remainder, True

        if sql_inline_pattern.search(token):
            remainder, _ = _consume_sql_chunk()
            return remainder, True

        return token, suppressed
    # ------------------------------------------------------------------

    async def event_stream():
        try:
            async for event in stream_agent(
                query,
                tenant=tenant,
                user_filters=user_filters,
                history=history,
                composer_enabled=composer_enabled,
                debug_thinking=debug_thinking,
                thread_id=conversation_id,
            ):
                if event.get("type") == "token":
                    raw_token = (
                        event.get("payload")
                        or event.get("value")
                        or event.get("content")
                        or ""
                    )
                    if raw_token:
                        sanitized_token, suppressed = _sanitize_stream_token(raw_token)
                        if suppressed:
                            logger.warning("[STREAM_GUARD] Suppressed SQL token from stream.")
                        if sanitized_token:
                            stream_tokens.append(sanitized_token)
                            payload = {"type": "token", "content": sanitized_token}
                            yield f"data: {json.dumps(payload)}\n\n"

                elif event.get("type") == "final":
                    result = normalise_payload(event.get("payload") or event.get("value") or {})
                    streamed_markdown = "".join(stream_tokens).strip()
                    if streamed_markdown and isinstance(result, dict):
                        existing_content = result.get("content")
                        content_text = existing_content.strip() if isinstance(existing_content, str) else ""
                        if not content_text or len(streamed_markdown) >= len(content_text):
                            result["content"] = streamed_markdown
                        existing_answer = result.get("answer_text")
                        answer_text = existing_answer.strip() if isinstance(existing_answer, str) else ""
                        if not answer_text or len(streamed_markdown) >= len(answer_text):
                            result["answer_text"] = streamed_markdown
                        bundle = result.get("result_bundle")
                        if not isinstance(bundle, dict):
                            bundle = {} if bundle in (None, "", []) else {"raw": bundle}
                        summary_value = bundle.get("summary") if isinstance(bundle, dict) else None
                        summary_text = summary_value.strip() if isinstance(summary_value, str) else ""
                        if isinstance(bundle, dict):
                            if not summary_text or len(streamed_markdown) >= len(summary_text):
                                bundle["summary"] = streamed_markdown
                            result["result_bundle"] = bundle
                    stream_tokens.clear()
                    elapsed = time.perf_counter() - start_time
                    result.setdefault("telemetry", {})["total_latency_s"] = elapsed
                    result.setdefault("telemetry", {})["latency_ms"] = elapsed * 1000
                    result["latency"] = round(elapsed, 2)
                    updated = _finalize_assistant_message(conversation_id, query, result, elapsed)
                    if isinstance(updated, dict):
                        last_final_result = updated
                    payload_json = safe_json(normalise_payload(updated))
                    payload = {"type": "final", "payload": payload_json}
                    final_text = payload_json.get("content") if isinstance(payload_json, dict) else None
                    if isinstance(final_text, str) and "select" in final_text.lower():
                        logging.warning(
                            "[LEAK_DETECTOR][BACKEND_RESPONSE] SQL text visible in streaming response:\n%s",
                            final_text[:200],
                        )
                    yield f"data: {json.dumps(payload)}\n\n"
            if last_final_result is not None:
                logger.warning(
                    "[DEBUG_PERSIST_CHECK] Final message payload keys: %s",
                    list(last_final_result.keys()),
                )
                if "payload" in last_final_result:
                    try:
                        payload_preview = json.dumps(
                            last_final_result.get("payload"),
                            indent=2,
                            default=str,
                        )[:400]
                    except Exception as exc:  # pragma: no cover - defensive logging
                        payload_preview = f"<unserializable payload: {exc}>"
                    logger.warning("[DEBUG_PERSIST_CHECK] Payload snippet: %s", payload_preview)
                else:
                    logger.warning("[DEBUG_PERSIST_CHECK] No payload field in result!")
            else:
                logger.warning("[DEBUG_PERSIST_CHECK] No final result captured before done event.")
            yield 'data: {"type": "done"}\n\n'
        except Exception as exc:  # pragma: no cover - streaming failure fallback
            if "No template matched" in str(exc):
                updated = _insert_greeting_response(conversation_id, query)
                payload_json = safe_json(normalise_payload(updated))
                payload = {"type": "final", "payload": payload_json}
                yield f"data: {json.dumps(payload)}\n\n"
                yield 'data: {"type": "done"}\n\n'
                return
            logger.error("[Conversations] Streaming error: %s", exc, exc_info=True)
            updated = _insert_error_response(conversation_id, query, str(exc))
            payload_json = safe_json(normalise_payload(updated))
            payload = {"type": "error", "error": str(exc), "payload": payload_json}
            yield f"data: {json.dumps(payload)}\n\n"
            yield 'data: {"type": "done"}\n\n'

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


@app.post("/api/conversations/{conversation_id}/messages")
async def post_message(conversation_id: str, payload: dict):
    logger.info("[Conversations] Incoming payload for %s: %s", conversation_id, payload)

    query_raw = (payload.get("query") or payload.get("content") or "").strip()
    if not query_raw:
        return JSONResponse(status_code=400, content={"error": "Question cannot be empty."})

    tenant = payload.get("tenant")
    user_filters = payload.get("user_filters")
    composer_enabled = payload.get("composer_enabled")
    stream_requested = bool(payload.get("stream"))
    debug_thinking = bool(
        payload.get("debug_thinking")
        or (isinstance(payload.get("depth"), str) and payload.get("depth") == "deep")
    )

    try:
        history = _record_user_message(conversation_id, query_raw)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - database failures
        logger.error("[Conversations] Unable to persist user message: %s", exc, exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Unable to record message."})

    if stream_requested:
        return _stream_response(
            conversation_id,
            query_raw,
            tenant,
            user_filters,
            history,
            composer_enabled,
            debug_thinking,
        )

    try:
        result, elapsed = _execute_langgraph(
            query_raw,
            tenant,
            user_filters,
            history,
            thread_id=conversation_id,
            composer_enabled=composer_enabled,
            debug_thinking=debug_thinking,
        )
        # [GRAPH_STATE_FIX] Promote markdown content from extras ONLY if missing at top level
        if not result.get("content"):
            extras = result.get("extras", {}) or {}
            answer_text = extras.get("answer_text")
            content_text = extras.get("content")
            if answer_text or content_text:
                result["content"] = answer_text or content_text
                logger.warning("[GRAPH_STATE_FIX] Promoted content from extras (RAG mode only)")
        updated = _finalize_assistant_message(conversation_id, query_raw, result, elapsed)
        safe_response = safe_json(normalise_payload(updated))
        return JSONResponse(content=safe_response)
    except Exception as exc:
        if "No template matched" in str(exc):
            updated = _insert_greeting_response(conversation_id, query_raw)
            safe_response = safe_json(normalise_payload(updated))
            return JSONResponse(content=safe_response, status_code=200)
        logger.error("[Conversations] Error handling query: %s", exc, exc_info=True)
        updated = _insert_error_response(conversation_id, query_raw, str(exc))
        safe_response = safe_json(normalise_payload(updated))
        return JSONResponse(status_code=500, content={"error": str(exc), "conversation": safe_response})



@app.delete("/api/conversations/{conversation_id}")
async def api_delete_conversation(conversation_id: str):
    try:
        with open_connection() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", [conversation_id])
            conn.execute("DELETE FROM conversations WHERE id = ?", [conversation_id])
            remaining = list_conversations(conn)
        safe_response = safe_json(normalise_payload(remaining))
        return JSONResponse(content=safe_response)
    except HTTPException:
        raise
    except Exception as exc:
        import traceback

        print("❌ Message Error:", traceback.format_exc())
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@app.delete("/api/conversations/{conversation_id}/messages/{message_id}")
async def api_delete_message(conversation_id: str, message_id: str):
    try:
        with open_connection() as conn:
            conn.execute("DELETE FROM messages WHERE id = ? AND conversation_id = ?", [message_id, conversation_id])
            # Update updated_at to last message timestamp when available
            last_timestamp = conn.execute(
                "SELECT MAX(timestamp) FROM messages WHERE conversation_id = ?",
                [conversation_id],
            ).fetchone()[0]
            if last_timestamp:
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    [last_timestamp, conversation_id],
                )
            conversation = fetch_conversation(conn, conversation_id)
        safe_response = safe_json(normalise_payload(conversation))
        return JSONResponse(content=safe_response)
    except HTTPException:
        raise
    except Exception as exc:
        import traceback

        print("❌ Message Error:", traceback.format_exc())
        return JSONResponse(content={"error": str(exc)}, status_code=500)


def _build_conversation_summary(conversation_id: str) -> ConversationSummaryResponse:
    with open_connection(read_only=True) as conn:
        conversation = fetch_conversation(conn, conversation_id)
        thread_candidates = _resolve_thread_candidates(conn, conversation_id)

    messages = conversation.get("messages") or []
    trace = _prepare_trace(messages, limit=None)
    if len(trace) < 2:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Not enough data to summarise.")

    concise_text, concise_sections = _build_concise_summary(thread_candidates, trace)
    detailed_text, detailed_topics = _build_detailed_summary(messages)
    detailed = detailed_text or concise_text

    return ConversationSummaryResponse(
        conversation_id=conversation_id,
        concise=concise_text,
        detailed=detailed,
        concise_sections=concise_sections,
        detailed_topics=detailed_topics,
    )


@app.post("/api/conversations/{conversation_id}/summary")
async def api_conversation_summary(conversation_id: str):
    try:
        summary = _build_conversation_summary(conversation_id)
        payload = summary.dict()
        safe_response = safe_json(payload)
        return JSONResponse(content=safe_response)
    except HTTPException:
        raise
    except Exception as exc:
        import traceback

        print("❌ Message Error:", traceback.format_exc())
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@app.post("/api/conversations/{conversation_id}/summary/email")
async def api_conversation_summary_email(conversation_id: str, req: ConversationSummaryEmailRequest):
    try:
        summary = _build_conversation_summary(conversation_id)
        variant = req.variant or "concise"
        summary_text = summary.concise if variant == "concise" else summary.detailed
        summary_text = (summary_text or summary.concise or summary.detailed or "No summary available.").strip()

        if variant == "concise":
            subject = "Concise Chat Summary from wtchtwr"
            intro = "Please find below your concise chat summary:"
        else:
            subject = "Detailed Chat Summary from wtchtwr"
            intro = "Please find below your detailed chat summary:"

        summary_type = f"{variant.title()} Summary"
        body, html_body = _build_email_content(
            summary_type,
            conversation_id,
            intro,
            summary_text,
            concise_sections=summary.concise_sections if variant == "concise" else None,
            detailed_topics=summary.detailed_topics if variant == "detailed" else None,
        )

        _ensure_email_configured()

        try:
            send_email(req.email, subject, body, html_body=html_body)
        except RuntimeError as exc:
            raise _email_http_exception(exc)

        return {"detail": f"{subject} emailed to {req.email}", "variant": variant}
    except HTTPException:
        raise
    except Exception as exc:
        import traceback

        print("❌ Message Error:", traceback.format_exc())
        return JSONResponse(content={"error": str(exc)}, status_code=500)


def export_table_to_csv(table: Dict[str, Any]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(table.get("columns", []))
    for row in table.get("data", []):
        writer.writerow([to_primitive(row.get(col)) for col in table.get("columns", [])])
    return buffer.getvalue().encode("utf-8")


def table_to_dataframe(table: Dict[str, Any]) -> pd.DataFrame:
    columns = table.get("columns") or []
    rows = table.get("data", []) or []
    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(columns=columns)
    if columns:
        for col in columns:
            if col not in df.columns:
                df[col] = None
        df = df[columns]
    return df


@app.post("/api/conversations/{conversation_id}/messages/{message_id}/export")
async def api_export_message(conversation_id: str, message_id: str, req: ExportMessageRequest):
    try:
        with open_connection(read_only=True) as conn:
            conversation = fetch_conversation(conn, conversation_id)
        message = next((m for m in conversation["messages"] if m["id"] == message_id), None)
        if not message or message["role"] != "assistant":
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assistant message not found.")
        payload = message.get("payload") or {}
        tables = payload.get("tables") or []
        if not tables:
            action = ExportActionResponse(delivery=req.delivery, detail="This response has no tabular data.").dict()
            safe_response = safe_json(action)
            return JSONResponse(content=safe_response)
        if req.table_index >= len(tables):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid table index.")

        table = tables[req.table_index]
        table_name = table.get("name") or "results"
        csv_bytes = export_table_to_csv(table)
        row_count = int(table.get("row_count") or len(table.get("data", [])))

        if req.delivery == "email":
            recipient = (req.email or "").strip()
            if not recipient:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail="Recipient email is required for email delivery."
                )
            _ensure_email_configured()
            email_mode = (req.email_mode or "csv").lower()
            include_csv = email_mode in {"csv", "both"}
            include_sql = email_mode in {"sql", "both"}

            attachments: List[Tuple[str, bytes, str]] = []
            delivery_mode = None
            download_link: Optional[str] = None

            if include_csv:
                try:
                    file_delivery = _prepare_csv_delivery(table_name, csv_bytes, recipient=recipient)
                except RuntimeError as exc:
                    raise _email_http_exception(exc)
                attachments = file_delivery.attachments
                delivery_mode = file_delivery.mode
                download_link = file_delivery.link
            sql_text = (payload.get("sql") or "").strip()

            summary_lines = [
                f"Table: {table_name}",
                f"Rows: {row_count}",
            ]
            if download_link:
                summary_lines.append(f"Download link: {download_link}")
            if include_sql and sql_text:
                summary_lines.extend(["", "SQL Query:", sql_text])

            intro = f"Here is the requested export for '{table_name}'."
            summary_text = "\n".join(summary_lines)
            body, html_body = _build_email_content("Conversation Export", conversation_id, intro, summary_text)
            subject = f"wtchtwr Conversation Export • {table_name}"

            try:
                send_email(recipient, subject, body, attachments=attachments, html_body=html_body)
            except RuntimeError as exc:
                raise _email_http_exception(exc)

            detail_parts = [f"Export emailed to {recipient}"]
            if delivery_mode == "drive":
                detail_parts.append("Shared via Google Drive link.")
            elif delivery_mode == "zip":
                detail_parts.append("Sent as ZIP attachment.")
            action = ExportActionResponse(delivery="email", detail=" ".join(detail_parts)).dict()
            safe_response = safe_json(action)
            return JSONResponse(content=safe_response)

        label = table_name or "results"
        if len(csv_bytes) <= _EMAIL_ATTACHMENT_LIMIT:
            df = table_to_dataframe(table)
            metadata_dict = exporter.stage_csv(df, label=label)
            metadata_dict["rows"] = row_count
            metadata_dict["session_only"] = False
            metadata = ExportMetadata(**metadata_dict)
            action = ExportActionResponse(delivery="download", metadata=metadata).dict()
            safe_response = safe_json(action)
            return JSONResponse(content=safe_response)

        zipped = _zip_bytes(f"{_safe_filename(label)}.csv", csv_bytes)
        if drive_uploader.enabled():
            link = drive_uploader.upload_bytes(
                name=f"{_safe_filename(label)}.zip",
                content=zipped,
                mimetype="application/zip",
            )
            logger.info("Large conversation export uploaded to Drive: %s", link)
            detail = f"Large export uploaded to Google Drive: {link}"
            action = ExportActionResponse(delivery="drive", detail=detail).dict()
            safe_response = safe_json(action)
            return JSONResponse(content=safe_response)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Export exceeds 25 MB and Google Drive upload is not configured. "
                "Set GDRIVE_* environment variables to enable large downloads."
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        import traceback

        print("❌ Message Error:", traceback.format_exc())
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@app.get("/api/exports/{token}")
async def api_download_export(token: str):
    media_type = "text/csv"
    try:
        payload, filename = exporter.get_csv(token)
    except KeyError:
        data = get_export_blob(token)
        if not data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Export expired or missing.")
        filename, mime_type, payload = data
        media_type = mime_type
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(io.BytesIO(payload), media_type=media_type, headers=headers)


# ---------------------------------------------------------------------------
# API endpoints – Data explorer
# ---------------------------------------------------------------------------


@app.get("/api/data-explorer/schema")
async def api_data_explorer_schema():
    """Return filtered list of allowed tables and columns."""
    try:
        schema = EXPLORER.list_tables()
        tables_payload = [
            {
                "name": meta.name,
                "columns": [
                    {
                        "name": col.name,
                        "data_type": col.data_type,
                        "description": col.description,
                    }
                    for col in meta.columns
                ],
            }
            for meta in schema.values()
        ]
        return {"tables": tables_payload}
    except DataExplorerError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)
    except Exception as exc:  # pragma: no cover - defensive
        import traceback

        print("❌ Schema error:", traceback.format_exc())
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@app.post("/api/data-explorer/query")
async def api_data_explorer_query(payload: Dict[str, Any]):
    """Run a structured query for the Data Export preview."""
    try:
        result = EXPLORER.structured_query(payload)
        rows = result.dataframe.to_dict(orient="records")
        response_payload = {
            "sql": result.sql,
            "columns": result.selected_columns,
            "rows": rows,
            "row_count": len(rows),
            "limit": result.limit,
            "tables": result.tables,
        }
        safe_response = safe_json(response_payload)
        return JSONResponse(content=safe_response)
    except DataExplorerError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)
    except Exception as exc:  # pragma: no cover - defensive
        import traceback

        print("❌ Query error:", traceback.format_exc())
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@app.post("/api/data-explorer/export")
async def api_data_explorer_export(payload: Dict[str, Any]):
    """Prepare a CSV export for the selected tables."""
    try:
        delivery = (payload.get("delivery") or "download").lower()
        if delivery not in {"download", "email"}:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported delivery option.")

        query_payload = {k: v for k, v in payload.items() if k not in {"delivery", "email"}}
        result = EXPLORER.structured_query(query_payload, limit_cap=None)
        csv_bytes = result.dataframe.to_csv(index=False).encode("utf-8")
        export_tables = result.tables or query_payload.get("tables") or []
        tables_label = "-".join(export_tables) if export_tables else "data-export"
        join_detail = (
            "Join configuration detected—results may not reflect join intent."
            if query_payload.get("joins")
            else None
        )

        if delivery == "email":
            recipient = (payload.get("email") or "").strip()
            if not recipient:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email recipient is required for email delivery.",
                )
            _ensure_email_configured()
            subject = "wtchtwr Data Explorer Export"
            tables_display = ", ".join(export_tables) if export_tables else "your selected data"
            row_count = len(result.dataframe.index)
            conversation_ref = str(payload.get("conversation_id") or "N/A")
            intro = f"Attached is your wtchtwr Data Explorer export for {tables_display}."
            summary_text = f"Rows: {row_count}\nGenerated at: {isoformat(utcnow())}"
            if join_detail:
                summary_text += f"\n{join_detail}"
            try:
                file_delivery = _prepare_csv_delivery(tables_label or "data-export", csv_bytes, recipient=recipient)
            except RuntimeError as exc:
                raise _email_http_exception(exc)
            if file_delivery.link:
                summary_text += f"\nDownload link: {file_delivery.link}"
            body, html_body = _build_email_content(
                "Data Explorer Export",
                conversation_ref,
                intro,
                summary_text,
            )
            try:
                send_email(
                    recipient,
                    subject,
                    body,
                    attachments=file_delivery.attachments,
                    html_body=html_body,
                )
            except RuntimeError as exc:
                raise _email_http_exception(exc)

            detail_parts = [f"Data export emailed to {recipient}"]
            if file_delivery.mode == "drive":
                detail_parts.append("Shared via Google Drive link.")
            elif file_delivery.mode == "zip":
                detail_parts.append("Sent as ZIP attachment.")
            if join_detail:
                detail_parts.append(join_detail)
            action = ExportActionResponse(delivery="email", detail=" ".join(detail_parts)).dict()
            safe_response = safe_json(action)
            return JSONResponse(content=safe_response)

        label = tables_label or "data-export"
        base_csv_name = f"{_safe_filename(label)}.csv"

        if len(csv_bytes) <= _EMAIL_ATTACHMENT_LIMIT:
            metadata_dict = exporter.stage_csv(result.dataframe, label=label)
            metadata_dict["session_only"] = False
            metadata = ExportMetadata(**metadata_dict)
            action = ExportActionResponse(delivery="download", metadata=metadata, detail=join_detail).dict()
            safe_response = safe_json(action)
            return JSONResponse(content=safe_response)

        zipped = _zip_bytes(base_csv_name, csv_bytes)
        if drive_uploader.enabled():
            link = drive_uploader.upload_bytes(
                name=f"{_safe_filename(label)}.zip",
                content=zipped,
                mimetype="application/zip",
            )
            logger.info("Large Data Explorer export uploaded to Drive: %s", link)
            detail_parts = [f"Large export uploaded to Google Drive: {link}"]
            if join_detail:
                detail_parts.append(join_detail)
            action = ExportActionResponse(delivery="drive", detail=" ".join(detail_parts)).dict()
            safe_response = safe_json(action)
            return JSONResponse(content=safe_response)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Export exceeds 25 MB and Google Drive upload is not configured. "
                "Set GDRIVE_* environment variables to enable large downloads."
            ),
        )
    except DataExplorerError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)
    except Exception as exc:  # pragma: no cover - defensive
        import traceback

        print("❌ Export error:", traceback.format_exc())
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@app.get("/api/data-explorer/column-values/{table}/{column}")
@app.get("/api/data-explorer/tables/{table}/columns/{column}/values")
async def api_data_explorer_column_values(table: str, column: str):
    """Return distinct column values for filter dropdowns."""
    try:
        values = EXPLORER.distinct_values(table, column)
        return {"values": values}
    except DataExplorerError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)
    except Exception as exc:  # pragma: no cover - defensive
        import traceback

        print("❌ Column value error:", traceback.format_exc())
        return JSONResponse(content={"error": str(exc)}, status_code=500)


# ---------------------------------------------------------------------------
# Misc endpoints
# ---------------------------------------------------------------------------


@app.post("/api/slackbot/start")
async def api_slackbot_start():
    return JSONResponse({"detail": "Slackbot trigger acknowledged."})


@app.get("/api/slackbot/status")
async def api_slackbot_status():
    global SLACK_RUNNING
    running = SLACK_RUNNING and bool(SLACK_THREAD and SLACK_THREAD.is_alive())
    if SLACK_RUNNING and not running:
        SLACK_RUNNING = False
    return {"running": running}


@app.on_event("startup")
async def startup_event() -> None:
    with open_connection() as conn:
        ensure_schema(conn)
        purge_expired_exports()
    _start_slack_bot_thread()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global SLACK_THREAD, SLACK_RUNNING
    SLACK_RUNNING = False
    thread = SLACK_THREAD
    if thread and thread.is_alive():
        thread.join(timeout=2)
    SLACK_THREAD = None
    logger.info("[SlackBot] Stopped gracefully.")

# [LEAK_DETECTOR]: Added logging to monitor SQL text leakage.
