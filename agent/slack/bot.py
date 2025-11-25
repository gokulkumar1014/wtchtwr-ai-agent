"""Slack bot for wtchtwr that reuses the same conversation pipeline as the web UI."""

from __future__ import annotations
import logging
import os
import re
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError
from slack_sdk.web import WebClient

from agent.config import load_config

# ---------------------------------------------------------------------------
# Globals and constants
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
MENTION_RE = re.compile(r"<@[^>]+>")
GREETING_PATTERN = re.compile(
    r"^(hi|hello|hey|good (?:morning|afternoon|evening)|howdy|yo|sup)[!. ]*$",
    re.IGNORECASE,
)

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
BACKEND_BASE_URL = os.getenv("SLACK_BACKEND_URL", "http://127.0.0.1:8000")
DASHBOARD_URL = f"{FRONTEND_URL}/dashboard"
HELP_URL = FRONTEND_URL
EXPORT_URL = f"{DASHBOARD_URL}#export"
BACKEND_CONNECT_TIMEOUT = float(os.getenv("SLACK_BACKEND_CONNECT_TIMEOUT", "10"))
BACKEND_READ_TIMEOUT = float(os.getenv("SLACK_BACKEND_TIMEOUT", "240"))
# Tuple -> (connect timeout, read timeout)
BACKEND_TIMEOUT = (BACKEND_CONNECT_TIMEOUT, BACKEND_READ_TIMEOUT)

GREETING_RESPONSE = "Hi there! I’m wtchtwr — your portfolio co-pilot. How can I help?"
WELCOME_MESSAGE = (
    "Hi there! I’m *wtchtwr* 👋\n\n"
    "wtchtwr is a portfolio intelligence co-pilot for STR operators. It blends NL→SQL analytics, review retrieval, sentiment scoring, portfolio triage, competitor benchmarking, and amenity diagnostics so you can interrogate every KPI from one prompt bar.\n\n"
    "Ask me things like:\n"
    "• \"Average occupancy rate for next 60 days for my listings in Manhattan\"\n"
    "• \"Which of my properties have paid parking?\"\n"
    "• \"Compare avg price in Midtown: mine vs market\"\n"
    "• \"Show the latest 5 reviews for listing 2595\"\n"
    "• \"Be a portfolio triage agent and diagnose Highbury listings in Manhattan\"\n\n"
    "Shortcuts: *help* for prompts · *dashboard* to open wtchtwr · *clear* to reset · *export* for export options."
)

MAX_CACHE_ENTRIES = 50
_RESULT_CACHE: Dict[str, Dict[str, Any]] = {}
_THREAD_CONVERSATIONS: Dict[str, str] = {}
_LAST_PAYLOAD: Optional[Dict[str, Any]] = None

# ---------------------------------------------------------------------------
# LangGraph runner
# ---------------------------------------------------------------------------

def _ensure_config_loaded() -> None:
    """Ensure shared agent configuration is initialised once."""
    load_config(refresh=False)


def _ensure_conversation(thread_key: str) -> str:
    """Create or reuse a conversation mapped to a Slack thread/channel."""
    if thread_key in _THREAD_CONVERSATIONS:
        return _THREAD_CONVERSATIONS[thread_key]
    try:
        resp = requests.post(f"{BACKEND_BASE_URL}/api/conversations", timeout=BACKEND_TIMEOUT)
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(f"Backend conversation bootstrap timed out after {BACKEND_READ_TIMEOUT} seconds.") from exc
    logger.info("[Slack] create conversation -> %s", resp.status_code)
    resp.raise_for_status()
    data = resp.json()
    convo_id = data.get("id")
    if not convo_id:
        raise RuntimeError("Unable to create conversation for Slack thread.")
    _THREAD_CONVERSATIONS[thread_key] = convo_id
    return convo_id


def _invoke_backend(question: str, *, thread_key: str) -> Dict[str, Any]:
    """Call the same backend conversation endpoint the web UI uses."""
    convo_id = _ensure_conversation(thread_key)
    payload = {"query": question, "stream": False}
    try:
        resp = requests.post(
            f"{BACKEND_BASE_URL}/api/conversations/{convo_id}/messages",
            json=payload,
            timeout=BACKEND_TIMEOUT,
        )
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"Backend took longer than {BACKEND_READ_TIMEOUT} seconds to respond. "
            "Try narrowing the question or raise SLACK_BACKEND_TIMEOUT."
        ) from exc
    logger.info("[Slack] backend conversation %s -> %s", convo_id, resp.status_code)
    if not resp.ok:
        logger.error("[Slack] backend error: %s", resp.text[:500])
    resp.raise_for_status()
    conversation = resp.json()
    messages = conversation.get("messages") or []
    assistant = next((m for m in reversed(messages) if m.get("role") == "assistant"), None)
    if not assistant:
        raise RuntimeError("No assistant message returned from backend.")
    return assistant

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _split_md_row(row: str) -> List[str]:
    """Split a markdown table row into cells."""
    trimmed = row.strip()
    if trimmed.startswith("|"):
        trimmed = trimmed[1:]
    if trimmed.endswith("|"):
        trimmed = trimmed[:-1]
    return [cell.strip() for cell in trimmed.split("|")]


def _render_markdown_table_block(lines: List[str]) -> str:
    """Convert a markdown table block to a monospace code block for Slack."""
    if not lines:
        return ""

    rows: List[List[str]] = []
    for ln in lines:
        cells = _split_md_row(ln)
        if cells and any(cells):
            rows.append(cells)

    # Drop separator row if present
    if len(rows) >= 2:
        separator = rows[1]
        if all(re.fullmatch(r":?-{2,}:?", cell) for cell in separator):
            rows.pop(1)

    if not rows:
        return ""

    col_count = max(len(r) for r in rows)
    widths = [0] * col_count
    for r in rows:
        for idx in range(col_count):
            val = r[idx] if idx < len(r) else ""
            widths[idx] = max(widths[idx], len(val))

    def fmt_row(row: List[str]) -> str:
        padded = []
        for idx in range(col_count):
            val = row[idx] if idx < len(row) else ""
            padded.append(val.ljust(widths[idx]))
        return " | ".join(padded)

    formatted_rows = [fmt_row(r) for r in rows]
    divider = " | ".join("-" * w for w in widths)
    body_lines = [formatted_rows[0], divider] + formatted_rows[1:]
    body = "\n".join(body_lines)
    return f"```text\n{body}\n```"


def _normalize_summary(message: Dict[str, Any]) -> str:
    payload = message.get("payload") or {}
    primary = (message.get("content") or message.get("nl_summary") or "").strip()
    if primary:
        return primary
    summary = (payload.get("summary") or "").strip()
    if summary:
        return summary
    return "No insight available yet — try refining the question."


def _coerce_columns(bundle: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> List[str]:
    columns = list(bundle.get("columns") or [])
    if not columns and rows:
        example = rows[0]
        if isinstance(example, dict):
            columns = list(example.keys())
    return columns


def _format_rows_preview(columns: Sequence[str], rows: Sequence[Dict[str, Any]], limit: int = 8) -> str:
    """Render a small markdown table preview."""
    if not rows:
        return "(no rows returned)"
    header = " | ".join(str(c) for c in columns) if columns else ""
    divider = " | ".join("---" for _ in columns) if columns else ""
    lines = []
    for row in rows[:limit]:
        if isinstance(row, dict):
            values = [row.get(c, "") for c in columns] if columns else list(row.values())
        else:
            values = row
        lines.append(" | ".join(str(v) for v in values))
    parts = [header, divider] if header else []
    parts.extend(lines)
    return "\n".join(parts).strip() or "(no rows returned)"


def _build_table_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    tables = payload.get("tables") or []
    normalized: List[Dict[str, Any]] = []
    for table in tables:
        if not isinstance(table, dict):
            continue
        name = table.get("name", "Result")
        columns = table.get("columns") or []
        rows = table.get("data") or []
        preview = table.get("preview") or ""
        row_count = table.get("row_count") or len(rows)
        if not preview and columns:
            preview = _format_rows_preview(columns, rows)
        normalized.append(
            {
                "name": name,
                "columns": columns,
                "data": rows[:50],
                "preview": preview or "(no rows returned)",
                "row_count": row_count,
            }
        )
    return normalized


def _build_payload(question: str, message: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble a unified Slack message payload from the web conversation response."""
    payload = message.get("payload") or {}
    telemetry = payload.get("telemetry") or payload.get("policy") or {}
    tables = _build_table_payload(payload)
    sql_text = payload.get("sql") or ""
    summary = _normalize_summary(message)
    return {
        "question": question,
        "policy": payload.get("policy") or telemetry.get("policy"),
        "sql": sql_text,
        "tables": tables,
        "summary": summary,
        "nl_summary": summary,
        "row_count": tables[0]["row_count"] if tables else 0,
        "response_type": payload.get("response_type") or payload.get("policy") or "sql",
        "telemetry": payload.get("telemetry") or {},
        "markdown_table": payload.get("markdown_table"),
        "filters": payload.get("applied_filters") or payload.get("filters") or {},
    }

# ---------------------------------------------------------------------------
# Core interaction handlers
# ---------------------------------------------------------------------------

def handle_question(question: str, *, thread_id: Optional[str] = None) -> Dict[str, Any]:
    """Resolve a Slack question via the same web conversation pipeline and return payload + summary."""
    global _LAST_PAYLOAD
    question_clean = (question or "").strip()
    if not question_clean:
        return {"error": "Please provide a question for me to analyse."}

    thread_key = thread_id or "default"
    try:
        assistant_message = _invoke_backend(question_clean, thread_key=thread_key)
    except Exception as exc:
        logger.exception("Slackbot failed to execute conversation pipeline.")
        return {"error": f"Something went wrong while running the query: {exc}"}

    payload = _build_payload(question_clean, assistant_message)
    _LAST_PAYLOAD = payload
    return {"payload": payload, "summary": payload["summary"]}


def _strip_mentions(text: str) -> str:
    return MENTION_RE.sub("", text or "").strip()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _store_payload(channel: str, payload: Dict[str, Any], thread_ts: Optional[str]) -> str:
    """Cache Slack thread responses for modals."""
    key = str(uuid.uuid4())
    _RESULT_CACHE[key] = {"payload": payload, "channel": channel, "thread_ts": thread_ts}
    if len(_RESULT_CACHE) > MAX_CACHE_ENTRIES:
        oldest_key = next(iter(_RESULT_CACHE))
        _RESULT_CACHE.pop(oldest_key, None)
    return key


def _get_cached_payload(key: str) -> Optional[Dict[str, Any]]:
    return _RESULT_CACHE.get(key)


# ---------------------------------------------------------------------------
# Conversation utilities
# ---------------------------------------------------------------------------

def _dashboard_message() -> str:
    return f":bar_chart: *wtchtwr Dashboard*\n<{DASHBOARD_URL}|Open the wtchtwr dashboard>"


def _clear_conversation(client: WebClient, channel: str, bot_user_id: str) -> str:
    """Delete previous messages in a channel (best-effort; Slack may restrict non-bot deletions)."""
    try:
        history = client.conversations_history(channel=channel, limit=200)
    except SlackApiError as exc:
        return f"I couldn't fetch the conversation history ({exc.response['error']})."

    deleted = 0
    for message in history.get("messages", []):
        ts = message.get("ts")
        if not ts:
            continue
        try:
            client.chat_delete(channel=channel, ts=ts)
            deleted += 1
        except SlackApiError:
            continue

    return ""


def _respond(client: WebClient, channel: str, text: str,
             thread_ts: Optional[str] = None,
             blocks: Optional[Iterable[Dict[str, Any]]] = None) -> None:
    kwargs = {"channel": channel, "text": text}
    if thread_ts:
        kwargs["thread_ts"] = thread_ts
    if blocks:
        kwargs["blocks"] = list(blocks)
    client.chat_postMessage(**kwargs)


def _chunk_text(text: str, limit: int = 2800) -> List[str]:
    """Chunk large markdown into Slack-safe sections without dropping content."""
    text = text or ""
    if len(text) <= limit:
        return [text] if text else []
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for line in text.splitlines():
        add_len = len(line) + 1  # newline
        if current_len + add_len > limit and current:
            chunks.append("\n".join(current))
            current = [line]
            current_len = len(line) + 1
        else:
            current.append(line)
            current_len += add_len
    if current:
        chunks.append("\n".join(current))
    return chunks


def _format_for_slack(text: str) -> str:
    """
    Transform web-optimized markdown into Slack-friendly mrkdwn.
    - Remove code fences and convert markdown tables to monospace blocks.
    - Simplify headings to bold lines.
    - Normalize bold markers.
    - Soft-wrap single newlines within paragraphs, keep double newlines.
    - Keep bullets readable.
    """
    if not text:
        return ""

    # Branding normalization
    text = text.replace("\r\n", "\n")
    text = re.sub(r"(?i)H\.?O\.?P\.?E\.?|Highbury Occupancy Property Engine", "wtchtwr", text)

    lines = text.split("\n")
    cleaned: List[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip()
        stripped = line.strip()
        # Skip code fences
        if stripped.startswith("```"):
            i += 1
            # skip until closing fence
            while i < n and not lines[i].strip().startswith("```"):
                i += 1
            i += 1
            continue
        # Convert markdown tables in summaries into Slack monospace tables
        if "|" in line and (i + 1) < n and re.search(r"\|\s*-", lines[i + 1]):
            j = i
            table_lines: List[str] = []
            while j < n and "|" in lines[j]:
                table_lines.append(lines[j])
                j += 1
            cleaned.append(_render_markdown_table_block(table_lines))
            i = j
            continue
        cleaned.append(line)
        i += 1

    processed: List[str] = []
    for line in cleaned:
        if line.startswith("```"):
            processed.append(line)
            continue
        line = re.sub(r"\*\*(.+?)\*\*", r"*\1*", line)
        heading_match = re.match(r"^\s*#{1,6}\s*(.+?)\s*$", line)
        if heading_match:
            heading_text = heading_match.group(1).strip()
            heading_text = re.sub(r"^\*+(.*?)\*+$", r"\1", heading_text).strip()
            processed.append(f"*{heading_text}*")
            continue
        bullet_match = re.match(r"^\s*[-*]\s+(.*)$", line)
        if bullet_match:
            processed.append(f"• {bullet_match.group(1).strip()}")
            continue
        processed.append(line)

    joined = "\n".join(processed)
    paragraphs = joined.split("\n\n")
    normalized: List[str] = []
    for para in paragraphs:
        para_strip = para.strip()
        if para_strip.startswith("```"):
            normalized.append(para_strip)
            continue
        soft = re.sub(r"\s*\n\s*", " ", para_strip).strip()
        if soft:
            normalized.append(soft)
    return "\n\n".join(normalized).strip()


def _format_table_as_list(table: Dict[str, Any], title: Optional[str] = None, max_rows: int = 5) -> str:
    """
    Convert a structured table into a Slack-friendly nested bullet list.
    """
    columns = [c.lower() for c in (table.get("columns") or [])]
    rows = table.get("data") or []
    if not columns or not rows:
        return ""
    rows = rows[:max_rows]

    def get(row: Dict[str, Any], keys: List[str]) -> str:
        for k in keys:
            if k in row:
                return str(row.get(k) or "")
            # try case-insensitive
            for actual in row.keys():
                if actual.lower() == k:
                    return str(row.get(actual) or "")
        return ""

    bullets: List[str] = []
    if title:
        bullets.append(f"*{title}*")

    # Detect tier-style tables
    for row in rows:
        row_dict = row if isinstance(row, dict) else {}
        listing = get(row_dict, ["listing_id", "listing", "id"])
        area = get(row_dict, ["area", "neighbourhood", "neighborhood"])
        issue = get(row_dict, ["issue"])
        action = get(row_dict, ["recommended action (next 30 days)", "recommended_action", "recommended action"])
        evidence = get(row_dict, ["evidence"])

        if listing or area or issue or action or evidence:
            header = f"• *Listing {listing} – {area}*" if listing or area else "• Listing"
            bullets.append(header.strip())
            if issue:
                bullets.append(f"  • Issue: {issue}")
            if action:
                bullets.append(f"  • Recommended action: {action}")
            if evidence:
                bullets.append(f'  • Evidence: "{evidence}"')
            continue

        # Fallback: generic row listing
        parts = [f"{col}: {row_dict.get(col, '')}" for col in table.get("columns", []) if row_dict.get(col, "")]
        if parts:
            bullets.append("• " + "; ".join(parts))

    return "\n".join(bullets).strip()


def _format_slack_answer(summary: str, payload: Dict[str, Any]) -> str:
    """
    Build a Slack-friendly mrkdwn string combining narrative + listified tables/snippets.
    """
    text = summary or ""
    text = _format_for_slack(text)
    # Prepend branding once
    if text:
        text = f"*wtchtwr — your property insights companion*\n\n{text}"

    tables = payload.get("tables") or []
    extras: List[str] = []

    # Render tables as lists (no code fences)
    for tbl in tables:
        cols = [c.lower() for c in tbl.get("columns") or []]
        if "snippet" in cols:
            # Treat as sample feedback
            snippet_lines = ["*Sample feedback*"]
            data_rows = tbl.get("data") or []
            for snip in data_rows[:5]:
                row = snip if isinstance(snip, dict) else {}
                listing = str(row.get("listing_id") or row.get("id") or "")
                hood = row.get("neighbourhood") or row.get("neighborhood") or ""
                month = row.get("month") or ""
                year = row.get("year") or ""
                sentiment = row.get("sentiment_label") or ""
                compound = row.get("compound")
                quote = (row.get("snippet") or row.get("text") or "").strip()
                quote = quote[:180]
                prefix_parts = []
                if listing:
                    prefix_parts.append(f"Listing {listing}")
                if hood:
                    prefix_parts.append(hood)
                if month or year:
                    prefix_parts.append(f"{month} {year}".strip())
                if sentiment:
                    prefix_parts.append(sentiment)
                if compound not in (None, ""):
                    prefix_parts.append(str(compound))
                prefix = " | ".join([p for p in prefix_parts if p])
                snippet_lines.append(f'• {prefix}: "{quote}"' if quote else f"• {prefix}")
            extras.append("\n".join(snippet_lines))
        else:
            title = tbl.get("name") or None
            extras_text = _format_table_as_list(tbl, title=title)
            if extras_text:
                extras.append(extras_text)

    if extras:
        text = text + "\n\n" + "\n\n".join(extras)
    return text.strip()


def render_slack_table(table: Dict[str, Any], max_rows: int = 8) -> str:
    """Render a structured table as a padded monospace table inside a single code block."""
    columns = table.get("columns") or []
    rows = table.get("data") or []
    if not columns:
        return ""
    rows = rows[:max_rows]
    # Ensure each row is a list aligned with columns
    formatted_rows: List[List[str]] = []
    for row in rows:
        if isinstance(row, dict):
            formatted_rows.append([str(row.get(col, "")) for col in columns])
        elif isinstance(row, (list, tuple)):
            vals = list(row)[: len(columns)]
            while len(vals) < len(columns):
                vals.append("")
            formatted_rows.append([str(v) for v in vals])
        else:
            formatted_rows.append([str(row)])

    widths = [len(str(col)) for col in columns]
    for r in formatted_rows:
        for idx, val in enumerate(r):
            if idx < len(widths):
                widths[idx] = max(widths[idx], len(val))

    def fmt_row(vals: List[str]) -> str:
        padded = []
        for idx, val in enumerate(vals):
            width = widths[idx] if idx < len(widths) else len(val)
            padded.append(val.ljust(width))
        return " | ".join(padded)

    lines_out = [fmt_row(columns)]
    lines_out.append(" | ".join("-" * w for w in widths))
    for r in formatted_rows:
        lines_out.append(fmt_row(r))

    body = "\n".join(lines_out)
    return f"```text\n{body}\n```"


def _build_blocks(summary: str, payload: Dict[str, Any], cache_key: str) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    tables = payload.get("tables") or []

    # Main narrative
    summary_text = summary or "Here’s what I found:"

    for chunk in _chunk_text(summary_text):
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": chunk}})
    if not blocks:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "Here’s what I found:"}})

    # Actions
    blocks.append({
        "type": "actions",
        "elements": [
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "View SQL and RAG sources"},
                "action_id": "view_sql_results",
                "value": cache_key,
            }
        ],
    })
    return blocks


def _open_modal(client: WebClient, trigger_id: str, payload: Dict[str, Any]) -> None:
    """Display SQL + table preview in Slack modal."""
    sql = str(payload.get("sql") or "").replace("\r\n", "\n")
    tables = payload.get("tables") or []
    full_summary = _format_for_slack(str(payload.get("summary") or ""))

    blocks = []

    # Full narrative (chunked)
    if full_summary:
        for chunk in _chunk_text(f"*Full response*\n{full_summary}", limit=2800):
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": chunk}})

    # SQL
    blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*SQL*\n```sql\n{sql[:2000]}\n```"}})

    # Tables
    for table in tables[:2]:
        name = table.get("name", "Result")
        preview = (table.get("preview") or "").replace("\r\n", "\n").strip()
        if not preview:
            columns = table.get("columns") or []
            rows = table.get("data") or []
            preview = _format_rows_preview(columns, rows, limit=8)
        preview = preview or "(no rows returned)"
        rows = table.get("row_count", 0)
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*{name}* (_{rows} rows_)\n```text\n{preview[:1600]}\n```"},
        })

    client.views_open(
        trigger_id=trigger_id,
        view={
            "type": "modal",
            "title": {"type": "plain_text", "text": "Query details"},
            "close": {"type": "plain_text", "text": "Close"},
            "blocks": blocks,
        },
    )


# ---------------------------------------------------------------------------
# Slack app initialization
# ---------------------------------------------------------------------------

def create_slack_app() -> App:
    bot_token = os.getenv("SLACK_BOT_TOKEN")
    signing_secret = os.getenv("SLACK_SIGNING_SECRET")
    app_token = os.getenv("SLACK_APP_TOKEN")
    if not bot_token or not signing_secret:
        raise RuntimeError("SLACK_BOT_TOKEN and SLACK_SIGNING_SECRET must be set in the environment.")

    app = App(token=bot_token, signing_secret=signing_secret)
    if app_token:
        # stash for optional Socket Mode startup (avoids public HTTP endpoint)
        setattr(app, "_app_token", app_token)

    @app.middleware  # log all user messages without consuming the event
    def log_raw_messages(body, logger, next):
        event = (body or {}).get("event") or {}
        if event.get("type") == "message" and not event.get("bot_id"):
            logger.info("🪵 Raw message event received: %s", event)
        return next()
    @app.event({"type": "message", "subtype": "message_deleted"})
    def handle_deleted_message_events(event, ack):
        ack()
    auth_info = app.client.auth_test()
    bot_user_id = auth_info.get("user_id")

    # Home tab
    @app.event("app_home_opened")
    def handle_home(event, ack, client, logger):
        ack()
        user_id = event.get("user")
        if not user_id:
            logger.debug("app_home_opened without user id")
            return
        try:
            client.views_publish(
                user_id=user_id,
                view={
                    "type": "home",
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*wtchtwr — your property insights companion*\nStart with a natural language question about your listings or the market.",
                            },
                        },
                        {"type": "divider"},
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "• Mention me in a channel or DM directly.\n"
                                        "• Type `help` for sample prompts.\n"
                                        "• Type `dashboard` to open the web dashboard.\n"
                                        "• Type `slackbot` in the web UI to reopen this workspace.",
                            },
                        },
                        {
                            "type": "actions",
                            "elements": [
                                {"type": "button", "text": {"type": "plain_text", "text": "Open Web Dashboard"}, "url": FRONTEND_URL},
                                {"type": "button", "text": {"type": "plain_text", "text": "InsideAirbnb Market"}, "url": DASHBOARD_URL},
                            ],
                        },
                    ],
                },
            )
        except SlackApiError as exc:
            logger.warning("Failed to publish Slack home view: %s", exc.response.get("error"))

    # Mentions in channels
    @app.event("app_mention")
    def handle_app_mention(event, client, ack):
        ack()
        if event.get("bot_id"):
            return
        question = _strip_mentions(event.get("text", ""))
        channel, ts = event["channel"], event.get("ts")

        if GREETING_PATTERN.match(question):
            _respond(client, channel=channel, text=GREETING_RESPONSE, thread_ts=ts)
            return

        lowered = question.lower().strip()
        if lowered == "help":
            _respond(client, channel=channel, text=f"{WELCOME_MESSAGE}\n<{HELP_URL}|Open wtchtwr>", thread_ts=ts)
        elif lowered in {"dashboard", "open dashboard", "show dashboard"}:
            _respond(client, channel=channel, text=_dashboard_message(), thread_ts=ts)
        elif lowered == "clear":
            _clear_conversation(client, channel, bot_user_id)
            return
        elif lowered == "export":
            _respond(client, channel=channel, text=f"Export via the web dashboard: <{EXPORT_URL}|Open export options>", thread_ts=ts)
        else:
            thread_ref = channel  # keep one conversation per channel/thread in Slack
            logger.info("[Slack] Handling mention in %s (%s): %s", channel, thread_ref, question)
            try:
                outcome = handle_question(question, thread_id=thread_ref) or {}
                if outcome.get("error"):
                    _respond(client, channel=channel, text=outcome.get("error"), thread_ts=ts)
                else:
                    payload = outcome.get("payload")
                    summary = outcome.get("summary") or outcome.get("answer_text") or ""
                    if payload:
                        summary = _format_slack_answer(summary, payload)
                    if payload is None:
                        _respond(client, channel=channel, text="No result returned for that question.", thread_ts=ts)
                    else:
                        cache_key = _store_payload(channel, payload, ts)
                        blocks = _build_blocks(summary or "Here you go:", payload, cache_key)
                        _respond(client, channel=channel, text=summary or "Here you go:", blocks=blocks, thread_ts=ts)
            except SlackApiError as exc:
                logger.exception("Slack send failed: %s", exc)
            except Exception as exc:
                logger.exception("Slack mention handler failed: %s", exc)
                _respond(client, channel=channel, text="Sorry, I hit an error processing that message.", thread_ts=ts)

    # Direct messages
    @app.message(re.compile(".*"))
    def handle_direct_message(message, client, ack):
        ack()
        if message.get("bot_id") or message.get("channel_type") != "im":
            return
        question = message.get("text", "").strip()
        channel = message["channel"]

        if GREETING_PATTERN.match(question):
            _respond(client, channel=channel, text=GREETING_RESPONSE)
            return

        lowered = question.lower()
        if lowered == "help":
            _respond(client, channel=channel, text=f"{WELCOME_MESSAGE}\n<{HELP_URL}|Open wtchtwr>")
        elif lowered in {"dashboard", "open dashboard", "show dashboard"}:
            _respond(client, channel=channel, text=_dashboard_message())
        elif lowered == "clear":
            _clear_conversation(client, channel, bot_user_id)
            return
        elif lowered == "export":
            _respond(client, channel=channel, text=f"Export via the web dashboard: <{EXPORT_URL}|Open export options>")
        else:
            thread_ref = channel  # keep one conversation per DM channel
            logger.info("[Slack] Handling DM %s (%s): %s", channel, thread_ref, question)
            try:
                outcome = handle_question(question, thread_id=thread_ref) or {}
                if outcome.get("error"):
                    _respond(client, channel=channel, text=outcome.get("error"))
                else:
                    payload = outcome.get("payload")
                    summary = outcome.get("summary") or outcome.get("answer_text") or ""
                    if payload:
                        summary = _format_slack_answer(summary, payload)
                    if payload is None:
                        _respond(client, channel=channel, text="No result returned for that question.")
                    else:
                        cache_key = _store_payload(channel, payload, message.get("ts"))
                        blocks = _build_blocks(summary or "Here you go:", payload, cache_key)
                        _respond(client, channel=channel, text=summary or "Here you go:", blocks=blocks)
            except SlackApiError as exc:
                logger.exception("Slack send failed: %s", exc)
                _respond(client, channel=channel, text="Slack send failed; check scopes and tokens.")
            except Exception as exc:
                logger.exception("Slack DM handler failed: %s", exc)
                _respond(client, channel=channel, text=f"Error: {exc}")

    # Modal open
    @app.action("view_sql_results")
    def action_view_sql_results(body, ack, client):
        ack()
        cache_key = body["actions"][0]["value"]
        cached = _get_cached_payload(cache_key)
        if not cached or not body.get("trigger_id"):
            return
        _open_modal(client, body["trigger_id"], cached["payload"])

    return app


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from slack_bolt.adapter.socket_mode import SocketModeHandler

    logging.basicConfig(level=logging.INFO)
    slack_app = create_slack_app()
    logger.info("Starting Slack bot with LangGraph orchestration (Socket Mode).")

    # Prefer Socket Mode if SLACK_APP_TOKEN is set
    app_token = os.getenv("SLACK_APP_TOKEN")
    if app_token:
        handler = SocketModeHandler(slack_app, app_token)
        handler.start()
    else:
        slack_app.start(port=int(os.environ.get("PORT", 3000)))
