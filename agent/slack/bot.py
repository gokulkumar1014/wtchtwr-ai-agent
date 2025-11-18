"""
LangGraph-backed Slack bot for H.O.P.E AI (Highbury Occupancy Property Engine).

Once validated, this module replaces all legacy Slack wiring in /app/slackbot
and decommissions remaining /app/nl2sql dependencies.
"""

from __future__ import annotations
import logging
import os
import re
import uuid
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence

from slack_bolt import App
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
DASHBOARD_URL = "https://insideairbnb.com/new-york-city/"
GREETING_RESPONSE = "Hi there, how may I help you today?"
WELCOME_MESSAGE = (
    "Hi there! I'm *H.O.P.E.* (Highbury Occupancy Property Engine) 👋\n\n"
    "You can ask me things like:\n"
    "• \"Average occupancy rate for next 60 days for my listings in Manhattan\"\n"
    "• \"Which of my properties have paid parking?\"\n"
    "• \"Compare avg price in Midtown: mine vs market\"\n"
    "• \"Show the latest 5 reviews for listing 2595\"\n\n"
    "Useful shortcuts:\n"
    "• Type *help* for available actions\n"
    "• Type *dashboard* to open the dashboard\n"
    "• Type *clear* to clean this thread\n"
    "_LangGraph orchestration active._"
)

MAX_CACHE_ENTRIES = 50
_RESULT_CACHE: Dict[str, Dict[str, Any]] = {}
_LAST_PAYLOAD: Optional[Dict[str, Any]] = None

# ---------------------------------------------------------------------------
# LangGraph runner
# ---------------------------------------------------------------------------

def _ensure_config_loaded() -> None:
    """Ensure shared agent configuration is initialised once."""
    load_config(refresh=False)


@lru_cache(maxsize=1)
def _langgraph_runner():
    """Lazy import of the LangGraph runner to avoid heavy imports on module load."""
    from agent.graph import run as run_agent
    return run_agent


def run_langgraph_query(question: str, *, thread_id: Optional[str] = None) -> Dict[str, Any]:
    """Execute the LangGraph pipeline for the given question."""
    _ensure_config_loaded()
    runner = _langgraph_runner()
    logger.info("[Slack] Dispatching LangGraph query")
    response = runner(question, thread_id=thread_id)
    logger.info("[Slack] LangGraph query completed (keys: %s)", list(response.keys()))
    return response or {}

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _normalize_summary(result: Dict[str, Any]) -> str:
    """Normalize LangGraph output to a concise summary string."""
    bundle = (result.get("result_bundle") or {}) if isinstance(result, dict) else {}
    primary = (result.get("answer_text") or result.get("content") or "").strip()
    if primary:
        return primary
    summary = (bundle.get("summary") or "").strip()
    if summary:
        return summary
    extras = result.get("extras") if isinstance(result, dict) else {}
    if isinstance(extras, dict):
        extra_text = (extras.get("answer_text") or extras.get("content") or "").strip()
        if extra_text:
            return extra_text
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


def _build_table_payload(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create Slack-friendly table preview payload."""
    rows = bundle.get("rows") or []
    rows = rows if isinstance(rows, list) else []
    columns = _coerce_columns(bundle, rows)
    markdown_table = (bundle.get("markdown_table") or "").strip()
    if not markdown_table and columns:
        markdown_table = _format_rows_preview(columns, rows)
    return [
        {
            "name": bundle.get("sql_table") or "Result",
            "columns": columns,
            "data": rows[:50],
            "preview": markdown_table or "(no rows returned)",
            "row_count": len(rows),
        }
    ]


def _build_payload(question: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble a unified Slack message payload."""
    bundle = (result.get("result_bundle") or {}) if isinstance(result, dict) else {}
    summary = _normalize_summary(result)
    tables = _build_table_payload(bundle) if bundle else []
    sql_text = bundle.get("sql") or (result.get("sql") or {}).get("text") or ""
    telemetry = result.get("telemetry") or {}

    payload = {
        "question": question,
        "policy": bundle.get("policy") or telemetry.get("policy"),
        "sql": sql_text,
        "tables": tables,
        "summary": summary,
        "nl_summary": summary,
        "row_count": tables[0]["row_count"] if tables else 0,
        "response_type": bundle.get("policy") or "sql",
        "telemetry": telemetry,
        "markdown_table": bundle.get("markdown_table"),
        "filters": bundle.get("applied_filters") or {},
    }
    return payload

# ---------------------------------------------------------------------------
# Core interaction handlers
# ---------------------------------------------------------------------------

def handle_question(question: str, *, thread_id: Optional[str] = None) -> Dict[str, Any]:
    """Resolve a Slack question via LangGraph and return payload + summary."""
    global _LAST_PAYLOAD
    question_clean = (question or "").strip()
    if not question_clean:
        return {"error": "Please provide a question for me to analyse."}

    try:
        result = run_langgraph_query(question_clean, thread_id=thread_id)
    except Exception as exc:
        logger.exception("Slackbot failed to execute LangGraph query.")
        return {"error": f"Something went wrong while running the query: {exc}"}

    bundle = (result.get("result_bundle") or {}) if isinstance(result, dict) else {}
    if bundle.get("error"):
        return {"error": bundle.get("summary") or bundle["error"]}

    payload = _build_payload(question_clean, result)
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
    return f":bar_chart: *NYC Market Dashboard*\n<{DASHBOARD_URL}|Open InsideAirbnb dashboard>"


def _clear_conversation(client: WebClient, channel: str, bot_user_id: str) -> str:
    """Delete previous HOPE messages in a channel."""
    try:
        history = client.conversations_history(channel=channel, limit=200)
    except SlackApiError as exc:
        return f"I couldn't fetch the conversation history ({exc.response['error']})."

    deleted = 0
    for message in history.get("messages", []):
        ts = message.get("ts")
        if ts and (message.get("user") == bot_user_id or message.get("bot_id")):
            try:
                client.chat_delete(channel=channel, ts=ts)
                deleted += 1
            except SlackApiError:
                continue

    if deleted:
        return "Cleared HOPE responses in this conversation."
    return "No HOPE messages to clear (I can only remove messages posted by me)."


def _respond(client: WebClient, channel: str, text: str,
             thread_ts: Optional[str] = None,
             blocks: Optional[Iterable[Dict[str, Any]]] = None) -> None:
    kwargs = {"channel": channel, "text": text}
    if thread_ts:
        kwargs["thread_ts"] = thread_ts
    if blocks:
        kwargs["blocks"] = list(blocks)
    client.chat_postMessage(**kwargs)


def _build_blocks(summary: str, payload: Dict[str, Any], cache_key: str) -> List[Dict[str, Any]]:
    return [
        {"type": "section", "text": {"type": "mrkdwn", "text": summary}},
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "View SQL & Results"},
                    "action_id": "view_sql_results",
                    "value": cache_key,
                }
            ],
        },
    ]


def _open_modal(client: WebClient, trigger_id: str, payload: Dict[str, Any]) -> None:
    """Display SQL + table preview in Slack modal."""
    sql = payload.get("sql") or ""
    tables = payload.get("tables") or []

    blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": f"*SQL*\n```{sql[:2000]}```"}}]
    for table in tables[:2]:
        name = table.get("name", "Result")
        preview = (table.get("preview") or "(no rows returned)")[:1800]
        rows = table.get("row_count", 0)
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*{name}* (_{rows} rows_)\n```{preview}```"},
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
    if not bot_token or not signing_secret:
        raise RuntimeError("SLACK_BOT_TOKEN and SLACK_SIGNING_SECRET must be set in the environment.")

    app = App(token=bot_token, signing_secret=signing_secret)

    @app.event("message")
    def debug_all_messages(event, say, logger):
        logger.info(f"🪵 Raw message event received: {event}")
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
                                "text": f"*{GREETING_RESPONSE}*\nStart with a natural language question about your listings or the market.",
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
            _respond(client, channel=channel, text=WELCOME_MESSAGE, thread_ts=ts)
        elif lowered in {"dashboard", "open dashboard", "show dashboard"}:
            _respond(client, channel=channel, text=_dashboard_message(), thread_ts=ts)
        elif lowered == "clear":
            message = _clear_conversation(client, channel, bot_user_id)
            _respond(client, channel=channel, text=message, thread_ts=ts)
        elif lowered == "export":
            _respond(client, channel=channel, text="Use the HOPE web dashboard or SQL modal to export.", thread_ts=ts)
        else:
            thread_ref = event.get("thread_ts") or ts or channel
            outcome = handle_question(question, thread_id=thread_ref)
            if "error" in outcome:
                _respond(client, channel=channel, text=outcome["error"], thread_ts=ts)
            else:
                payload, summary = outcome["payload"], outcome["summary"]
                cache_key = _store_payload(channel, payload, ts)
                blocks = _build_blocks(summary, payload, cache_key)
                _respond(client, channel=channel, text=summary, blocks=blocks, thread_ts=ts)

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
            _respond(client, channel=channel, text=WELCOME_MESSAGE)
        elif lowered in {"dashboard", "open dashboard", "show dashboard"}:
            _respond(client, channel=channel, text=_dashboard_message())
        elif lowered == "clear":
            msg = _clear_conversation(client, channel, bot_user_id)
            _respond(client, channel=channel, text=msg)
        elif lowered == "export":
            _respond(client, channel=channel, text="Use the HOPE web dashboard or SQL modal to export.")
        else:
            thread_ref = message.get("thread_ts") or message.get("ts") or channel
            outcome = handle_question(question, thread_id=thread_ref)
            if "error" in outcome:
                _respond(client, channel=channel, text=outcome["error"])
            else:
                payload, summary = outcome["payload"], outcome["summary"]
                cache_key = _store_payload(channel, payload, message.get("ts"))
                blocks = _build_blocks(summary, payload, cache_key)
                _respond(client, channel=channel, text=summary, blocks=blocks)

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
