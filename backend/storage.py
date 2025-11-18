"""
Persistent storage utilities for conversations and thread mappings in H.O.P.E backend.
"""

from __future__ import annotations
import json
import math
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .models import Conversation, Message

# ---------------------------------------------------------------------------
# Static Texts
# ---------------------------------------------------------------------------

WELCOME_TEXT = (
    "Hi there! wtchtwr 👋\n\n"
    "You can ask me things like:\n"
    "• \"Average occupancy rate for next 60 days for my listings in Manhattan\"\n"
    "• \"Which of my properties have paid parking?\"\n"
    "• \"Compare avg price in Midtown: mine vs market\"\n"
    "• \"Show the latest 5 reviews for listing 2595\"\n\n"
    "Useful shortcuts: Type help for knowing what things you can do using the chatbot · "
    "dashboard for veiwing the dashboard · slackbot for going to the slack app. How can I help today?"
)

HELP_TEXT = (
    "Here’s what wtchtwr can help you explore today:\n"
    "• Guided prompts span occupancy, pricing, revenue, amenities, sentiment, and benchmarking.\n"
    "• Ask follow-up questions or combine filters (borough, neighbourhood, property type, date windows).\n\n"
    "Sample prompts to get started:\n"
    "• \"Average occupancy rate for next 60 days for my listings in Manhattan\"\n"
    "• \"Which of my properties have paid parking?\"\n"
    "• \"Compare avg price in Midtown: mine vs market\"\n"
    "• \"Show the latest 5 reviews for listing 2595\"\n"
    "• \"Give me the occupancy trend for Williamsburg studios over the past 90 days\"\n"
    "• \"What is the average nightly rate for Highbury listings with 2 bedrooms in Brooklyn?\"\n"
    "• \"List my listings with occupancy below 40% last month\"\n"
    "• \"Which Highbury listing delivered the highest revenue in Queens last quarter?\"\n"
    "• \"Highlight the amenities guests mention most often in reviews for listing 2595\"\n"
    "• \"Which neighbourhood has the best review scores rating across my portfolio?\"\n"
    "• \"Show average cleaning fee difference between Highbury and market in Harlem\"\n"
    "• \"Do any of my units offer free parking and washer/dryer?\"\n"
    "• \"What percentage of Highbury listings allow pets?\"\n"
    "• \"Benchmark Williamsburg vs Chelsea occupancy for the next 60 days\"\n"
    "• \"Summarise cleanliness sentiment for listing 2595\"\n"
    "• \"Surface recent review highlights mentioning 'noise' in Chelsea\"\n"
    "• \"Compare average booking lead time: HPG vs market\"\n"
    "• \"Which of my listings have occupancy above 85% this month?\"\n"
    "• \"Average revenue per available room for SoHo lofts\"\n"
    "• \"What is the cancellation rate for market listings in Brooklyn over the last 60 days?\"\n"
    "• \"Which Highbury listings include a dedicated workspace?\"\n"
    "• \"Show host fees trend for my portfolio this year\"\n"
    "• \"Break down kitchen amenity coverage across boroughs\"\n"
    "• \"List the 10 most recent bookings for listing 3021\"\n"
    "• \"How many of my listings have a cleanliness score above 4.8?\"\n"
    "• \"Which competitor listings near Times Square undercut my nightly rate?\"\n"
    "• \"Show weekend vs weekday pricing for listing 2595 last month\"\n"
    "• \"Highlight market average occupancy for luxury listings in Tribeca\"\n"
    "• \"Identify listings with more than 3 negative reviews about Wi-Fi\"\n"
    "• \"List the average bathrooms, bedrooms, and accommodates for my lofts in Midtown\"\n"
    "• \"Share a detailed amenities comparison between my listings and the market in Brooklyn\"\n"
)

# ---------------------------------------------------------------------------
# Constants & Utilities
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CONVERSATIONS_FILE = DATA_DIR / "conversations_store.json"
THREAD_MAP_FILE = DATA_DIR / "thread_map.json"
CUTOFF_DAYS = 30


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _welcome_message() -> Message:
    return Message(
        id=str(uuid4()),
        role="assistant",
        nl_summary=WELCOME_TEXT,
        timestamp=_now_iso(),
    )


# ---------------------------------------------------------------------------
# JSON Helpers
# ---------------------------------------------------------------------------

def _sanitize_for_json(value: Any) -> Any:
    """Ensure value is JSON serializable and finite."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, set):
        return [_sanitize_for_json(v) for v in value]
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


# ---------------------------------------------------------------------------
# Conversation persistence
# ---------------------------------------------------------------------------

def load_conversations() -> List[Conversation]:
    """Load recent conversations from disk, prune old ones."""
    if not CONVERSATIONS_FILE.exists():
        return []

    try:
        raw_data = json.loads(CONVERSATIONS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=CUTOFF_DAYS)
    conversations: List[Conversation] = []

    for convo in raw_data:
        try:
            updated_at = datetime.fromisoformat(convo["updated_at"].replace("Z", "+00:00"))
        except Exception:
            continue
        if updated_at < cutoff:
            continue

        convo.setdefault("messages", [])
        if not convo["messages"]:
            convo["messages"] = [json.loads(_welcome_message().json())]
        conversations.append(Conversation(**convo))

    # Sort most recent first
    conversations.sort(key=lambda c: c.updated_at, reverse=True)

    for c in conversations:
        if not c.messages:
            c.messages = [_welcome_message()]
    return conversations


def save_conversations(conversations: List[Conversation]) -> None:
    """Persist conversations to disk."""
    CONVERSATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = [_sanitize_for_json(c.model_dump()) for c in conversations]
    CONVERSATIONS_FILE.write_text(json.dumps(payload, indent=2))


def create_conversation(title: Optional[str] = None) -> Conversation:
    now = _now_iso()
    return Conversation(
        id=str(uuid4()),
        title=title or "New conversation",
        created_at=now,
        updated_at=now,
        messages=[_welcome_message()],
    )


def append_message(conversation: Conversation, message: Message) -> Conversation:
    conversation.messages.append(message)
    conversation.updated_at = message.timestamp

    if message.role == "user" and message.content:
        snippet = message.content.strip()
        if len(snippet) > 60:
            snippet = snippet[:57].rstrip() + "…"
        conversation.title = snippet or conversation.title
    return conversation


def delete_conversation(convo_id: str, conversations: List[Conversation]) -> List[Conversation]:
    """Delete conversation by ID."""
    return [c for c in conversations if c.id != convo_id]


def refresh_conversation_title(conversation: Conversation) -> Conversation:
    """Refresh title based on last user message."""
    user_msgs = [m for m in conversation.messages if m.role == "user" and m.content]
    if user_msgs:
        snippet = user_msgs[-1].content.strip()
        if len(snippet) > 60:
            snippet = snippet[:57].rstrip() + "…"
        conversation.title = snippet or "New conversation"
    else:
        conversation.title = "New conversation"
    return conversation


# ---------------------------------------------------------------------------
# Thread ↔ Conversation Map
# ---------------------------------------------------------------------------

def update_thread_map(thread_id: str, convo_id: str) -> None:
    THREAD_MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    data: Dict[str, str] = {}
    if THREAD_MAP_FILE.exists():
        try:
            data = json.loads(THREAD_MAP_FILE.read_text())
        except json.JSONDecodeError:
            data = {}
    data[thread_id] = convo_id
    THREAD_MAP_FILE.write_text(json.dumps(data, indent=2))


def get_conversation_id(thread_id: str) -> Optional[str]:
    """Return conversation ID linked to a given Slack thread."""
    if not THREAD_MAP_FILE.exists():
        return None
    try:
        data = json.loads(THREAD_MAP_FILE.read_text())
    except json.JSONDecodeError:
        return None
    return data.get(thread_id)
