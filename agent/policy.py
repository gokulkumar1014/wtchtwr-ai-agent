"""Policy helpers for state normalization and planning (HOPE Agent)."""

from __future__ import annotations
import logging
import re
from typing import Dict, List, Any, Optional

from .config import load_config
from .intents import (
    _SQL_KEYWORDS,
    _RAG_KEYWORDS,
    _HYBRID_CONNECTORS,
    _SENTIMENT_WORD_MAP,
    _SENTIMENT_REVIEW_TERMS,
)
from .types import State

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MONTH_CANONICAL = {
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
}


def _normalise_months(months: List[str]) -> List[str]:
    """Uppercase + keep only canonical month codes, preserve order & de-dupe."""
    seen, out = set(), []
    for m in months or []:
        key = str(m).strip().upper()
        if key in _MONTH_CANONICAL and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _bool_or_none(val) -> bool | None:
    """Convert strings like 'true'/'false' to booleans safely."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        s = val.strip().lower()
        if s in {"1", "true", "y", "yes"}:
            return True
        if s in {"0", "false", "n", "no"}:
            return False
    return None


_SENTIMENT_ALLOWED = {"positive", "neutral", "negative"}


def _normalise_sentiment(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in _SENTIMENT_ALLOWED:
        return text
    return _SENTIMENT_WORD_MAP.get(text)


def _detect_sentiment_query(text: str | None) -> Optional[str]:
    if not text:
        return None
    lowered = text.lower()
    if not any(term in lowered for term in _SENTIMENT_REVIEW_TERMS):
        return None
    for keyword, label in _SENTIMENT_WORD_MAP.items():
        if re.search(fr"\b{re.escape(keyword)}\b", lowered):
            return label
    return None


# ---------------------------------------------------------------------------
# Entity Normalization
# ---------------------------------------------------------------------------

def resolve_entities(state: State) -> State:
    """
    Normalize filters extracted by intents.py and add deterministic defaults.

    We do not invent data; we simply sanitize what was already found or provided by UI.
    """
    filters = dict(state.get("filters", {}))

    # Borough normalization
    boroughs = filters.get("borough") or []
    if not isinstance(boroughs, list):
        boroughs = [boroughs]
    filters["borough"] = [str(b).strip().title() for b in boroughs if str(b).strip()]

    # Month normalization
    months = filters.get("month") or []
    if not isinstance(months, list):
        months = [months]
    filters["month"] = _normalise_months(months)

    # Year normalization
    years = filters.get("year") or []
    if not isinstance(years, list):
        years = [years]
    try:
        filters["year"] = sorted({int(y) for y in years if str(y).isdigit()})
    except Exception:
        filters["year"] = []

    # Listing ID cleanup
    listing_id = filters.get("listing_id")
    if listing_id is not None:
        text_id = str(listing_id).strip()
        filters["listing_id"] = int(text_id) if text_id.isdigit() else None
    else:
        filters["listing_id"] = None

    # Boolean normalization for is_highbury
    filters["is_highbury"] = _bool_or_none(filters.get("is_highbury"))

    sentiment_label = _normalise_sentiment(filters.get("sentiment_label"))
    if not sentiment_label:
        sentiment_label = _detect_sentiment_query(state.get("query"))
    filters["sentiment_label"] = sentiment_label

    # Ensure scope alignment
    scope = state.get("scope") or state.get("tenant") or "Market"
    state["scope"] = scope

    state["filters"] = filters
    _LOGGER.debug("[ENTITY] Normalized filters: %s", filters)
    return state


# ---------------------------------------------------------------------------
# Planning Logic
# ---------------------------------------------------------------------------

def plan_steps(state: State) -> State:
    """
    Decide final routing plan from (intent, scope, UI toggles) without re-parsing text.

    Routing rules:
      - intent == "REVIEWS_RAG"  -> mode = "rag"
      - intent == "FACT_SQL"     -> mode = "sql"
      - intent == "HYBRID"/"FACT_SQL_RAG" -> mode = "hybrid"

    SQL table selection:
      - scope == "Highbury" -> "highbury_listings"
      - scope == "Hybrid"   -> "both"
      - else                -> "listings"
    """
    cfg = load_config()

    intent = (state.get("intent") or "FACT_SQL").upper()
    scope = (state.get("scope") or "Market").capitalize()
    filters = dict(state.get("filters", {}) or {})

    # Handle conversational shortcuts
    if intent in {"THANKS", "SMALLTALK", "GREETING"}:
        plan = {
            "mode": "chat",
            "policy": "CONVERSATION",
            "sql_table": None,
            "top_k": cfg.top_k_default,
            "use_sentiment": False,
        }
        state.update({
            "plan": plan,
            "policy": "CONVERSATION",
            "scope": "General",
        })
        telemetry = state.setdefault("telemetry", {})
        telemetry.update({"policy": "CONVERSATION", "mode": "chat"})
        return state

    if intent == "EXPANSION_SCOUT":
        plan = {
            "mode": "expansion_scout",
            "policy": "EXPANSION_SCOUT",
            "sql_table": None,
            "top_k": cfg.top_k_default,
            "use_sentiment": False,
        }
        state.update({"plan": plan, "policy": "EXPANSION_SCOUT"})
        telemetry = state.setdefault("telemetry", {})
        telemetry.update({"policy": "EXPANSION_SCOUT", "mode": "expansion_scout"})
        return state

    user_filters = (state.get("_input", {}) or {}).get("user_filters", {}) or {}

    force_rag = _bool_or_none(user_filters.get("reviews")) is True
    force_hybrid = _bool_or_none(user_filters.get("hybrid")) is True

    query_text = state.get("query") or ""
    lowered_query = query_text.lower()
    tokens = set(re.findall(r"\b[a-zA-Z]+\b", lowered_query))

    # Heuristic cues
    has_sql_tokens = bool(tokens & _SQL_KEYWORDS)
    has_rag_tokens = bool(tokens & _RAG_KEYWORDS)
    rag_soft_cues = has_rag_tokens or any(
        cue in lowered_query
        for cue in ["review", "reviews", "feedback", "comment", "guest", "sentiment", "opinion"]
    )
    numeric_tokens = {
        "ratios", "ratio", "trends", "mean", "median", "average", "averages",
        "percent", "percentage", "trend",
    }
    has_numeric = bool(tokens & numeric_tokens)
    hybrid_connector_terms = set(_HYBRID_CONNECTORS) | {
        "versus", "as well as", "plus reviews", "both revenue and reviews",
        "benchmark", "along with", "together with", "compare", "while also", "and what",
    }
    hybrid_connector_hit = any(term in lowered_query for term in hybrid_connector_terms)

    mixed_cues = has_sql_tokens and has_rag_tokens
    rag_numeric_override = rag_soft_cues and has_numeric

    # -----------------------------------------------------------------------
    # Mode decision
    # -----------------------------------------------------------------------
    def _string_value(raw: Any) -> Optional[str]:
        if isinstance(raw, str):
            text = raw.strip()
            return text or None
        if isinstance(raw, (list, tuple)):
            for item in raw:
                text = _string_value(item)
                if text:
                    return text
        return None

    kpi_hint = _string_value(filters.get("kpi"))
    triage_mode = intent in {"PORTFOLIO_TRIAGE", "PORTFOLIO_TRIAGE_ADVANCED"}

    if triage_mode:
        scope = "Highbury"
        mode = "portfolio_triage"
    elif intent == "SENTIMENT_REVIEWS":
        mode = "rag"
    elif intent in {"HYBRID", "FACT_SQL_RAG"}:
        mode = "hybrid"
    elif intent == "REVIEWS_RAG":
        mode = "rag"
    else:
        mode = "sql"

    if mode == "sql" and rag_soft_cues and not has_sql_tokens:
        mode = "rag"
    if mixed_cues or hybrid_connector_hit or rag_numeric_override:
        mode = "hybrid"
    if force_hybrid:
        mode = "hybrid"
    elif force_rag:
        mode = "rag"

    # -----------------------------------------------------------------------
    # SQL Table decision
    # -----------------------------------------------------------------------
    if scope == "Highbury":
        sql_table = "highbury_listings"
    elif scope == "Hybrid":
        sql_table = "both"
    else:
        sql_table = "listings"

    # -----------------------------------------------------------------------
    # Policy mapping
    # -----------------------------------------------------------------------
    policy_map = {
        ("sql", "Highbury"): "SQL_HIGHBURY",
        ("sql", "Hybrid"): "SQL_COMPARE",
        ("sql", "Market"): "SQL_MARKET",
        ("rag", "Highbury"): "RAG_HIGHBURY",
        ("rag", "Hybrid"): "RAG_HYBRID",
        ("rag", "Market"): "RAG_MARKET",
        ("hybrid", "Highbury"): "SQL_RAG_HIGHBURY",
        ("hybrid", "Hybrid"): "SQL_RAG_COMPARE",
        ("hybrid", "Market"): "SQL_RAG_MARKET",
        ("chat", "General"): "CONVERSATION",
        ("portfolio_triage", "Highbury"): "PORTFOLIO_TRIAGE",
    }
    policy = policy_map.get((mode, scope), f"{mode.upper()}_{scope.upper()}")

    _LOGGER.info("[POLICY] mode=%s intent=%s scope=%s policy=%s", mode, intent, scope, policy)

    # -----------------------------------------------------------------------
    # Telemetry and plan construction
    # -----------------------------------------------------------------------
    telemetry = state.setdefault("telemetry", {})
    requested_top_k = telemetry.get("top_k") or cfg.top_k_default
    try:
        top_k = max(1, min(50, int(requested_top_k)))
    except Exception:
        top_k = cfg.top_k_default

    sentiment_filter_applied = bool(filters.get("sentiment_label"))
    use_sentiment = sentiment_filter_applied or any(
        t in lowered_query for t in ["sentiment", "positive", "negative", "tone"]
    )

    plan = {
        "mode": mode,
        "policy": policy,
        "sql_table": sql_table,
        "top_k": top_k,
        "use_sentiment": use_sentiment,
    }

    if triage_mode:
        plan["kpi"] = kpi_hint or "occupancy_rate_90"
        plan["use_sentiment"] = True

    state.update({
        "plan": plan,
        "policy": policy,
        "scope": scope,
    })

    telemetry.update({
        "policy": policy,
        "top_k": top_k,
    })

    state.setdefault("sql", {})["table"] = sql_table
    _LOGGER.debug("[POLICY] Plan decided: %s", plan)
    return state


# ---------------------------------------------------------------------------
# LangGraph conditional routing helper
# ---------------------------------------------------------------------------

def _state_value(state: State | object, key: str, default=None):
    if hasattr(state, key):
        return getattr(state, key, default)
    if isinstance(state, dict):
        return state.get(key, default)
    return default


def choose_path(state: State | object) -> str:
    """Return routing identifier for LangGraph conditional edges."""
    plan = _state_value(state, "plan", {}) or {}
    mode = str(plan.get("mode") or "").lower()

    if mode == "portfolio_triage":
        return "portfolio_triage"
    if mode in {"rag", "hybrid"}:
        return mode

    intent = str(_state_value(state, "intent", "") or "").upper()
    if intent == "EXPANSION_SCOUT":
        return "expansion_scout"
    if intent in {"REVIEWS_RAG", "SENTIMENT_REVIEWS"}:
        return "rag"
    if intent in {"HYBRID", "FACT_SQL_RAG"}:
        return "hybrid"

    return "nl2sql"


__all__ = ["resolve_entities", "plan_steps", "choose_path"]