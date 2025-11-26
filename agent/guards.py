"""Guard and ingress/egress helpers for the wtchtwr AI Agent."""
from __future__ import annotations

import html
import logging
import re
import time
from typing import Any, Dict, List, Optional

from .config import load_config
from .types import State, ThinkingStep

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex and Filter Constants
# ---------------------------------------------------------------------------

_HTML_RE = re.compile(r"<[^>]+>")
_EMAIL_RE = re.compile(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9_.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\b(?:\+?\d[\s.-]?){7,}\b")
_DIGIT_SPAN_RE = re.compile(r"\b\d{6,}\b")

_ALLOWED_FILTER_KEYS = {
    "borough": list,
    "neighbourhood": list,
    "month": list,
    "year": list,
    "listing_id": (int, type(None)),
    "is_highbury": (bool, type(None)),
}

# ---------------------------------------------------------------------------
# Query Sanitization and Filter Coercion
# ---------------------------------------------------------------------------

def _sanitize_query(query: str) -> str:
    """Remove HTML, PII, and formatting noise from user queries."""
    query = html.unescape(query or "")
    query = _HTML_RE.sub(" ", query)
    query = _EMAIL_RE.sub("[redacted email]", query)

    def _phone_replacer(match: re.Match[str]) -> str:
        # Preserve numeric ids when preceded by listing/property references.
        start = match.start()
        prefix = query[max(0, start - 24) : start].lower()
        if "listing" in prefix or "property" in prefix:
            return match.group(0)
        return "[redacted phone]"

    query = _PHONE_RE.sub(_phone_replacer, query)
    query = query.replace("\n", " ")
    return " ".join(query.split())


def _normalise_tenant(tenant: Optional[str]) -> Optional[str]:
    """Normalize tenant field (auto/highbury/market/both)."""
    if tenant is None:
        return None
    normal = tenant.strip().lower()
    mapping = {
        "auto": None,
        "": None,
        "market": "market",
        "highbury": "highbury",
        "both": "both",
    }
    return mapping.get(normal, None)


def _coerce_filters(user_filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Ensure filter keys are well-typed and normalized."""
    result = {}
    if not user_filters:
        return {
            "borough": [],
            "neighbourhood": [],
            "month": [],
            "year": [],
            "listing_id": None,
            "is_highbury": None,
        }

    for key, expected in _ALLOWED_FILTER_KEYS.items():
        value = user_filters.get(key)
        if value is None:
            result[key] = [] if expected is list else None
            continue

        if expected is list:
            result[key] = value if isinstance(value, list) else [value]
        else:
            result[key] = value if isinstance(value, expected) else None

    return result

# ---------------------------------------------------------------------------
# State Lifecycle
# ---------------------------------------------------------------------------

def ingress(query: str, tenant: Optional[str], user_filters: Optional[Dict[str, Any]]) -> State:
    """Initialize a new agent state."""
    config = load_config()
    sanitized_query = _sanitize_query(query)
    filters = _coerce_filters(user_filters)

    # Handle top_k override if supplied by frontend
    top_k_override = None
    if user_filters and "top_k" in user_filters:
        try:
            top_k_override = int(user_filters["top_k"])
        except (TypeError, ValueError):
            pass

    state = {
        "query": sanitized_query,
        "tenant": _normalise_tenant(tenant),
        "intent": None,
        "filters": filters,
        "plan": {},
        "sql": {"text": None, "df": None, "explain": None},
        "rag": {"hits": [], "summary": None, "citations": []},
        "viz": {"table_html": None, "chart_spec": None},
        "draft": None,
        "critique": {"ok": True, "issues": []},
        "history": [],
        "answer": None,
        "telemetry": {
            "policy": "",
            "latency_ms": 0,
            "tokens": {},
            "top_k": config.top_k_default,
        },
        "_start_time": time.perf_counter(),
        "_retry": False,
    }

    if top_k_override is not None:
        state["telemetry"]["top_k"] = top_k_override

    _LOGGER.debug("Ingress initialized with filters=%s", filters)
    return state


def guardrails(state: State) -> State:
    """Apply lightweight guardrails before expensive processing."""
    query = state.get("query", "") or ""

    # Length guard
    if len(query) > 500:
        message = "Sorry, that query is too long. Please shorten it."
        state["draft"] = state["answer"] = message
        state.setdefault("critique", {"ok": False, "issues": []})
        state["critique"]["ok"] = False
        state["critique"].setdefault("issues", []).append("Query length exceeded policy.")
        state["guardrail_blocked"] = True
        _LOGGER.warning("Guardrail blocked: query too long")
        return state

    # PII guard
    def _numeric_pii_detected(text: str) -> bool:
        for match in _DIGIT_SPAN_RE.finditer(text):
            start = match.start()
            prefix = text[max(0, start - 30) : start].lower()
            if "listing" in prefix or "property" in prefix:
                continue
            return True
        return False

    if _EMAIL_RE.search(query) or _numeric_pii_detected(query):
        message = "I’m sorry, but I can’t assist with that request."
        state["draft"] = state["answer"] = message
        state.setdefault("critique", {"ok": False, "issues": []})
        state["critique"]["ok"] = False
        state["critique"].setdefault("issues", []).append("Potential PII detected.")
        state["guardrail_blocked"] = True
        _LOGGER.warning("Guardrail blocked: PII pattern detected")

    return state


def egress(state: State | str) -> Dict[str, Any]:
    """Summarize the conversational state for downstream consumers (frontend, Slack, etc.)."""
    payload = state if isinstance(state, dict) else {"answer": str(state)}

    latency_ms = 0
    start = payload.get("_start_time")
    if isinstance(start, (int, float)):
        latency_ms = int((time.perf_counter() - start) * 1000)

    telemetry = payload.get("telemetry", {}) or {}
    telemetry["latency_ms"] = latency_ms

    bundle = payload.get("result_bundle", {}) or {}
    answer_text = (
        payload.get("answer_text")
        or payload.get("answer")
        or payload.get("draft")
        or "No response generated."
    )
    usage = payload.get("answer_usage") or {}
    warning = payload.get("answer_warning")

    thinking_debug = False
    if isinstance(payload, dict):
        thinking_debug = bool(
            payload.get("debug_thinking")
            or (payload.get("_input", {}) or {}).get("debug_thinking")
        )

    response = {
        "answer_text": answer_text,
        "answer_warning": warning,
        "result_bundle": bundle,
        "policy": bundle.get("policy") or telemetry.get("policy", ""),
        "sql_text": bundle.get("sql"),
        "sql_params": bundle.get("sql_params", []),
        "rag_snippets": bundle.get("rag_snippets", []),
        "telemetry": {**telemetry, "llm_usage": usage},
    }

    thinking_entries: List[Dict[str, Any]] = []
    raw_thinking = payload.get("thinking") if isinstance(payload, dict) else None
    if isinstance(raw_thinking, list):
        for step in raw_thinking:
            payload_step: Optional[Dict[str, Any]] = None
            if isinstance(step, ThinkingStep):
                payload_step = step.to_payload()
            elif isinstance(step, dict):
                payload_step = dict(step)
            if not payload_step:
                continue
            meta = payload_step.get("meta") if isinstance(payload_step.get("meta"), dict) else {}
            thinking_entries.append(
                {
                    "phase": payload_step.get("phase") or "unknown",
                    "title": payload_step.get("title") or "",
                    "detail": payload_step.get("detail") or "",
                    "meta": meta,
                    "elapsed_ms": payload_step.get("elapsed_ms"),
                }
            )
    if thinking_entries and thinking_debug:
        response["thinking_trace"] = thinking_entries

    _LOGGER.debug("Egress response built: latency=%sms", latency_ms)
    return response


def fallback_expansion_text() -> str:
    """Deterministic guidance when external expansion signals are unavailable."""
    return (
        "Could not retrieve external signals. Here are 3 operator heuristics for evaluating new markets:\n"
        "- Prioritize neighborhoods with visible infrastructure commitments (transit, mixed-use projects).\n"
        "- Look for steady tourism pull: hotel pipeline compression, rising boutique inventory, and walkability to anchors.\n"
        "- Favor regulation-stable zones with consistent short-term rental precedents and clear permitting rules."
    )


__all__ = ["ingress", "guardrails", "egress", "fallback_expansion_text"]
