"""
Intent classification and lightweight entity extraction for the H.O.P.E AI Agent.
Determines whether a query maps to SQL, RAG, or hybrid analysis, 
and extracts temporal, spatial, and listing-level entities for downstream nodes.
"""

from __future__ import annotations
import logging
import re
from typing import Dict, List, Optional

from .types import State
from .config import MONTHS_MAP, SEASON_MAP, get_month_abbrev

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenization and cue maps
# ---------------------------------------------------------------------------

_TEMPORAL_CUES = {"during", "in", "from"}
_TEMPORAL_SKIP_TOKENS = {
    "through", "and", "to", "late", "early", "til", "till", "the", "of",
    "month", "until", "into", "throughout", "months",
}

_TOKEN_PATTERN = re.compile(r"\b[\w'-]+\b")
_NEGATED_HIGHBURY_PATTERN = re.compile(r"\b(?:non[-\s]?highbury|not\s+highbury|except\s+highbury)\b", re.IGNORECASE)
_COMPARE_SCOPE_PATTERN = re.compile(r"\bhighbury\b[^.]*\bmarket\b|\bmarket\b[^.]*\bhighbury\b", re.IGNORECASE)
_COMPARISON_TOKENS = ("compare", "comparison", " vs ", "versus", "benchmark", "against", "relative", "stack up")
_OUR_PATTERN = re.compile(r"\b(?:our|ours|we|my)\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Split text into lowercase tokens."""
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text or "")]


def _dedupe_preserve(items: List[str]) -> List[str]:
    """Remove duplicates while preserving order."""
    seen, ordered = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _has_temporal_cue(tokens: List[str], index: int) -> bool:
    """Return True if a month token has a nearby temporal cue (e.g., 'in March')."""
    for offset in range(1, min(index, 6) + 1):
        lookback = tokens[index - offset]
        if lookback in _TEMPORAL_CUES:
            return True
        if lookback in _TEMPORAL_SKIP_TOKENS or lookback in MONTHS_MAP:
            continue
    return False


def _extract_months(text: str) -> List[str]:
    """Extract canonical month codes and expand seasonal references."""
    tokens = _tokenize(text)
    months = []
    for idx, token in enumerate(tokens):
        canon = MONTHS_MAP.get(token)
        if canon and _has_temporal_cue(tokens, idx):
            months.append(canon)

    lowered = text.lower()
    for season, month_codes in SEASON_MAP.items():
        if re.search(fr"\b{re.escape(season)}\b", lowered):
            months.extend(month_codes)
    return _dedupe_preserve(months)


_SENTIMENT_WORD_MAP = {
    "positive": "positive",
    "good": "positive",
    "great": "positive",
    "amazing": "positive",
    "negative": "negative",
    "bad": "negative",
    "issue": "negative",
    "issues": "negative",
    "complaint": "negative",
    "complaints": "negative",
    "terrible": "negative",
    "neutral": "neutral",
}
_SENTIMENT_REVIEW_TERMS = {
    "review",
    "reviews",
    "feedback",
    "comment",
    "comments",
    "sentiment",
    "guests",
    "guest",
}


def _detect_sentiment_filter(text: str | None) -> Optional[str]:
    """Return sentiment label if query explicitly calls for tone-specific reviews."""
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
# Entity / geo maps
# ---------------------------------------------------------------------------

_BOROUGH_MAP = {
    "manhattan": "Manhattan",
    "midtown": "Manhattan",
    "uptown": "Manhattan",
    "downtown": "Manhattan",
    "upper east": "Manhattan",
    "upper west": "Manhattan",
    "harlem": "Manhattan",
    "inwood": "Manhattan",
    "soho": "Manhattan",
    "chelsea": "Manhattan",
    "tribeca": "Manhattan",
    "brooklyn": "Brooklyn",
    "bk": "Brooklyn",
    "williamsburg": "Brooklyn",
    "greenpoint": "Brooklyn",
    "bed-stuy": "Brooklyn",
    "bed stuy": "Brooklyn",
    "park slope": "Brooklyn",
    "dumbo": "Brooklyn",
    "bushwick": "Brooklyn",
    "queens": "Queens",
    "astoria": "Queens",
    "jamaica": "Queens",
    "sunnyside": "Queens",
    "flushing": "Queens",
    "lic": "Queens",
}

_NEIGHBOURHOOD_TOKENS = {
    "sunnyside", "jamaica", "bushwick", "soho", "harlem", "astoria", "east village",
    "dumbo", "park slope", "upper east", "lic", "fort greene", "west village",
    "midtown", "ues", "flushing", "greenpoint", "chelsea", "bed-stuy",
    "inwood", "gramercy", "upper west", "uws", "tribeca",
}

# ---------------------------------------------------------------------------
# Intent lexicons
# ---------------------------------------------------------------------------

_SQL_KEYWORDS = {
    "percentage", "total", "revenue", "price", "benchmark", "adr", "share",
    "average", "top", "growth", "count", "occupancy", "median", "trend",
    "distribution", "rate", "change",
}

_RAG_KEYWORDS = {
    "comment", "quotes", "stories", "communication", "quote", "snippet", "sentiment",
    "why", "guests", "pros", "subway", "pain", "complaints", "praise", "cons",
    "experience", "clean", "feel", "mention", "story", "snippets", "feedback",
    "noise", "cleanliness",
}

_RAG_PHRASES = {"pain points", "near subway"}
_HYBRID_CONNECTORS = {
    "react", "and how do guests", "plus reviews", "as well", "and reviews",
    "and feedback", "combined with", "while also", "and also", "versus sentiment",
    "along with", "and comments", "together with",
}
# Amenity & comparison extensions
_AMENITY_TOPICS = {
    "amenities", "amenity", "features", "facilities",
    "wifi", "air conditioning", "parking", "pool", "gym",
    "kitchen", "laundry", "washer", "dryer",
}

_AMENITY_SQL_TRIGGERS = {
    "common", "most", "average", "distribution", "count",
    "top", "number of", "percentage of", "how many", "share of",
}

_COMPARISON_KEYWORDS = {
    "compare", "comparison", "versus", "vs", "benchmark", "against", "difference",
}

_HYBRID_INTENT_HINTS = {
    "influence", "impact", "affect", "why", "reason", "relationship", "correlation",
    "feedback on", "what do guests think about", "guest opinion", "reviews about",
    "reflect", "correlate", "link between", "relationship", "impact of",
    "influence of", "associated with", "reviews reflect",
}

_EXPANSION_KEYWORDS = (
    "expand",
    "new market",
    "new neighborhood",
    "where should we go next",
    "where should highbury expand",
    "best neighborhood to invest",
    "what area is growing",
    "future neighborhods",
    "future neighborhoods",
)

_COMPARISON_KEYWORD_TOKENS = {kw for kw in _COMPARISON_KEYWORDS if " " not in kw}
_COMPARISON_KEYWORD_PHRASES = {kw for kw in _COMPARISON_KEYWORDS if " " in kw}

_AMENITY_RAG_PHRASES = {
    "what do guests", "how do guests feel", "guest say about", "feedback on",
}

_GREETING_PHRASES = {"hi", "hey", "hello", "good morning", "good evening"}
_THANKS_PHRASES = {"thank you", "thanks", "appreciate it", "thanks a lot", "much appreciated"}
_SMALLTALK_PHRASES = {"how does this work", "help", "who are you", "what can you do", "tell me about yourself"}
_TRIAGE_KEYWORDS = {"triage", "action plan", "playbook", "backlog", "portfolio health", "diagnose"}
_TRIAGE_PHRASES = (
    "portfolio triage",
    "portfolio analysis",
    "triage for",
    "triage portfolio",
    "portfolio health check",
    "health check",
    "rank portfolio",
    "rank our portfolio",
    "rank portfolio based on",
)

_TRIAGE_KPI_PATTERNS: List[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b30[-\s]?day\s+occupancy\b"), "occupancy_rate_30"),
    (re.compile(r"\b60[-\s]?day\s+occupancy\b"), "occupancy_rate_60"),
    (re.compile(r"\b90[-\s]?day\s+occupancy\b"), "occupancy_rate_90"),
    (re.compile(r"\b365[-\s]?day\s+occupancy\b"), "occupancy_rate_365"),
    (re.compile(r"\boccupancy\s*(?:rate)?\s*(?:30|30d)\b"), "occupancy_rate_30"),
    (re.compile(r"\boccupancy\s*(?:rate)?\s*(?:60|60d)\b"), "occupancy_rate_60"),
    (re.compile(r"\boccupancy\s*(?:rate)?\s*(?:90|90d)\b"), "occupancy_rate_90"),
    (re.compile(r"\boccupancy\s*(?:rate)?\s*(?:365|year)\b"), "occupancy_rate_365"),
    (re.compile(r"\b30[-\s]?day\s+revenue\b"), "estimated_revenue_30"),
    (re.compile(r"\b60[-\s]?day\s+revenue\b"), "estimated_revenue_60"),
    (re.compile(r"\b(?:adr|adr test)\b"), "estimated_revenue_30"),
    (re.compile(r"\brevenue\b"), "estimated_revenue_30"),
    (re.compile(r"\breview(?:s)?\s+score\b"), "review_scores_rating"),
    (re.compile(r"\brating\b"), "review_scores_rating"),
]

_SCOPE_HIGHBURY_TOKENS = {"we", "our listings", "our kpi", "highbury", "my", "portfolio", "ours", "our"}
_SCOPE_MARKET_TOKENS = {"neighbourhoods", "citywide", "market", "nyc", "others", "borough", "neighborhoods"}
_SCOPE_HYBRID_TOKENS = {"benchmark", "compare", "comparison", "relative", "against", "stack up", "vs", "versus"}

_LISTING_RE = re.compile(
    r"\b(?:listing|property)(?:\s*(?:id|#))?(?:\s*[-–—:\"']*\s*)?(?P<id>\d{4,})\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _merge_unique(existing, additions):
    out = list(existing) if isinstance(existing, list) else []
    for v in additions:
        if v not in out:
            out.append(v)
    return out


def _has_phrase(text: str | None, patterns) -> bool:
    """Case-insensitive substring match for any phrase."""
    if not text:
        return False
    lowered = text.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def _detect_triage_kpi(text: Optional[str]) -> Optional[str]:
    """Map free-form KPI hints in triage queries to canonical columns."""
    if not text:
        return None
    lowered = text.lower()
    for pattern, column in _TRIAGE_KPI_PATTERNS:
        if pattern.search(lowered):
            return column
    return None


def _is_portfolio_triage(text: str) -> bool:
    lowered = text.lower()
    if any(phrase in lowered for phrase in _TRIAGE_PHRASES):
        return True
    if "portfolio" in lowered and any(keyword in lowered for keyword in _TRIAGE_KEYWORDS):
        return True
    if "health check" in lowered and ("portfolio" in lowered or "listings" in lowered):
        return True
    return False


def _detect_scope(text: str | None, tenant_hint: str | None) -> str:
    """Determine scope (Highbury, Market, or Hybrid)."""
    if not text:
        return tenant_hint.capitalize() if tenant_hint else "Market"

    lower = text.lower()
    tokens = set(_tokenize(text))
    has_highbury = any(tok in lower for tok in _SCOPE_HIGHBURY_TOKENS)
    has_market = any(tok in lower for tok in _SCOPE_MARKET_TOKENS)
    negated_highbury = bool(_NEGATED_HIGHBURY_PATTERN.search(text))
    connector_hybrid = _has_phrase(lower, _SCOPE_HYBRID_TOKENS)
    comparison_hint = bool(tokens & _COMPARISON_KEYWORD_TOKENS) or _has_phrase(text, _COMPARISON_KEYWORD_PHRASES)
    cross_scope = "highbury" in lower and "market" in lower

    if cross_scope:
        scope = "Hybrid"
    elif tenant_hint == "both" or (has_highbury and has_market) or connector_hybrid or comparison_hint:
        scope = "Hybrid"
    elif tenant_hint == "highbury" or (has_highbury and not has_market):
        scope = "Highbury"
    elif tenant_hint == "market" or has_market:
        scope = "Market"
    else:
        scope = "Market"

    if scope == "Hybrid" and negated_highbury and not has_market:
        scope = "Market"

    _LOGGER.debug("[INTENT] Scope detection: %s", scope)
    return scope


def _detect_intent(text: str | None) -> str:
    """Detect whether query is SQL, RAG, or Hybrid."""
    if not text:
        return "FACT_SQL"

    lowered = text.lower()
    tokens = set(re.findall(r"\b[a-zA-Z]+\b", lowered))

    sql_tokens = _SQL_KEYWORDS | {
        "trend", "value", "performance", "growth", "decline", "rate", "profit",
        "earnings", "income", "cost", "expensive", "cheap",
    }
    rag_tokens = _RAG_KEYWORDS | {
        "complaints", "recommendations", "noise", "transport", "cleanliness",
        "opinion", "stories", "host", "comfort", "experience", "amenities",
    }

    has_sql = bool(tokens & sql_tokens)
    has_rag = bool(tokens & rag_tokens) or _has_phrase(lowered, _RAG_PHRASES)

    connector_hits = [p for p in _HYBRID_CONNECTORS if p in lowered]
    has_hybrid_connector = bool(connector_hits)

    comparison_hits = [p for p in _COMPARISON_TOKENS if p in lowered]
    comparison_hits.extend(list(tokens & _COMPARISON_KEYWORD_TOKENS))
    comparison_hits.extend([kw for kw in _COMPARISON_KEYWORD_PHRASES if kw in lowered])
    has_comparison = bool(comparison_hits)

    hybrid_hint = _has_phrase(lowered, _HYBRID_INTENT_HINTS)

    # Amenity & comparison extensions
    amenity_topic = _has_phrase(lowered, _AMENITY_TOPICS)
    amenity_sql_trigger = _has_phrase(lowered, _AMENITY_SQL_TRIGGERS)
    amenity_sql = amenity_topic and (amenity_sql_trigger or has_sql or has_comparison)
    amenity_rag = amenity_topic and (has_rag or hybrid_hint)

    effective_sql = has_sql or amenity_sql or has_comparison
    hybrid_overlap = effective_sql and (has_rag or has_hybrid_connector or hybrid_hint)
    hybrid_compare = has_comparison and (has_rag or hybrid_hint or has_hybrid_connector)

    if amenity_rag and not effective_sql:
        return "REVIEWS_RAG"

    if hybrid_compare:
        _LOGGER.debug("[INTENT] Hybrid comparison triggers=%s", comparison_hits)
        return "FACT_SQL_RAG_HYBRID_COMPARE"

    if has_comparison and effective_sql:
        return "FACT_SQL_COMPARE"

    if hybrid_overlap:
        reason = []
        if has_rag:
            reason.append("sql_rag_overlap")
        if has_hybrid_connector:
            reason.append("hybrid_connector")
        if hybrid_hint:
            reason.append("why_or_impact_phrase")
        _LOGGER.debug("[INTENT] Hybrid triggers=%s", reason)
        return "FACT_SQL_RAG_HYBRID"

    if amenity_sql:
        return "FACT_SQL_AMENITIES"

    if has_rag:
        return "REVIEWS_RAG"

    if effective_sql:
        return "FACT_SQL"

    return "FACT_SQL"


def _detect_intent_refined(text: str | None) -> str:
    """Enhanced contextual routing after Codex augmentation."""
    if not text:
        return "FACT_SQL"

    lowered = text.lower()
    has_amenity_topic = any(topic in lowered for topic in _AMENITY_TOPICS)
    has_amenity_sql = has_amenity_topic and any(trigger in lowered for trigger in _AMENITY_SQL_TRIGGERS)
    has_amenity_rag = has_amenity_topic and any(phrase in lowered for phrase in _AMENITY_RAG_PHRASES)
    has_comparison = any(keyword in lowered for keyword in _COMPARISON_KEYWORDS)
    has_hybrid_hint = any(hint in lowered for hint in _HYBRID_INTENT_HINTS)
    has_review_terms = any(term in lowered for term in _RAG_KEYWORDS) or _has_phrase(lowered, _RAG_PHRASES)
    has_sql_terms = any(term in lowered for term in _SQL_KEYWORDS)

    if has_amenity_sql and not has_amenity_rag:
        return "FACT_SQL_AMENITIES"

    if has_amenity_rag:
        return "REVIEWS_RAG"

    if (has_review_terms and has_sql_terms) or has_hybrid_hint:
        return "FACT_SQL_RAG_HYBRID"

    if has_comparison and not has_review_terms:
        return "FACT_SQL_COMPARE"

    if has_comparison and has_review_terms:
        return "FACT_SQL_RAG_HYBRID_COMPARE"

    return _detect_intent(text)


def _is_expansion_query(text: str) -> bool:
    """Return True when the query is asking about expansion / new markets."""
    if not text:
        return False
    normalized = text.lower()
    return any(keyword in normalized for keyword in _EXPANSION_KEYWORDS)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def classify_intent(state: State) -> State:
    """Enrich agent state with detected scope, intent, and lightweight filters."""
    raw_query = state.get("query", "") or ""
    tenant_hint = (state.get("tenant") or "").strip().lower()
    normalized = re.sub(r"\s+", " ", raw_query.strip().lower())

    # Handle conversational queries
    if not normalized or normalized in _GREETING_PHRASES:
        state.update({"scope": "General", "intent": "GREETING", "filters": {}})
        _LOGGER.info("[INTENT] GREETING → %r", raw_query)
        return state

    if any(p in normalized for p in _THANKS_PHRASES):
        state.update({"scope": "General", "intent": "THANKS", "filters": {}})
        _LOGGER.info("[INTENT] THANKS → %r", raw_query)
        return state

    if any(p in normalized for p in _SMALLTALK_PHRASES):
        state.update({"scope": "General", "intent": "SMALLTALK", "filters": {}})
        _LOGGER.info("[INTENT] SMALLTALK → %r", raw_query)
        return state

    filters = dict(state.get("filters", {}))
    negated_highbury = bool(_NEGATED_HIGHBURY_PATTERN.search(raw_query))
    contains_highbury = "highbury" in normalized
    contains_market = "market" in normalized
    contains_our = bool(_OUR_PATTERN.search(raw_query))

    if _is_expansion_query(normalized):
        state.update({"scope": "Highbury", "intent": "EXPANSION_SCOUT", "filters": filters})
        _LOGGER.info("[INTENT] EXPANSION_SCOUT detected for query=%r", raw_query)
        return state

    compare_scope = (
        contains_highbury
        and contains_market
        and bool(_COMPARE_SCOPE_PATTERN.search(raw_query))
        and (
            any(tok in normalized for tok in _COMPARISON_TOKENS)
            or _has_phrase(raw_query, _COMPARISON_KEYWORDS)
        )
    )

    scope = _detect_scope(raw_query, tenant_hint)
    intent = _detect_intent_refined(raw_query)
    sentiment_hint = _detect_sentiment_filter(raw_query)
    if sentiment_hint and not filters.get("sentiment_label"):
        filters["sentiment_label"] = sentiment_hint
    if sentiment_hint:
        intent = "SENTIMENT_REVIEWS"

    if _is_portfolio_triage(normalized):
        filters["is_highbury"] = True
        kpi_override = _detect_triage_kpi(raw_query) or _detect_triage_kpi(normalized)
        if kpi_override:
            filters["kpi"] = kpi_override
        scope = "Highbury"
        intent = "PORTFOLIO_TRIAGE_ADVANCED"
        state.update({"scope": scope, "intent": intent, "filters": filters})
        _LOGGER.info("[INTENT] PORTFOLIO_TRIAGE_ADVANCED detected for query=%r", raw_query)
        return state

    if compare_scope:
        telemetry = state.setdefault("telemetry", {})
        telemetry["compare_scope"] = True
        scope = "Hybrid"
        if intent in {"HYBRID", "REVIEWS_RAG"}:
            intent = "REVIEWS_RAG"

    _LOGGER.info("[INTENT] Detected: intent=%s | scope=%s | query=%r", intent, scope, raw_query)

    # Borough and neighbourhood extraction
    borough_matches = [canon for key, canon in _BOROUGH_MAP.items() if key in normalized]
    if borough_matches:
        filters["borough"] = _merge_unique(filters.get("borough", []), borough_matches)

    hood_hits = [
        hood
        for hood in _NEIGHBOURHOOD_TOKENS
        if re.search(fr"\b{hood.replace(' ', r'[\s-]+')}\b", normalized)
    ]
    if hood_hits:
        filters["neighbourhood_hint"] = _merge_unique(filters.get("neighbourhood_hint", []), hood_hits)

    # Temporal extraction
    months = _extract_months(raw_query)
    if months:
        filters["month"] = _merge_unique(filters.get("month", []), months)

    listing_matches = list(_LISTING_RE.finditer(raw_query))
    listing_spans: List[tuple[int, int]] = [m.span() for m in listing_matches]
    years = []
    for m in re.finditer(r"(?:20)?(\d{2})", raw_query):
        year_val = int(m.group(1))
        if year_val < 100:
            year_val += 2000
        span = m.span()
        if any(not (span[1] <= ls[0] or span[0] >= ls[1]) for ls in listing_spans):
            continue
        years.append(year_val)
    if years:
        filters["year"] = _merge_unique(filters.get("year", []), _dedupe_preserve(years))

    # Listing ID extraction
    if listing_matches:
        chosen_match = listing_matches[0]
        try:
            filters["listing_id"] = int(chosen_match.group("id"))
        except ValueError:
            _LOGGER.debug("[INTENT] Invalid listing ID: %s", chosen_match.group("id"))

    # Highbury flag inference
    highbury_flag = filters.get("is_highbury")
    if negated_highbury:
        highbury_flag = False
    elif contains_our:
        highbury_flag = True
    elif compare_scope:
        highbury_flag = False
    elif contains_highbury:
        highbury_flag = True
    elif highbury_flag is None:
        highbury_flag = False

    filters["is_highbury"] = highbury_flag

    if contains_our and not filters.get("borough"):
        filters["borough"] = ["Manhattan"]
    if scope == "Highbury" and not negated_highbury and not compare_scope:
        filters["is_highbury"] = True

    state.update({"scope": scope, "intent": intent, "filters": filters})
    _LOGGER.info(
        "[INTENT+SCOPE] query=%r | intent=%s | scope=%s | filters=%s",
        raw_query,
        intent,
        scope,
        filters,
    )
    return state


__all__ = ["classify_intent"]
