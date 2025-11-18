"""
Metadata filter extraction utilities for H.O.P.E AI Agent.

Purpose:
- Parse a natural-language query (e.g., “Top 5 reviews in September 2025 for Highbury listing 6848”)
  into structured metadata filters compatible with vector.py filtering.
- Robust to partial matches, case differences, and short year/month forms (e.g., “Aug 25”, “August 2025”).
"""

from __future__ import annotations
import logging
import re
from typing import Any, Dict, List
import pandas as pd

from ..config import MONTHS_MAP, SEASON_MAP

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patterns & constants
# ---------------------------------------------------------------------------

_YEAR_FULL_PATTERN = re.compile(r"\b(20\d{2})\b")
_YEAR_SHORT_PATTERN = re.compile(r"(?<!\d)(?:'|’)?(\d{2})(?!\d)")
_LISTING_PATTERN = re.compile(
    r"\b(?:listing|property)(?:\s*(?:id|#))?(?:\s*[-–—:]*\s*)?(?P<id>\d{4,})\b",
    re.IGNORECASE,
)
_LONG_LISTING_PATTERN = re.compile(
    r"\b(?:listing|property)(?:\s*(?:id|#))?(?:\s*[-–—:]*\s*)?(?P<long_id>\d{6,})\b",
    re.IGNORECASE,
)
_NUMBER_PATTERN = re.compile(r"\b\d{3,}\b", re.IGNORECASE)

_MONTH_ALIASES = {
    **MONTHS_MAP,
    "sept": "SEP",
    "sep.": "SEP",
    "aug.": "AUG",
    "oct.": "OCT",
}

_TEMPORAL_CUES = {"in", "during", "from"}
_TEMPORAL_SKIP_TOKENS = {
    "into", "throughout", "of", "to", "months", "till", "through", "until", "the",
    "month", "early", "and", "til", "late",
}

_TOKEN_PATTERN = re.compile(r"\b[\w'-]+\b")
_NEGATED_HIGHBURY_PATTERN = re.compile(
    r"\b(?:non[-\s]?highbury|not\s+highbury|except\s+highbury)\b", re.IGNORECASE
)
_COMPARE_SCOPE_PATTERN = re.compile(
    r"\bhighbury\b[^.]*\bmarket\b|\bmarket\b[^.]*\bhighbury\b", re.IGNORECASE
)
_COMPARISON_TOKENS = ("compare", "comparison", " vs ", "versus", "benchmark", "against", "relative", "stack up")

# Borough inference map from neighborhood keywords
NEIGHBORHOOD_TO_BOROUGH: Dict[str, str] = {
    "bedford stuyvesant": "Brooklyn",
    "harlem": "Manhattan",
    "midtown": "Manhattan",
    "williamsburg": "Brooklyn",
    "bushwick": "Brooklyn",
    "crown heights": "Brooklyn",
    "hells kitchen": "Manhattan",
    "chelsea": "Manhattan",
    "financial district": "Manhattan",
    "east village": "Manhattan",
    "lower east side": "Manhattan",
    "upper west side": "Manhattan",
    "east harlem": "Manhattan",
    "astoria": "Queens",
    "jamaica": "Queens",
    "park slope": "Brooklyn",
    "flushing": "Queens",
    "canarsie": "Brooklyn",
    "east new york": "Brooklyn",
    "flatbush": "Brooklyn",
    "sunset park": "Brooklyn",
    "long island city": "Queens",
    "east elmhurst": "Queens",
    "upper east side": "Manhattan",
    "east flatbush": "Brooklyn",
}

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _normalise_text(value: Any) -> str:
    """Return lowercase text without extra spaces."""
    return str(value).strip().lower()


def _normalise_lookup(value: Any) -> str:
    """Normalise text for dictionary lookups (strip punctuation and collapse whitespace)."""
    norm = _normalise_text(value)
    norm = re.sub(r"[’']", "", norm)
    norm = re.sub(r"[-_/]", " ", norm)
    norm = re.sub(r"\s+", " ", norm)
    return norm


def _dedupe_preserve(items: List[str]) -> List[str]:
    """Remove duplicates while preserving the original order."""
    seen, ordered = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase word-like components."""
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]


def _has_temporal_cue(tokens: List[str], index: int) -> bool:
    """Check if a month token is preceded by a temporal cue like 'in', 'during', or 'from'."""
    for offset in range(1, min(index, 6) + 1):
        candidate = tokens[index - offset]
        if candidate in _TEMPORAL_CUES:
            return True
        if candidate in _TEMPORAL_SKIP_TOKENS or candidate in _MONTH_ALIASES:
            continue
    return False


def _set_borough_filter(filters: Dict[str, Any], borough: str) -> None:
    """Merge borough value into filters without losing previous selections."""
    if not borough:
        return
    current = filters.get("borough")
    if current is None:
        filters["borough"] = borough
    elif isinstance(current, list):
        if borough not in current:
            current.append(borough)
    elif current != borough:
        filters["borough"] = [current, borough]


def _infer_years(text: str) -> List[int]:
    """Infer 2-digit or 4-digit years intelligently."""
    years: List[int] = []

    # 4-digit
    for y in _YEAR_FULL_PATTERN.findall(text):
        try:
            years.append(int(y))
        except ValueError:
            continue

    # 2-digit
    for y in _YEAR_SHORT_PATTERN.findall(text):
        try:
            val = int(y)
            if 0 <= val <= 99:
                if val < 30:
                    val += 2000
                else:
                    val += 1900
                if val not in years:
                    years.append(val)
        except ValueError:
            continue

    return sorted(set(years))


def _infer_months(text: str) -> List[str]:
    """Find month or season references with temporal cues and return canonical codes."""
    tokens = _tokenize(text)
    months = []

    for idx, token in enumerate(tokens):
        canon = _MONTH_ALIASES.get(token)
        if canon and _has_temporal_cue(tokens, idx):
            months.append(canon)

    lowered = text.lower()
    for season, season_months in SEASON_MAP.items():
        if re.search(fr"\b{re.escape(season)}\b", lowered):
            months.extend(season_months)

    return _dedupe_preserve(months)


def _alias_matches(text: str, alias: Any, *, context: str) -> bool:
    """Return True if alias appears in text with safe word boundaries."""
    if not isinstance(alias, str):
        alias = str(alias) if alias is not None else ""
    alias_norm = _normalise_text(alias)
    if not alias_norm:
        return False

    parts = [re.escape(part) for part in alias_norm.split()]
    if not parts:
        return False

    pattern_body = r"\s+".join(parts)
    pattern = re.compile(fr"(?<![\w]){pattern_body}(?![\w])", re.IGNORECASE)

    compact = alias_norm.replace(" ", "")
    if len(compact) < 3:
        # Skip short aliases that cause false positives
        if pattern.search(text):
            tokens = re.findall(r"\b\w+\b", text.lower())
            if alias_norm in tokens:
                return True
            _LOGGER.debug(
                "Skipping short %s alias '%s' to avoid partial match in query %r",
                context, alias, text,
            )
        return False

    return bool(pattern.search(text))


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def extract_metadata_filters(query: str, meta_df: pd.DataFrame | None = None) -> Dict[str, Any]:
    """Infer structured metadata filters from a natural-language query."""
    if not isinstance(query, str) or not query.strip():
        return {}

    lowered = query.lower()
    filters: Dict[str, Any] = {}

    # Years
    years = _infer_years(lowered)
    if years:
        filters["year"] = years[-1] if len(years) == 1 else years

    # Months
    months = _infer_months(query)
    if months:
        filters["month"] = months[-1] if len(months) == 1 else months

    # Listing ID
    listing_match = _LONG_LISTING_PATTERN.search(query) or _LISTING_PATTERN.search(query)
    if listing_match:
        try:
            group_name = "long_id" if listing_match.groupdict().get("long_id") else "id"
            filters["listing_id"] = str(int(listing_match.group(group_name)))
        except ValueError:
            pass
    else:
        # fallback numeric inference
        for raw in _NUMBER_PATTERN.findall(lowered):
            try:
                if len(raw) >= 6:
                    num = int(raw)
                    if num not in years:
                        filters["listing_id"] = str(num)
                        break
            except ValueError:
                continue

    # Highbury scope inference
    contains_highbury = "highbury" in lowered
    negated_highbury = bool(_NEGATED_HIGHBURY_PATTERN.search(query))
    contains_market = "market" in lowered
    contains_our = bool(re.search(r"\b(?:our|ours|we|my)\b", lowered))
    compare_scope = (
        contains_highbury
        and contains_market
        and bool(_COMPARE_SCOPE_PATTERN.search(query))
        and any(token in lowered for token in _COMPARISON_TOKENS)
    )

    if negated_highbury:
        filters["is_highbury"] = False
    elif contains_our:
        filters["is_highbury"] = True
    elif compare_scope:
        filters["is_highbury"] = False
    elif contains_highbury:
        filters["is_highbury"] = True
    else:
        filters["is_highbury"] = False

    # Borough / neighbourhood extraction via metadata
    neighbourhood_match = None
    if meta_df is not None and not meta_df.empty:
        if "neighbourhood_group" in meta_df.columns:
            for group in meta_df["neighbourhood_group"].dropna().unique():
                if _alias_matches(query, group, context="borough"):
                    filters["borough"] = group
                    break

        if "neighbourhood" in meta_df.columns:
            for hood in meta_df["neighbourhood"].dropna().unique():
                if _alias_matches(query, hood, context="neighbourhood"):
                    filters["neighbourhood"] = hood
                    neighbourhood_match = hood
                    break

    if neighbourhood_match:
        mapped_borough = NEIGHBORHOOD_TO_BOROUGH.get(_normalise_lookup(neighbourhood_match))
        if mapped_borough:
            _set_borough_filter(filters, mapped_borough)

    if contains_our and not filters.get("borough"):
        filters["borough"] = "Manhattan"

    _LOGGER.debug("[FILTERS] Extracted metadata filters: %s", filters)
    return filters


__all__ = ["extract_metadata_filters"]
