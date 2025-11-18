"""Portfolio triage node that orchestrates SQL + RAG probes."""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .nl2sql_llm import execute_duckdb
from .types import GraphState, add_thinking_step
from .vector_qdrant import exec_rag as legacy_exec_rag
from .vector_qdrant import summarize_hits as legacy_summarize_hits

_LOGGER = logging.getLogger(__name__)

_MAX_LISTING_CARDS = 5
_MAX_RAG_PER_LISTING = 3
_DEFAULT_KPI = "occupancy_rate_90"

_KPI_OPTIONS: Dict[str, Dict[str, Any]] = {
    "occupancy_rate_30": {
        "label": "30-day occupancy",
        "unit": "%",
        "decimals": 1,
        "description": "Projected occupancy over the next 30 days",
    },
    "occupancy_rate_60": {
        "label": "60-day occupancy",
        "unit": "%",
        "decimals": 1,
        "description": "Projected occupancy over the next 60 days",
    },
    "occupancy_rate_90": {
        "label": "90-day occupancy",
        "unit": "%",
        "decimals": 1,
        "description": "Projected occupancy over the next 90 days",
    },
    "occupancy_rate_365": {
        "label": "365-day occupancy",
        "unit": "%",
        "decimals": 1,
        "description": "Projected occupancy over the next 12 months",
    },
    "estimated_revenue_30": {
        "label": "30-day revenue",
        "unit": "$",
        "decimals": 0,
        "description": "Projected revenue over the next 30 days",
    },
    "estimated_revenue_60": {
        "label": "60-day revenue",
        "unit": "$",
        "decimals": 0,
        "description": "Projected revenue over the next 60 days",
    },
    "review_scores_rating": {
        "label": "review score",
        "unit": "pts",
        "decimals": 1,
        "description": "Average guest review rating",
    },
}

_KPI_ALIASES: Dict[str, str] = {
    "30 day occupancy": "occupancy_rate_30",
    "30d occupancy": "occupancy_rate_30",
    "occ30": "occupancy_rate_30",
    "occupancy rate 30": "occupancy_rate_30",
    "60 day occupancy": "occupancy_rate_60",
    "60d occupancy": "occupancy_rate_60",
    "occ60": "occupancy_rate_60",
    "90 day occupancy": "occupancy_rate_90",
    "90d occupancy": "occupancy_rate_90",
    "occ90": "occupancy_rate_90",
    "365 day occupancy": "occupancy_rate_365",
    "365d occupancy": "occupancy_rate_365",
    "year occupancy": "occupancy_rate_365",
    "revenue 30": "estimated_revenue_30",
    "30 day revenue": "estimated_revenue_30",
    "revenue30": "estimated_revenue_30",
    "revenue 60": "estimated_revenue_60",
    "60 day revenue": "estimated_revenue_60",
    "review score": "review_scores_rating",
    "rating": "review_scores_rating",
    "scores rating": "review_scores_rating",
}

_THEME_KEYWORDS = {
    "cleanliness": ("clean", "dirty", "dust", "odor", "smell"),
    "noise": ("noise", "noisy", "loud", "traffic"),
    "wifi": ("wifi", "internet", "connection"),
    "comfort": ("bed", "sofa", "mattress", "temperature", "heat", "cold"),
    "access": ("check-in", "checkin", "stairs", "lock", "key", "door", "elevator"),
}


def _sanitize_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace("%", "").replace("$", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _resolve_table(state: GraphState) -> str:
    table = (state.plan or {}).get("sql_table")
    if table:
        return table
    scope = (state.scope or "").lower()
    if scope == "highbury":
        return "highbury_listings"
    return "listings_cleaned"


def _normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    lowered = text.lower()
    for char in "-_/":
        lowered = lowered.replace(char, " ")
    return " ".join(lowered.split())


def _match_kpi_from_text(text: Optional[str]) -> Optional[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return None
    for alias, column in _KPI_ALIASES.items():
        if alias in normalized:
            return column
    for column in _KPI_OPTIONS.keys():
        if column in normalized:
            return column
    return None


def _resolve_kpi_choice(state: GraphState) -> Dict[str, Any]:
    candidates: List[str] = []
    filters = state.filters or {}
    for key in ("kpi", "metric", "sort", "sort_by", "rank_by", "focus_metric"):
        value = filters.get(key)
        if isinstance(value, str):
            candidates.append(value)
    plan = state.plan or {}
    for key in ("kpi", "metric", "sort_by"):
        value = plan.get(key)
        if isinstance(value, str):
            candidates.append(value)
    raw_query = (state.raw_input or {}).get("query")
    if isinstance(raw_query, str):
        candidates.append(raw_query)
    if state.query:
        candidates.append(state.query)

    for text in candidates:
        column = _match_kpi_from_text(text)
        if column and column in _KPI_OPTIONS:
            config = dict(_KPI_OPTIONS[column])
            config["column"] = column
            return config

    fallback = dict(_KPI_OPTIONS[_DEFAULT_KPI])
    fallback["column"] = _DEFAULT_KPI
    return fallback


def _dedupe_preserve(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        lowered = value.lower()
        if lowered not in seen:
            seen.add(lowered)
            ordered.append(lowered)
    return ordered


def _extract_scope_filters(filters: Optional[Dict[str, Any]]) -> Tuple[str, Dict[str, List[str]]]:
    filters = filters or {}
    borough_source = filters.get("borough") or []
    if not isinstance(borough_source, list):
        borough_source = [borough_source]
    neighbourhood_source = filters.get("neighbourhood") or []
    if not isinstance(neighbourhood_source, list):
        neighbourhood_source = [neighbourhood_source]
    hood_hints = filters.get("neighbourhood_hint") or []
    if not isinstance(hood_hints, list):
        hood_hints = [hood_hints]

    boroughs = _dedupe_preserve(str(b).strip() for b in borough_source if str(b).strip())
    neighbourhoods = _dedupe_preserve(
        str(n).strip() for n in (neighbourhood_source + hood_hints) if str(n).strip()
    )

    label_parts = ["Highbury"]
    if boroughs:
        label_parts.append(" / ".join(b.title() for b in boroughs))
    if neighbourhoods:
        label_parts.append(" / ".join(n.title() for n in neighbourhoods))
    scope_label = " / ".join(label_parts)

    return scope_label, {"boroughs": boroughs, "neighbourhoods": neighbourhoods}


def _escape_literal(value: str) -> str:
    return value.replace("'", "''")


def _build_scope_where_clause(scope_filters: Dict[str, List[str]]) -> str:
    clauses: List[str] = []
    boroughs = scope_filters.get("boroughs") or []
    if boroughs:
        joined = ", ".join(f"'{_escape_literal(b)}'" for b in boroughs)
        clauses.append(f"lower(neighbourhood_group) IN ({joined})")
    neighbourhoods = scope_filters.get("neighbourhoods") or []
    if neighbourhoods:
        joined = ", ".join(f"'{_escape_literal(n)}'" for n in neighbourhoods)
        clauses.append(f"lower(neighbourhood) IN ({joined})")
    return f"WHERE {' AND '.join(clauses)}" if clauses else ""


def _execute_rows(sql: str) -> List[Dict[str, Any]]:
    try:
        result = execute_duckdb(sql)
    except Exception as exc:  # pragma: no cover - defensive guard
        _LOGGER.error("Portfolio triage DuckDB query failed: %s", exc)
        return []
    return result.get("rows") or []


def _listing_identifier(row: Dict[str, Any]) -> Optional[str]:
    for key in ("listing_id", "listings_id", "id"):
        value = row.get(key)
        if value in (None, ""):
            continue
        return str(value)
    return None


def _normalize_listing_rows(rows: List[Dict[str, Any]], kpi_column: str) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        listing_id = _listing_identifier(row)
        if not listing_id:
            continue
        neighbourhood = row.get("neighbourhood") or row.get("neighborhood")
        borough = row.get("neighbourhood_group") or row.get("borough")
        kpi_val = _sanitize_float(row.get("selected_kpi"))
        if kpi_val is None:
            kpi_val = _sanitize_float(row.get(kpi_column))
        normalized.append(
            {
                "listing_id": listing_id,
                "listing_name": row.get("listing_name") or row.get("name") or row.get("host_name"),
                "neighbourhood": neighbourhood,
                "borough": borough,
                "kpi_value": kpi_val,
                "metrics": {
                    "kpi_value": kpi_val,
                    "price": _sanitize_float(row.get("price_in_usd") or row.get("price")),
                    "revenue_30": _sanitize_float(row.get("estimated_revenue_30")),
                    "revenue_60": _sanitize_float(row.get("estimated_revenue_60")),
                    "revenue_90": _sanitize_float(row.get("estimated_revenue_90")),
                    "revenue_365": _sanitize_float(row.get("estimated_revenue_365")),
                    "review_score": _sanitize_float(row.get("review_scores_rating")),
                    "occupancy_rate_30": _sanitize_float(row.get("occupancy_rate_30")),
                    "occupancy_rate_60": _sanitize_float(row.get("occupancy_rate_60")),
                    "occupancy_rate_90": _sanitize_float(row.get("occupancy_rate_90")),
                    "occupancy_rate_365": _sanitize_float(row.get("occupancy_rate_365")),
                },
            }
        )
    return normalized


def _parse_distribution(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    row = rows[0]
    return {
        "listing_count": int(_sanitize_float(row.get("listing_count")) or 0),
        "avg": _sanitize_float(row.get("avg_kpi")),
        "median": _sanitize_float(row.get("median_kpi")),
        "min": _sanitize_float(row.get("min_kpi")),
        "max": _sanitize_float(row.get("max_kpi")),
        "stddev": _sanitize_float(row.get("stddev_kpi")),
    }


def _format_metric_value(value: Optional[float], *, unit: Optional[str], decimals: int = 1) -> Optional[str]:
    if value is None:
        return None
    if unit == "%":
        return f"{value:.{decimals}f}%"
    if unit == "$":
        pattern = f"{value:,.{decimals}f}" if decimals else f"{value:,.0f}"
        return f"${pattern}"
    if unit == "pts":
        return f"{value:.{decimals}f}"
    return f"{value:.{decimals}f}"


def _index_market_benchmarks(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        borough = row.get("neighbourhood_group") or row.get("borough")
        neighbourhood = row.get("neighbourhood") or row.get("neighborhood")
        if not neighbourhood:
            continue
        key = f"{(borough or '').strip().lower()}::{neighbourhood.strip().lower()}"
        index[key] = {
            "market_median_kpi": _sanitize_float(row.get("market_median_kpi")),
            "market_median_price_usd": _sanitize_float(row.get("market_median_price_usd")),
            "market_median_revenue_30": _sanitize_float(row.get("market_median_revenue_30")),
            "market_avg_review_score": _sanitize_float(row.get("market_avg_review_score")),
            "label": f"{(neighbourhood or '').strip()} ({(borough or '').strip()})".strip(),
            "neighbourhood": (neighbourhood or "").strip(),
            "borough": (borough or "").strip(),
        }
    return index


def _snippet_preview(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    snippet = str(text).strip()
    return snippet if len(snippet) <= 200 else snippet[:197].rstrip() + "..."


def _format_review_quotes(hits: List[Dict[str, Any]]) -> List[str]:
    quotes: List[str] = []
    for hit in hits[:_MAX_RAG_PER_LISTING]:
        borough = hit.get("borough") or hit.get("neighbourhood_group") or "n/a"
        month = hit.get("month") or ""
        year = hit.get("year") or ""
        snippet = _snippet_preview(hit.get("snippet") or hit.get("text") or hit.get("comments"))
        if not snippet:
            continue
        context = borough
        if month or year:
            context = f"{context} | {month} {year}".strip()
        quotes.append(f"({context.strip()}) {snippet}")
    return quotes


def _infer_theme(snippets: List[Dict[str, Any]]) -> Optional[str]:
    if not snippets:
        return None
    sample = (snippets[0].get("snippet") or snippets[0].get("text") or "").lower()
    for label, keywords in _THEME_KEYWORDS.items():
        if any(keyword in sample for keyword in keywords):
            return label
    return None


def _fetch_listing_sentiment(
    base_state: GraphState,
    listing_id: str,
    sentiment_label: str,
    *,
    top_k: int = _MAX_RAG_PER_LISTING,
) -> List[Dict[str, Any]]:
    rag_state: Dict[str, Any] = {
        "query": f"{sentiment_label.capitalize()} guest feedback for listing {listing_id}",
        "filters": {"listing_id": listing_id, "sentiment_label": sentiment_label},
        "rag_needed": True,
        "telemetry": {},
        "plan": {"top_k": max(top_k, 3)},
        "scope": base_state.scope,
        "debug_thinking": base_state.debug_thinking,
    }
    try:
        ragged = legacy_exec_rag(rag_state)
        ragged = legacy_summarize_hits(ragged)
        hits = ragged.get("rag_snippets") or []
        return hits[:top_k]
    except Exception as exc:  # pragma: no cover - qdrant failure
        _LOGGER.warning("Listing sentiment retrieval failed for %s: %s", listing_id, exc)
        return []


def _pricing_gap_percent(price: Optional[float], market_price: Optional[float]) -> Optional[float]:
    if price is None or market_price in (None, 0):
        return None
    if market_price == 0:
        return None
    return ((price - market_price) / market_price) * 100.0


def _recommended_adr_range(
    current_price: Optional[float],
    market_price: Optional[float],
    sentiment_label: Optional[str],
) -> Optional[Dict[str, Any]]:
    if current_price is None or current_price <= 0:
        return None
    sentiment = (sentiment_label or "").lower()
    if sentiment == "positive":
        low = current_price * 1.05
        high = current_price * 1.15
        mode = "increase"
    else:
        low = current_price * 0.85
        high = current_price * 0.95
        mode = "decrease"
    return {
        "mode": mode,
        "test_low": low,
        "test_high": high,
        "current_price": current_price,
        "market_price": market_price,
    }


def _summarize_sentiment_hits(hits: List[Dict[str, Any]], sentiment_label: str) -> Dict[str, Any]:
    def _avg(key: str) -> Optional[float]:
        values = [hit.get(key) for hit in hits if isinstance(hit.get(key), (int, float))]
        return float(sum(values) / len(values)) if values else None

    compound_scores = [hit.get("compound") for hit in hits if isinstance(hit.get("compound"), (int, float))]
    compound_avg = float(sum(compound_scores) / len(compound_scores)) if compound_scores else None
    return {
        "label": sentiment_label,
        "hit_count": len(hits),
        "positive": _avg("positive"),
        "neutral": _avg("neutral"),
        "negative": _avg("negative"),
        "compound": compound_avg,
    }


def _build_actions(
    tier_type: str,
    kpi_info: Dict[str, Any],
    kpi_delta: Optional[float],
    price_gap: Optional[float],
    theme: Optional[str],
    adr_recommendation: Optional[Dict[str, Any]],
) -> List[str]:
    def _adr_range_label() -> Optional[str]:
        if not adr_recommendation:
            return None
        low = adr_recommendation.get("test_low")
        high = adr_recommendation.get("test_high")
        if isinstance(low, (int, float)) and isinstance(high, (int, float)):
            low_text = _format_metric_value(low, unit="$", decimals=0)
            high_text = _format_metric_value(high, unit="$", decimals=0)
            if low_text and high_text:
                return f"{low_text}–{high_text}"
        return None

    actions: List[str] = []
    kpi_gap_display = None
    if kpi_delta is not None:
        kpi_gap_display = _format_metric_value(abs(kpi_delta), unit=kpi_info.get("unit"), decimals=kpi_info.get("decimals", 1))
    price_gap_display = None
    if price_gap is not None:
        price_gap_display = f"{abs(price_gap):.0f}%"
    current_price_text = _format_metric_value(
        (adr_recommendation or {}).get("current_price"),
        unit="$",
        decimals=0,
    )
    market_price_text = _format_metric_value(
        (adr_recommendation or {}).get("market_price"),
        unit="$",
        decimals=0,
    )
    adr_range_text = _adr_range_label()

    if tier_type == "problem":
        if theme:
            actions.append(f"Address {theme} feedback highlighted in guest reviews within the next 7 days.")
        else:
            actions.append("Resolve the top guest pain points cited in reviews within the next 7 days.")
        if kpi_gap_display:
            actions.append(f"Close the {kpi_gap_display} {kpi_info['label']} gap versus market medians.")
        if price_gap is not None:
            if price_gap > 5:
                actions.append(f"Consider tactical ADR cuts (~{price_gap_display}) until sentiment recovers.")
            elif price_gap < -5:
                actions.append("Hold rates — portfolio pricing already trails market.")
        if adr_range_text:
            actions.append(
                f"Reset ADR into the {adr_range_text} band "
                f"(current {current_price_text or 'n/a'}, market {market_price_text or 'n/a'}) until sentiment improves."
            )
        actions.append("Document fixes and re-list the improvements in the weekly operations pulse.")
    else:
        if theme:
            actions.append(f"Double down on the {theme} theme guests love — make it part of the booking pitch.")
        actions.append("Protect review flywheel with proactive mid-stay check-ins and rapid responses.")
        if adr_range_text:
            actions.append(
                f"Run ADR experiments in the {adr_range_text} range "
                f"(current {current_price_text or 'n/a'}, market {market_price_text or 'n/a'}) while demand and sentiment stay strong."
            )
        elif price_gap is not None and price_gap < -3:
            actions.append("Test +10–15% ADR lifts over the next 2 weeks while occupancy momentum holds.")
        else:
            actions.append("Pilot a premium package (photos, add-ons) to convert strength into higher ADR.")

    return actions


def _build_listing_insight(
    row: Dict[str, Any],
    *,
    kpi_info: Dict[str, Any],
    tier_type: str,
    sentiment_label: str,
    benchmark: Optional[Dict[str, Any]],
    base_state: GraphState,
    sentiment_tracker: Dict[str, Any],
    rag_hits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    listing_id = row.get("listing_id")
    hits = _fetch_listing_sentiment(base_state, listing_id, sentiment_label)
    if hits:
        rag_hits.extend(hits)
    sentiment_meta = _summarize_sentiment_hits(hits, sentiment_label)
    theme = _infer_theme(hits)
    quotes = _format_review_quotes(hits)
    kpi_value = row.get("metrics", {}).get("kpi_value")
    market_kpi = (benchmark or {}).get("market_median_kpi")
    kpi_delta = None
    if kpi_value is not None and market_kpi is not None:
        kpi_delta = kpi_value - market_kpi
    current_price = row.get("metrics", {}).get("price")
    market_price = (benchmark or {}).get("market_median_price_usd")
    price_gap = _pricing_gap_percent(current_price, market_price)
    adr_recommendation = _recommended_adr_range(current_price, market_price, sentiment_label)
    actions = _build_actions(tier_type, kpi_info, kpi_delta, price_gap, theme, adr_recommendation)

    totals: Counter = sentiment_tracker.setdefault("totals", Counter())
    totals[sentiment_label] += sentiment_meta.get("hit_count", 0)
    listing_summaries = sentiment_tracker.setdefault("listings", [])
    listing_summaries.append(
        {
            "listing_id": listing_id,
            "listing_name": row.get("listing_name"),
            "label": sentiment_meta.get("label"),
            "hit_count": sentiment_meta.get("hit_count", 0),
            "compound": sentiment_meta.get("compound"),
        }
    )

    return {
        "listing_id": listing_id,
        "listing_name": row.get("listing_name"),
        "neighbourhood": row.get("neighbourhood"),
        "borough": row.get("borough"),
        "metrics": row.get("metrics"),
        "kpi_value": kpi_value,
        "market_kpi_median": market_kpi,
        "kpi_vs_market_delta": kpi_delta,
        "pricing_gap_percent": price_gap,
        "tier": "tier2" if tier_type == "winner" else "tier1",
        "tier_type": tier_type,
        "sentiment": sentiment_meta,
        "theme": theme,
        "sample_reviews": quotes,
        "actions": actions,
        "market_context": benchmark or {},
        "adr_recommendation": adr_recommendation,
    }


def _listing_overview(entry: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        "listing_id": entry.get("listing_id"),
        "listing_name": entry.get("listing_name"),
        "neighbourhood": entry.get("neighbourhood"),
        "borough": entry.get("borough"),
        "kpi_value": entry.get("kpi_value"),
        "market_kpi_median": entry.get("market_kpi_median"),
        "kpi_vs_market_delta": entry.get("kpi_vs_market_delta"),
        "pricing_gap_percent": entry.get("pricing_gap_percent"),
        "theme": entry.get("theme"),
        "sample_reviews": entry.get("sample_reviews"),
        "adr_recommendation": entry.get("adr_recommendation"),
        "market_context": entry.get("market_context"),
    }
    sentiment = entry.get("sentiment") or {}
    if sentiment:
        summary["sentiment_label"] = sentiment.get("label")
        summary["sentiment_strength"] = sentiment.get("compound")
        summary["sentiment_hits"] = sentiment.get("hit_count")
    return summary


def _build_distribution_cards(dist: Dict[str, Any], kpi_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    series: List[Dict[str, Any]] = []
    ordering = [
        ("listing_count", "Listings analysed", None, 0),
        ("median", f"Median {kpi_info['label']}", kpi_info.get("unit"), kpi_info.get("decimals", 1)),
        ("avg", f"Average {kpi_info['label']}", kpi_info.get("unit"), kpi_info.get("decimals", 1)),
        ("min", "Portfolio low", kpi_info.get("unit"), kpi_info.get("decimals", 1)),
        ("max", "Portfolio high", kpi_info.get("unit"), kpi_info.get("decimals", 1)),
    ]
    for key, label, unit, decimals in ordering:
        value = dist.get(key)
        if value is None:
            continue
        formatted = (
            _format_metric_value(value, unit=unit, decimals=decimals if decimals is not None else 1)
            if unit
            else str(int(value))
        )
        series.append({"label": label, "value": formatted, "raw_value": value})
    return series


def _build_market_snapshot(market_index: Dict[str, Dict[str, Any]], kpi_info: Dict[str, Any]) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    for entry in market_index.values():
        entries.append(
            {
                "label": entry.get("label"),
                "neighbourhood": entry.get("neighbourhood"),
                "borough": entry.get("borough"),
                "market_kpi_median": entry.get("market_median_kpi"),
                "market_price_median": entry.get("market_median_price_usd"),
                "market_revenue_30_median": entry.get("market_median_revenue_30"),
                "market_review_score_avg": entry.get("market_avg_review_score"),
            }
        )
    entries.sort(key=lambda item: (item.get("market_kpi_median") or 0), reverse=True)
    return {"kpi_label": kpi_info.get("label"), "entries": entries}


def _build_sentiment_summary(sentiment_tracker: Dict[str, Any]) -> Dict[str, Any]:
    totals: Counter = sentiment_tracker.get("totals", Counter())
    listings = sentiment_tracker.get("listings", [])
    listings_sorted = sorted(listings, key=lambda item: item.get("hit_count", 0), reverse=True)
    return {
        "total_positive": totals.get("positive", 0),
        "total_negative": totals.get("negative", 0),
        "total_neutral": totals.get("neutral", 0),
        "listings": listings_sorted,
    }


def _build_action_backlog(tier1: List[Dict[str, Any]], tier2: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    backlog: List[Dict[str, Any]] = []
    for entry in tier1 + tier2:
        backlog.append(
            {
                "listing_id": entry.get("listing_id"),
                "listing_name": entry.get("listing_name"),
                "tier": entry.get("tier"),
                "kpi_value": entry.get("kpi_value"),
                "market_kpi_median": entry.get("market_kpi_median"),
                "delta": entry.get("kpi_vs_market_delta"),
                "pricing_gap_percent": entry.get("pricing_gap_percent"),
                "theme": entry.get("theme"),
                "sentiment": entry.get("sentiment"),
                "sample_reviews": entry.get("sample_reviews"),
                "actions": entry.get("actions"),
                "market_context": entry.get("market_context"),
                "adr_recommendation": entry.get("adr_recommendation"),
                "metrics": entry.get("metrics"),
            }
        )
    return backlog


def _build_playbook(kpi_info: Dict[str, Any], tier1: List[Dict[str, Any]], tier2: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    kpi_label = kpi_info.get("label", "portfolio KPI")
    def _listing_labels(entries: List[Dict[str, Any]]) -> str:
        labels = [entry.get("listing_id") for entry in entries[:3] if entry.get("listing_id")]
        return ", ".join(labels) if labels else "priority listings"

    if tier1:
        plan.append(
            {
                "window": "Week 1",
                "focus": "Stabilize Tier 1 listings",
                "actions": [
                    f"Triage top blockers on {_listing_labels(tier1)} to close the {kpi_label} gap.",
                    "Deploy same-week QA sweeps (cleaning, access, HVAC) and drop ADR into each listing's recommended correction band until reviews recover.",
                ],
            }
        )
    if tier2:
        plan.append(
            {
                "window": "Week 2",
                "focus": "Monetize Tier 2 upside",
                "actions": [
                    f"Use each Tier 2 listing's adr_recommendation band to run tightly controlled ADR experiments on {_listing_labels(tier2)}.",
                    "Bundle refreshed creative (photos, listing copy) to justify higher nightly rates.",
                ],
            }
        )
    plan.append(
        {
            "window": "Week 3–4",
            "focus": "Lock in gains + prep next health check",
            "actions": [
                "Share a readout of fixes + wins, then cascade successful plays to the rest of the portfolio.",
                "Revisit unresolved Tier 1 items and set the next triage cadence (30 days).",
            ],
        }
    )
    return plan


def _triage_summary_text(triage: Dict[str, Any], kpi_info: Dict[str, Any]) -> str:
    scope = triage.get("scope") or "Highbury"
    glance = triage.get("portfolio_at_glance") or {}
    top_len = len(glance.get("top5_overview") or [])
    bottom_len = len(glance.get("bottom5_overview") or [])
    kpi_label = kpi_info.get("label", "portfolio KPI")
    return (
        f"Ranked the {scope} portfolio by {kpi_label}: "
        f"{top_len} upside winners identified and {bottom_len} watchlist listings need action."
    )


def _fetch_distribution_data(table: str, where_clause: str, kpi_column: str) -> Dict[str, Any]:
    where_sql = f" {where_clause}" if where_clause else ""
    sql = f"""
        SELECT
            MIN(listings_id) AS listing_id,
            COUNT(*) AS listing_count,
            AVG({kpi_column}) AS avg_kpi,
            MEDIAN({kpi_column}) AS median_kpi,
            MIN({kpi_column}) AS min_kpi,
            MAX({kpi_column}) AS max_kpi,
            STDDEV_POP({kpi_column}) AS stddev_kpi
        FROM {table}{where_sql}
    """
    rows = _execute_rows(sql)
    return _parse_distribution(rows)


def _fetch_ranked_rows(table: str, where_clause: str, kpi_column: str, direction: str) -> List[Dict[str, Any]]:
    order = "DESC" if direction.lower() == "desc" else "ASC"
    where_sql = f" {where_clause}" if where_clause else ""
    sql = f"""
        SELECT
            listings_id AS listing_id,
            host_name AS listing_name,
            neighbourhood_group,
            neighbourhood,
            price_in_usd,
            estimated_revenue_30,
            estimated_revenue_60,
            estimated_revenue_90,
            estimated_revenue_365,
            review_scores_rating,
            occupancy_rate_30,
            occupancy_rate_60,
            occupancy_rate_90,
            occupancy_rate_365,
            {kpi_column} AS selected_kpi
        FROM {table}{where_sql}
        ORDER BY {kpi_column} {order}
        LIMIT {_MAX_LISTING_CARDS}
    """
    rows = _execute_rows(sql)
    return _normalize_listing_rows(rows, kpi_column)


def _fetch_market_rows(neighbourhoods: List[str], kpi_column: str) -> List[Dict[str, Any]]:
    cleaned = sorted({n.strip().lower() for n in neighbourhoods if n})
    if not cleaned:
        return []
    joined = ", ".join(f"'{_escape_literal(name)}'" for name in cleaned)
    sql = f"""
        SELECT
            MIN(listings_id) AS listing_id,
            neighbourhood_group,
            neighbourhood,
            MEDIAN({kpi_column}) AS market_median_kpi,
            MEDIAN(price_in_usd) AS market_median_price_usd,
            MEDIAN(estimated_revenue_30) AS market_median_revenue_30,
            AVG(review_scores_rating) AS market_avg_review_score
        FROM listings_cleaned
        WHERE lower(neighbourhood) IN ({joined})
        GROUP BY neighbourhood_group, neighbourhood
    """
    return _execute_rows(sql)


def run_portfolio_triage(state: GraphState) -> GraphState:
    """Assemble structured portfolio triage insights using the new KPI-driven pipeline."""
    state.result_bundle = state.result_bundle or {}
    state.telemetry = state.telemetry or {}
    state.extras = state.extras or {}

    kpi_info = _resolve_kpi_choice(state)
    table = _resolve_table(state)
    scope_label, scope_filters = _extract_scope_filters(state.filters)
    where_clause = _build_scope_where_clause(scope_filters)

    triage: Dict[str, Any] = {
        "scope": scope_label,
        "kpi_used": kpi_info["column"],
        "portfolio_at_glance": {
            "kpi_label": kpi_info["label"],
            "kpi_distribution": [],
            "top5_overview": [],
            "bottom5_overview": [],
            "sentiment_summary": {},
            "market_benchmarks": {},
        },
        "action_backlog": [],
        "playbook_30d": [],
    }

    rag_hits: List[Dict[str, Any]] = []
    sentiment_tracker: Dict[str, Any] = {}

    try:
        distribution = _fetch_distribution_data(table, where_clause, kpi_info["column"])
        triage["portfolio_at_glance"]["kpi_distribution"] = _build_distribution_cards(distribution, kpi_info)
        add_thinking_step(
            state,
            phase="portfolio_triage",
            title="Computed KPI distribution",
            detail=f"Computed {kpi_info['label']} stats for scope",
            meta={"rows": distribution.get("listing_count", 0)},
        )

        top_rows = _fetch_ranked_rows(table, where_clause, kpi_info["column"], direction="desc")
        bottom_rows = _fetch_ranked_rows(table, where_clause, kpi_info["column"], direction="asc")
        add_thinking_step(
            state,
            phase="portfolio_triage",
            title="Ranked listings",
            detail=f"Sorted Highbury listings by {kpi_info['label']}",
            meta={"top_rows": len(top_rows), "bottom_rows": len(bottom_rows)},
        )

        neighbourhoods = [row.get("neighbourhood") or "" for row in (top_rows + bottom_rows)]
        market_rows = _fetch_market_rows(neighbourhoods, kpi_info["column"])
        market_index = _index_market_benchmarks(market_rows)
        triage["portfolio_at_glance"]["market_benchmarks"] = _build_market_snapshot(market_index, kpi_info)
        add_thinking_step(
            state,
            phase="portfolio_triage",
            title="Benchmarked neighbourhoods",
            detail="Compared scope listings vs neighbourhood medians",
            meta={"market_areas": len(market_rows)},
        )

        tier1_cards: List[Dict[str, Any]] = []
        tier2_cards: List[Dict[str, Any]] = []
        for row in bottom_rows:
            key = f"{(row.get('borough') or '').strip().lower()}::{(row.get('neighbourhood') or '').strip().lower()}"
            benchmark = market_index.get(key)
            tier1_cards.append(
                _build_listing_insight(
                    row,
                    kpi_info=kpi_info,
                    tier_type="problem",
                    sentiment_label="negative",
                    benchmark=benchmark,
                    base_state=state,
                    sentiment_tracker=sentiment_tracker,
                    rag_hits=rag_hits,
                )
            )
        for row in top_rows:
            key = f"{(row.get('borough') or '').strip().lower()}::{(row.get('neighbourhood') or '').strip().lower()}"
            benchmark = market_index.get(key)
            tier2_cards.append(
                _build_listing_insight(
                    row,
                    kpi_info=kpi_info,
                    tier_type="winner",
                    sentiment_label="positive",
                    benchmark=benchmark,
                    base_state=state,
                    sentiment_tracker=sentiment_tracker,
                    rag_hits=rag_hits,
                )
            )

        triage["portfolio_at_glance"]["top5_overview"] = [_listing_overview(entry) for entry in tier2_cards]
        triage["portfolio_at_glance"]["bottom5_overview"] = [_listing_overview(entry) for entry in tier1_cards]
        triage["portfolio_at_glance"]["sentiment_summary"] = _build_sentiment_summary(sentiment_tracker)
        triage["action_backlog"] = _build_action_backlog(tier1_cards, tier2_cards)
        triage["playbook_30d"] = _build_playbook(kpi_info, tier1_cards, tier2_cards)
        triage_summary = _triage_summary_text(triage, kpi_info)
        state.result_bundle["summary"] = triage_summary
        state.result_bundle["portfolio_triage"] = triage
        state.result_bundle.setdefault("rag_snippets", rag_hits)
        state.extras["portfolio_triage"] = triage
        state.extras["rag_snippets"] = rag_hits
        state.telemetry["triage_sql_rows"] = {
            "distribution": distribution.get("listing_count", 0),
            "ranking_desc": len(top_rows),
            "ranking_asc": len(bottom_rows),
            "market_neighbourhoods": len(market_rows),
            "rag_hits": len(rag_hits),
        }
        add_thinking_step(
            state,
            phase="portfolio_triage",
            title="Built Tier 1 + Tier 2 backlog",
            detail="Attached market + sentiment context to listings",
            meta={"tier1": len(tier1_cards), "tier2": len(tier2_cards), "rag_hits": len(rag_hits)},
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        _LOGGER.error("Portfolio triage failed: %s", exc, exc_info=True)
        triage["error"] = str(exc)
        state.extras["portfolio_triage"] = triage
        state.result_bundle["portfolio_triage"] = triage
        state.telemetry["triage_error"] = str(exc)

    return state


__all__ = ["run_portfolio_triage"]
