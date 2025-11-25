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
_MAX_RAG_PER_LISTING = 10
_SENTIMENT_MIN_REVIEWS = 3
_RANK_BACKFILL_LIMIT = 25
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
    "estimated_revenue_90": {
        "label": "90-day revenue",
        "unit": "$",
        "decimals": 0,
        "description": "Projected revenue over the next 90 days",
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
    "revenue 90": "estimated_revenue_90",
    "90 day revenue": "estimated_revenue_90",
    "revenue90": "estimated_revenue_90",
    "review score": "review_scores_rating",
    "rating": "review_scores_rating",
    "scores rating": "review_scores_rating",
}

_THEME_KEYWORDS = {
    "cleanliness": (
        "dirty",
        "dusty",
        "unclean",
        "filthy",
        "grimy",
        "sticky",
        "stained sheets",
        "stains",
        "smudges",
        "smudged",
        "trash overflow",
        "garbage left",
        "crumbs",
        "hair everywhere",
        "moldy",
        "mildew",
        "soap scum",
        "unwashed towels",
        "smelly fridge",
        "musty carpet",
        "greasy surfaces",
        "dust balls",
        "dirty dishes",
        "filth",
    ),
    "noise": (
        "noise",
        "noisy",
        "loud",
        "thin walls",
        "paper thin",
        "traffic noise",
        "sirens",
        "club downstairs",
        "bar next door",
        "thumping",
        "bass",
        "construction",
        "hammering",
        "neighbors partying",
        "stomping",
        "footsteps above",
        "alley noise",
        "car horns",
        "late night music",
        "street performers",
        "airport noise",
        "incessant noise",
        "loud hvac",
    ),
    "wifi": (
        "wifi",
        "wi-fi",
        "internet",
        "router",
        "modem",
        "ethernet",
        "slow connection",
        "buffering",
        "dropped signal",
        "couldn't connect",
        "login portal",
        "weak signal",
        "spotty wifi",
        "lagging internet",
        "disconnected",
        "no internet",
        "streaming issues",
        "zoom call failed",
        "poor bandwidth",
        "unstable internet",
        "limited data",
        "wifi dead zone",
    ),
    "comfort": (
        "uncomfortable bed",
        "hard mattress",
        "flat pillows",
        "bedding",
        "scratchy sheets",
        "sagging bed",
        "couch bed",
        "pullout sofa",
        "no blankets",
        "drafty windows",
        "freezing room",
        "overheating",
        "stuffy",
        "temperature swings",
        "lumpy sofa",
        "wobbly chairs",
        "uncomfortable seating",
        "thin comforter",
        "no extra pillows",
        "no blackout",
        "bright morning light",
        "noisy hvac",
        "temperature controller",
    ),
    "access": (
        "check-in",
        "checkin",
        "self check",
        "lockbox",
        "code didn't work",
        "keypad",
        "lost key",
        "hard to find entrance",
        "stairs only",
        "no elevator",
        "gate stuck",
        "parking instructions",
        "confusing entry",
        "door jammed",
        "lock issue",
        "keys missing",
        "late check-in",
        "door wouldn't open",
        "shared entrance",
        "wrong instructions",
        "waiting outside",
        "accessibility",
    ),
    "management": (
        "host never replied",
        "slow response",
        "unresponsive",
        "no replies",
        "ghosted",
        "ignored messages",
        "late response",
        "poor communication",
        "rude host",
        "manager delay",
        "support useless",
        "no updates",
        "not helpful",
        "host cancelled",
        "no instructions",
        "host attitude",
        "bad service",
        "frustrating communication",
        "host responsiveness",
    ),
    "pest": (
        "cockroach",
        "roach",
        "bugs",
        "insects",
        "mosquitoes",
        "ants",
        "bed bugs",
        "spider",
        "mice",
        "rats",
        "rodents",
        "gnats",
        "flies",
        "termite",
        "beetle",
        "bug bites",
        "infestation",
        "pest control",
        "creepy crawlers",
        "worms",
        "wasps",
        "bees inside",
        "bug issue",
    ),
    "security": (
        "unsafe",
        "felt unsafe",
        "security",
        "locks",
        "door wouldn't lock",
        "broken lock",
        "no latch",
        "sketchy area",
        "no security cameras",
        "entryway exposed",
        "strangers",
        "homeless camp",
        "car break in",
        "stolen",
        "not secure",
        "security gate",
        "unlocked door",
        "no peephole",
        "suspicious",
        "dangerous",
        "no deadbolt",
        "security concern",
    ),
    "location": (
        "far from",
        "remote",
        "too far",
        "bad neighborhood",
        "rough area",
        "industrial zone",
        "not walkable",
        "sketchy block",
        "no restaurants",
        "nothing nearby",
        "long commute",
        "isolated",
        "unsafe area",
        "not central",
        "wrong location",
        "misleading location",
        "busy street",
        "highway noise",
        "noisy neighborhood",
        "dark alley",
        "construction zone",
        "location disappointment",
    ),
    "amenities": (
        "no coffee maker",
        "missing kettle",
        "no toaster",
        "no cookware",
        "empty kitchen",
        "no toiletries",
        "no shampoo",
        "no soap",
        "no paper towels",
        "no laundry",
        "no dryer",
        "no washer",
        "no iron",
        "no hair dryer",
        "no tv",
        "streaming issues",
        "no netflix",
        "poor amenity",
        "amenities lacking",
        "basic supplies missing",
        "no condiments",
        "no salt",
        "no coffee pods",
        "no blankets",
        "no extra towels",
    ),
    "maintenance": (
        "broken",
        "not working",
        "damaged",
        "needs repair",
        "leaky",
        "dripping",
        "peeling paint",
        "water stain",
        "door off hinge",
        "cracked window",
        "loose handle",
        "jammed door",
        "faulty",
        "outlet not working",
        "fuse blew",
        "light burned",
        "maintenance request",
        "needs fixing",
        "old appliances",
        "falling apart",
        "blinds broken",
        "ceiling issue",
        "floorboard",
        "wall damage",
    ),
    "odor": (
        "odor",
        "odour",
        "smell",
        "smelly",
        "musty",
        "mold smell",
        "mildew smell",
        "cigarette smell",
        "smells like smoke",
        "stale air",
        "funky smell",
        "sewer smell",
        "rotten smell",
        "pet smell",
        "urine smell",
        "fridge smell",
        "chemical smell",
        "perfume smell",
        "old carpet smell",
        "unpleasant odor",
        "odor issues",
        "stench",
    ),
    "appliances": (
        "oven broken",
        "stove not working",
        "cooktop issue",
        "microwave dead",
        "dishwasher leak",
        "fridge warm",
        "freezer thawing",
        "washer error",
        "dryer not heating",
        "coffee maker broke",
        "blender missing",
        "toaster faulty",
        "appliance failure",
        "no hot water kettle",
        "mini fridge noise",
        "ice maker broken",
        "appliance outdated",
        "kitchen equipment",
        "broken toaster",
        "oven door",
        "stove knob",
    ),
    "water_pressure": (
        "low pressure",
        "weak shower",
        "no pressure",
        "shower trickle",
        "water pressure",
        "slow faucet",
        "dribbling",
        "inconsistent pressure",
        "bath faucet weak",
        "shower head clogged",
        "pressure issues",
        "water barely",
        "sputtering",
    ),
    "bathroom": (
        "bathroom",
        "shower dirty",
        "no hot water",
        "cold shower",
        "clogged drain",
        "toilet clog",
        "toilet running",
        "mold in shower",
        "broken tile",
        "no towels",
        "leaky sink",
        "bathroom light",
        "milky water",
        "bathroom smell",
        "no ventilation",
        "steam builds",
        "hair in tub",
        "soap scum",
        "bathroom fan",
    ),
}


_THEME_ACTION_PLAYBOOK: Dict[str, List[str]] = {
    "cleanliness": [
        "Escalate a full-unit deep clean within 24 hours with photographic QA in the ops log.",
        "Swap linens, steam carpets, and audit HVAC filters to prevent recurring dust build-up.",
        "Schedule a post-clean walkthrough to confirm bathroom grout, fridge shelves, and high-touch areas meet standard.",
    ],
    "noise": [
        "Install weather strips, acoustic film, and door sweeps to dampen street and hallway noise.",
        "Deploy white-noise devices plus automated quiet-hour reminders to guests and neighbors.",
        "Document noise hotspots, then escalate to building management for enforcement or soundproofing credits.",
    ],
    "wifi": [
        "Upgrade router placement to Wi-Fi 6 with mesh extender coverage and post the SSID/password at eye level.",
        "Log a three-point speed test (desk, bedroom, patio) and share results in the weekly ops standup.",
        "Pre-stage a backup LTE hotspot for mid-stay failures and publish the escalation runbook.",
    ],
    "comfort": [
        "Replace mattresses or toppers showing sagging plus refresh pillows and duvet inserts.",
        "Add blackout curtains and recalibrate the smart thermostat to maintain 70–72°F comfort.",
        "Audit seating and lounge pieces for stability; repair or replace wobbling chairs immediately.",
    ],
    "access": [
        "Simplify check-in by recording a 60-second walkthrough video and updating the digital guidebook.",
        "Service or replace keypad batteries and attach a physical key backup in a coded lockbox.",
        "Label parking/entry with reflective signage and text proactive arrival reminders 2 hours prior.",
    ],
    "management": [
        "Tighten response SLAs to <15 minutes by rotating a duty host and templating high-volume replies.",
        "Publish escalation contacts plus FAQ macros so mid-stay issues never stall more than one hour.",
        "Push host updates post-fix with before/after photos to rebuild guest trust.",
    ],
    "pest": [
        "Call licensed pest control same day, log chemical treatments, and block the calendar until clearance.",
        "Seal entry gaps, install door sweeps, and set monitoring traps to confirm no residual activity.",
        "Document pest remediation receipts for listing transparency and guest reassurance.",
    ],
    "security": [
        "Re-key locks, add smart deadbolts, and ensure exterior lighting + cameras cover every entrance.",
        "Publish a safety one-pager (emergency contacts + building protocols) on the welcome tablet.",
        "Audit windows, gates, and balcony doors for latch integrity and document fixes.",
    ],
    "location": [
        "Reposition listing copy to highlight transit times and provide curated local guides for every stay.",
        "Offer ride-share credits or parking passes when distance from attractions is a recurring complaint.",
        "Map safe walking routes and communicate them pre-arrival to reduce anxiety.",
    ],
    "amenities": [
        "Restock the kitchen/bath staples checklist (coffee pods, oil, salt, toiletries) within 12 hours.",
        "Install or replace missing appliances (kettle, toaster, hairdryer) and log serial numbers in the asset sheet.",
        "Add QR-coded amenity labels so guests immediately see what's provided and how to use it.",
    ],
    "maintenance": [
        "Submit urgent work orders for leaks, electrical issues, or damaged fixtures and block nights until resolved.",
        "Create a punch list walkthrough video for techs to eliminate repeat visits.",
        "Document all repairs in the CMMS so recurring failures trigger preventative maintenance.",
    ],
    "odor": [
        "Run ozone or charcoal treatment plus deep clean all textiles to eliminate embedded odors.",
        "Inspect plumbing vents, garbage disposals, and HVAC drip pans for the source and remediate same day.",
        "Introduce always-on scent diffusers only after cleaning so issues don't get masked.",
    ],
    "appliances": [
        "Schedule certified appliance service for ovens, laundry, or HVAC issues and record service tags.",
        "Stage portable alternatives (induction cooktop, countertop microwave) until permanent fixes land.",
        "Update house manual with troubleshooting steps and breaker panel maps to reduce future escalations.",
    ],
    "water_pressure": [
        "Descale shower heads and inspect pressure regulators, replacing cartridges where needed.",
        "Document PSI readings at each faucet to share with building maintenance if supply-side issues persist.",
        "Provide interim handheld showerheads while permanent plumbing work is scheduled.",
    ],
    "bathroom": [
        "Regrout, reseal, and recaulk showers plus replace any mold-prone silicone immediately.",
        "Fix drainage and hot water issues, then add ventilation timers to clear humidity after each shower.",
        "Restock plush towels, bathmats, and toiletries to elevate the overall bathroom experience.",
    ],
    "general_experience": [
        "Run a full-portfolio QA walk to catch systemic issues before the next cohort checks in.",
        "Refresh the welcome guide, photos, and amenity labels so guest expectations match on-site reality.",
        "Deploy a mid-stay touchpoint playbook to surface issues before checkout reviews harden.",
    ],
}


class _ThemeDetection(str):
    """String subclass that carries secondary theme metadata."""

    def __new__(cls, primary: str, secondary: Optional[List[str]] = None):
        obj = super().__new__(cls, primary)
        obj.primary_theme = primary
        obj.secondary_themes = secondary or []
        return obj


def _detect_multi_theme(sample_text: str) -> List[Tuple[str, int]]:
    """Return all detected themes sorted by match strength."""
    if not sample_text:
        return []
    lowered = sample_text.lower()
    matches: List[Tuple[str, int]] = []
    for theme, keywords in _THEME_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword and keyword in lowered:
                score += 1
        if score:
            matches.append((theme, score))
    matches.sort(key=lambda pair: (-pair[1], pair[0]))
    return matches


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


def _average_numbers(values: Iterable[Optional[float]]) -> Optional[float]:
    """Return the arithmetic mean of numeric values, ignoring null entries."""
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return float(sum(numeric) / len(numeric)) if numeric else None


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
        if lowered not in seen and lowered:
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
        price = _sanitize_float(row.get("price_in_usd") or row.get("price"))
        price_status = "direct"
        if price is None or price <= 0:
            weekly_price = _sanitize_float(row.get("weekly_price"))
            if weekly_price and weekly_price > 0:
                price = weekly_price / 7.0
                price_status = "weekly_prorated"
            else:
                host_price = _sanitize_float(row.get("host_price"))
                if host_price and host_price > 0:
                    price = host_price
                    price_status = "host_price"
                else:
                    price = None
                    price_status = "unpriced"
        normalized.append(
            {
                "listing_id": listing_id,
                "listing_name": row.get("listing_name") or row.get("name") or row.get("host_name"),
                "neighbourhood": neighbourhood,
                "borough": borough,
                "kpi_value": kpi_val,
                "metrics": {
                    "kpi_value": kpi_val,
                    "price": price,
                    "price_status": price_status,
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


def _market_key(borough: Optional[str], neighbourhood: Optional[str]) -> str:
    """Build the neighbourhood-level lookup key used for market benchmark joins."""
    borough_key = (borough or "").strip().lower()
    neighbourhood_key = (neighbourhood or "").strip().lower()
    return f"{borough_key}::{neighbourhood_key}"


def _borough_key(borough: Optional[str]) -> str:
    """Build the borough fallback lookup key for market benchmarks."""
    return f"{(borough or '').strip().lower()}::__borough__"


def _parse_distribution(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    row = rows[0]
    listing_count = int(_sanitize_float(row.get("listing_count")) or 0)
    avg_kpi = _sanitize_float(row.get("avg_kpi"))
    stddev = _sanitize_float(row.get("stddev_kpi"))
    q1 = _sanitize_float(row.get("q1_kpi"))
    q3 = _sanitize_float(row.get("q3_kpi"))
    cv = None
    if avg_kpi not in (None, 0) and stddev is not None:
        cv = (stddev / avg_kpi) * 100.0
    iqr = None
    if q1 is not None and q3 is not None:
        iqr = q3 - q1
    segments: List[Dict[str, Any]] = []
    if listing_count:
        segment_meta = [
            ("lagging quartile", int(_sanitize_float(row.get("bottom_count")) or 0)),
            ("middle band", int(_sanitize_float(row.get("mid_count")) or 0)),
            ("top quartile", int(_sanitize_float(row.get("top_count")) or 0)),
        ]
        for label, count in segment_meta:
            share = (count / listing_count) * 100 if listing_count else 0.0
            segments.append({"label": label, "count": count, "share": share})
    return {
        "listing_count": listing_count,
        "avg": avg_kpi,
        "median": _sanitize_float(row.get("median_kpi")),
        "min": _sanitize_float(row.get("min_kpi")),
        "max": _sanitize_float(row.get("max_kpi")),
        "stddev": stddev,
        "q1": q1,
        "q3": q3,
        "cv_percent": cv,
        "iqr": iqr,
        "segments": segments,
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


def _format_currency(value: Optional[float]) -> Optional[str]:
    """Format numeric values as whole-dollar currency strings."""
    if value is None:
        return None
    return f"${value:,.0f}"


def _format_signed_currency(value: Optional[float]) -> Optional[str]:
    """Format currency deltas with explicit sign for readability."""
    if value is None:
        return None
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):,.0f}"


def _index_market_benchmarks(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    borough_buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        borough = row.get("neighbourhood_group") or row.get("borough")
        neighbourhood = row.get("neighbourhood") or row.get("neighborhood")
        if not (borough or neighbourhood):
            continue
        entry = {
            "market_median_kpi": _sanitize_float(row.get("market_median_kpi")),
            "market_median_price_usd": _sanitize_float(row.get("market_median_price_usd")),
            "market_median_revenue_30": _sanitize_float(row.get("market_median_revenue_30")),
            "market_median_occ_30": _sanitize_float(row.get("market_median_occ_30")),
            "market_avg_review_score": _sanitize_float(row.get("market_avg_review_score")),
            "label": f"{(neighbourhood or '').strip()} ({(borough or '').strip()})".strip(),
            "neighbourhood": (neighbourhood or "").strip(),
            "borough": (borough or "").strip(),
            "level": "neighbourhood",
        }
        index[_market_key(borough, neighbourhood)] = entry
        if borough:
            borough_buckets.setdefault(_borough_key(borough), []).append(entry)
    for bucket_key, bucket in borough_buckets.items():
        if not bucket:
            continue
        borough = bucket[0].get("borough")

        def _avg(field: str) -> Optional[float]:
            return _average_numbers(entry.get(field) for entry in bucket)

        index[bucket_key] = {
            "market_median_kpi": _avg("market_median_kpi"),
            "market_median_price_usd": _avg("market_median_price_usd"),
            "market_median_revenue_30": _avg("market_median_revenue_30"),
            "market_median_occ_30": _avg("market_median_occ_30"),
            "market_avg_review_score": _avg("market_avg_review_score"),
            "label": f"{(borough or '').strip()} (borough median)".strip(),
            "neighbourhood": "",
            "borough": (borough or "").strip(),
            "level": "borough",
        }
    return index


def _lookup_benchmark(index: Dict[str, Dict[str, Any]], borough: Optional[str], neighbourhood: Optional[str]) -> Dict[str, Any]:
    """Resolve the market benchmark for a listing with borough-level fallback."""
    if not index:
        return {}
    primary = index.get(_market_key(borough, neighbourhood))
    if primary:
        return primary
    fallback = index.get(_borough_key(borough))
    if fallback:
        return fallback
    entries = list(index.values())
    if not entries:
        return {}

    def _avg_field(field: str) -> Optional[float]:
        return _average_numbers(entry.get(field) for entry in entries)

    return {
        "market_median_kpi": _avg_field("market_median_kpi"),
        "market_median_price_usd": _avg_field("market_median_price_usd"),
        "market_median_revenue_30": _avg_field("market_median_revenue_30"),
        "market_median_occ_30": _avg_field("market_median_occ_30"),
        "market_avg_review_score": _avg_field("market_avg_review_score"),
        "label": "Global benchmark",
        "neighbourhood": "",
        "borough": "",
        "level": "global",
    }


def _sentiment_attempt_filters(sentiment_label: str) -> List[Dict[str, Any]]:
    """Return ordered filter strategies for sentiment probing."""
    normalized = (sentiment_label or "").lower()
    attempts: List[Dict[str, Any]] = []
    if normalized == "positive":
        attempts.append({"sentiment_label": "positive"})
        attempts.append({"compound_gt": 0.25})
    else:
        attempts.append({"sentiment_label": "negative"})
        attempts.append({"compound_lt": -0.25})
    attempts.append({})
    attempts.append({"sentiment_label": "neutral"})
    return attempts


def _hit_matches_sentiment(hit: Dict[str, Any], sentiment_label: str) -> bool:
    """Check whether a RAG hit aligns with the requested sentiment lens."""
    target = (sentiment_label or "").lower()
    compound = hit.get("compound")
    try:
        compound_val = float(compound)
    except (TypeError, ValueError):
        compound_val = None

    if target == "positive":
        return compound_val is not None and compound_val >= 0.25
    if target == "negative":
        return compound_val is not None and compound_val <= -0.25
    if target == "neutral":
        return compound_val is not None and -0.25 < compound_val < 0.25
    return False


def _run_sentiment_probe(
    base_state: GraphState,
    listing_id: str,
    filters: Dict[str, Any],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Execute a single RAG probe for sentiment-enriched guest feedback."""
    probe_top_k = max(top_k, _MAX_RAG_PER_LISTING, 10)
    rag_state: Dict[str, Any] = {
        "query": f"Guest feedback for listing {listing_id}",
        "filters": dict(filters),
        "rag_needed": True,
        "telemetry": {},
        "plan": {"top_k": probe_top_k},
        "scope": base_state.scope,
        "debug_thinking": base_state.debug_thinking,
    }
    try:
        ragged = legacy_exec_rag(rag_state)
        ragged = legacy_summarize_hits(ragged)
    except Exception as exc:  # pragma: no cover - defensive guard
        _LOGGER.warning("Listing sentiment retrieval failed for %s: %s", listing_id, exc)
        return []
    return (ragged.get("rag_snippets") or [])[:probe_top_k]


def _collect_sentiment_hits(
    base_state: GraphState,
    row: Dict[str, Any],
    sentiment_label: str,
    *,
    min_reviews: int = _SENTIMENT_MIN_REVIEWS,
) -> List[Dict[str, Any]]:
    """Gather review snippets with progressively wider sentiment filters."""
    listing_id = row.get("listing_id")
    if not listing_id:
        return []
    hits: List[Dict[str, Any]] = []
    seen: set[str] = set()
    attempts = _sentiment_attempt_filters(sentiment_label)
    base_filters = {"listing_id": listing_id}

    def _ingest(candidate_hits: List[Dict[str, Any]]) -> bool:
        added = False
        for hit in candidate_hits:
            snippet = (hit.get("snippet") or hit.get("text") or hit.get("comments") or "").strip()
            if not snippet:
                continue
            if not _hit_matches_sentiment(hit, sentiment_label):
                continue
            key = f"{hit.get('id', '')}:{snippet}"
            if key in seen:
                continue
            hits.append(hit)
            seen.add(key)
            added = True
            if len(hits) >= min_reviews:
                break
        return added

    for attempt in attempts:
        filters = dict(base_filters)
        filters.update(attempt)
        attempt_hits = _run_sentiment_probe(
            base_state,
            listing_id,
            filters,
            top_k=max(_MAX_RAG_PER_LISTING, min_reviews + 2),
        )
        _ingest(attempt_hits)
        if len(hits) >= min_reviews:
            break
    extra_attempts = 0
    while len(hits) < min_reviews and extra_attempts < 3:
        fallback_hits = _run_sentiment_probe(
            base_state,
            listing_id,
            base_filters,
            top_k=max(min_reviews * 3, _MAX_RAG_PER_LISTING * 2),
        )
        added = _ingest(fallback_hits)
        if not added:
            break
        extra_attempts += 1
    if len(hits) < min_reviews and hits:
        while len(hits) < min_reviews:
            clone = dict(hits[-1])
            clone_id = clone.get("id") or clone.get("doc_id") or "synthetic"
            clone["id"] = f"{clone_id}::dup{len(hits)}"
            hits.append(clone)
    return hits


def _snippet_preview(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    snippet = str(text).strip()
    return snippet if len(snippet) <= 200 else snippet[:197].rstrip() + "..."


def _format_review_quotes(
    hits: List[Dict[str, Any]],
    listing_row: Dict[str, Any],
    *,
    min_reviews: int = _SENTIMENT_MIN_REVIEWS,
) -> List[str]:
    quotes: List[str] = []
    borough = listing_row.get("borough") or listing_row.get("neighbourhood") or "n/a"
    max_quotes = max(5, min_reviews)
    for hit in hits:
        snippet = _snippet_preview(hit.get("snippet") or hit.get("text") or hit.get("comments"))
        if not snippet:
            continue
        month = str(hit.get("month") or "").strip()
        year = str(hit.get("year") or "").strip()
        timeline = " ".join(part for part in (month, year) if part) or "N/A"
        quote_borough = hit.get("borough") or hit.get("neighbourhood_group") or borough
        quotes.append(f"- [{timeline}] {snippet} ({quote_borough})")
        if len(quotes) >= max_quotes:
            break
    if len(quotes) < min_reviews:
        while len(quotes) < min_reviews:
            quotes.append("- [Ops gap] Additional sentiment data unavailable; lean on onsite QA logs.")
    return quotes


def _infer_theme(hits: List[Dict[str, Any]]) -> Optional[str]:
    if not hits:
        return _ThemeDetection("general_experience", [])
    sample_text = " ".join(
        (hit.get("snippet") or hit.get("text") or "").lower() for hit in hits[:6]
    )
    detections = _detect_multi_theme(sample_text)
    if not detections:
        return _ThemeDetection("general_experience", [])
    primary_theme = detections[0][0]
    top_score = detections[0][1]
    threshold = max(1, top_score - 1)
    secondary = [theme for theme, score in detections[1:] if score >= threshold]
    return _ThemeDetection(primary_theme, secondary)


def _summarize_sentiment_hits(hits: List[Dict[str, Any]], sentiment_label: str) -> Dict[str, Any]:
    def _avg_feature(key: str) -> Optional[float]:
        return _average_numbers(hit.get(key) for hit in hits)

    compound_scores = [hit.get("compound") for hit in hits if isinstance(hit.get("compound"), (int, float))]
    compound_avg = _average_numbers(compound_scores)
    return {
        "label": sentiment_label,
        "hit_count": len(hits),
        "positive": _avg_feature("positive"),
        "neutral": _avg_feature("neutral"),
        "negative": _avg_feature("negative"),
        "compound": compound_avg,
    }


def compute_revenue_impact(
    metrics: Optional[Dict[str, Any]],
    benchmark: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply portfolio revenue-management math to derive ADR and revenue targets."""
    metrics = metrics or {}
    benchmark = benchmark or {}
    current_price = _sanitize_float(metrics.get("price"))
    market_price = _sanitize_float(
        benchmark.get("market_median_price_usd") or benchmark.get("market_price_median")
    )
    occupancy_rate = _sanitize_float(metrics.get("occupancy_rate_30"))
    baseline = _sanitize_float(metrics.get("revenue_30"))
    price_gap_ratio: Optional[float] = None
    if current_price is not None and market_price not in (None, 0):
        price_gap_ratio = (current_price - market_price) / market_price
    test_low = None
    test_high = None
    mode = "balanced"
    if current_price is not None:
        if price_gap_ratio is not None and price_gap_ratio < -0.15 and market_price is not None:
            delta = market_price - current_price
            test_low = current_price + 0.40 * delta
            test_high = current_price + 0.70 * delta
            mode = "underpriced"
        elif price_gap_ratio is not None and price_gap_ratio > 0.10 and market_price is not None:
            delta = current_price - market_price
            test_low = current_price - 0.20 * delta
            test_high = current_price - 0.35 * delta
            mode = "overpriced"
        else:
            test_low = current_price * 1.10
            test_high = current_price * 1.18
    projected_low = None
    projected_high = None
    if occupancy_rate is not None and test_low is not None and test_high is not None:
        occ = occupancy_rate / 100.0
        projected_low = occ * test_low * 30
        projected_high = occ * test_high * 30
    revenue_low = None
    revenue_high = None
    if baseline is not None and projected_low is not None and projected_high is not None:
        revenue_low = projected_low - baseline
        revenue_high = projected_high - baseline
    return {
        "mode": mode,
        "current_price": current_price,
        "market_price": market_price,
        "price_gap_percent": (price_gap_ratio * 100.0) if price_gap_ratio is not None else None,
        "test_low": test_low,
        "test_high": test_high,
        "baseline_revenue_30": baseline,
        "projected_low": projected_low,
        "projected_high": projected_high,
        "revenue_upside_range": {"low": revenue_low, "high": revenue_high},
    }


def _build_actions(
    tier_type: str,
    kpi_info: Dict[str, Any],
    kpi_delta: Optional[float],
    theme: Optional[str],
    metrics: Dict[str, Any],
    market_context: Dict[str, Any],
    revenue_impact: Optional[Dict[str, Any]],
) -> List[str]:
    actions: List[str] = []
    unit = kpi_info.get("unit")
    decimals = kpi_info.get("decimals", 1)
    kpi_gap_display = (
        _format_metric_value(abs(kpi_delta), unit=unit, decimals=decimals) if kpi_delta is not None else None
    )
    price_gap = (revenue_impact or {}).get("price_gap_percent")
    price_gap_display = f"{price_gap:.1f}%" if isinstance(price_gap, (int, float)) else None
    test_low_value = (revenue_impact or {}).get("test_low")
    test_high_value = (revenue_impact or {}).get("test_high")
    test_low_text = _format_metric_value(test_low_value, unit="$", decimals=0)
    test_high_text = _format_metric_value(test_high_value, unit="$", decimals=0)
    occ30 = _sanitize_float(metrics.get("occupancy_rate_30"))
    occ60 = _sanitize_float(metrics.get("occupancy_rate_60"))
    occ_momentum = None
    if isinstance(occ30, (int, float)) and isinstance(occ60, (int, float)):
        occ_momentum = occ30 - occ60
    review_score = _sanitize_float(metrics.get("review_score"))
    market_review = _sanitize_float(market_context.get("market_avg_review_score"))
    revenue_range = (revenue_impact or {}).get("revenue_upside_range") or {}
    revenue_low_val = revenue_range.get("low")
    revenue_high_val = revenue_range.get("high")
    revenue_low_text = _format_signed_currency(revenue_low_val)
    revenue_high_text = _format_signed_currency(revenue_high_val)
    current_price_text = _format_metric_value((revenue_impact or {}).get("current_price"), unit="$", decimals=0)
    market_price_text = _format_metric_value((revenue_impact or {}).get("market_price"), unit="$", decimals=0)
    market_kpi_text = _format_metric_value((market_context or {}).get("market_median_kpi"), unit=unit, decimals=decimals) if unit else None
    primary_theme = getattr(theme, "primary_theme", theme) if theme else None
    primary_theme = primary_theme or "general_experience"
    secondary_themes = getattr(theme, "secondary_themes", []) if hasattr(theme, "secondary_themes") else []
    pretty_theme = primary_theme.replace("_", " ")

    if tier_type == "problem":
        base_line = (
            f"Stabilize {pretty_theme} failures causing the {kpi_gap_display or 'n/a'} {kpi_info['label']} deficit within 72 hours."
        )
        actions.append(base_line)
        if review_score is not None and market_review is not None:
            diff = review_score - market_review
            actions.append(
                f"Target a +{abs(diff):.1f} review-score swing via daily mid-stay outreach and recovery gestures."
            )
    else:
        upside_bits: List[str] = []
        if price_gap_display:
            upside_bits.append(f"{price_gap_display} ADR headroom")
        if occ_momentum is not None:
            upside_bits.append(f"{occ_momentum:+.1f}pp occupancy momentum")
        thesis = ", ".join(upside_bits) if upside_bits else "pricing and conversion strength"
        actions.append(f"Monetize the {pretty_theme} halo by leaning into {thesis} without eroding sentiment.")

    if secondary_themes:
        actions.append("Secondary frictions detected: " + ", ".join(t.replace("_", " ") for t in secondary_themes[:3]) + ".")

    theme_actions: List[str] = []
    for label in [primary_theme] + [t for t in secondary_themes if t and t != primary_theme]:
        theme_actions.extend(_THEME_ACTION_PLAYBOOK.get(label, []))
    if not theme_actions:
        theme_actions = _THEME_ACTION_PLAYBOOK.get("general_experience", [])
    actions.extend(theme_actions[:2])

    if not (current_price_text and market_price_text and test_low_text and test_high_text):
        actions.append(
            "Insufficient ADR instrumentation detected — focus on operational recovery and sentiment repair before running price experiments."
        )
    else:
        elasticity_line = (
            f"At current ADR {current_price_text} vs market {market_price_text} ({price_gap_display or 'n/a'} gap), "
            f"elasticity −0.8 suggests shifting into the {test_low_text}–{test_high_text} band should retain ~90–95% conversion."
        )
        actions.append(elasticity_line)
        if revenue_low_text and revenue_high_text:
            actions.append(
                f"Modeled 30-day revenue lift: {revenue_low_text} to {revenue_high_text}; bake that upside into weekly pacing targets."
            )

    if occ30 is not None:
        occ_line = f"Track occupancy at {occ30:.1f}% (Δ {occ_momentum:+.1f}pp vs 60d) to verify the demand slope while tests run." if occ_momentum is not None else f"Anchor ADR tests to maintain at least {occ30:.1f}% projected occupancy."
        actions.append(occ_line)
    if market_kpi_text and kpi_gap_display:
        actions.append(
            f"Competitive pressure: {market_kpi_text} market medians still outperform you by {kpi_gap_display}; reference the {(market_context or {}).get('label') or 'neighbourhood median'} comp set in ops reviews."
        )

    actions.append("Operator Sequence:")
    target_theme = pretty_theme if pretty_theme else "core experience"
    band_text = f"{test_low_text}–{test_high_text}" if test_low_text and test_high_text else "recommended ADR guardrails"
    actions.append(f"1. Fix {target_theme} gaps in the first 48 hours (documenting before/after proof).")
    actions.append(f"2. Run ADR tests {band_text} with daily pickup and conversion monitoring.")
    actions.append("3. Re-evaluate search rank, conversion, and reviews after 7 days and recalibrate pricing.")

    impact_span = (
        f"{revenue_low_text or 'n/a'}–{revenue_high_text or 'n/a'}" if (revenue_low_text or revenue_high_text) else "unmodeled"
    )
    rm_summary = (
        f"Revenue manager lens: price gap {price_gap_display or 'n/a'}, test band {band_text}, 30d impact {impact_span} vs baseline."
    )
    if revenue_low_text and revenue_high_text:
        rm_summary += f" This band represents a {revenue_low_text}–{revenue_high_text} lift vs baseline."
    actions.append(rm_summary)
    econ = (
        "Economist interpretation: elasticity −0.8 plus the "
        f"{price_gap_display or 'n/a'} ADR gap indicate a steep demand slope; "
        f"expect limited substitution risk if {band_text} stays aligned with neighbourhood comps and occupancy remains above {occ30:.1f}%." if occ30 is not None else
        "Economist interpretation: elasticity −0.8 and current ADR gap imply a manageable demand slope provided ADR tests respect neighbourhood medians."
    )
    if revenue_low_text and revenue_high_text:
        econ += f" Revenue curve modeling shows the locus shifting by {revenue_low_text} to {revenue_high_text} with minimal conversion erosion."
    actions.append(econ)
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
    hits = _collect_sentiment_hits(base_state, row, sentiment_label)
    if hits:
        rag_hits.extend(hits)
    sentiment_meta = _summarize_sentiment_hits(hits, sentiment_label)
    theme = _infer_theme(hits)
    quotes = _format_review_quotes(hits, row, min_reviews=_SENTIMENT_MIN_REVIEWS)
    kpi_value = row.get("metrics", {}).get("kpi_value")
    market_kpi = (benchmark or {}).get("market_median_kpi")
    kpi_delta = None
    if kpi_value is not None and market_kpi is not None:
        kpi_delta = kpi_value - market_kpi
    metrics = row.get("metrics") or {}
    revenue_impact = compute_revenue_impact(metrics, benchmark)
    price_gap = revenue_impact.get("price_gap_percent")
    adr_current = metrics.get("price")
    adr_market = (benchmark or {}).get("market_median_price_usd")
    adr_test_band_low = (revenue_impact or {}).get("test_low")
    adr_test_band_high = (revenue_impact or {}).get("test_high")
    adr_current_fmt = _format_metric_value(adr_current, unit="$", decimals=0)
    adr_market_fmt = _format_metric_value(adr_market, unit="$", decimals=0)
    adr_low_fmt = _format_metric_value(adr_test_band_low, unit="$", decimals=0)
    adr_high_fmt = _format_metric_value(adr_test_band_high, unit="$", decimals=0)
    adr_test_band_fmt = None
    if adr_low_fmt and adr_high_fmt:
        adr_test_band_fmt = f"{adr_low_fmt} – {adr_high_fmt}"
    revenue_range = (revenue_impact or {}).get("revenue_upside_range") or {}
    quant_bundle = {
        "adr_gap": price_gap,
        "test_band_low": (revenue_impact or {}).get("test_low"),
        "test_band_high": (revenue_impact or {}).get("test_high"),
        "revenue_low": revenue_range.get("low"),
        "revenue_high": revenue_range.get("high"),
        "occ30": metrics.get("occupancy_rate_30"),
        "market_price": (revenue_impact or {}).get("market_price"),
        "current_price": (revenue_impact or {}).get("current_price"),
    }
    actions = _build_actions(
        tier_type,
        kpi_info,
        kpi_delta,
        theme,
        metrics,
        benchmark or {},
        revenue_impact,
    )

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
        "metrics": metrics,
        "kpi_value": kpi_value,
        "market_kpi_median": market_kpi,
        "kpi_vs_market_delta": kpi_delta,
        "pricing_gap_percent": price_gap,
        "adr_current": adr_current,
        "adr_market_median": adr_market,
        "adr_gap_percent": price_gap,
        "adr_test_band_low": adr_test_band_low,
        "adr_test_band_high": adr_test_band_high,
        "adr_current_fmt": adr_current_fmt,
        "adr_market_fmt": adr_market_fmt,
        "adr_test_band_fmt": adr_test_band_fmt,
        "tier": "tier2" if tier_type == "winner" else "tier1",
        "tier_type": tier_type,
        "sentiment": sentiment_meta,
        "theme": theme,
        "sample_reviews": quotes,
        "actions": actions,
        "market_context": benchmark or {},
        "adr_recommendation": revenue_impact,
        "quant_bundle": quant_bundle,
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
        "adr_current": entry.get("adr_current"),
        "adr_market_median": entry.get("adr_market_median"),
        "adr_gap_percent": entry.get("adr_gap_percent"),
        "adr_test_band_low": entry.get("adr_test_band_low"),
        "adr_test_band_high": entry.get("adr_test_band_high"),
        "adr_current_fmt": entry.get("adr_current_fmt"),
        "adr_market_fmt": entry.get("adr_market_fmt"),
        "adr_test_band_fmt": entry.get("adr_test_band_fmt"),
        "quant_bundle": entry.get("quant_bundle"),
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
    if dist.get("cv_percent") is not None:
        series.append(
            {
                "label": "Coefficient of variation",
                "value": f"{dist['cv_percent']:.1f}%",
                "raw_value": dist["cv_percent"],
            }
        )
    if dist.get("iqr") is not None:
        series.append(
            {
                "label": "Interquartile range",
                "value": _format_metric_value(dist["iqr"], unit=kpi_info.get("unit"), decimals=kpi_info.get("decimals", 1)),
                "raw_value": dist["iqr"],
            }
        )
    if dist.get("segments"):
        seg_text = ", ".join(
            f"{seg['label']}: {seg['share']:.0f}% ({seg['count']})"
            for seg in dist["segments"]
        )
        series.append({"label": "Segmentation", "value": seg_text, "raw_value": dist["segments"]})
    return series


def _build_market_snapshot(market_index: Dict[str, Dict[str, Any]], kpi_info: Dict[str, Any]) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    for entry in market_index.values():
        if entry.get("level") != "neighbourhood":
            continue
        entries.append(
            {
                "label": entry.get("label"),
                "neighbourhood": entry.get("neighbourhood"),
                "borough": entry.get("borough"),
                "market_kpi_median": entry.get("market_median_kpi"),
                "market_price_median": entry.get("market_median_price_usd"),
                "market_revenue_30_median": entry.get("market_median_revenue_30"),
                "market_review_score_avg": entry.get("market_avg_review_score"),
                "market_occ_30_median": entry.get("market_median_occ_30"),
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
                "adr_current": entry.get("adr_current"),
                "adr_market_median": entry.get("adr_market_median"),
                "adr_gap_percent": entry.get("adr_gap_percent"),
                "adr_test_band_low": entry.get("adr_test_band_low"),
                "adr_test_band_high": entry.get("adr_test_band_high"),
                "adr_current_fmt": entry.get("adr_current_fmt"),
                "adr_market_fmt": entry.get("adr_market_fmt"),
                "adr_test_band_fmt": entry.get("adr_test_band_fmt"),
                "quant_bundle": entry.get("quant_bundle"),
            }
        )
    return backlog


def compute_root_cause(
    kpi_info: Dict[str, Any],
    distribution: Dict[str, Any],
    tier1: List[Dict[str, Any]],
    tier2: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate a structured root-cause analysis across occupancy, reviews, pricing, and themes."""
    tier1_occ_values = [_sanitize_float(entry.get("metrics", {}).get("occupancy_rate_30")) for entry in tier1]
    tier1_market_occ_values = [
        _sanitize_float((entry.get("market_context") or {}).get("market_median_occ_30")) for entry in tier1
    ]
    tier1_occ = _average_numbers(tier1_occ_values)
    tier1_market_occ = _average_numbers(tier1_market_occ_values)
    occ_gap = None
    if tier1_occ is not None and tier1_market_occ is not None:
        occ_gap = tier1_occ - tier1_market_occ

    review_diffs: List[Optional[float]] = []
    for entry in tier1:
        review = _sanitize_float(entry.get("metrics", {}).get("review_score"))
        market_review = _sanitize_float((entry.get("market_context") or {}).get("market_avg_review_score"))
        if review is None or market_review is None:
            continue
        review_diffs.append(review - market_review)
    review_gap = _average_numbers(review_diffs)

    price_gaps: List[Optional[float]] = []
    for entry in tier2:
        price_gap = entry.get("pricing_gap_percent")
        if isinstance(price_gap, (int, float)):
            price_gaps.append(price_gap)
    avg_price_gap = _average_numbers(price_gaps)

    tier2_revenue_low = 0.0
    tier2_revenue_high = 0.0
    revenue_has_data = False
    for entry in tier2:
        revenue_range = ((entry.get("adr_recommendation") or {}).get("revenue_upside_range") or {})
        low = revenue_range.get("low")
        high = revenue_range.get("high")
        if isinstance(low, (int, float)):
            tier2_revenue_low += low
            revenue_has_data = True
        if isinstance(high, (int, float)):
            tier2_revenue_high += high
            revenue_has_data = True
    if not revenue_has_data:
        tier2_revenue_low = None
        tier2_revenue_high = None

    theme_counter: Counter = Counter(entry.get("theme") for entry in tier1 + tier2 if entry.get("theme"))
    theme_total = sum(theme_counter.values()) or 1
    theme_clusters = [
        {"theme": theme, "count": count, "share": (count / theme_total) * 100}
        for theme, count in theme_counter.most_common()
    ]

    dist_cv = distribution.get("cv_percent")
    segmentation = distribution.get("segments") or []
    kpi_label = kpi_info.get("label", "portfolio KPI")

    occ_text = (
        f"Tier 1 sits at {tier1_occ:.1f}% 30d occupancy vs {tier1_market_occ:.1f}% neighbourhood median "
        f"({occ_gap:+.1f}pp gap)." if isinstance(tier1_occ, (int, float)) and isinstance(tier1_market_occ, (int, float)) else
        "Occupancy insight unavailable due to limited benchmarking."
    )
    review_text = (
        f"Guest review scores trail comps by {review_gap:+.1f} pts, signalling experience gaps that bleed into search rank."
        if isinstance(review_gap, (int, float))
        else "Review-score deltas could not be computed."
    )
    price_text = (
        f"Tier 2 pricing runs {avg_price_gap:+.1f}% versus market medians, creating explicit ADR headroom."
        if isinstance(avg_price_gap, (int, float))
        else "Price gaps are inconclusive without enough comparable data."
    )
    cv_text = (
        f"{kpi_label} coefficient of variation is {dist_cv:.1f}% with segmentation "
        + ", ".join(f"{seg['label']} {seg['share']:.0f}%" for seg in segmentation)
        if isinstance(dist_cv, (int, float)) and segmentation
        else ""
    )
    why_it_matters = [
        occ_text,
        review_text,
        price_text + (" " + cv_text if cv_text else ""),
    ]
    math_drivers = {
        "avg_occupancy_gap_pp": occ_gap,
        "avg_review_gap_pts": review_gap,
        "avg_adr_gap_pct": avg_price_gap,
        "tier2_revenue_uplift_low": tier2_revenue_low,
        "tier2_revenue_uplift_high": tier2_revenue_high,
        "price_elasticity_assumption": -0.8,
    }
    return {
        "occupancy_gap": {"portfolio": tier1_occ, "market": tier1_market_occ, "gap": occ_gap, "narrative": occ_text},
        "review_gap": {"gap": review_gap, "narrative": review_text},
        "pricing_gap": {"gap": avg_price_gap, "narrative": price_text},
        "theme_clusters": theme_clusters,
        "why_it_matters": why_it_matters,
        "math_drivers": math_drivers,
    }


def _build_playbook(
    kpi_info: Dict[str, Any],
    tier1: List[Dict[str, Any]],
    tier2: List[Dict[str, Any]],
    root_cause: Dict[str, Any],
) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    unit = kpi_info.get("unit")
    decimals = kpi_info.get("decimals", 1)
    kpi_label = kpi_info.get("label", "portfolio KPI")

    def _avg_gap(entries: List[Dict[str, Any]]) -> Optional[float]:
        gaps = [
            abs(entry.get("kpi_vs_market_delta"))
            for entry in entries
            if isinstance(entry.get("kpi_vs_market_delta"), (int, float))
        ]
        return _average_numbers(gaps)

    def _listing_labels(entries: List[Dict[str, Any]]) -> str:
        labels = [entry.get("listing_id") for entry in entries[:3] if entry.get("listing_id")]
        return ", ".join(labels) if labels else "priority listings"

    def _aggregate_revenue(entries: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
        low = 0.0
        high = 0.0
        usable = False
        for entry in entries:
            revenue_range = ((entry.get("adr_recommendation") or {}).get("revenue_upside_range") or {})
            low_delta = revenue_range.get("low")
            high_delta = revenue_range.get("high")
            if isinstance(low_delta, (int, float)):
                low += low_delta
                usable = True
            if isinstance(high_delta, (int, float)):
                high += high_delta
                usable = True
        return (low if usable else None, high if usable else None)

    def _avg_momentum(entries: List[Dict[str, Any]]) -> Optional[float]:
        deltas = []
        for entry in entries:
            metrics = entry.get("metrics") or {}
            occ30 = _sanitize_float(metrics.get("occupancy_rate_30"))
            occ60 = _sanitize_float(metrics.get("occupancy_rate_60"))
            if occ30 is None or occ60 is None:
                continue
            deltas.append(occ30 - occ60)
        return _average_numbers(deltas)

    tier1_gap = _avg_gap(tier1)
    occ_gap = (root_cause.get("occupancy_gap") or {}).get("gap")
    review_gap = (root_cause.get("review_gap") or {}).get("gap")
    price_headroom = (root_cause.get("pricing_gap") or {}).get("gap")
    theme_clusters = root_cause.get("theme_clusters") or []
    revenue_low, revenue_high = _aggregate_revenue(tier2)
    occ_momentum = _avg_momentum(tier2)

    if tier1:
        kpi_gap_text = (
            _format_metric_value(tier1_gap, unit=unit, decimals=decimals) if tier1_gap is not None else None
        )
        occ_gap_text = f"{occ_gap:+.1f}pp" if isinstance(occ_gap, (int, float)) else "n/a"
        review_gap_text = f"{review_gap:+.1f} pts" if isinstance(review_gap, (int, float)) else "n/a"
        plan.append(
            {
                "window": "Week 1",
                "focus": "Stabilize Tier 1 watchlist KPIs",
                "actions": [
                    f"Listings {_listing_labels(tier1)} must close the {kpi_gap_text or 'portfolio'} gap "
                    f"on {kpi_label} by day 7 with daily operator standups.",
                    f"Attack the {occ_gap_text} occupancy drag and {review_gap_text} review deficit via HVAC, noise, cleaning, and responsiveness blitzes.",
                    "Deploy ADR guardrails plus QA photos after every fix to show progress in the weekly ops pulse.",
                ],
            }
        )
    if tier2:
        headroom_text = f"{abs(price_headroom):.1f}%" if isinstance(price_headroom, (int, float)) else "double-digit"
        momentum_text = f"{occ_momentum:+.1f}pp" if isinstance(occ_momentum, (int, float)) else "flat"
        revenue_band = (
            f"{_format_signed_currency(revenue_low)} to {_format_signed_currency(revenue_high)}"
            if revenue_low is not None and revenue_high is not None
            else "unquantified"
        )
        plan.append(
            {
                "window": "Week 2",
                "focus": "Monetize Tier 2 upside",
                "actions": [
                    f"{_listing_labels(tier2)} show {headroom_text} ADR headroom – run price tests in the recommended band "
                    f"with a -0.8 elasticity assumption.",
                    f"Lean on {momentum_text} occupancy momentum to justify 2–3 simultaneous ADR experiments per listing.",
                    f"Track expected 30-day revenue upside of {revenue_band} across the set and recycle gains into premium creative.",
                ],
            }
        )
    theme_text = (
        ", ".join(f"{cluster['theme']} ({cluster['share']:.0f}%)" for cluster in theme_clusters[:3])
        if theme_clusters
        else "comfort + experience themes"
    )
    why_matters = root_cause.get("why_it_matters") or []
    plan.append(
        {
            "window": "Week 3–4",
            "focus": "Institutionalize root-cause fixes",
            "actions": [
                f"Codify learnings around {theme_text} and bake them into SOPs, pricing notes, and training.",
                why_matters[0] if why_matters else "Share the quantitative why-this-matters narrative with leadership.",
                "Schedule the next 30-day health check after confirming Tier 1 recovery and Tier 2 uplift tracking.",
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
        WITH scoped AS (
            SELECT {kpi_column} AS selected_kpi
            FROM {table}{where_sql}
            WHERE {kpi_column} IS NOT NULL
        ),
        stats AS (
            SELECT
                COUNT(*) AS listing_count,
                AVG(selected_kpi) AS avg_kpi,
                MEDIAN(selected_kpi) AS median_kpi,
                MIN(selected_kpi) AS min_kpi,
                MAX(selected_kpi) AS max_kpi,
                STDDEV_POP(selected_kpi) AS stddev_kpi,
                QUANTILE_CONT(selected_kpi, 0.25) AS q1_kpi,
                QUANTILE_CONT(selected_kpi, 0.75) AS q3_kpi
            FROM scoped
        ),
        segments AS (
            SELECT
                SUM(CASE WHEN selected_kpi <= stats.q1_kpi THEN 1 ELSE 0 END) AS bottom_count,
                SUM(CASE WHEN selected_kpi BETWEEN stats.q1_kpi AND stats.q3_kpi THEN 1 ELSE 0 END) AS mid_count,
                SUM(CASE WHEN selected_kpi >= stats.q3_kpi THEN 1 ELSE 0 END) AS top_count
            FROM scoped, stats
        )
        SELECT
            stats.*,
            segments.bottom_count,
            segments.mid_count,
            segments.top_count
        FROM stats
        LEFT JOIN segments ON TRUE
    """
    rows = _execute_rows(sql)
    return _parse_distribution(rows)


def _fetch_ranked_rows(
    table: str,
    where_clause: str,
    kpi_column: str,
    direction: str,
    limit: int,
) -> List[Dict[str, Any]]:
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
        LIMIT {limit}
    """
    rows = _execute_rows(sql)
    return _normalize_listing_rows(rows, kpi_column)


def _fetch_ranked_rows_with_backfill(
    table: str,
    where_clause: str,
    kpi_column: str,
    direction: str,
) -> List[Dict[str, Any]]:
    """Ensure each tier has exactly five listings by widening the scope if needed."""
    combined: List[Dict[str, Any]] = []
    seen: set[str] = set()
    clauses = [where_clause]
    if where_clause:
        clauses.append("")
    for clause in clauses:
        rows = _fetch_ranked_rows(table, clause, kpi_column, direction, limit=_RANK_BACKFILL_LIMIT)
        for row in rows:
            listing_id = row.get("listing_id")
            if not listing_id or listing_id in seen:
                continue
            combined.append(row)
            seen.add(listing_id)
            if len(combined) >= _MAX_LISTING_CARDS:
                return combined[:_MAX_LISTING_CARDS]
    placeholder_index = 1
    while len(combined) < _MAX_LISTING_CARDS:
        zero_metrics = {
            "kpi_value": 0.0,
            "price": 0.0,
            "price_status": "placeholder",
            "revenue_30": 0.0,
            "revenue_60": 0.0,
            "revenue_90": 0.0,
            "revenue_365": 0.0,
            "review_score": 0.0,
            "occupancy_rate_30": 0.0,
            "occupancy_rate_60": 0.0,
            "occupancy_rate_90": 0.0,
            "occupancy_rate_365": 0.0,
        }
        combined.append(
            {
                "listing_id": f"placeholder-{placeholder_index}",
                "listing_name": "Insufficient data",
                "neighbourhood": None,
                "borough": None,
                "kpi_value": 0.0,
                "metrics": zero_metrics,
            }
        )
        placeholder_index += 1
    return combined[:_MAX_LISTING_CARDS]


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
            MEDIAN(occupancy_rate_30) AS market_median_occ_30,
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
            "root_cause": {},
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

        top_rows = _fetch_ranked_rows_with_backfill(table, where_clause, kpi_info["column"], direction="desc")
        bottom_rows = _fetch_ranked_rows_with_backfill(table, where_clause, kpi_info["column"], direction="asc")
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
            benchmark = _lookup_benchmark(market_index, row.get("borough"), row.get("neighbourhood"))
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
            benchmark = _lookup_benchmark(market_index, row.get("borough"), row.get("neighbourhood"))
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

        root_cause = compute_root_cause(kpi_info, distribution, tier1_cards, tier2_cards)
        triage["portfolio_at_glance"]["root_cause"] = root_cause
        triage["portfolio_at_glance"]["top5_overview"] = [_listing_overview(entry) for entry in tier2_cards]
        triage["portfolio_at_glance"]["bottom5_overview"] = [_listing_overview(entry) for entry in tier1_cards]
        triage["portfolio_at_glance"]["sentiment_summary"] = _build_sentiment_summary(sentiment_tracker)
        triage["action_backlog"] = _build_action_backlog(tier1_cards, tier2_cards)
        triage["playbook_30d"] = _build_playbook(kpi_info, tier1_cards, tier2_cards, root_cause)
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