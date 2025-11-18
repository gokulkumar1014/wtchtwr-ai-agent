from __future__ import annotations

import threading
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable

import duckdb
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict

from .models import Conversation  # noqa: F401  # keep pydantic models discoverable


DB_PATH = "db/airbnb.duckdb"
CACHE_DIR = Path("data/cache")
FEATURES_CACHE_PATH = CACHE_DIR / "dashboard_features.parquet"

_MASTER_DF_LOCK = threading.Lock()
_MASTER_DF: Optional[pd.DataFrame] = None

METRIC_LABELS = {
    "review_scores_accuracy": "Accuracy",
    "review_scores_cleanliness": "Cleanliness",
    "review_scores_checkin": "Check-in",
    "review_scores_communication": "Communication",
    "review_scores_location": "Location",
    "review_scores_value": "Value",
}

RATING_BUCKET_EDGES = [0, 3.5, 4.0, 4.5, 4.8, 5.1]
RATING_BUCKET_LABELS = ["≤3.5", "3.5–4.0", "4.0–4.5", "4.5–4.8", "≥4.8"]

OCCUPANCY_METRICS = {
    "occupancy_rate_30": "Next 30 days",
    "occupancy_rate_60": "Next 60 days",
    "occupancy_rate_90": "Next 90 days",
    "occupancy_rate_365": "Next 365 days",
}

REVENUE_METRICS = {
    "estimated_revenue_30": "Revenue 30 days",
    "estimated_revenue_60": "Revenue 60 days",
    "estimated_revenue_90": "Revenue 90 days",
    "estimated_revenue_365": "Revenue 365 days",
}

_CACHE_TTL_SECONDS = 120
_DASHBOARD_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}


class RangeFilter(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    min: Optional[float] = None
    max: Optional[float] = None


class DashboardFilters(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    neighborhood_groups: List[str] = Field(default_factory=list, alias="neighborhoodGroups")
    neighborhoods: List[str] = Field(default_factory=list)
    property_types: List[str] = Field(default_factory=list, alias="propertyTypes")
    room_types: List[str] = Field(default_factory=list, alias="roomTypes")
    accommodates: RangeFilter = RangeFilter()
    bathrooms: RangeFilter = RangeFilter()
    bedrooms: RangeFilter = RangeFilter()
    beds: RangeFilter = RangeFilter()
    price: RangeFilter = RangeFilter()
    host_names: List[str] = Field(default_factory=list, alias="hostNames")  # kept for compatibility
    bathroom_details: List[str] = Field(default_factory=list, alias="bathroomDetails")


class DashboardComparison(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    mode: str = Field("market", pattern="^(market|hosts)$")
    hosts: List[str] = Field(default_factory=list)


class DashboardRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    filters: DashboardFilters = DashboardFilters()
    comparison: DashboardComparison = DashboardComparison()


def _ensure_feature_cache() -> None:
    if FEATURES_CACHE_PATH.exists():
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    query = """
        COPY (
            SELECT
                lc.listings_id,
                lc.host_id,
                lc.host_name,
                lower(lc.host_name) AS host_name_lower,
                lc.neighbourhood,
                lower(lc.neighbourhood) AS neighbourhood_lower,
                lc.neighbourhood_group,
                lower(lc.neighbourhood_group) AS neighbourhood_group_lower,
                lc.latitude,
                lc.longitude,
                lc.property_type,
                lower(lc.property_type) AS property_type_lower,
                lc.room_type,
                lower(lc.room_type) AS room_type_lower,
                lc.accommodates,
                lc.bathrooms,
                lc.bathroom_details,
                lc.bedrooms,
                lc.beds,
                lc.price_in_usd,
                lc.review_scores_rating,
                lc.review_scores_accuracy,
                lc.review_scores_cleanliness,
                lc.review_scores_checkin,
                lc.review_scores_communication,
                lc.review_scores_location,
                lc.review_scores_value,
                lc.occupancy_rate_30,
                lc.occupancy_rate_60,
                lc.occupancy_rate_90,
                lc.occupancy_rate_365,
                lc.estimated_revenue_30,
                lc.estimated_revenue_60,
                lc.estimated_revenue_90,
                lc.estimated_revenue_365,
                te.property_name
            FROM listings_cleaned AS lc
            LEFT JOIN text_extract AS te USING (listings_id)
        ) TO '{path}'
        (FORMAT PARQUET)
    """.format(path=str(FEATURES_CACHE_PATH).replace("'", "''"))
    with duckdb.connect(DB_PATH, read_only=True) as con:
        con.execute(query)


def _get_master_dataframe() -> pd.DataFrame:
    global _MASTER_DF
    with _MASTER_DF_LOCK:
        if _MASTER_DF is not None:
            return _MASTER_DF.copy()
        _ensure_feature_cache()
        df = pd.read_parquet(FEATURES_CACHE_PATH)
        df["bathroom_details"] = df["bathroom_details"].fillna("").astype(str).str.strip()
        df["bathroom_details_lower"] = df["bathroom_details"].str.lower()
        rating_bucket = pd.cut(
            df["review_scores_rating"],
            bins=RATING_BUCKET_EDGES,
            labels=RATING_BUCKET_LABELS,
            include_lowest=True,
            right=False,
        )
        df["rating_bucket"] = rating_bucket.astype(str)
        df.loc[rating_bucket.isna(), "rating_bucket"] = "missing"
        df["is_highbury"] = df["host_name_lower"] == "highbury"
        _MASTER_DF = df
    return df.copy()


def _apply_range_filter(series: pd.Series, range_filter: RangeFilter) -> pd.Series:
    mask = pd.Series(True, index=series.index)
    if range_filter.min is not None:
        mask &= series.fillna(range_filter.min - 1) >= range_filter.min
    if range_filter.max is not None:
        mask &= series.fillna(range_filter.max + 1) <= range_filter.max
    return mask


def _apply_filters(df: pd.DataFrame, filters: DashboardFilters) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if filters.neighborhood_groups:
        groups = {g.lower() for g in filters.neighborhood_groups}
        mask &= df["neighbourhood_group_lower"].isin(groups)
    if filters.neighborhoods:
        neighborhoods = {n.lower() for n in filters.neighborhoods}
        mask &= df["neighbourhood_lower"].isin(neighborhoods)
    if filters.property_types:
        props = {p.lower() for p in filters.property_types}
        mask &= df["property_type_lower"].isin(props)
    if filters.room_types:
        rooms = {r.lower() for r in filters.room_types}
        mask &= df["room_type_lower"].isin(rooms)

    mask &= _apply_range_filter(df["accommodates"], filters.accommodates)
    mask &= _apply_range_filter(df["bathrooms"], filters.bathrooms)
    mask &= _apply_range_filter(df["bedrooms"], filters.bedrooms)
    mask &= _apply_range_filter(df["beds"], filters.beds)
    mask &= _apply_range_filter(df["price_in_usd"], filters.price)

    if filters.bathroom_details:
        bathroom_terms = [term.lower() for term in filters.bathroom_details]
        bathroom_mask = pd.Series(False, index=df.index)
        for term in bathroom_terms:
            if term:
                bathroom_mask |= df["bathroom_details_lower"].str.contains(term, na=False)
        mask &= bathroom_mask

    # We intentionally **ignore** filters.host_names here now to keep the main filter
    # panel free of host-based filtering. Comparison panel controls hosts.
    return df[mask].copy()


def _filtered_slices(df: pd.DataFrame, filters: DashboardFilters, comparison: DashboardComparison) -> Dict[str, pd.DataFrame]:
    filtered = _apply_filters(df, filters)
    highbury_df = filtered[filtered["is_highbury"]].copy()
    comparison_df = filtered[~filtered["is_highbury"]].copy()

    if comparison.mode == "hosts":
        target_hosts = {h.lower() for h in comparison.hosts}
        if target_hosts:
            comparison_df = comparison_df[comparison_df["host_name_lower"].isin(target_hosts)]

    return {
        "highbury": highbury_df,
        "comparison": comparison_df,
        "combined": pd.concat([highbury_df, comparison_df], ignore_index=True),
    }


def _mean_or_none(series: pd.Series) -> Optional[float]:
    if series is None or series.empty:
        return None
    value = series.dropna().mean()
    if pd.isna(value):
        return None
    return float(round(value, 4))


def _quantiles(series: pd.Series) -> Dict[str, Optional[float]]:
    cleaned = series.dropna()
    if cleaned.empty:
        return {"min": None, "q1": None, "median": None, "q3": None, "max": None}
    q = cleaned.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).tolist()
    return {"min": round(q[0], 4), "q1": round(q[1], 4), "median": round(q[2], 4), "q3": round(q[3], 4), "max": round(q[4], 4)}


def _rating_distribution(series: pd.Series) -> List[Dict[str, Any]]:
    if series.empty:
        return [{"label": label, "count": 0} for label in RATING_BUCKET_LABELS]
    counts = series.value_counts()
    return [{"label": label, "count": int(counts.get(label, 0))} for label in RATING_BUCKET_LABELS]


def _host_counts(df: pd.DataFrame, *, only_hosts: Optional[Iterable[str]] = None, default_limit: int = 25) -> List[Dict[str, Any]]:
    """
    If only_hosts is provided (case-insensitive names), return counts for exactly those hosts
    (including ones with 0 matches). Otherwise return top-N by count.
    """
    if df.empty:
        if only_hosts:
            return [{"hostName": name, "listings": 0} for name in only_hosts]
        return []

    counts = df.groupby("host_name").size()

    if only_hosts:
        # Map lower->original name found in data (fallback to requested casing)
        names_lower = [h.lower() for h in only_hosts]
        lower_to_original = {h.lower(): h for h in counts.index}
        result: List[Dict[str, Any]] = []
        for req in names_lower:
            original = lower_to_original.get(req, None)
            display = original if original is not None else next((h for h in only_hosts if h.lower() == req), req)
            result.append({"hostName": display, "listings": int(counts.get(original, 0) if original else 0)})
        return result

    top = counts.sort_values(ascending=False).head(default_limit)
    return [{"hostName": name, "listings": int(val)} for name, val in top.items()]


def _compute_summary(highbury_df: pd.DataFrame, comparison_df: pd.DataFrame, comparison: DashboardComparison) -> Dict[str, Any]:
    def build_metric_entries(metrics: Dict[str, str]) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for metric, label in metrics.items():
            entries.append(
                {
                    "metric": metric,
                    "label": label,
                    "highbury": _mean_or_none(highbury_df[metric]) if not highbury_df.empty else None,
                    "comparison": _mean_or_none(comparison_df[metric]) if not comparison_df.empty else None,
                }
            )
        return entries

    price_stats = {
        "highbury": _quantiles(highbury_df["price_in_usd"]),
        "comparison": _quantiles(comparison_df["price_in_usd"]),
    }

    # Rating summary
    def _dist(df_: pd.DataFrame) -> List[Dict[str, Any]]:
        if df_.empty:
            return [{"label": label, "count": 0} for label in RATING_BUCKET_LABELS]
        counts = df_["rating_bucket"].value_counts()
        return [{"label": label, "count": int(counts.get(label, 0))} for label in RATING_BUCKET_LABELS]

    highbury_distribution = _dist(highbury_df)
    comparison_distribution = _dist(comparison_df)
    distribution_lookup = {
        "highbury": {d["label"]: d["count"] for d in highbury_distribution},
        "comparison": {d["label"]: d["count"] for d in comparison_distribution},
    }
    rating_summary = {
        "highburyAverage": _mean_or_none(highbury_df["review_scores_rating"]),
        "comparisonAverage": _mean_or_none(comparison_df["review_scores_rating"]),
        "distribution": [
            {"label": label, "highbury": int(distribution_lookup["highbury"].get(label, 0)), "comparison": int(distribution_lookup["comparison"].get(label, 0))}
            for label in RATING_BUCKET_LABELS
        ],
    }

    # Host counts
    if comparison.mode == "hosts" and comparison.hosts:
        # Ensure the table shows **Highbury** plus all selected hosts, even if zero.
        only_hosts = ["Highbury"] + comparison.hosts
        combined_counts = _host_counts(pd.concat([highbury_df, comparison_df], ignore_index=True), only_hosts=only_hosts)
    else:
        combined_counts = _host_counts(pd.concat([highbury_df, comparison_df], ignore_index=True), default_limit=25)

    return {
        "totals": {
            "highburyListings": int(highbury_df.shape[0]),
            "comparisonListings": int(comparison_df.shape[0]),
        },
        "occupancyCards": build_metric_entries(OCCUPANCY_METRICS),
        "revenueCards": build_metric_entries(REVENUE_METRICS),
        "reviewScores": build_metric_entries(METRIC_LABELS),
        "priceSummary": price_stats,
        "ratingSummary": rating_summary,
        "hostCounts": {
            "highbury": _host_counts(highbury_df, only_hosts=["Highbury"]),
            "comparison": _host_counts(comparison_df, default_limit=25),
            "combined": combined_counts,
        },
    }


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _map_payload(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"listings": [], "total": 0, "rendered": 0, "comparisonSampled": 0}

    highbury_df = df[df["is_highbury"]]
    comparison_df = df[~df["is_highbury"]]
    render_df = pd.concat([highbury_df, comparison_df], ignore_index=True)

    listings: List[Dict[str, Any]] = []
    for _, row in render_df.iterrows():
        listings.append(
            {
                "listingId": int(row["listings_id"]),
                "lat": float(row["latitude"]),
                "lng": float(row["longitude"]),
                "group": "Highbury" if row["is_highbury"] else "Comparison",
                "hostName": row["host_name"],
                "propertyName": (row.get("property_name") or "").strip() or None,
                "neighborhood": row.get("neighbourhood"),
                "neighborhoodGroup": row.get("neighbourhood_group"),
                "propertyType": row.get("property_type"),
                "roomType": row.get("room_type"),
                "price": _as_float(row.get("price_in_usd")),
                "occupancyRate90": _as_float(row.get("occupancy_rate_90")),
                "reviewScore": _as_float(row.get("review_scores_rating")),
            }
        )
    return {"listings": listings, "total": int(df.shape[0]), "rendered": int(render_df.shape[0]), "comparisonSampled": int(comparison_df.shape[0])}


def _available_filters(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "neighborhoodGroups": sorted(filter(None, df["neighbourhood_group"].dropna().unique().tolist())),
        "neighborhoods": sorted(filter(None, df["neighbourhood"].dropna().unique().tolist())),
        "propertyTypes": sorted(filter(None, df["property_type"].dropna().unique().tolist())),
        "roomTypes": sorted(filter(None, df["room_type"].dropna().unique().tolist())),
        "hostNames": sorted(filter(None, df["host_name"].dropna().unique().tolist())),
        "bathroomDetails": sorted(
            filter(
                None,
                (
                    detail.strip()
                    for detail in df["bathroom_details"].dropna().unique().tolist()
                    if isinstance(detail, str) and detail.strip()
                ),
            )
        ),
    }


def _cache_key(request: DashboardRequest) -> str:
    payload = request.model_dump(mode="json")
    return json.dumps(payload, sort_keys=True)


def build_dashboard_response(request: DashboardRequest) -> Dict[str, Any]:
    cache_key = _cache_key(request)
    now = time.time()
    cached = _DASHBOARD_CACHE.get(cache_key)
    if cached and now - cached[0] <= _CACHE_TTL_SECONDS:
        return cached[1]

    master_df = _get_master_dataframe()
    slices = _filtered_slices(master_df, request.filters, request.comparison)
    highbury_df = slices["highbury"]
    comparison_df = slices["comparison"]
    combined_df = slices["combined"]

    map_payload = _map_payload(combined_df)
    summary = _compute_summary(highbury_df, comparison_df, request.comparison)
    filters_available = _available_filters(combined_df)

    response = {"summary": summary, "map": map_payload, "availableFilters": filters_available}
    _DASHBOARD_CACHE[cache_key] = (now, response)
    if len(_DASHBOARD_CACHE) > 64:
        oldest_key = next(iter(_DASHBOARD_CACHE))
        if oldest_key != cache_key:
            _DASHBOARD_CACHE.pop(oldest_key, None)
    return response


def load_filter_options() -> Dict[str, Any]:
    df = _get_master_dataframe()
    return {
        "neighborhoodGroups": sorted(filter(None, df["neighbourhood_group"].dropna().unique().tolist())),
        "neighborhoods": sorted(filter(None, df["neighbourhood"].dropna().unique().tolist())),
        "propertyTypes": sorted(filter(None, df["property_type"].dropna().unique().tolist())),
        "roomTypes": sorted(filter(None, df["room_type"].dropna().unique().tolist())),
        "hostNames": sorted(filter(None, df["host_name"].dropna().unique().tolist())),
        "bathroomDetails": sorted(
            filter(
                None,
                (
                    detail.strip()
                    for detail in df["bathroom_details"].dropna().unique().tolist()
                    if isinstance(detail, str) and detail.strip()
                ),
            )
        ),
        "ranges": {
            "accommodates": {"min": int(df["accommodates"].min(skipna=True) or 0), "max": int(df["accommodates"].max(skipna=True) or 0)},
            "bathrooms": {"min": float(df["bathrooms"].min(skipna=True) or 0), "max": float(df["bathrooms"].max(skipna=True) or 0)},
            "bedrooms": {"min": float(df["bedrooms"].min(skipna=True) or 0), "max": float(df["bedrooms"].max(skipna=True) or 0)},
            "beds": {"min": float(df["beds"].min(skipna=True) or 0), "max": float(df["beds"].max(skipna=True) or 0)},
            "price": {"min": float(df["price_in_usd"].min(skipna=True) or 0), "max": float(df["price_in_usd"].max(skipna=True) or 0)},
        },
    }