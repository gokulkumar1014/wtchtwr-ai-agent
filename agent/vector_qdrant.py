"""Vector retrieval helpers backed by Qdrant."""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from pydantic import ValidationError
from sentence_transformers import SentenceTransformer

from .config import MONTHS_MAP, load_config, normalize_filters
from .types import State, add_thinking_step
from .utils.filters import extract_metadata_filters

LOGGER = logging.getLogger(__name__)

_CLIENT: Optional[QdrantClient] = None
_METADATA: Optional[pd.DataFrame] = None
_MODEL: Optional[SentenceTransformer] = None


# ---------------------------- Resource loading ----------------------------

def ensure_ready() -> None:
    """Eagerly load Qdrant client, metadata, and embedding model."""
    _load_resources()


def _load_resources() -> tuple[QdrantClient, pd.DataFrame, SentenceTransformer, Any]:
    """Lazily instantiate the Qdrant client, metadata frame, and encoder."""
    global _CLIENT, _METADATA, _MODEL
    cfg = load_config()

    if _CLIENT is None:
        _CLIENT = QdrantClient(
            url=cfg.qdrant_url,
            api_key=cfg.qdrant_api_key or None,
            timeout=cfg.qdrant_timeout_s,
        )

    if _METADATA is None:
        path = cfg.reviews_metadata_path
        if not path.exists():
            raise FileNotFoundError(f"Metadata parquet not found at {path}")
        _METADATA = pd.read_parquet(path)

    if _MODEL is None:
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    return _CLIENT, _METADATA, _MODEL, cfg


# ---------------------------- Normalisation helpers ----------------------------

def _ensure_in_list(raw: Any) -> List[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        values = raw
    else:
        values = [raw]
    cleaned: List[Any] = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, str):
            text = v.strip()
            if text:
                cleaned.append(text)
        else:
            cleaned.append(v)
    return cleaned


def _normalize_month_input(raw_months: Iterable[Any]) -> List[str]:
    """Map user month words → canonical 3-letter uppercase codes."""
    canonical: List[str] = []
    for value in raw_months:
        key = str(value).strip().lower()
        if not key:
            continue
        mapped = MONTHS_MAP.get(key)
        if mapped and mapped not in canonical:
            canonical.append(mapped)
    return canonical


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "t"}:
            return True
        if lowered in {"false", "0", "no", "n", "f", ""}:
            return False
    return False


def _score_to_distance(score: Optional[float]) -> Optional[float]:
    if score is None:
        return None
    try:
        # Convert cosine similarity (higher is better) into distance (lower is better)
        value = float(score)
        return float(round(1.0 - value, 6))
    except (TypeError, ValueError):
        return None


def _encode_query(model: SentenceTransformer, text: str) -> np.ndarray:
    embedding = model.encode([text], normalize_embeddings=True)
    if embedding.dtype != np.float32:
        embedding = embedding.astype(np.float32)
    return embedding[0]


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return None
        return result
    except (TypeError, ValueError):
        return None


def _extract_range_filter(filters: Dict[str, Any], key: str) -> Optional[models.Range]:
    """Construct a Range condition for numeric sentiment scores."""
    params: Dict[str, float] = {}
    raw = filters.get(key)
    if isinstance(raw, dict):
        for op in ("gt", "gte", "lt", "lte"):
            val = _safe_float(raw.get(op))
            if val is not None:
                params[op] = val
    for suffix, op in RANGE_SUFFIX_MAP.items():
        val = _safe_float(filters.get(f"{key}{suffix}"))
        if val is not None:
            params[op] = val
    if not params:
        return None
    return models.Range(**params)


# ---------------------------- Qdrant filter builder ----------------------------

NUMERIC_FILTER_KEYS = {"listing_id", "comment_id"}
SENTIMENT_LABELS = {"positive", "neutral", "negative"}
RANGE_SUFFIX_MAP = {
    "_gt": "gt",
    "_gte": "gte",
    "_lt": "lt",
    "_lte": "lte",
}


def _normalize_metadata_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure metadata filters align with Qdrant payload expectations."""
    normalized = dict(filters or {})
    listing_value = normalized.get("listing_id")

    def _coerce_listing(value: Any) -> Optional[int]:
        if value is None:
            return None
        coerced, is_numeric = _coerce_numeric_filter_value("listing_id", value)
        if coerced in (None, ""):
            return None
        if not is_numeric:
            try:
                return int(coerced)
            except Exception:
                return None
        return coerced

    if isinstance(listing_value, list):
        cleaned = []
        for item in listing_value:
            coerced = _coerce_listing(item)
            if coerced is not None:
                cleaned.append(coerced)
        if cleaned:
            normalized["listing_id"] = cleaned
            LOGGER.info("[QDRANT] Normalized listing_id list to ints: %s", cleaned)
        else:
            normalized.pop("listing_id", None)
    else:
        coerced = _coerce_listing(listing_value)
        if coerced is not None:
            normalized["listing_id"] = coerced
            if coerced != listing_value:
                LOGGER.info("[QDRANT] Normalized listing_id to int: %s", coerced)
        elif listing_value is not None:
            normalized.pop("listing_id", None)

    return normalized


def _coerce_numeric_filter_value(key: str, value: Any) -> tuple[Any, bool]:
    """Ensure numeric metadata like listing/comment IDs stay numeric for Qdrant."""
    if key not in NUMERIC_FILTER_KEYS:
        return value, False
    if value is None:
        return None, True
    try:
        if isinstance(value, bool):
            raise ValueError("boolean is not valid for numeric filter")
        if isinstance(value, (int, np.integer)):
            return int(value), True
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                raise ValueError("invalid float")
            return int(value), True
        text = str(value).strip()
        if not text:
            return None, True
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return int(text), True
        float_val = float(text)
        if float_val.is_integer():
            return int(float_val), True
        raise ValueError("non-integer string")
    except Exception:
        LOGGER.warning("Qdrant filter: could not coerce %s=%r to int", key, value)
        return value, True

def _match_from_values(values: List[Any]) -> Optional[models.Match]:
    if not values:
        return None
    if len(values) == 1:
        value = values[0]
        return models.MatchValue(value=value)

    return models.MatchAny(any=values)


def _build_filter(filters: Dict[str, Any]) -> Optional[models.Filter]:
    """Convert filter dictionary into a Qdrant Filter."""
    normalized_filters = _normalize_metadata_filters(filters)
    must_conditions: List[models.FieldCondition] = []

    # Years
    years: List[int] = []
    for item in _ensure_in_list(normalized_filters.get("year")):
        try:
            years.append(int(item))
        except (TypeError, ValueError):
            continue
    match_years = _match_from_values(years)
    if match_years is not None:
        must_conditions.append(models.FieldCondition(key="year", match=match_years))

    # Months
    months = _normalize_month_input(_ensure_in_list(normalized_filters.get("month")))
    if months:
        must_conditions.append(models.FieldCondition(key="month", match=_match_from_values(months)))

    # Boroughs
    boroughs = [str(b).strip().title() for b in _ensure_in_list(normalized_filters.get("borough")) if str(b).strip()]
    if boroughs:
        must_conditions.append(
            models.FieldCondition(key="neighbourhood_group", match=_match_from_values(boroughs))
        )

    # Neighbourhood
    neighbourhoods = [str(n).strip() for n in _ensure_in_list(normalized_filters.get("neighbourhood")) if str(n).strip()]
    if neighbourhoods:
        must_conditions.append(models.FieldCondition(key="neighbourhood", match=_match_from_values(neighbourhoods)))

    # Listing IDs
    listing_ids: List[Any] = []
    for item in _ensure_in_list(normalized_filters.get("listing_id")):
        if item is None:
            continue
        coerced, is_numeric = _coerce_numeric_filter_value("listing_id", item)
        if coerced in (None, ""):
            continue
        listing_ids.append((coerced, is_numeric))
    if listing_ids:
        values = [value for value, _ in listing_ids]
        if len(values) == 1:
            must_conditions.append(
                models.FieldCondition(key="listing_id", match=models.MatchValue(value=values[0]))
            )
        else:
            must_conditions.append(
                models.FieldCondition(key="listing_id", match=models.MatchAny(any=values))
            )

    # Comment IDs
    comment_ids: List[Any] = []
    for item in _ensure_in_list(normalized_filters.get("comment_id")):
        if item is None or (isinstance(item, str) and not item.strip()):
            continue
        coerced, is_numeric = _coerce_numeric_filter_value("comment_id", item)
        if coerced in (None, ""):
            continue
        comment_ids.append((coerced, is_numeric))
    if comment_ids:
        values = [value for value, _ in comment_ids]
        if len(values) == 1:
            must_conditions.append(
                models.FieldCondition(key="comment_id", match=models.MatchValue(value=values[0]))
            )
        else:
            must_conditions.append(
                models.FieldCondition(key="comment_id", match=models.MatchAny(any=values))
            )

    # Highbury flag
    if "is_highbury" in normalized_filters and normalized_filters.get("is_highbury") is not None:
        must_conditions.append(
            models.FieldCondition(
                key="is_highbury",
                match=models.MatchValue(value=bool(normalized_filters.get("is_highbury"))),
            )
        )

    # Sentiment label filtering
    sentiment_candidates = [
        str(val).strip().lower()
        for val in _ensure_in_list(normalized_filters.get("sentiment_label"))
        if str(val).strip()
    ]
    sentiment_values = [val for val in sentiment_candidates if val in SENTIMENT_LABELS]
    if sentiment_values:
        match = _match_from_values(sentiment_values)
        if match is not None:
            must_conditions.append(models.FieldCondition(key="sentiment_label", match=match))

    # Sentiment score ranges
    for sentiment_key in ("positive", "neutral", "negative", "compound"):
        range_filter = _extract_range_filter(normalized_filters, sentiment_key)
        if range_filter is not None:
            must_conditions.append(models.FieldCondition(key=sentiment_key, range=range_filter))

    if not must_conditions:
        return None

    return models.Filter(must=must_conditions)


# ---------------------------- Search result shaping ----------------------------

def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
                return int(stripped)
            float_candidate = float(stripped)
            if float_candidate.is_integer():
                return int(float_candidate)
            return None
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            if value.is_integer():
                return int(value)
            return None
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _safe_str(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def _points_to_hits(points: Sequence[models.ScoredPoint], top_k: int) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for point in points:
        payload = point.payload or {}
        borough = _safe_str(payload.get("borough") or payload.get("neighbourhood_group"))
        neighbourhood = _safe_str(payload.get("neighbourhood"))
        month = _safe_str(payload.get("month"))
        year = _safe_int(payload.get("year"))
        snippet = _safe_str(payload.get("comments")) or ""
        sentiment_label = _safe_str(payload.get("sentiment_label"))
        compound = _safe_float(payload.get("compound"))
        hit = {
            "listing_id": _safe_str(payload.get("listing_id")),
            "comment_id": _safe_int(payload.get("comment_id") or point.id),
            "borough": borough,
            "neighbourhood_group": borough,
            "neighbourhood": neighbourhood,
            "month": month,
            "year": year,
            "snippet": snippet[:220],
            "distance": _score_to_distance(point.score),
            "is_highbury": _to_bool(payload.get("is_highbury")),
            "sentiment_label": sentiment_label.lower() if sentiment_label else None,
            "compound": compound,
            "positive": _safe_float(payload.get("positive")),
            "neutral": _safe_float(payload.get("neutral")),
            "negative": _safe_float(payload.get("negative")),
        }
        hits.append(hit)

    def _sort_key(item: Dict[str, Any]) -> tuple[float, int]:
        distance = item.get("distance")
        dist_val = float(distance) if isinstance(distance, (int, float)) else float("inf")
        year_val = item.get("year") or 0
        return (dist_val, -int(year_val))

    hits_sorted = sorted(hits, key=_sort_key)
    if top_k > 0:
        hits_sorted = hits_sorted[:top_k]
    return hits_sorted


# ---------------------------- RAG control ----------------------------

def need_rag(state: State) -> State:
    """Compute whether the RAG branch should execute."""
    plan = state.get("plan", {}) or {}
    mode = plan.get("mode", "sql")
    sql_state = state.get("sql", {}) or {}
    sql_df = sql_state.get("df")

    needed = mode in {"rag", "hybrid"} or (
        mode == "sql" and sql_df is not None and getattr(sql_df, "empty", False)
    )
    state["rag_needed"] = needed
    LOGGER.debug("RAG needed: %s", needed)
    return state


def exec_rag(state: State) -> State:
    """Qdrant retrieval with metadata filters and timing telemetry."""
    total_start = time.time()
    telemetry = state.setdefault("telemetry", {})

    if not state.get("rag_needed", False):
        state["rag_snippets"] = []
        telemetry["faiss_index_type"] = "QDRANT_HNSW"
        telemetry.update({
            "rag_retrieval_time_s": 0.0,
            "rag_relax_time_s": 0.0,
            "rag_total_time_s": 0.0,
        })
        return state

    try:
        client, metadata, model, cfg = _load_resources()
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.error("Qdrant resources unavailable: %s", exc)
        state["rag_snippets"] = []
        telemetry["rag_error"] = str(exc)
        return state

    telemetry["faiss_index_type"] = "QDRANT_HNSW"

    filters_raw = dict(state.get("filters", {}) or {})
    filters = normalize_filters(filters_raw)
    sentiment_hint = filters_raw.get("sentiment_label")
    if sentiment_hint and not filters.get("sentiment_label"):
        filters["sentiment_label"] = str(sentiment_hint).strip().lower()

    scope = state.get("scope") or state.get("policy")
    market_only = scope == "Market" and filters.get("is_highbury") is None
    has_listing_filter = bool(filters.get("listing_id"))
    if scope == "Highbury" and filters.get("is_highbury") is None:
        filters["is_highbury"] = True
    elif market_only and not has_listing_filter:
        filters["is_highbury"] = False

    top_k = int(state.get("plan", {}).get("top_k", cfg.top_k_default))
    if top_k < 10:
        top_k = 10
    query_text = state.get("query", "") or ""

    metadata_filters = extract_metadata_filters(query_text, metadata)
    if metadata_filters:
        for key, value in metadata_filters.items():
            if not cfg.use_metadata_filters and key not in {"listing_id", "comment_id"}:
                continue
            existing = filters.get(key)
            if existing in (None, "", [], {}, set()):
                filters[key] = value
    if filters.get("listing_id") is not None:
        filters.pop("is_highbury", None)

    query_vector = _encode_query(model, query_text)
    total_vectors = len(metadata) if metadata is not None else top_k * 10
    search_k = min(max(top_k * 6, 30), total_vectors)

    qdrant_filter = _build_filter(filters)

    retrieval_start = time.time()
    try:
        points = client.search(
            collection_name=cfg.qdrant_collection,
            query_vector=query_vector.tolist(),
            limit=search_k,
            query_filter=qdrant_filter,
            with_payload=True,
            search_params=models.SearchParams(hnsw_ef=96),
        )
    except (ValidationError, UnexpectedResponse) as exc:
        LOGGER.error("Qdrant search failed: %s", exc)
        state["rag_snippets"] = []
        telemetry["rag_error"] = str(exc)
        return state
    except Exception as exc:  # pragma: no cover - remote failure
        LOGGER.error("Qdrant search unexpected error: %s", exc)
        state["rag_snippets"] = []
        telemetry["rag_error"] = str(exc)
        return state

    retrieval_time = time.time() - retrieval_start

    relaxed = False
    relax_time = 0.0
    if not points and (filters.get("month") or filters.get("year")):
        relax_start = time.time()
        relaxed_filters = dict(filters)
        relaxed_filters.pop("month", None)
        relaxed_filters.pop("year", None)
        relaxed_filter_obj = _build_filter(relaxed_filters)
        try:
            points = client.search(
                collection_name=cfg.qdrant_collection,
                query_vector=query_vector.tolist(),
                limit=search_k,
                query_filter=relaxed_filter_obj,
                with_payload=True,
                search_params=models.SearchParams(hnsw_ef=96),
            )
        except (ValidationError, UnexpectedResponse) as exc:
            LOGGER.error("Qdrant relaxed search failed: %s", exc)
            points = []
        except Exception as exc:  # pragma: no cover - remote failure
            LOGGER.error("Qdrant relaxed search unexpected error: %s", exc)
            points = []
        relax_time = time.time() - relax_start
        if points:
            filters = relaxed_filters
            qdrant_filter = relaxed_filter_obj
            relaxed = True

    hits = _points_to_hits(points, top_k=min(max(top_k, 10), 10))

    # Normalize listing_id comparisons post-retrieval to avoid int/str mismatches.
    listing_filter_values = filters.get("listing_id")
    if listing_filter_values:
        allowed_ids = {str(v).strip() for v in _ensure_in_list(listing_filter_values) if v is not None}
        if allowed_ids:
            hits = [hit for hit in hits if str(hit.get("listing_id")).strip() in allowed_ids]

    total_time = time.time() - total_start
    state["rag_snippets"] = hits

    applied_filters = {k: v for k, v in filters.items() if v not in (None, "", [], {}, set())}

    rag_state = state.setdefault("rag", {})
    rag_state.update({
        "hits": hits,
        "relaxed_once": relaxed,
        "filters": applied_filters,
        "candidate_count": len(points),
        "filter_object": qdrant_filter.dict() if qdrant_filter else None,
    })

    telemetry.update({
        "applied_filters": {k: str(v) for k, v in applied_filters.items()},
        "filtered_count": int(len(points)),
        "rag_retrieval_time_s": round(retrieval_time, 3),
        "rag_relax_time_s": round(relax_time, 3),
        "rag_total_time_s": round(total_time, 3),
        "faiss_search_k": search_k,
    })

    LOGGER.info(
        "🔍 Qdrant retrieved %d hits in %.2fs (retrieval %.2fs, relax %.2fs)",
        len(hits),
        total_time,
        retrieval_time,
        relax_time,
    )

    add_thinking_step(
        state,
        phase="rag_search",
        title="Searched guest feedback",
        detail=f"Collected {len(hits)} review snippets with metadata filters.",
        meta={
            "hits": len(hits),
            "relaxed_once": relaxed,
            "filters": list(applied_filters.keys()),
        },
    )

    return state


# ---------------------------- Light summariser ----------------------------

_POSITIVE_TERMS = {"great", "good", "love", "excellent", "amazing", "friendly", "clean"}
_NEGATIVE_TERMS = {"bad", "dirty", "noisy", "issue", "complaint", "poor", "cold", "hot"}


def summarize_hits(state: State) -> State:
    """Summarize and normalize vector hits: sort, de-dupe, and confidence."""
    hits: List[Dict[str, Any]] = state.get("rag_snippets") or state.get("rag", {}).get("hits", [])
    rag_state = state.setdefault("rag", {})

    if not hits:
        state["rag_snippets"] = []
        rag_state.update({
            "summary": "No relevant reviews found for the current filters.",
            "citations": [],
            "weak_evidence": True,
            "evidence_count": 0,
            "confidence": "weak",
        })
        return state

    def _key(item: Dict[str, Any]) -> tuple[float, int]:
        distance = item.get("distance")
        dist_val = float(distance) if isinstance(distance, (int, float)) else float("inf")
        year_val = item.get("year") or 0
        return (dist_val, -int(year_val))

    hits_sorted = sorted(hits, key=_key)

    seen = set()
    deduped: List[Dict[str, Any]] = []
    for hit in hits_sorted:
        key = (hit.get("listing_id"), hit.get("comment_id"))
        if key not in seen:
            seen.add(key)
            deduped.append({
                "listing_id": hit.get("listing_id"),
                "comment_id": hit.get("comment_id"),
                "snippet": hit.get("snippet") or hit.get("text"),
                "borough": hit.get("borough") or hit.get("neighbourhood"),
                "neighbourhood_group": hit.get("neighbourhood_group"),
                "neighbourhood": hit.get("neighbourhood"),
                "month": hit.get("month"),
                "year": hit.get("year"),
                "distance": hit.get("distance"),
                "score": hit.get("score"),
                "sentiment_label": hit.get("sentiment_label"),
                "compound": hit.get("compound"),
                "positive": hit.get("positive"),
                "neutral": hit.get("neutral"),
                "negative": hit.get("negative"),
                "is_highbury": hit.get("is_highbury"),
            })

    pos, neg = 0, 0
    for hit in deduped:
        txt = (hit.get("snippet") or "").lower()
        if any(term in txt for term in _POSITIVE_TERMS):
            pos += 1
        if any(term in txt for term in _NEGATIVE_TERMS):
            neg += 1

    evidence_count = len(deduped)
    if evidence_count < 2:
        confidence = "weak"
    elif pos and neg:
        confidence = "mixed"
    elif pos:
        confidence = "positive"
    elif neg:
        confidence = "negative"
    else:
        confidence = "neutral"

    best = deduped[0]
    borough = best.get("borough") or best.get("neighbourhood")
    month = best.get("month")
    year = best.get("year")

    if borough and year:
        summary = f"Top review evidence from {borough} ({month or 'month unknown'} {year})."
    elif borough:
        summary = f"Top review evidence from {borough}."
    else:
        summary = "Top review evidence identified."

    rag_state.update({
        "hits": deduped,
        "summary": summary,
        "citations": [f"{hit.get('listing_id')}:{hit.get('comment_id')}" for hit in deduped if hit.get("listing_id")],
        "weak_evidence": evidence_count < 2,
        "evidence_count": evidence_count,
        "confidence": confidence,
    })
    state["rag_snippets"] = deduped
    return state


__all__ = ["ensure_ready", "exec_rag", "need_rag", "summarize_hits", "_build_filter"]
