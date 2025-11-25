"""
Configuration utilities for the wtchtwr AI Agent.
Ensures all environment variables and file paths are consistently initialized
for LangGraph orchestration (compose → graph → nl2sql_llm → vector_qdrant).
"""

from __future__ import annotations
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    def load_dotenv(*args: Any, **kwargs: Any) -> None:
        return None

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normalization Maps
# ---------------------------------------------------------------------------

MONTHS_MAP: Dict[str, str] = {
    "jan": "JAN", "january": "JAN",
    "feb": "FEB", "february": "FEB",
    "mar": "MAR", "march": "MAR",
    "apr": "APR", "april": "APR",
    "may": "MAY",
    "jun": "JUN", "june": "JUN",
    "jul": "JUL", "july": "JUL",
    "aug": "AUG", "august": "AUG",
    "sep": "SEP", "sept": "SEP", "september": "SEP",
    "oct": "OCT", "october": "OCT",
    "nov": "NOV", "november": "NOV",
    "dec": "DEC", "december": "DEC",
}

SEASON_MAP: Dict[str, List[str]] = {
    "winter": ["DEC", "JAN", "FEB"],
    "spring": ["MAR", "APR", "MAY"],
    "summer": ["JUN", "JUL", "AUG"],
    "fall": ["SEP", "OCT", "NOV"],
}

DATA_SCHEMA: Dict[str, str] = {
    "listing_id": "int",
    "comment_id": "int",
    "comments": "str",
    "year": "int",
    "month": "str",
    "neighbourhood": "str",
    "neighbourhood_group": "str",
    "is_highbury": "bool",
}

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def get_month_abbrev(value: str) -> Optional[str]:
    """Return the normalized 3-letter month abbreviation (e.g., 'August' → 'AUG')."""
    if not value:
        return None
    key = str(value).strip().lower()
    return MONTHS_MAP.get(key, key.upper())


def normalize_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize common filters (month/year/highbury) before vector or SQL filtering."""
    if not filters:
        return {}

    normalized = dict(filters)

    # Normalize month(s)
    if "month" in normalized:
        val = normalized["month"]
        if isinstance(val, list):
            normalized["month"] = [get_month_abbrev(v) or v for v in val]
        else:
            normalized["month"] = get_month_abbrev(val) or val

    # Normalize boolean highbury flag
    if "is_highbury" in normalized:
        v = normalized["is_highbury"]
        if isinstance(v, str):
            normalized["is_highbury"] = v.strip().lower() in {"yes", "y", "true", "1"}

    # Normalize year as int
    if "year" in normalized:
        try:
            normalized["year"] = int(normalized["year"])
        except Exception:
            pass

    return normalized


# ---------------------------------------------------------------------------
# Whitelist and Defaults
# ---------------------------------------------------------------------------

DEFAULT_WHITELIST: Dict[str, List[str]] = {
    "listings": [
        "listings_id", "host_id", "host_name", "neighbourhood", "neighbourhood_group",
        "latitude", "longitude", "property_type", "room_type", "accommodates",
        "bathrooms", "bedrooms", "beds", "price_in_usd",
        "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
        "review_scores_checkin", "review_scores_communication",
        "review_scores_location", "review_scores_value",
        "occupancy_rate_30", "occupancy_rate_60", "occupancy_rate_90", "occupancy_rate_365",
        "host_listings_count",
        "estimated_revenue_30", "estimated_revenue_60",
        "estimated_revenue_90", "estimated_revenue_365",
    ],
    "highbury_listings": [
        "listings_id", "host_id", "host_name", "neighbourhood", "neighbourhood_group",
        "latitude", "longitude", "property_type", "room_type", "accommodates",
        "bathrooms", "bedrooms", "beds", "price_in_usd",
        "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
        "review_scores_checkin", "review_scores_communication",
        "review_scores_location", "review_scores_value",
        "occupancy_rate_30", "occupancy_rate_60", "occupancy_rate_90", "occupancy_rate_365",
        "host_listings_count",
        "estimated_revenue_30", "estimated_revenue_60",
        "estimated_revenue_90", "estimated_revenue_365",
    ],
}


# ---------------------------------------------------------------------------
# Dataclass Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    """Immutable configuration for the HOPE Agent runtime."""

    duckdb_path: Path
    chroma_dir: Path
    reviews_embeddings_path: Path
    reviews_metadata_path: Path
    streaming_enabled: bool
    use_metadata_filters: bool
    openai_model: str
    openai_fallback_model: str
    openai_api_key: Optional[str]
    top_k_default: int
    similarity_min: float
    max_rows: int
    qdrant_url: str
    qdrant_collection: str
    qdrant_api_key: Optional[str]
    qdrant_timeout_s: float
    nl2sql_whitelist: Dict[str, List[str]] = field(default_factory=dict)
    default_years: List[int] = field(default_factory=list)
    default_boroughs: List[str] = field(default_factory=list)
    telemetry: bool = False
    chat_max_turns: int = 6
    stream_composer: bool = True
    chroma_collection: str = "airbnb_reviews_all"
    tavily_api_key: Optional[str] = None
    tavily_max_results: int = 3

    @property
    def duckdb_path_str(self) -> str:
        return str(self.duckdb_path)

    @property
    def chroma_dir_str(self) -> str:
        return str(self.chroma_dir)

    @property
    def reviews_embeddings_path_str(self) -> str:
        return str(self.reviews_embeddings_path)

    @property
    def reviews_metadata_path_str(self) -> str:
        return str(self.reviews_metadata_path)

    @property
    def qdrant_url_str(self) -> str:
        return self.qdrant_url


_cached_config: Optional[Config] = None


# ---------------------------------------------------------------------------
# Environment Handling
# ---------------------------------------------------------------------------

def _ensure_env_silence() -> None:
    """Disable telemetry for local components (Chroma, TensorFlow, etc.)."""
    os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")
    os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "false")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def load_config(refresh: bool = False) -> Config:
    """
    Load configuration from environment and cache the result.
    Parameters
    ----------
    refresh : bool
        If True, re-read environment variables and reinitialize Config.
    """
    global _cached_config
    if _cached_config is not None and not refresh:
        return _cached_config

    load_dotenv(override=False)
    _ensure_env_silence()

    project_root = Path(os.getenv("HOPE_AGENT_ROOT", Path.cwd()))

    duckdb_path = Path(os.getenv("HOPE_AGENT_DUCKDB", "db/airbnb.duckdb"))
    chroma_dir = Path(os.getenv("HOPE_AGENT_CHROMA", "vec/airbnb_reviews"))
    reviews_embeddings_path = Path(os.getenv("HOPE_AGENT_EMBEDDINGS_PATH", "vec/reviews_embeddings.npy"))
    reviews_metadata_path = Path(os.getenv("HOPE_AGENT_METADATA_PATH", "vec/reviews_metadata.parquet"))

    qdrant_url = os.getenv("HOPE_AGENT_QDRANT_URL", "http://localhost:6333").strip()
    qdrant_collection = os.getenv("HOPE_AGENT_QDRANT_COLLECTION", "airbnb_reviews").strip()
    qdrant_api_key = os.getenv("HOPE_AGENT_QDRANT_API_KEY")
    qdrant_timeout_s = float(os.getenv("HOPE_AGENT_QDRANT_TIMEOUT_S", "10.0"))

    # Resolve paths relative to project root
    for path_var in [duckdb_path, chroma_dir, reviews_embeddings_path, reviews_metadata_path]:
        if not path_var.is_absolute():
            path_var = project_root / path_var

    primary_model_default = "gpt-5.1"
    fallback_model_default = "gpt-4o"
    openai_model = os.getenv("HOPE_AGENT_OPENAI_MODEL", primary_model_default).strip()
    openai_fallback_model = os.getenv("HOPE_AGENT_OPENAI_FALLBACK_MODEL", fallback_model_default).strip()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    tavily_max_results = int(os.getenv("TAVILY_MAX_RESULTS", "3"))

    streaming_enabled = os.getenv("HOPE_AGENT_STREAMING_ENABLED", "true").strip().lower() in {"true", "1", "yes"}
    use_metadata_filters = os.getenv("HOPE_AGENT_USE_METADATA_FILTERS", "true").strip().lower() in {"true", "1", "yes"}
    top_k_default = int(os.getenv("HOPE_AGENT_TOP_K", "10"))
    similarity_min = float(os.getenv("HOPE_AGENT_SIMILARITY_MIN", "0.45"))
    max_rows = int(os.getenv("HOPE_AGENT_MAX_ROWS", "500"))

    default_years_env = os.getenv("HOPE_AGENT_DEFAULT_YEARS", "")
    default_years = [int(y.strip()) for y in default_years_env.split(",") if y.strip()]

    default_boroughs_env = os.getenv("HOPE_AGENT_DEFAULT_BOROUGHS", "")
    default_boroughs = [b.strip() for b in default_boroughs_env.split(",") if b.strip()]

    telemetry = os.getenv("HOPE_AGENT_TELEMETRY", "false").lower() in {"1", "yes", "true"}
    chat_max_turns = int(os.getenv("HOPE_AGENT_CHAT_MAX_TURNS", "6"))

    stream_requested = os.getenv("HOPE_AGENT_STREAM_COMPOSER", "true").strip().lower() in {"1", "yes", "true"}
    stream_composer = bool(openai_api_key) and stream_requested

    # Merge whitelist with any extra columns defined via env
    whitelist = {t: list(c) for t, c in DEFAULT_WHITELIST.items()}
    extra_columns = os.getenv("HOPE_AGENT_EXTRA_COLUMNS")
    if extra_columns:
        for pair in extra_columns.split(","):
            if not pair or "." not in pair:
                _LOGGER.warning("Skipping malformed whitelist entry: %s", pair)
                continue
            table, column = [s.strip() for s in pair.split(".", 1)]
            whitelist.setdefault(table, []).append(column)

    chroma_collection = os.getenv("HOPE_AGENT_COLLECTION", "airbnb_reviews_all")

    _cached_config = Config(
        duckdb_path=duckdb_path,
        chroma_dir=chroma_dir,
        reviews_embeddings_path=reviews_embeddings_path,
        reviews_metadata_path=reviews_metadata_path,
        streaming_enabled=streaming_enabled,
        use_metadata_filters=use_metadata_filters,
        openai_model=openai_model,
        openai_fallback_model=openai_fallback_model,
        openai_api_key=openai_api_key,
        top_k_default=top_k_default,
        similarity_min=similarity_min,
        max_rows=max_rows,
        qdrant_url=qdrant_url,
        qdrant_collection=qdrant_collection,
        qdrant_api_key=qdrant_api_key,
        qdrant_timeout_s=qdrant_timeout_s,
        nl2sql_whitelist=whitelist,
        default_years=default_years,
        default_boroughs=default_boroughs,
        telemetry=telemetry,
        chat_max_turns=chat_max_turns,
        stream_composer=stream_composer,
        chroma_collection=chroma_collection,
        tavily_api_key=tavily_api_key,
        tavily_max_results=tavily_max_results,
    )

    _LOGGER.debug(
        "Loaded configuration: duckdb=%s | chroma_dir=%s | qdrant=%s | collection=%s",
        duckdb_path, chroma_dir, qdrant_url, qdrant_collection,
    )
    return _cached_config


__all__ = [
    "Config",
    "load_config",
    "MONTHS_MAP",
    "DATA_SCHEMA",
    "get_month_abbrev",
    "normalize_filters",
]