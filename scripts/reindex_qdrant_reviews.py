import logging
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("reindex_qdrant_reviews")

COLLECTION_NAME = "airbnb_reviews"
EMB_PATH = Path("vec/airbnb_reviews/reviews_embeddings.npy")
META_PATH = Path("vec/airbnb_reviews/reviews_metadata.parquet")
VECTOR_SIZE = 384
BATCH_SIZE = 512


MONTH_MAP = {
    "JAN": "JAN",
    "FEB": "FEB",
    "MAR": "MAR",
    "APR": "APR",
    "MAY": "MAY",
    "JUN": "JUN",
    "JUL": "JUL",
    "AUG": "AUG",
    "SEP": "SEP",
    "OCT": "OCT",
    "NOV": "NOV",
    "DEC": "DEC",
}


def _normalize_str(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_month(value) -> str | None:
    text = _normalize_str(value)
    if not text:
        return None
    text = text.upper()[:3]
    return MONTH_MAP.get(text, text)


def _normalize_title(value) -> str | None:
    text = _normalize_str(value)
    return text.title() if text else None


def _normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def load_metadata() -> pd.DataFrame:
    LOGGER.info("Loading metadata from %s", META_PATH)
    df = pd.read_parquet(META_PATH)

    def _get_col(name: str, fallback: str | None = None):
        if name in df.columns:
            return df[name]
        if fallback and fallback in df.columns:
            LOGGER.warning("Column %s missing; using %s", name, fallback)
            return df[fallback]
        LOGGER.warning("Column %s missing; filling empty values", name)
        return pd.Series([None] * len(df), index=df.index)

    borough_series = _get_col("borough", fallback="neighbourhood_group")
    neighbourhood_group_series = _get_col("neighbourhood_group")
    neighbourhood_series = _get_col("neighbourhood")

    df = df.assign(
        listing_id=_get_col("listing_id").astype("int64"),
        comment_id=_get_col("comment_id").astype("int64"),
        year=_get_col("year").astype("int64"),
        month=_get_col("month").apply(_normalize_month),
        borough=borough_series.apply(_normalize_title),
        neighbourhood=neighbourhood_series.apply(_normalize_title),
        neighbourhood_group=neighbourhood_group_series.apply(_normalize_title),
        is_highbury=_get_col("is_highbury").apply(_normalize_bool),
        comments=_get_col("comments").fillna(""),
    )
    return df


def load_embeddings() -> np.ndarray:
    LOGGER.info("Loading embeddings from %s", EMB_PATH)
    embeddings = np.load(EMB_PATH)
    if embeddings.shape[1] != VECTOR_SIZE:
        raise ValueError(f"Expected embeddings with dim {VECTOR_SIZE}, got {embeddings.shape[1]}")
    return embeddings.astype(np.float32)


def prepare_client() -> QdrantClient:
    client = QdrantClient(url="http://localhost:6333", timeout=120.0)
    if client.collection_exists(COLLECTION_NAME):
        LOGGER.info("Deleting existing collection %s", COLLECTION_NAME)
        client.delete_collection(COLLECTION_NAME)
    LOGGER.info("Creating collection %s", COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
        on_disk_payload=True,
    )
    return client


def batch_iter(df: pd.DataFrame, embeddings: np.ndarray) -> Iterator[Tuple[pd.DataFrame, np.ndarray]]:
    total = len(df)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        yield df.iloc[start:end], embeddings[start:end]


def upsert_batches(client: QdrantClient, df: pd.DataFrame, embeddings: np.ndarray) -> None:
    total = len(df)
    inserted = 0
    for batch_df, batch_vecs in batch_iter(df, embeddings):
        points = []
        for (_, row), vec in zip(batch_df.iterrows(), batch_vecs):
            points.append(
                models.PointStruct(
                    id=int(row.comment_id),
                    vector=vec.tolist(),
                    payload={
                        "listing_id": int(row.listing_id),
                        "comment_id": int(row.comment_id),
                        "year": int(row.year),
                        "month": row.month,
                        "borough": row.borough,
                        "neighbourhood": row.neighbourhood,
                        "neighbourhood_group": row.neighbourhood_group,
                        "is_highbury": bool(row.is_highbury),
                        "comments": row.comments,
                    },
                )
            )
        client.upsert(collection_name=COLLECTION_NAME, wait=True, points=points)
        inserted += len(points)
        LOGGER.info("Upserted %d/%d", inserted, total)


def verify(client: QdrantClient, df: pd.DataFrame) -> None:
    listing_id = int(df.iloc[0].listing_id)
    borough = "Manhattan"
    year = int(df.iloc[0].year)
    month = df.iloc[0].month
    dummy_vec = [0.0] * VECTOR_SIZE

    def _search(filter_obj: models.Filter, label: str):
        res = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=dummy_vec,
            query_filter=filter_obj,
            limit=5,
            with_payload=True,
        )
        if not res:
            raise RuntimeError(f"Verification {label} returned no hits")
        LOGGER.info("Verification %s hits: %d", label, len(res))
        LOGGER.info("First payload: %s", res[0].payload)

    _search(
        models.Filter(must=[models.FieldCondition(key="listing_id", match=models.MatchValue(value=listing_id))]),
        "listing_id",
    )
    _search(
        models.Filter(must=[models.FieldCondition(key="borough", match=models.MatchValue(value=borough))]),
        "borough",
    )
    _search(
        models.Filter(
            must=[
                models.FieldCondition(key="listing_id", match=models.MatchValue(value=listing_id)),
                models.FieldCondition(key="year", match=models.MatchValue(value=year)),
                models.FieldCondition(key="month", match=models.MatchValue(value=month)),
            ]
        ),
        "listing/year/month",
    )


def main() -> None:
    metadata = load_metadata()
    embeddings = load_embeddings()
    if len(metadata) != len(embeddings):
        raise RuntimeError(
            f"Embeddings count {len(embeddings)} != metadata count {len(metadata)}"
        )

    client = prepare_client()
    upsert_batches(client, metadata, embeddings)
    verify(client, metadata)
    LOGGER.info("Reindex complete — %d points ingested", len(metadata))


if __name__ == "__main__":
    main()
