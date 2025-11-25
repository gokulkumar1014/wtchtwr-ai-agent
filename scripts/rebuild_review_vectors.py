"""
Rebuild the Airbnb reviews vector store and Qdrant collection from scratch.

Steps performed:
1. Load and normalize sentiment-scored reviews.
2. Generate SentenceTransformer embeddings for each comment.
3. Persist refreshed embeddings/metadata artifacts.
4. Recreate the Qdrant collection and ingest points in batches.
"""

from __future__ import annotations

import logging
import math
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.config import MONTHS_MAP, load_config

LOGGER = logging.getLogger("rebuild_review_vectors")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIMENSION = 384
EMBED_BATCH_SIZE = 1024
QDRANT_UPLOAD_BATCH_SIZE = 2000

DATASET_PATH = Path("data/clean/review_sentiment_scores.parquet")
QDRANT_STORAGE_ROOT = Path("qdrant_storage/collections")

METADATA_COLUMNS = [
    "comment_id",
    "listing_id",
    "comments",
    "year",
    "month",
    "neighbourhood",
    "neighbourhood_group",
    "is_highbury",
    "negative",
    "neutral",
    "positive",
    "compound",
    "sentiment_label",
]

FLOAT_COLUMNS = ["negative", "neutral", "positive", "compound"]


@dataclass(frozen=True)
class Paths:
    embeddings: Path
    metadata: Path
    qdrant_collection_dir: Path
    data: Path


def _resolve_paths(cfg) -> Paths:
    def _normalize_path(path_value: Path) -> Path:
        path = Path(path_value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    embeddings_path = _normalize_path(cfg.reviews_embeddings_path)
    metadata_path = _normalize_path(cfg.reviews_metadata_path)
    qdrant_collection_dir = PROJECT_ROOT / QDRANT_STORAGE_ROOT / cfg.qdrant_collection
    data_path = PROJECT_ROOT / DATASET_PATH

    return Paths(
        embeddings=embeddings_path,
        metadata=metadata_path,
        qdrant_collection_dir=qdrant_collection_dir,
        data=data_path,
    )


def _normalize_month_value(value: str | None) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    mapped = MONTHS_MAP.get(lowered)
    if mapped:
        return mapped
    return text[:3].upper()


def _cleanup_previous_outputs(paths: Paths, preserve_artifacts: bool = False) -> None:
    for file_path in (paths.embeddings, paths.metadata):
        if preserve_artifacts:
            LOGGER.info("Preserving existing file %s", file_path)
            continue
        if file_path.exists():
            LOGGER.info("Deleting existing file %s", file_path)
            file_path.unlink()

    if paths.qdrant_collection_dir.exists():
        LOGGER.info("Removing Qdrant storage directory %s", paths.qdrant_collection_dir)
        try:
            shutil.rmtree(paths.qdrant_collection_dir, onerror=_handle_remove_readonly)
        except PermissionError:
            LOGGER.warning(
                "Permission denied removing %s; retrying with sudo", paths.qdrant_collection_dir
            )
            _remove_dir_with_sudo(paths.qdrant_collection_dir)


def _handle_remove_readonly(func, path, exc_info):
    exc = exc_info[1]
    if isinstance(exc, PermissionError):
        path_obj = Path(path)
        try:
            path_obj.chmod(stat.S_IRWXU)
        except Exception:
            pass
        func(path)
    else:
        raise exc


def _remove_dir_with_sudo(target: Path) -> None:
    try:
        subprocess.run(["sudo", "rm", "-rf", str(target)], check=True)
    except FileNotFoundError as err:
        raise PermissionError(
            f"Unable to delete {target}: sudo not available; remove manually or fix ownership."
        ) from err
    except subprocess.CalledProcessError as err:
        raise PermissionError(
            f"Sudo removal of {target} failed; please delete it manually."
        ) from err
    if target.exists():
        raise PermissionError(
            f"Failed to delete {target}; ensure you have ownership or remove manually."
        )


def _load_and_prepare_metadata(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Source data not found at {data_path}")

    LOGGER.info("Loading sentiment-scored reviews from %s", data_path)
    df = pd.read_parquet(data_path)

    missing_columns = [col for col in METADATA_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.copy()

    df["listing_id"] = pd.to_numeric(df["listing_id"], errors="raise").astype("int64")
    df["comment_id"] = pd.to_numeric(df["comment_id"], errors="raise").astype("int64")
    df["year"] = pd.to_numeric(df["year"], errors="raise").astype("int64")

    df["month"] = df["month"].apply(_normalize_month_value).fillna("UNK")

    df["comments"] = df["comments"].fillna("").astype(str)
    df["neighbourhood"] = df["neighbourhood"].fillna("").astype(str)
    df["neighbourhood_group"] = df["neighbourhood_group"].fillna("").astype(str)
    df["sentiment_label"] = df["sentiment_label"].fillna("").astype(str)

    if "is_highbury" not in df.columns:
        raise ValueError("is_highbury column missing from dataset")
    df["is_highbury"] = pd.to_numeric(df["is_highbury"], errors="coerce").fillna(0).astype(int).astype(bool)

    for col in FLOAT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    metadata = df[METADATA_COLUMNS].copy()
    metadata["month"] = metadata["month"].astype(str)

    return metadata


def _encode_comments(texts: Iterable[str]) -> np.ndarray:
    model = SentenceTransformer(MODEL_NAME)
    LOGGER.info("Encoding %d comments with %s", len(texts), MODEL_NAME)
    embeddings = model.encode(texts, batch_size=EMBED_BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True)
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    return embeddings


def _prepare_qdrant_client(cfg) -> QdrantClient:
    LOGGER.info("Connecting to Qdrant at %s", cfg.qdrant_url)
    return QdrantClient(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key or None,
        timeout=cfg.qdrant_timeout_s,
    )


def _collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """
    Older qdrant-client releases lacked collection_exists() so we fallback to get_collection().
    """
    if hasattr(client, "collection_exists"):
        return client.collection_exists(collection_name=collection_name)
    try:
        client.get_collection(collection_name=collection_name)
        return True
    except UnexpectedResponse as exc:
        message = str(exc).lower()
        if "not found" in message or "doesn't exist" in message or getattr(exc, "status_code", None) == 404:
            return False
        LOGGER.warning("get_collection failed; assuming collection exists: %s", exc)
        return True
    
def _recreate_collection(client: QdrantClient, collection_name: str) -> None:
    LOGGER.info("Recreating Qdrant collection %s", collection_name)
    try:
        exists = _collection_exists(client, collection_name)
    except UnexpectedResponse as exc:
        LOGGER.warning("collection_exists failed: %s", exc)
        exists = True

    if exists:
        try:
            client.delete_collection(collection_name=collection_name)
        except UnexpectedResponse as exc:
            if "No such file or directory" in str(exc):
                LOGGER.warning("Delete reported missing storage, continuing: %s", exc)
            else:
                raise

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=VECTOR_DIMENSION, distance=models.Distance.COSINE),
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
        on_disk_payload=True,
    )


def _upload_batches(
    client: QdrantClient,
    collection_name: str,
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
) -> None:
    total = len(metadata)
    for start in range(0, total, QDRANT_UPLOAD_BATCH_SIZE):
        end = min(start + QDRANT_UPLOAD_BATCH_SIZE, total)
        batch_meta = metadata.iloc[start:end]
        batch_vectors = embeddings[start:end]
        points = []
        for (row_index, row), vector in zip(batch_meta.iterrows(), batch_vectors):
            payload = {
                "listing_id": int(row.listing_id),
                "comment_id": int(row.comment_id),
                "comments": row.comments,
                "year": int(row.year),
                "month": row.month,
                "neighbourhood": row.neighbourhood,
                "neighbourhood_group": row.neighbourhood_group,
                "is_highbury": bool(row.is_highbury),
                "negative": float(row.negative),
                "neutral": float(row.neutral),
                "positive": float(row.positive),
                "compound": float(row.compound),
                "sentiment_label": row.sentiment_label,
            }
            points.append(
                models.PointStruct(
                    id=int(row.comment_id),
                    vector=vector.tolist(),
                    payload=payload,
                )
            )
        LOGGER.info("Upserting points %d-%d/%d", start + 1, end, total)
        client.upsert(collection_name=collection_name, wait=True, points=points)


def _save_artifacts(embeddings: np.ndarray, metadata: pd.DataFrame, paths: Paths) -> None:
    paths.embeddings.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing embeddings → %s", paths.embeddings)
    np.save(paths.embeddings, embeddings)

    LOGGER.info("Writing metadata → %s", paths.metadata)
    metadata.to_parquet(paths.metadata, index=False)


def _print_summary(metadata: pd.DataFrame, embeddings: np.ndarray, collection_name: str, qdrant_total: int) -> None:
    print(f"Rows processed: {len(metadata):,}")
    print(f"Embeddings shape: {embeddings.shape}")
    print("Sentiment label counts:")
    for label, count in metadata["sentiment_label"].value_counts().items():
        print(f"  {label}: {count:,}")
    print(f"Qdrant upload complete: collection '{collection_name}' now contains {qdrant_total:,} points")


def _load_existing_artifacts(paths: Paths) -> Tuple[pd.DataFrame, np.ndarray]:
    if not paths.metadata.exists():
        raise FileNotFoundError(f"Metadata parquet not found at {paths.metadata}")
    if not paths.embeddings.exists():
        raise FileNotFoundError(f"Embeddings file not found at {paths.embeddings}")
    LOGGER.info("Loading existing metadata from %s", paths.metadata)
    metadata = pd.read_parquet(paths.metadata)
    LOGGER.info("Loading existing embeddings from %s", paths.embeddings)
    embeddings = np.load(paths.embeddings)
    return metadata, embeddings


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Rebuild Airbnb review vectors and Qdrant collection.")
    parser.add_argument(
        "--reuse-artifacts",
        action="store_true",
        help="Reuse existing embeddings/metadata artifacts instead of regenerating them.",
    )
    args = parser.parse_args()

    cfg = load_config(refresh=True)
    paths = _resolve_paths(cfg)

    _cleanup_previous_outputs(paths, preserve_artifacts=args.reuse_artifacts)

    if args.reuse_artifacts:
        metadata, embeddings = _load_existing_artifacts(paths)
    else:
        metadata = _load_and_prepare_metadata(paths.data)
        embeddings = _encode_comments(metadata["comments"].tolist())
        if embeddings.shape[0] != len(metadata):
            raise RuntimeError("Embeddings count does not match metadata rows")
        _save_artifacts(embeddings, metadata, paths)

    client = _prepare_qdrant_client(cfg)
    _recreate_collection(client, cfg.qdrant_collection)
    _upload_batches(client, cfg.qdrant_collection, metadata, embeddings)

    final_count = client.count(collection_name=cfg.qdrant_collection, exact=True).count
    if final_count != len(metadata):
        raise RuntimeError(
            f"Qdrant count {final_count} does not match metadata rows {len(metadata)}"
        )
    _print_summary(metadata, embeddings, cfg.qdrant_collection, final_count)


if __name__ == "__main__":
    main()
