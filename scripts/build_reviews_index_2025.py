"""
requirements: sentence-transformers, chromadb, pandas, numpy, tqdm, pyarrow|fastparquet

Usage examples:
  python scripts/build_reviews_index_2025.py --reset --limit 500000
"""

from __future__ import annotations

import os
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import argparse
import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings


try:
    import torch
except ImportError:  # pragma: no cover - torch is installed with sentence-transformers
    torch = None  # type: ignore[assignment]

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
COLLECTION_NAME = "reviews_2025_minilm"
BATCH_SIZE = 256


np.random.seed(42)
if torch is not None:
    torch.manual_seed(42)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def get_embedder() -> Callable[[List[str]], np.ndarray]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - fail fast for missing deps
        raise SystemExit("sentence-transformers is required to run this script") from exc

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    def _embed(texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, EMBED_DIM), dtype=np.float32)
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        embeddings = embeddings / norms
        return embeddings.astype(np.float32)

    return _embed


def embed_texts(texts: List[str]) -> np.ndarray:
    embedder = get_embedder()
    return embedder(texts)


def ensure_parquet(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Parquet file not found: {path}")


def load_reviews(limit: int | None) -> pd.DataFrame:
    data_path = repo_root() / "data" / "clean" / "reviews_enriched.parquet"
    ensure_parquet(data_path)

    df = pd.read_parquet(data_path)
    df = df[df["year"] == 2025].copy()
    if df.empty:
        raise SystemExit("No reviews found for year 2025 in the parquet file.")

    if limit is not None:
        df = df.head(limit).copy()

    df.reset_index(drop=True, inplace=True)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Chroma index of Airbnb review comments for 2025.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of rows to ingest.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the target Chroma collection before ingesting.",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="vec/airbnb_reviews_2025",
        help="Directory for the Chroma persistence layer (relative to repo root).",
    )
    return parser.parse_args()


def prepare_collection(client: chromadb.PersistentClient, reset: bool) -> Collection:
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Dropped existing collection '{COLLECTION_NAME}'.")
        except Exception:
            pass  # Collection might not exist yet.

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def chunk_indices(total: int, chunk_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        yield start, end


def clean_id(raw_id: pd.Series) -> str | None:
    if pd.isna(raw_id):
        return None
    try:
        return str(int(raw_id))
    except (TypeError, ValueError):
        return str(raw_id)


def main() -> None:
    args = parse_args()

    df = load_reviews(limit=args.limit)

    destination = repo_root() / args.persist_dir
    destination.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(destination), settings=Settings(anonymized_telemetry=False))
    collection = prepare_collection(client, reset=args.reset)

    get_embedder()  # Prime the model before iterating.

    selected_rows = len(df)
    rows_ingested = 0
    skipped_null_text = 0
    skipped_id_issues = 0

    seen_ids: set[str] = set()

    iterator = tqdm(
        chunk_indices(selected_rows, BATCH_SIZE),
        total=math.ceil(selected_rows / BATCH_SIZE),
        desc="Indexing reviews",
    )

    for start, end in iterator:
        batch = df.iloc[start:end]
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[dict] = []

        for idx, row in batch.iterrows():
            comment_id = clean_id(row.get("comment_id"))
            if not comment_id:
                skipped_id_issues += 1
                continue
            if comment_id in seen_ids:
                skipped_id_issues += 1
                continue

            text = row.get("comments")
            if pd.isna(text) or not str(text).strip():
                skipped_null_text += 1
                continue

            ids.append(comment_id)
            documents.append(str(text))
            metadatas.append(
                {
                    "listing_id": None if pd.isna(row.get("listing_id")) else int(row.get("listing_id")),
                    "comment_id": None if pd.isna(row.get("comment_id")) else int(row.get("comment_id")),
                    "year": int(row.get("year")) if not pd.isna(row.get("year")) else None,
                    "month": None if pd.isna(row.get("month")) else str(row.get("month")),
                    "neighbourhood": None if pd.isna(row.get("neighbourhood")) else str(row.get("neighbourhood")),
                    "neighbourhood_group": None if pd.isna(row.get("neighbourhood_group")) else str(row.get("neighbourhood_group")),
                    "is_highbury": None if pd.isna(row.get("is_highbury")) else bool(row.get("is_highbury")),
                }
            )

        if not ids:
            continue

        existing = set()
        if not args.reset:
            try:
                existing = set(collection.get(ids=ids, include=[]).get("ids", []))
            except Exception:
                existing = set()

        if existing:
            filtered_ids: List[str] = []
            filtered_documents: List[str] = []
            filtered_metadatas: List[dict] = []
            for cid, doc, meta in zip(ids, documents, metadatas):
                if cid in existing:
                    seen_ids.add(cid)
                    skipped_id_issues += 1
                    continue
                filtered_ids.append(cid)
                filtered_documents.append(doc)
                filtered_metadatas.append(meta)
                seen_ids.add(cid)
            ids, documents, metadatas = filtered_ids, filtered_documents, filtered_metadatas
        else:
            seen_ids.update(ids)

        if not ids:
            continue

        embeddings = embed_texts(documents)
        if embeddings.shape[1] != EMBED_DIM:
            raise SystemExit(f"Unexpected embedding dimension {embeddings.shape[1]} (expected {EMBED_DIM}).")

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings.tolist(),
        )
        rows_ingested += len(ids)

        iterator.set_postfix_str(f"ingested={rows_ingested}")

    # no client.persist() needed

    print(f"Rows selected: {selected_rows}")
    print(f"Rows ingested: {rows_ingested}")
    print(f"Rows skipped (null comments): {skipped_null_text}")
    print(f"Rows skipped (missing/duplicate/already indexed ids): {skipped_id_issues}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Aborted by user.")
