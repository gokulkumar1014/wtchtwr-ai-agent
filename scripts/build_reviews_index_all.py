"""Build a ChromaDB index over the full Airbnb reviews corpus.

Example::
    python scripts/build_reviews_index_all.py --reset --batch-size 512
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
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - csv fallback
    pq = None  # type: ignore[assignment]

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
DEFAULT_PERSIST_DIR = "vec/airbnb_reviews"
DEFAULT_COLLECTION = "airbnb_reviews_all"
DEFAULT_BATCH_SIZE = 512


np.random.seed(42)
if torch is not None:
    torch.manual_seed(42)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Chroma index for all Airbnb reviews (no year filter).",
    )
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR, help="Chroma persistence directory.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name.")
    parser.add_argument(
        "--parquet",
        default="data/clean/reviews_enriched.parquet",
        help="Primary parquet source for reviews.",
    )
    parser.add_argument(
        "--csv",
        default="data/clean/reviews_enriched.csv",
        help="CSV fallback when parquet is unavailable.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Embedding batch size.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of rows processed.")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate the collection before ingesting.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Log progress every N ingested rows (default 5000).",
    )
    return parser.parse_args()


def ensure_input(parquet_path: Path, csv_path: Path) -> Path:
    if parquet_path.exists():
        return parquet_path
    if csv_path.exists():
        return csv_path
    raise SystemExit(
        f"Neither parquet nor csv review files were found.\nMissing: {parquet_path} and {csv_path}"
    )


@lru_cache(maxsize=1)
def get_embedder(batch_size: int):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("sentence-transformers is required to run this script") from exc

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    def _embed(texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, EMBED_DIM), dtype=np.float32)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        embeddings = embeddings / norms
        return embeddings.astype(np.float32)

    return _embed


def clean_text(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    text = raw.replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ")
    text = " ".join(text.split())
    return text.strip()


def build_prefix(row: Dict[str, Any]) -> str:
    borough = row.get("neighbourhood_group", "?")
    month = row.get("month", "?")
    year = row.get("year", "?")
    return f"[{borough}|{month}|{year}] "


def iter_source(
    source_path: Path,
    batch_size: int,
    limit: int | None,
) -> Iterable[pd.DataFrame]:
    if source_path.suffix.lower() == ".parquet":
        if pq is None:
            raise SystemExit("pyarrow is required to stream parquet files")
        dataset = pq.ParquetFile(source_path)
        rows_consumed = 0
        for batch in dataset.iter_batches(batch_size=batch_size * 4):
            df = batch.to_pandas()
            if limit is not None:
                remaining = limit - rows_consumed
                if remaining <= 0:
                    break
                if len(df) > remaining:
                    df = df.iloc[:remaining]
            rows_consumed += len(df)
            yield df
            if limit is not None and rows_consumed >= limit:
                break
    else:  # CSV fallback
        chunk_iter = pd.read_csv(source_path, chunksize=batch_size * 4)
        rows_consumed = 0
        for chunk in chunk_iter:
            if limit is not None:
                remaining = limit - rows_consumed
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining]
            rows_consumed += len(chunk)
            yield chunk
            if limit is not None and rows_consumed >= limit:
                break


def prepare_collection(client: chromadb.PersistentClient, name: str, reset: bool) -> Collection:
    if reset:
        try:
            client.delete_collection(name)
            print(f"Dropped existing collection '{name}'.")
        except Exception:
            pass
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def collect_existing_ids(collection: Collection, candidates: List[str]) -> set[str]:
    existing: set[str] = set()
    chunk = 500
    for start in range(0, len(candidates), chunk):
        batch = candidates[start : start + chunk]
        if not batch:
            continue
        result = collection.get(ids=batch)
        ids = (result or {}).get("ids") or []
        existing.update(ids)
    return existing


def count_rows(path: Path, limit: int | None) -> int:
    if limit is not None:
        return limit
    if path.suffix.lower() == ".parquet" and pq is not None:
        return pq.ParquetFile(path).metadata.num_rows
    # Fallback: count CSV lines (minus header)
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle) - 1


def main() -> None:
    args = parse_args()
    root = repo_root()
    parquet_path = root / args.parquet
    csv_path = root / args.csv
    source_path = ensure_input(parquet_path, csv_path)

    embed_batch = max(1, args.batch_size)
    embedder = get_embedder(embed_batch)

    persist_dir = root / args.persist_dir
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir), settings=Settings(anonymized_telemetry=False))
    collection = prepare_collection(client, args.collection, args.reset)

    total_expected = count_rows(source_path, args.limit)
    print(f"Preparing to ingest {total_expected} review rows from {source_path}")

    total_processed = 0
    total_ingested = 0
    total_skipped = 0

    for chunk in iter_source(source_path, embed_batch, args.limit):
        chunk = chunk[[
            "listing_id",
            "comment_id",
            "comments",
            "year",
            "month",
            "neighbourhood",
            "neighbourhood_group",
            "is_highbury",
        ]].copy()

        chunk["comments"] = chunk["comments"].apply(clean_text)
        chunk = chunk[chunk["comments"] != ""]

        candidates: List[str] = []
        row_buffer: List[pd.Series] = []
        for row in chunk.itertuples(index=False):
            listing_id = row.listing_id
            comment_id = row.comment_id
            if pd.isna(listing_id) or pd.isna(comment_id):
                total_skipped += 1
                continue
            listing_str = str(int(listing_id)) if isinstance(listing_id, (int, float)) and not isinstance(listing_id, bool) else str(listing_id)
            comment_str = str(int(comment_id)) if isinstance(comment_id, (int, float)) and not isinstance(comment_id, bool) else str(comment_id)
            if not listing_str or listing_str.lower() == "nan" or not comment_str or comment_str.lower() == "nan":
                total_skipped += 1
                continue
            doc_id = f"{listing_str}:{comment_str}"
            candidates.append(doc_id)
            row_buffer.append(row)

        if not candidates:
            continue

        existing = collect_existing_ids(collection, candidates)
        to_upsert = []
        documents = []
        metadatas: List[Dict[str, Any]] = []

        for row, doc_id in zip(row_buffer, candidates):
            if doc_id in existing:
                total_skipped += 1
                continue
            row_dict = row._asdict()
            prefix = build_prefix(row_dict)
            documents.append(prefix + row.comments)
            metadatas.append(
                {
                    "listing_id": doc_id.split(":", 1)[0],
                    "comment_id": doc_id.split(":", 1)[1],
                    "year": int(row.year) if not pd.isna(row.year) else None,
                    "month": str(row.month) if not pd.isna(row.month) else None,
                    "borough": row.neighbourhood_group,
                    "neighbourhood": row.neighbourhood,
                    "is_highbury": bool(row.is_highbury) if not pd.isna(row.is_highbury) else False,
                }
            )
            to_upsert.append(doc_id)

        if not to_upsert:
            total_processed += len(chunk)
            continue

        embeddings = embedder(documents)
        collection.upsert(ids=to_upsert, metadatas=metadatas, documents=documents, embeddings=embeddings)
        total_ingested += len(to_upsert)
        total_processed += len(chunk)

        if args.progress_every and total_processed % args.progress_every < embed_batch:
            print(
                "Progress: processed=%d ingested=%d skipped=%d"
                % (total_processed, total_ingested, total_skipped)
            )

    final_size = collection.count()
    print(
        f"Ingestion complete. processed={total_processed} ingested={total_ingested} "
        f"skipped={total_skipped} collection_total={final_size}"
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
