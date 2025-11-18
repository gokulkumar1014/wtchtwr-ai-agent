"""
requirements: sentence-transformers, chromadb, pandas, numpy, tqdm, pyarrow|fastparquet

Usage examples:
  python scripts/query_reviews_2025.py --q "noise complaints about street" --top-k 8 --borough Manhattan
  python scripts/query_reviews_2025.py --q "why guests love parking" --highbury yes --month JUL
"""

from __future__ import annotations

import os
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import argparse
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

import chromadb
from chromadb.config import Settings


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
COLLECTION_NAME = "reviews_2025_minilm"


np.random.seed(42)

try:
    import torch
except ImportError:  # pragma: no cover - torch ships with sentence-transformers
    torch = None  # type: ignore[assignment]

if torch is not None:
    torch.manual_seed(42)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def get_embedder() -> Callable[[List[str]], np.ndarray]:
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
            batch_size=256,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the 2025 Airbnb reviews Chroma index.",
    )
    parser.add_argument(
        "--q",
        required=True,
        help="Natural language query string (required).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top matches to return (default: 5).",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="vec/airbnb_reviews_2025",
        help="Directory where the Chroma index is stored (relative to repo root).",
    )
    parser.add_argument(
        "--borough",
        type=str,
        help="Filter by neighbourhood group (borough).",
    )
    parser.add_argument(
        "--month",
        type=str,
        help="Filter by review month (e.g., JAN, FEB).",
    )
    parser.add_argument(
        "--highbury",
        choices=("yes", "no"),
        help="Filter by Highbury flag (yes/no).",
    )
    parser.add_argument(
        "--listing-id",
        type=int,
        help="Filter by listing id.",
    )
    return parser.parse_args()


def build_where_filter(
    borough: Optional[str],
    month: Optional[str],
    highbury: Optional[str],
    listing_id: Optional[int],
) -> Optional[dict]:
    where: dict = {}
    if borough:
        where["neighbourhood_group"] = borough
    if month:
        where["month"] = month
    if highbury:
        where["is_highbury"] = highbury.lower() == "yes"
    if listing_id is not None:
        where["listing_id"] = listing_id
    return where or None


def truncate(text: str, length: int = 160) -> str:
    clean = " ".join(text.split())
    if len(clean) <= length:
        return clean
    return clean[: length - 1].rstrip() + "…"


def format_similarity(distance: float) -> float:
    similarity = 1.0 - distance
    return max(min(similarity, 1.0), -1.0)


def main() -> None:
    args = parse_args()

    persist_dir = repo_root() / args.persist_dir
    if not persist_dir.exists():
        raise SystemExit(f"Persist directory not found: {persist_dir}")

    client = chromadb.PersistentClient(path=str(persist_dir), settings=Settings(anonymized_telemetry=False))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise SystemExit(
            f"Collection '{COLLECTION_NAME}' not found at {persist_dir}. Build the index first."
        ) from exc

    where = build_where_filter(args.borough, args.month, args.highbury, args.listing_id)

    query_embedding = embed_texts([args.q])
    if query_embedding.shape != (1, EMBED_DIM):
        raise SystemExit("Unexpected query embedding shape.")

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=args.top_k,
        where=where,
        include=["metadatas", "documents", "distances"],
    )

    ids = results.get("ids", [[]])
    if not ids or not ids[0]:
        print("No matching reviews found. Try relaxing your filters or rebuilding the index.")
        client.persist()
        return

    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    documents = results.get("documents", [[]])[0]

    print(f"Top {len(ids[0])} results for query: {args.q!r}")
    header = (
        f"{'Rank':<5}{'Similarity':<12}{'Listing':<10}{'Month':<8}"
        f"{'Borough':<18}{'Highbury':<10}{'Comment ID':<15}Snippet"
    )
    print(header)
    print("-" * len(header))

    for rank, (distance, meta, doc, cid) in enumerate(
        zip(distances, metadatas, documents, ids[0]), start=1
    ):
        similarity = format_similarity(distance)
        listing = meta.get("listing_id") if meta else None
        month = meta.get("month") if meta else ""
        borough = meta.get("neighbourhood_group") if meta else ""
        highbury = meta.get("is_highbury") if meta else ""
        comment_id = meta.get("comment_id") if meta else cid
        snippet = truncate(doc or "")
        print(
            f"{rank:<5}{similarity:<12.4f}{str(listing):<10}{month:<8}"
            f"{str(borough):<18}{str(highbury):<10}{str(comment_id):<15}{snippet}"
        )

    # client.persist()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Aborted by user.")
