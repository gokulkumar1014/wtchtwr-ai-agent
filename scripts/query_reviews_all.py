"""Query the full Airbnb reviews Chroma index (all years)."""

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
from typing import Any, Callable, Dict, List, Optional

import numpy as np

import chromadb
from chromadb.config import Settings

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
DEFAULT_PERSIST_DIR = "vec/airbnb_reviews"
DEFAULT_COLLECTION = "airbnb_reviews_all"


np.random.seed(42)

try:
    import torch
except ImportError:  # pragma: no cover
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


def embed_query(text: str) -> np.ndarray:
    embedder = get_embedder()
    return embedder([text])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the Airbnb reviews index (all years).")
    parser.add_argument("--q", required=True, help="Natural language query (required).")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top matches to return.")
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR, help="Chroma persistence directory.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name.")
    parser.add_argument("--borough", action="append", help="Filter by borough (neighbourhood_group).")
    parser.add_argument("--month", action="append", help="Filter by review month (e.g., JAN).")
    parser.add_argument("--year", type=int, action="append", help="Filter by review year.")
    parser.add_argument("--highbury", choices=("yes", "no"), help="Whether to restrict to Highbury reviews.")
    parser.add_argument("--listing-id", type=str, help="Restrict to a specific listing id.")
    parser.add_argument(
        "--include",
        nargs="*",
        default=["metadatas", "documents", "distances"],
        help="Chroma include payload (default: metadatas documents distances).",
    )
    return parser.parse_args()


def build_where(filters: Dict[str, Any]) -> Dict[str, Any]:
    clauses: List[Dict[str, Any]] = []

    def optional(field: str, values: Optional[List[Any]]) -> None:
        if not values:
            return
        if len(values) == 1:
            clauses.append({field: values[0]})
        else:
            clauses.append({"$or": [{field: v} for v in values]})

    optional("year", filters.get("year"))
    optional("month", filters.get("month"))
    optional("borough", filters.get("borough"))

    highbury = filters.get("highbury")
    if highbury == "yes":
        clauses.append({"is_highbury": True})
    elif highbury == "no":
        clauses.append({"is_highbury": False})

    listing_id = filters.get("listing_id")
    if listing_id:
        clauses.append({"listing_id": str(listing_id)})

    if not clauses:
        return {}
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


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
        collection = client.get_collection(args.collection)
    except Exception as exc:
        raise SystemExit(
            f"Collection '{args.collection}' not found at {persist_dir}. Build the index first."
        ) from exc

    query_embedding = embed_query(args.q)
    if query_embedding.shape != (1, EMBED_DIM):
        raise SystemExit("Unexpected query embedding shape.")

    filters = {
        "borough": args.borough or [],
        "month": args.month or [],
        "year": args.year or [],
        "highbury": args.highbury,
        "listing_id": args.listing_id,
    }
    where = build_where(filters)

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=args.top_k,
        where=where or None,
        include=args.include,
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
        f"{'Rank':<5}{'Similarity':<12}{'Listing':<10}{'Year':<8}{'Month':<8}"
        f"{'Borough':<18}{'Highbury':<10}{'Comment ID':<15}Snippet"
    )
    print(header)
    print("-" * len(header))

    for rank, (distance, metadata, doc) in enumerate(zip(distances, metadatas, documents), start=1):
        similarity = format_similarity(distance)
        listing_id = metadata.get("listing_id", "?")
        year = metadata.get("year", "?")
        month = metadata.get("month", "?")
        borough = metadata.get("borough", "?")
        is_highbury = metadata.get("is_highbury", False)
        comment_id = metadata.get("comment_id", "?")
        snippet = truncate(doc or "")
        print(
            f"{rank:<5}{similarity:<12.3f}{listing_id:<10}{year:<8}{month:<8}{borough:<18}{str(is_highbury):<10}{comment_id:<15}{snippet}"
        )


if __name__ == "__main__":  # pragma: no cover
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
