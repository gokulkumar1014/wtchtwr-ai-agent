#!/usr/bin/env python3
"""
Load listings into DuckDB.

- Creates/overwrites db/airbnb.duckdb
- CREATE OR REPLACE TABLE listings           FROM data/clean/listings_cleaned.parquet
- CREATE OR REPLACE TABLE highbury_listings  FROM data/clean/highbury_listings.parquet
- CREATE OR REPLACE VIEW  highbury_from_main AS SELECT * FROM listings WHERE host_name='Highbury'
"""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DB_NAME = "airbnb.duckdb"


def repo_root() -> Path:
    """Find repository root (folder that contains /data)."""
    p = Path(__file__).resolve().parent
    while not (p / "data").exists() and p != p.parent:
        p = p.parent
    return p


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Expected input file: {path}")
    return path


def main() -> None:
    root = repo_root()
    clean_dir = root / "data" / "clean"
    db_dir = (root / "db")
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / DB_NAME

    listings_parquet  = require_file(clean_dir / "listings_cleaned.parquet")
    highbury_parquet  = require_file(clean_dir / "highbury_listings.parquet")

    # Start from a clean DB file (safe because we fully rebuild tables below)
    if db_path.exists():
        logging.info("Removing existing DB: %s", db_path)
        db_path.unlink()

    logging.info("Connecting to DuckDB: %s", db_path)
    con = duckdb.connect(str(db_path))

    try:
        # Load main listings
        logging.info("Creating table 'listings' from %s", listings_parquet.name)
        con.execute(
            """
            CREATE OR REPLACE TABLE listings AS
            SELECT * FROM read_parquet(?)
            """,
            [str(listings_parquet)],
        )

        # Load Highbury subset (pre-extracted file)
        logging.info("Creating table 'highbury_listings' from %s", highbury_parquet.name)
        con.execute(
            """
            CREATE OR REPLACE TABLE highbury_listings AS
            SELECT * FROM read_parquet(?)
            """,
            [str(highbury_parquet)],
        )

        # Convenience view: slice of main table by the normalized host name
        logging.info("Creating view 'highbury_from_main' (host_name = 'Highbury')")
        con.execute(
            """
            CREATE OR REPLACE VIEW highbury_from_main AS
            SELECT * FROM listings
            WHERE host_name = 'Highbury'
            """
        )

        # Sanity counts
        n_listings  = con.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
        n_highbury1 = con.execute("SELECT COUNT(*) FROM highbury_listings").fetchone()[0]
        n_highbury2 = con.execute("SELECT COUNT(*) FROM highbury_from_main").fetchone()[0]

        logging.info("Rows -> listings: %s | highbury_listings: %s | highbury_from_main: %s",
                     n_listings, n_highbury1, n_highbury2)

    finally:
        con.close()
        logging.info("DuckDB connection closed.")


if __name__ == "__main__":
    main()
