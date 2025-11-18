#!/usr/bin/env python3
"""
Create or refresh a neighbourhood lookup table derived from listings_cleaned.

The resulting table `neighbourhood_lookup` contains unique pairs of
neighbourhood_group and neighbourhood, sorted alphabetically, and can be
used by NL→SQL routing or UI filters.
"""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from amenity_utils import DB_PATH


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"DuckDB database not found at {DB_PATH}. "
            "Run scripts/load_duckdb.py first."
        )

    with duckdb.connect(str(DB_PATH)) as con:
        logging.info("Rebuilding neighbourhood_lookup table from listings_cleaned.")
        con.execute("DROP TABLE IF EXISTS neighbourhood_lookup")
        con.execute(
            """
            CREATE TABLE neighbourhood_lookup AS
            SELECT DISTINCT
                neighbourhood_group,
                neighbourhood
            FROM listings_cleaned
            WHERE neighbourhood_group IS NOT NULL
              AND neighbourhood IS NOT NULL
            ORDER BY neighbourhood_group, neighbourhood
            """
        )
        logging.info(
            "neighbourhood_lookup now contains %s rows.",
            con.execute("SELECT COUNT(*) FROM neighbourhood_lookup").fetchone()[0],
        )


if __name__ == "__main__":
    main()
