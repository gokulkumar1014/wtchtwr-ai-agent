#!/usr/bin/env python3
"""
Create and maintain the amenities_raw table and remove denormalized columns.

Steps performed:
1. Ensure the DuckDB database exists and contains listings_cleaned / highbury_listings.
2. Create `amenities_raw` (listing_id, amenities_text) if it does not already exist.
3. Populate the table from listings_cleaned (source of truth).
4. Drop the `amenities` column from listings tables to keep schemas lean.
5. Create compatibility views `<table>_with_amenities` if consumers still require it.
"""
from __future__ import annotations

import logging

import duckdb

from amenity_utils import DB_PATH


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return (
        con.execute(
            """
            SELECT COUNT(*) > 0
            FROM information_schema.tables
            WHERE table_schema = 'main'
              AND table_name = ?
            """,
            [table.lower()],
        ).fetchone()[0]
    )


def column_exists(con: duckdb.DuckDBPyConnection, table: str, column: str) -> bool:
    rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
    return any(row[1] == column for row in rows)


def create_amenities_raw(con: duckdb.DuckDBPyConnection) -> None:
    logging.info("Creating table amenities_raw from listings_cleaned amenities column.")
    con.execute("DROP TABLE IF EXISTS amenities_raw")
    con.execute(
        """
        CREATE TABLE amenities_raw (
            listing_id BIGINT,
            amenities_text TEXT
        )
        """
    )
    con.execute(
        """
        INSERT INTO amenities_raw (listing_id, amenities_text)
        SELECT DISTINCT
            listings_id AS listing_id,
            amenities AS amenities_text
        FROM listings_cleaned
        """
    )

    try:
        con.execute(
            "ALTER TABLE amenities_raw ADD CONSTRAINT amenities_raw_pk PRIMARY KEY (listing_id)"
        )
    except duckdb.NotImplementedException:
        logging.info(
            "DuckDB lacks ALTER TABLE ADD PRIMARY KEY; ensuring uniqueness via UNIQUE INDEX."
        )
        con.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS amenities_raw_listing_idx ON amenities_raw(listing_id)"
        )


def drop_column_if_present(con: duckdb.DuckDBPyConnection, table: str, column: str) -> None:
    if column_exists(con, table, column):
        logging.info("Dropping column %s.%s", table, column)
        con.execute(f'ALTER TABLE {table} DROP COLUMN "{column}"')


def drop_view_if_exists(con: duckdb.DuckDBPyConnection, view_name: str) -> None:
    logging.info("Dropping view if exists: %s", view_name)
    con.execute(f"DROP VIEW IF EXISTS {view_name}")


def drop_dependent_views(con: duckdb.DuckDBPyConnection, table: str) -> None:
    """Remove any views that reference the specified table name."""
    rows = con.execute(
        """
        SELECT table_name
        FROM information_schema.views
        WHERE table_schema = 'main'
          AND view_definition ILIKE '%' || ? || '%'
        """,
        [table],
    ).fetchall()
    for (view_name,) in rows:
        drop_view_if_exists(con, view_name)


def create_view_with_amenities(con: duckdb.DuckDBPyConnection, base_table: str) -> None:
    view_name = f"{base_table}_with_amenities"
    logging.info("Creating compatibility view %s", view_name)
    con.execute(
        f"""
        CREATE OR REPLACE VIEW {view_name} AS
        SELECT b.*, r.amenities_text AS amenities
        FROM {base_table} b
        LEFT JOIN amenities_raw r
          ON b.listings_id = r.listing_id
        """
    )


def create_highbury_view(con: duckdb.DuckDBPyConnection) -> None:
    logging.info("Creating view highbury_from_main")
    con.execute(
        """
        CREATE OR REPLACE VIEW highbury_from_main AS
        SELECT *
        FROM listings_cleaned
        WHERE host_name = 'Highbury'
        """
    )


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"DuckDB database not found at {DB_PATH}. "
            "Run scripts/load_duckdb.py before setting up amenities."
        )

    con = duckdb.connect(str(DB_PATH))

    try:
        required_tables = ("listings_cleaned", "highbury_listings")
        for table in required_tables:
            if not table_exists(con, table):
                raise RuntimeError(
                    f"Expected table '{table}' not found in DuckDB. "
                    "Make sure scripts/load_duckdb.py has been executed."
                )

        if not table_exists(con, "amenities_raw"):
            create_amenities_raw(con)
        else:
            logging.info("Table amenities_raw already exists; skipping recreation.")

        rebuild_required = False
        drop_view_if_exists(con, "highbury_from_main")
        drop_view_if_exists(con, "listings_cleaned_with_amenities")
        drop_view_if_exists(con, "highbury_listings_with_amenities")
        drop_dependent_views(con, "listings_cleaned")
        drop_dependent_views(con, "highbury_listings")

        try:
            drop_column_if_present(con, "listings_cleaned", "amenities")
            drop_column_if_present(con, "highbury_listings", "amenities")
        except duckdb.DependencyException:
            logging.warning(
                "Unable to drop amenities column due to dependencies; will rebuild database from source."
            )
            rebuild_required = True

        if rebuild_required:
            logging.info("Closing connection before rebuild.")
            con.close()
            from load_duckdb import main as reload_db

            logging.info("Rebuilding DuckDB database via load_duckdb.py")
            reload_db()
            logging.info("Rebuild complete. Amenities column removed during load.")
            return

        create_view_with_amenities(con, "listings_cleaned")
        create_view_with_amenities(con, "highbury_listings")
        create_highbury_view(con)

        logging.info("amenities_raw setup complete.")
    finally:
        try:
            con.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
