from __future__ import annotations

import csv
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent.config import load_config
from agent.nl2sql_llm import generate_sql_from_question

cfg = load_config()
DB_PATH = Path(cfg.duckdb_path)



MAX_PREVIEW_ROWS = 500
MAX_EXPORT_ROWS = 20000
DEFAULT_LIMIT = 200
DATA_DICTIONARY_PATH = Path("config/data_dictionary.csv")

# Tables that are safe to expose via the explorer along with the column(s) that
# link back to listings_cleaned. This keeps joins deterministic and prevents
# arbitrary SQL injection.
TABLE_JOIN_KEYS: Dict[str, Dict[str, str]] = {
    "listings_cleaned": {"primary": "listings_id"},
    "highbury_listings": {"primary": "listings_id"},
    "reviews_enriched": {"foreign": "listings_id"},
    "amenities_norm": {"foreign": "listing_id"},
    "calendar_aggregate": {"foreign": "listing_id"},
    "amenities_raw": {"foreign": "listing_id"},
    "text_extract": {"foreign": "listings_id"},
}

TABLE_KEYWORDS = {
    "review": "reviews_enriched",
    "reviews": "reviews_enriched",
    "sentiment": "reviews_enriched",
    "amenity": "amenities_norm",
    "amenities": "amenities_norm",
    "calendar": "calendar_aggregate",
    "booking": "calendar_aggregate",
    "availability": "calendar_aggregate",
    "listing": "listings_cleaned",
    "listings": "listings_cleaned",
    "highbury": "highbury_listings",
    "property": "listings_cleaned",
    "text": "text_extract",
}

YEAR_RE = re.compile(r"\b(20\d{2})\b")

FILTER_OPERATORS = {
    "equals": "=",
    "not_equals": "!=",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
}


class DataExplorerError(ValueError):
    """Raised when the explorer cannot satisfy a request."""


@dataclass
class TableColumn:
    name: str
    data_type: str
    description: Optional[str] = None


@dataclass
class TableMeta:
    name: str
    columns: List[TableColumn] = field(default_factory=list)

    def has_column(self, column: str) -> bool:
        return any(col.name == column for col in self.columns)


@dataclass
class QueryResult:
    sql: str
    dataframe: pd.DataFrame
    selected_columns: List[str]
    tables: List[str]
    limit: Optional[int]


class DataExplorer:
    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._schema_lock = threading.Lock()
        self._schema: Dict[str, TableMeta] | None = None
        self._column_descriptions = self._load_dictionary()

    def list_tables(self) -> Dict[str, TableMeta]:
        with self._schema_lock:
            if self._schema is None:
                self._schema = self._load_schema()
        return self._schema

    def structured_query(self, payload: Dict[str, Any], *, limit_cap: Optional[int] = MAX_PREVIEW_ROWS) -> QueryResult:
        schema = self.list_tables()
        tables = payload.get("tables") or []
        if not tables:
            raise DataExplorerError("Select at least one table to query.")
        unique_tables: List[str] = []
        for table in tables:
            if table not in schema:
                raise DataExplorerError(f"Table '{table}' is not available for export.")
            if table not in unique_tables:
                unique_tables.append(table)

        columns_payload = payload.get("columns") or []
        if not columns_payload:
            # Default to the first three columns from each table.
            columns_payload = []
            for table in unique_tables:
                default_cols = [col.name for col in schema[table].columns[:3]]
                columns_payload.extend({"table": table, "column": col} for col in default_cols)

        selections = self._build_selections(columns_payload, schema)
        joins = payload.get("joins") or []
        filters = payload.get("filters") or []
        sorts = payload.get("sort") or []
        limit_raw = payload.get("limit")
        if limit_cap is None:
            limit = None if limit_raw in (None, "", 0) else max(1, int(limit_raw))
        else:
            candidate = int(limit_raw or DEFAULT_LIMIT)
            limit = max(1, min(candidate, limit_cap))

        sql, params = self._build_sql(
            tables=unique_tables,
            selections=selections,
            filters=filters,
            sorts=sorts,
            joins=joins,
            limit=limit,
            schema=schema,
        )
        df = self._run_sql(sql, params)
        selected_labels = [sel["label"] for sel in selections]
        return QueryResult(sql=sql, dataframe=df, selected_columns=selected_labels, tables=unique_tables, limit=limit)

    def freeform_query(self, question: str, *, limit: Optional[int] = DEFAULT_LIMIT) -> QueryResult:
        llm_sql = generate_sql_from_question(question)
        print("[LLM] Generated SQL:\n", llm_sql)
        return QueryResult(sql=llm_sql, dataframe=pd.DataFrame(), selected_columns=[], tables=[], limit=None)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _load_schema(self) -> Dict[str, TableMeta]:
        metadata: Dict[str, TableMeta] = {}
        try:
            with duckdb.connect(str(self.db_path)) as con:
                rows = con.execute(
                    """
                    SELECT table_name, column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = 'main'
                    ORDER BY table_name, ordinal_position
                    """
                ).fetchall()
        except duckdb.Error as exc:  # pragma: no cover - defensive
            raise DataExplorerError(f"Unable to inspect schema: {exc}") from exc

        for table_name, column_name, data_type in rows:
            if table_name not in TABLE_JOIN_KEYS:
                continue
            table_meta = metadata.setdefault(table_name, TableMeta(name=table_name))
            description = self._column_descriptions.get((table_name, column_name))
            table_meta.columns.append(TableColumn(name=column_name, data_type=data_type, description=description))
        return metadata

    def _load_dictionary(self) -> Dict[Tuple[str, str], str]:
        descriptions: Dict[Tuple[str, str], str] = {}
        if not DATA_DICTIONARY_PATH.exists():
            return descriptions
        with DATA_DICTIONARY_PATH.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                table = (row.get("table") or "").strip()
                column = (row.get("column") or "").strip()
                description = (row.get("description") or "").strip()
                if table and column and description:
                    descriptions[(table, column)] = description
        return descriptions

    def distinct_values(self, table: str, column: str) -> List[str]:
        schema = self.list_tables()
        if table not in schema:
            raise DataExplorerError(f"Table '{table}' is not available for export.")
        if not schema[table].has_column(column):
            raise DataExplorerError(f"Column '{column}' does not exist on table '{table}'.")
        sql = f"SELECT DISTINCT {column} AS value FROM {table} WHERE {column} IS NOT NULL ORDER BY value"
        df = self._run_sql(sql, None)
        if "value" not in df.columns:
            return []
        values: List[str] = []
        for raw in df["value"].tolist():
            if raw is None:
                continue
            values.append(str(raw))
        return values

    def _build_selections(self, columns_payload: Sequence[Dict[str, str]], schema: Dict[str, TableMeta]) -> List[Dict[str, str]]:
        selections: List[Dict[str, str]] = []
        seen_aliases: Dict[str, int] = {}
        for entry in columns_payload:
            table = entry.get("table")
            column = entry.get("column")
            if not table or not column:
                continue
            if table not in schema:
                raise DataExplorerError(f"Table '{table}' is not available for export.")
            if not schema[table].has_column(column):
                raise DataExplorerError(f"Column '{column}' does not exist on table '{table}'.")
            alias_base = column
            if any(sel["column"] == column for sel in selections):
                alias_count = seen_aliases.get(column, 0) + 1
                seen_aliases[column] = alias_count
                alias_base = f"{column}_{alias_count}"
            selections.append({"table": table, "column": column, "label": alias_base})
        if not selections:
            raise DataExplorerError("Select at least one column to preview.")
        return selections

    def _build_sql(
        self,
        *,
        tables: List[str],
        selections: List[Dict[str, str]],
        filters: Sequence[Dict[str, Any]],
        sorts: Sequence[Dict[str, Any]],
        joins: Sequence[Dict[str, Any]],
        limit: Optional[int],
        schema: Dict[str, TableMeta],
        scoped_filter: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[Any]]:
        aliases = {table: f"t{idx+1}" for idx, table in enumerate(tables)}
        select_clause = ",\n    ".join(
            f"{aliases[sel['table']]}.{sel['column']} AS {sel['label']}" for sel in selections
        )
        from_clause = f"{tables[0]} {aliases[tables[0]]}"
        join_sql: List[str] = []
        join_params: List[Any] = []

        for table in tables[1:]:
            join_sql.append(self._auto_join(base_table=tables[0], target_table=table, aliases=aliases))

        for join in joins:
            left_table = join.get("left_table")
            right_table = join.get("right_table")
            left_column = join.get("left_column")
            right_column = join.get("right_column")
            if not (left_table and right_table and left_column and right_column):
                continue
            if left_table not in aliases or right_table not in aliases:
                raise DataExplorerError("Join tables must be part of the selection.")
            condition = (
                f"{aliases[left_table]}.{left_column} = {aliases[right_table]}.{right_column}"
            )
            join_sql.append(f"LEFT JOIN {right_table} {aliases[right_table]} ON {condition}")

        where_parts: List[str] = []
        where_params: List[Any] = []
        for filt in filters:
            clause, params = self._build_filter(filt, aliases)
            if clause:
                where_parts.append(clause)
                where_params.extend(params)

        if scoped_filter:
            scope_clause, scope_params = self._build_scope_clause(scoped_filter, aliases, tables, schema=schema)
            where_parts.append(scope_clause)
            where_params.extend(scope_params)

        order_clause = ""
        order_parts: List[str] = []
        for sort in sorts:
            table = sort.get("table")
            column = sort.get("column")
            direction = sort.get("direction", "asc").upper()
            if direction not in {"ASC", "DESC"}:
                direction = "ASC"
            if not table or table not in aliases:
                continue
            if not schema[table].has_column(column):
                continue
            order_parts.append(f"{aliases[table]}.{column} {direction}")
        if order_parts:
            order_clause = " ORDER BY " + ", ".join(order_parts)

        where_clause = ""
        if where_parts:
            where_clause = " WHERE " + " AND ".join(where_parts)

        limit_clause = f"\nLIMIT {limit}" if limit is not None else ""
        sql = (
            f"SELECT\n    {select_clause}\nFROM {from_clause}\n"
            + ("\n".join(join_sql) + "\n" if join_sql else "")
            + where_clause
            + order_clause
            + limit_clause
        )
        return sql, join_params + where_params

    def _auto_join(self, base_table: str, target_table: str, aliases: Dict[str, str]) -> str:
        base_alias = aliases[base_table]
        target_alias = aliases[target_table]
        base_key = self._listing_column(base_table, prefer_primary=True)
        target_key = self._listing_column(target_table)
        return f"LEFT JOIN {target_table} {target_alias} ON {base_alias}.{base_key} = {target_alias}.{target_key}"

    def _build_filter(self, filt: Dict[str, Any], aliases: Dict[str, str]) -> Tuple[str, List[Any]]:
        table = filt.get("table")
        column = filt.get("column")
        operator = filt.get("operator", "equals")
        value = filt.get("value")
        if not table or table not in aliases or not column:
            return "", []

        if operator == "between" and isinstance(value, (list, tuple)) and len(value) == 2:
            return (
                f"{aliases[table]}.{column} BETWEEN ? AND ?",
                [value[0], value[1]],
            )
        if operator in {"in", "not_in"}:
            values_list: List[Any]
            if isinstance(value, (list, tuple, set)):
                values_list = [v for v in value if v not in (None, "")]
            elif value not in (None, ""):
                values_list = [value]
            else:
                values_list = []
            if not values_list:
                return "1=0", [] if operator == "in" else ("1=1", [])
            placeholders = ", ".join("?" for _ in values_list)
            op = "IN" if operator == "in" else "NOT IN"
            return (f"{aliases[table]}.{column} {op} ({placeholders})", list(values_list))
        if operator in FILTER_OPERATORS:
            return (
                f"{aliases[table]}.{column} {FILTER_OPERATORS[operator]} ?",
                [value],
            )
        if operator == "contains":
            return (f"LOWER({aliases[table]}.{column}) LIKE LOWER(?)", [f"%{value}%"])
        if operator == "not_contains":
            return (f"LOWER({aliases[table]}.{column}) NOT LIKE LOWER(?)", [f"%{value}%"])
        if operator == "starts_with":
            return (f"LOWER({aliases[table]}.{column}) LIKE LOWER(?)", [f"{value}%"])
        if operator == "not_starts_with":
            return (f"LOWER({aliases[table]}.{column}) NOT LIKE LOWER(?)", [f"{value}%"])
        if operator == "ends_with":
            return (f"LOWER({aliases[table]}.{column}) LIKE LOWER(?)", [f"%{value}"])
        if operator == "not_ends_with":
            return (f"LOWER({aliases[table]}.{column}) NOT LIKE LOWER(?)", [f"%{value}"])
        return "", []

    def _run_sql(self, sql: str, params: Sequence[Any] | None) -> pd.DataFrame:
        with duckdb.connect(str(self.db_path)) as con:
            return con.execute(sql, params or []).fetchdf()

    def _match_columns(self, text: str, table: str) -> List[str]:
        schema = self.list_tables()
        available = schema.get(table)
        if not available:
            return []
        found: List[str] = []
        for column in available.columns:
            if column.name in text:
                found.append(column.name)
        return found

    def _infer_table(self, text: str) -> str:
        normalized = text.lower()
        for keyword, table in TABLE_KEYWORDS.items():
            if keyword in normalized:
                return table
        return "listings_cleaned"

    def _infer_scope(self, text: str) -> Optional[Dict[str, Any]]:
        lowered = text.lower()
        if "my" in lowered or "highbury" in lowered:
            return {"is_highbury": True}
        if "market" in lowered or "competitor" in lowered:
            return {"is_highbury": False}
        return None

    def _build_scope_clause(
        self,
        scoped_filter: Dict[str, Any],
        aliases: Dict[str, str],
        tables: List[str],
        *,
        schema: Dict[str, TableMeta],
    ) -> Tuple[str, List[Any]]:
        is_highbury = scoped_filter.get("is_highbury", True)
        target_table = scoped_filter.get("table") or tables[0]
        if target_table not in aliases:
            target_table = tables[0]
        alias = aliases[target_table]
        listing_column = scoped_filter.get("listing_column") or self._listing_column(target_table)
        if not schema.get(target_table) or not schema[target_table].has_column(listing_column):
            listing_column = self._listing_column(tables[0])
            alias = aliases[tables[0]]
        comparator = "1" if is_highbury else "0"
        scope_alias = "scope_lookup"
        clause = (
            f"{alias}.{listing_column} IN (SELECT listings_id FROM listings_cleaned {scope_alias} WHERE {scope_alias}.is_highbury = {comparator})"
        )
        return clause, []

    def _table_has_column(self, table: str, column: str, schema: Dict[str, TableMeta]) -> bool:
        return table in schema and schema[table].has_column(column)

    def _listing_column(self, table: str, *, prefer_primary: bool = False) -> str:
        join_meta = TABLE_JOIN_KEYS.get(table, {})
        if prefer_primary and join_meta.get("primary"):
            return join_meta["primary"]
        return join_meta.get("foreign") or join_meta.get("primary") or "listings_id"



EXPLORER = DataExplorer()
