# DuckDB Database

The DuckDB file is omitted from Git because it is large and regenerated from the cleaned CSV/Parquet assets in `data/clean/`.

## Expected Layout
```
db/
└── airbnb.duckdb
```

## Download
1. Download the pre-built database from Google Drive: <https://drive.google.com/drive/u/1/folders/1SWkw-wFlu7g9KUriNsotY3RBUrWxaajg>
2. Place the resulting `airbnb.duckdb` file (or extract the archive) into this `db/` directory.

## Rebuilding Locally
If you have the raw/clean CSVs:

```bash
source .venv/bin/activate       # ensure dependencies installed
python scripts/load_duckdb.py
```

This script loads the cleaned tables into a fresh DuckDB database and writes it to `db/airbnb.duckdb`.