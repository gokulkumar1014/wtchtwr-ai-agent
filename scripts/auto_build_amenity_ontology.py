#!/usr/bin/env python3
"""
Generate the auto-derived amenities ontology YAML from raw amenity variants.

The output YAML captures canonical amenity names, categories, and variant mappings:
- config/amenities_ontology.yaml
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import yaml

from amenity_utils import (
    AmenityRecord,
    classify_amenity,
    connect_db,
    ensure_directory,
    fetch_unique_amenity_variants,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

OUTPUT_PATH = Path("config/amenities_ontology.yaml")


def build_ontology() -> Dict[str, Dict]:
    """Return a mapping of canonical amenity entries keyed by slug."""
    with connect_db() as con:
        variants = fetch_unique_amenity_variants(con)

    logging.info("Found %s distinct amenity variants.", len(variants))
    ontology: Dict[str, Dict] = {}

    for variant in variants:
        record: AmenityRecord = classify_amenity(variant)
        entry = ontology.setdefault(
            record.canonical_id,
            {
                "amenity_id": record.canonical_id,
                "canonical_name": record.canonical_name,
                "category": record.category,
                "variants": [],
            },
        )
        entry["variants"].append(
            {
                "value": record.value,
                "normalized": record.normalized,
                "subtype": record.subtype,
            }
        )

    # Sort variants inside each canonical bucket for deterministic output
    for entry in ontology.values():
        entry["variants"].sort(key=lambda item: item["normalized"])

    return dict(sorted(ontology.items(), key=lambda kv: kv[1]["canonical_name"]))


def write_yaml(data: Dict[str, Dict]) -> Path:
    ensure_directory(OUTPUT_PATH.parent)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "amenities": list(data.values()),
    }
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)
    return OUTPUT_PATH


def main() -> None:
    ontology = build_ontology()
    output_path = write_yaml(ontology)
    logging.info("Amenity ontology written to %s", output_path)


if __name__ == "__main__":
    main()
