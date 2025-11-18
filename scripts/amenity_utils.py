#!/usr/bin/env python3
"""
Shared helpers for amenity normalization workflows.

This module centralizes the text normalization, canonical mapping rules,
and DuckDB access utilities reused across amenity scripts.
"""
from __future__ import annotations

import ast
import html
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import duckdb


REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "db" / "airbnb.duckdb"

# General-purpose regex helpers
PUNCTUATION_RE = re.compile(r"[\"'`“”‘’,]")
NON_ALPHANUM_RE = re.compile(r"[^a-z0-9\s]")
MULTISPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class AmenityRecord:
    """Structured representation of a single amenity variant."""

    value: str
    normalized: str
    canonical_id: str
    canonical_name: str
    category: str
    subtype: Optional[str]


def connect_db() -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection pointed at the project database."""
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"DuckDB database not found at {DB_PATH}. "
            "Run scripts/load_duckdb.py first."
        )
    return duckdb.connect(str(DB_PATH))


def parse_amenity_list(amenities_text: str) -> List[str]:
    """Safely parse an amenities JSON-like list from CSV."""
    if amenities_text is None:
        return []
    text = amenities_text.strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [text]
    if isinstance(parsed, (list, tuple)):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [str(parsed).strip()]


def clean_variant(raw: str) -> str:
    """Normalize spacing and unicode quirks without lowercasing."""
    text = html.unescape(raw or "")
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def normalize_variant(raw: str) -> str:
    """Lowercase, ascii-friendly version used for dictionary matching."""
    text = clean_variant(raw).lower()
    text = PUNCTUATION_RE.sub("", text)
    text = NON_ALPHANUM_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def slugify(text: str) -> str:
    """Convert text into a slug identifier (lowercase with underscores)."""
    norm = normalize_variant(text)
    return norm.replace(" ", "_")


CATEGORY_KEYWORDS: Dict[str, Sequence[str]] = {
    "kitchen": [
        "kitchen",
        "microwave",
        "oven",
        "stove",
        "cooktop",
        "dishwasher",
        "coffee",
        "espresso",
        "toaster",
        "blender",
        "refrigerator",
        "freezer",
        "dishes",
        "silverware",
        "wine glass",
        "dining table",
        "kettle",
        "baking",
        "cooking basics",
        "rice maker",
    ],
    "bathroom": [
        "shampoo",
        "conditioner",
        "body soap",
        "toiletries",
        "bath",
        "bathtub",
        "tub",
        "toilet",
        "bidet",
        "shower",
        "hair dryer",
        "hot water",
    ],
    "laundry": [
        "washer",
        "dryer",
        "laundry",
        "iron",
        "ironing",
        "clothing storage",
        "closet",
        "wardrobe",
        "hangers",
    ],
    "climate": [
        "air conditioning",
        "ac",
        "heater",
        "heat",
        "heating",
        "fan",
        "ceiling fan",
        "portable fans",
        "fireplace",
    ],
    "safety": [
        "smoke alarm",
        "carbon monoxide",
        "first aid",
        "fire extinguisher",
        "security",
        "lock",
        "surveillance",
        "camera",
        "safe",
    ],
    "access": [
        "self check",
        "keypad",
        "lockbox",
        "private entrance",
        "host greets",
        "elevator",
        "step-free",
        "accessible",
        "wheelchair",
    ],
    "entertainment": [
        "tv",
        "hdtv",
        "streaming",
        "game console",
        "speakers",
        "sound system",
        "books",
        "board game",
        "piano",
    ],
    "connectivity": [
        "wifi",
        "wi fi",
        "internet",
        "ethernet",
        "charging",
        "usb",
    ],
    "outdoor": [
        "patio",
        "balcony",
        "garden",
        "outdoor",
        "bbq",
        "grill",
        "backyard",
        "sun deck",
    ],
    "parking": [
        "parking",
        "garage",
        "driveway",
        "carport",
        "charging station",
    ],
    "workspace": [
        "workspace",
        "desk",
        "office",
        "monitor",
        "printer",
    ],
    "family": [
        "crib",
        "high chair",
        "pack n play",
        "baby",
        "children",
    ],
    "sleep": [
        "bed linens",
        "extra pillows",
        "blankets",
        "room darkening",
        "ear plugs",
        "mattress",
    ],
    "services": [
        "luggage",
        "long term stays",
        "cleaning available",
        "daily housekeeping",
        "concierge",
    ],
}

# Specific pattern rules to collapse common variants into canonical names.
PATTERN_CANONICAL: Sequence[Tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"\bfree street parking\b"), "Parking (Free street)", "parking"),
    (re.compile(r"\bpaid parking\b"), "Parking (Paid)", "parking"),
    (re.compile(r"\bparking garage\b"), "Parking (Garage)", "parking"),
    (re.compile(r"\bparking\b"), "Parking", "parking"),
    (re.compile(r"\bwasher\b|\blaundry\b"), "Washer", "laundry"),
    (re.compile(r"\bdryer\b"), "Dryer", "laundry"),
    (re.compile(r"\bdishwasher\b"), "Dishwasher", "kitchen"),
    (re.compile(r"\bcoffee maker\b"), "Coffee maker", "kitchen"),
    (re.compile(r"\bespresso\b|nespresso"), "Coffee maker", "kitchen"),
    (re.compile(r"\bwifi\b|\bwi fi\b|\binternet\b"), "WiFi", "connectivity"),
    (re.compile(r"\bethernet\b"), "Ethernet connection", "connectivity"),
    (re.compile(r"\btelevision\b|\btv\b|\bhdtv\b"), "TV", "entertainment"),
    (re.compile(r"\bstreaming\b|roku|netflix|prime video|hbo"), "Streaming services", "entertainment"),
    (re.compile(r"\bmicrowave\b"), "Microwave", "kitchen"),
    (re.compile(r"\boven\b"), "Oven", "kitchen"),
    (re.compile(r"\bstove\b|cooktop"), "Stove", "kitchen"),
    (re.compile(r"\brefrigerator\b|\bfridge\b"), "Refrigerator", "kitchen"),
    (re.compile(r"\bfreezer\b"), "Freezer", "kitchen"),
    (re.compile(r"\bdining table\b"), "Dining table", "kitchen"),
    (re.compile(r"\bkitchen\b"), "Kitchen", "kitchen"),
    (re.compile(r"\bhair dryer\b"), "Hair dryer", "bathroom"),
    (re.compile(r"\bbody soap\b|\bsoap\b"), "Body soap", "bathroom"),
    (re.compile(r"\bshampoo\b"), "Shampoo", "bathroom"),
    (re.compile(r"\bconditioner\b"), "Conditioner", "bathroom"),
    (re.compile(r"\bhot water\b"), "Hot water", "bathroom"),
    (re.compile(r"\bair conditioning\b|\bac unit\b|\bwindow ac\b"), "Air conditioning", "climate"),
    (re.compile(r"\bheating\b|\bheater\b"), "Heating", "climate"),
    (re.compile(r"\bfan\b"), "Fans", "climate"),
    (re.compile(r"\bsmoke alarm\b"), "Smoke alarm", "safety"),
    (re.compile(r"\bcarbon monoxide\b"), "Carbon monoxide alarm", "safety"),
    (re.compile(r"\bfirst aid\b"), "First aid kit", "safety"),
    (re.compile(r"\bfire extinguisher\b"), "Fire extinguisher", "safety"),
    (re.compile(r"\bsecurity camera\b"), "Security camera", "safety"),
    (re.compile(r"\blockbox\b|\bkeypad\b|\bself check in\b"), "Self check-in", "access"),
    (re.compile(r"\bprivate entrance\b"), "Private entrance", "access"),
    (re.compile(r"\belevator\b"), "Elevator", "access"),
    (re.compile(r"\bdedicated workspace\b|\bworkspace\b|\bdesk\b"), "Dedicated workspace", "workspace"),
    (re.compile(r"\bluggage dropoff\b"), "Luggage dropoff", "services"),
    (re.compile(r"\blong term stays\b"), "Long term stays", "services"),
    (re.compile(r"\bbed linens\b"), "Bed linens", "sleep"),
    (re.compile(r"\bextra pillows\b|\bblankets\b"), "Extra pillows and blankets", "sleep"),
    (re.compile(r"\broom[- ]?darkening\b"), "Room-darkening shades", "sleep"),
    (re.compile(r"\bpatio\b|\bterrace\b|\bdeck\b|\bal fresco\b"), "Outdoor seating", "outdoor"),
    (re.compile(r"\bgarden\b|\bbackyard\b|\byard\b"), "Backyard", "outdoor"),
    (re.compile(r"\bgrill\b|\bbbq\b"), "BBQ grill", "outdoor"),
    (re.compile(r"\bcrib\b|\bbaby crib\b"), "Crib", "family"),
    (re.compile(r"\bhigh chair\b"), "High chair", "family"),
]

# Specific normalized values that deserve canonical overrides.
CANONICAL_NORMAL_OVERRIDES: Dict[str, Tuple[str, str]] = {
    "paid parking garage off premises": ("Parking (Paid)", "parking"),
    "paid parking off premises": ("Parking (Paid)", "parking"),
    "free street parking": ("Parking (Free street)", "parking"),
    "free parking on premises": ("Parking (Free)", "parking"),
    "paid washer in building": ("Washer", "laundry"),
    "paid dryer in building": ("Dryer", "laundry"),
    "laundromat nearby": ("Laundromat nearby", "laundry"),
    "fast wifi 144 mbps": ("WiFi", "connectivity"),
    "fast wifi 273 mbps": ("WiFi", "connectivity"),
    "fast wifi": ("WiFi", "connectivity"),
    "wifi": ("WiFi", "connectivity"),
    "ethernet connection": ("Ethernet connection", "connectivity"),
    "hdtv with standard cable roku amazon prime video netflix hbo max": ("Streaming services", "entertainment"),
}


def infer_category(normalized: str, canonical_id: str) -> str:
    """Infer a category name based on keywords."""
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return category
    # fallback to canonical slug if matches known category
    for category in CATEGORY_KEYWORDS:
        if category in canonical_id:
            return category
    return "misc"


def classify_amenity(raw: str) -> AmenityRecord:
    """Return canonical classification details for a raw amenity string."""
    cleaned = clean_variant(raw)
    normalized = normalize_variant(cleaned)

    # Patreon overrides for exact normalized strings
    if normalized in CANONICAL_NORMAL_OVERRIDES:
        canonical_name, category = CANONICAL_NORMAL_OVERRIDES[normalized]
        canonical_id = slugify(canonical_name)
        subtype = compute_subtype(cleaned, canonical_name)
        return AmenityRecord(
            value=cleaned,
            normalized=normalized,
            canonical_id=canonical_id,
            canonical_name=canonical_name,
            category=category,
            subtype=subtype,
        )

    for pattern, canonical_name, category in PATTERN_CANONICAL:
        if pattern.search(normalized):
            canonical_id = slugify(canonical_name)
            category_name = category or infer_category(normalized, canonical_id)
            subtype = compute_subtype(cleaned, canonical_name)
            return AmenityRecord(
                value=cleaned,
                normalized=normalized,
                canonical_id=canonical_id,
                canonical_name=canonical_name,
                category=category_name,
                subtype=subtype,
            )

    canonical_name = cleaned.split(":", 1)[0]
    canonical_name = canonical_name.split("-", 1)[0].strip()
    if not canonical_name:
        canonical_name = cleaned
    canonical_id = slugify(canonical_name)
    category = infer_category(normalized, canonical_id)
    subtype = compute_subtype(cleaned, canonical_name)
    return AmenityRecord(
        value=cleaned,
        normalized=normalized,
        canonical_id=canonical_id,
        canonical_name=canonical_name.title(),
        category=category,
        subtype=subtype,
    )


def compute_subtype(original: str, canonical_name: str) -> Optional[str]:
    """Derive a subtype string (difference between original and canonical)."""
    residual = re.sub(
        rf"\b{re.escape(canonical_name)}\b",
        " ",
        original,
        flags=re.IGNORECASE,
    )
    residual = MULTISPACE_RE.sub(" ", residual).strip(" -–:")
    if residual and residual.lower() != canonical_name.lower():
        return residual
    return None


def fetch_unique_amenity_variants(
    conn: duckdb.DuckDBPyConnection,
) -> List[str]:
    """Load the distinct amenity variants from the `amenities_raw` table."""
    table_exists = conn.execute(
        """
        SELECT COUNT(*) > 0
        FROM information_schema.tables
        WHERE table_schema = 'main'
          AND table_name = 'amenities_raw'
        """
    ).fetchone()[0]
    if not table_exists:
        raise RuntimeError(
            "Table 'amenities_raw' not found. Run scripts/setup_amenities_raw.py first."
        )
    rows = conn.execute("SELECT amenities_text FROM amenities_raw").fetchall()
    variants: List[str] = []
    for (amenities_text,) in rows:
        variants.extend(parse_amenity_list(amenities_text))
    return sorted(set(filter(None, variants)))


def ensure_directory(path: Path) -> None:
    """Ensure target directory exists."""
    path.mkdir(parents=True, exist_ok=True)

