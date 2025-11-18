from __future__ import annotations

import io
import threading
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple
from uuid import uuid4

import pandas as pd


_EXPORT_TTL = timedelta(minutes=15)
_MAX_EXPORTS = 24


class _ExportEntry:
    __slots__ = ("content", "expires_at", "filename", "rows")

    def __init__(self, *, content: bytes, expires_at: datetime, filename: str, rows: int) -> None:
        self.content = content
        self.expires_at = expires_at
        self.filename = filename
        self.rows = rows


_store: "OrderedDict[str, _ExportEntry]" = OrderedDict()
_lock = threading.Lock()


def _cleanup(now: datetime) -> None:
    expired = [token for token, entry in _store.items() if entry.expires_at <= now]
    for token in expired:
        _store.pop(token, None)


def _slugify(text: str | None) -> str:
    if not text:
        return "export"
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in text) or "export"
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "export"


def stage_csv(df: pd.DataFrame, *, label: str | None = None) -> Dict[str, object]:
    """Persist a dataframe in memory and return metadata for download."""

    now = datetime.now(timezone.utc)
    with _lock:
        _cleanup(now)

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    content = buffer.getvalue().encode("utf-8")
    token = uuid4().hex
    expires_at = now + _EXPORT_TTL
    slug = _slugify(label)
    filename = f"{slug[:32]}-{token[:8]}.csv"
    entry = _ExportEntry(content=content, expires_at=expires_at, filename=filename, rows=len(df))

    with _lock:
        _store[token] = entry
        while len(_store) > _MAX_EXPORTS:
            _store.popitem(last=False)

    return {
        "token": token,
        "format": "csv",
        "rows": len(df),
        "expires_at": expires_at.isoformat(),
        "filename": filename,
        "session_only": True,
    }


def get_csv(token: str) -> Tuple[bytes, str]:
    now = datetime.now(timezone.utc)
    with _lock:
        entry = _store.get(token)
        if not entry:
            raise KeyError(token)
        if entry.expires_at <= now:
            _store.pop(token, None)
            raise KeyError(token)
        return entry.content, entry.filename