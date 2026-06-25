"""
SQLite-backed cache for provider responses.

Cache key: (provider, endpoint, params_hash)
Every row stores the as_of_date so callers can enforce date-gating in backtests.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from config.settings import CACHE_DIR

log = logging.getLogger(__name__)

_DB_PATH = CACHE_DIR / "data_cache.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            cache_key   TEXT PRIMARY KEY,
            provider    TEXT NOT NULL,
            endpoint    TEXT NOT NULL,
            as_of_date  TEXT NOT NULL,
            fetched_at  TEXT NOT NULL,
            ttl_hours   INTEGER NOT NULL,
            payload     TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def _make_key(provider: str, endpoint: str, params: dict) -> str:
    raw = json.dumps({"provider": provider, "endpoint": endpoint, "params": params}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def get_cached(
    provider: str,
    endpoint: str,
    params: dict,
    ttl_hours: int = 24,
) -> Any | None:
    """Return cached payload if fresh, else None."""
    key = _make_key(provider, endpoint, params)
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT payload, fetched_at, ttl_hours FROM cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
    except sqlite3.Error as exc:
        log.warning("Cache read error: %s", exc)
        return None

    if row is None:
        return None

    fetched_at = datetime.fromisoformat(row["fetched_at"])
    effective_ttl = row["ttl_hours"]
    if datetime.utcnow() - fetched_at > timedelta(hours=effective_ttl):
        return None

    return json.loads(row["payload"])


def set_cached(
    provider: str,
    endpoint: str,
    params: dict,
    payload: Any,
    ttl_hours: int = 24,
    as_of_date: str | None = None,
) -> None:
    """Store *payload* in cache."""
    key = _make_key(provider, endpoint, params)
    now = datetime.utcnow().isoformat()
    as_of = as_of_date or now[:10]

    try:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO cache (cache_key, provider, endpoint, as_of_date, fetched_at, ttl_hours, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    fetched_at = excluded.fetched_at,
                    ttl_hours  = excluded.ttl_hours,
                    payload    = excluded.payload
                """,
                (key, provider, endpoint, as_of, now, ttl_hours, json.dumps(payload, default=str)),
            )
    except sqlite3.Error as exc:
        log.warning("Cache write error: %s", exc)


def invalidate(provider: str, endpoint: str, params: dict) -> None:
    key = _make_key(provider, endpoint, params)
    try:
        with _connect() as conn:
            conn.execute("DELETE FROM cache WHERE cache_key = ?", (key,))
    except sqlite3.Error as exc:
        log.warning("Cache invalidate error: %s", exc)


def clear_all() -> None:
    try:
        with _connect() as conn:
            conn.execute("DELETE FROM cache")
        log.info("Cache cleared.")
    except sqlite3.Error as exc:
        log.warning("Cache clear error: %s", exc)
