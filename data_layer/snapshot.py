"""
V6 Phase 2 — Point-in-Time Snapshot System

Every --refresh-data run writes a full dated snapshot to data/history/.
Snapshots are NEVER overwritten. They are the foundation of the true
walk-forward backtest (analysis/walk_forward.py).

Retention: snapshots older than _RETENTION_DAYS are pruned on write.
Validation: each snapshot is verified before being written.

File naming: data/history/YYYY-MM-DD_snapshot.json
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_HISTORY_DIR    = Path(__file__).parent.parent / "data" / "history"
_RETENTION_DAYS = 730   # 2 years of snapshots


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_snapshot(payload: dict) -> list[str]:
    """
    Validate a snapshot payload before writing.
    Returns a list of error strings (empty = valid).
    """
    errors = []

    if not payload.get("generated_at"):
        errors.append("Missing 'generated_at' field")

    universe = payload.get("universe", [])
    if not isinstance(universe, list):
        errors.append("'universe' is not a list")
        return errors

    if len(universe) == 0:
        errors.append("'universe' is empty — no tickers to snapshot")
        return errors

    tickers_seen: set[str] = set()
    missing_composite: list[str] = []
    for entry in universe:
        t = entry.get("ticker", "")
        if not t:
            errors.append("Entry with missing ticker found in universe")
            continue
        if t in tickers_seen:
            errors.append(f"Duplicate ticker '{t}' in universe")
        tickers_seen.add(t)
        if entry.get("composite") is None:
            missing_composite.append(t)

    if len(missing_composite) > 10:
        errors.append(
            f"{len(missing_composite)} tickers have no composite score — "
            "snapshot may be incomplete"
        )

    return errors


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def save_snapshot(scored_universe_payload: dict) -> Path | None:
    """
    Write a dated snapshot for today.

    If a snapshot already exists for today, skip (never overwrite).
    Returns the path written, or None if skipped/failed.
    """
    today = date.today().isoformat()
    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    dest = _HISTORY_DIR / f"{today}_snapshot.json"

    if dest.exists():
        log.info("Snapshot for %s already exists — not overwriting (%s)", today, dest.name)
        return None

    errors = validate_snapshot(scored_universe_payload)
    if errors:
        for e in errors:
            log.warning("Snapshot validation warning: %s", e)
        # Block on hard errors (no composite scores at all)
        hard_errors = [e for e in errors if "empty" in e or "not a list" in e]
        if hard_errors:
            log.error("Snapshot NOT saved — hard validation failures: %s", hard_errors)
            return None

    # Add snapshot metadata
    payload = dict(scored_universe_payload)
    payload["snapshot_date"]   = today
    payload["snapshot_source"] = "refresh-data"

    try:
        with dest.open("w") as f:
            json.dump(payload, f, indent=2)
        log.info("Point-in-time snapshot saved: %s (%d tickers)",
                 dest.name, len(payload.get("universe", [])))
        _prune_old_snapshots()
        return dest
    except Exception as exc:
        log.error("Failed to write snapshot %s: %s", dest, exc)
        return None


# ---------------------------------------------------------------------------
# Retention / pruning
# ---------------------------------------------------------------------------

def _prune_old_snapshots() -> int:
    """Delete snapshots older than _RETENTION_DAYS. Returns count deleted."""
    cutoff = (date.today() - timedelta(days=_RETENTION_DAYS)).isoformat()
    deleted = 0
    for p in _HISTORY_DIR.glob("????-??-??_snapshot.json"):
        stem = p.stem.replace("_snapshot", "")
        try:
            if stem < cutoff:
                p.unlink()
                deleted += 1
                log.debug("Pruned old snapshot: %s", p.name)
        except Exception:
            pass
    if deleted:
        log.info("Pruned %d snapshot(s) older than %d days", deleted, _RETENTION_DAYS)
    return deleted


# ---------------------------------------------------------------------------
# Read / list
# ---------------------------------------------------------------------------

def list_snapshot_dates() -> list[str]:
    """Return sorted list of available snapshot date strings."""
    if not _HISTORY_DIR.exists():
        return []
    dates = []
    for p in _HISTORY_DIR.glob("????-??-??_snapshot.json"):
        stem = p.stem.replace("_snapshot", "")
        try:
            date.fromisoformat(stem)
            dates.append(stem)
        except ValueError:
            pass
    return sorted(dates)


def load_snapshot(snap_date: str) -> dict | None:
    """Load a snapshot by date string. Returns None if not found."""
    path = _HISTORY_DIR / f"{snap_date}_snapshot.json"
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except Exception as exc:
        log.warning("Failed to load snapshot %s: %s", path.name, exc)
        return None


def get_snapshot_summary() -> dict[str, Any]:
    """Return a summary dict for display in reports."""
    dates = list_snapshot_dates()
    return {
        "count":      len(dates),
        "first_date": dates[0]  if dates else None,
        "last_date":  dates[-1] if dates else None,
        "dates":      dates,
    }
