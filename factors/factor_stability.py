"""
V5 Phase 4 — Factor Stability Engine

Caches composite scores in data/cache/score_history.json keyed by date.
On each run, compares today's composite against 7-day, 30-day, and 90-day
snapshots. Wild swings penalize the stability_score.

Stability score (0-100): 100 = rock-steady, 0 = extreme instability.
Penalizes deviations > 10 composite points from any historical snapshot.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

_HISTORY_FILE = Path(__file__).parent.parent / "data" / "cache" / "score_history.json"
_MAX_SWING    = 20.0   # deviation this large → score = 0
_WARN_SWING   = 10.0   # deviation this large → start penalizing


@dataclass
class FactorStabilityResult:
    ticker: str
    stability_score: float       # 0-100
    composite_today: float
    composite_7d:  float | None = None
    composite_30d: float | None = None
    composite_90d: float | None = None
    swing_7d:   float | None = None
    swing_30d:  float | None = None
    swing_90d:  float | None = None
    label: str = ""             # "Stable" / "Moderate" / "Unstable"

    def to_dict(self) -> dict:
        return {
            "stability_score":  round(self.stability_score, 1),
            "composite_today":  round(self.composite_today, 2),
            "composite_7d":     round(self.composite_7d, 2) if self.composite_7d is not None else None,
            "composite_30d":    round(self.composite_30d, 2) if self.composite_30d is not None else None,
            "composite_90d":    round(self.composite_90d, 2) if self.composite_90d is not None else None,
            "swing_7d":         round(self.swing_7d, 2) if self.swing_7d is not None else None,
            "swing_30d":        round(self.swing_30d, 2) if self.swing_30d is not None else None,
            "swing_90d":        round(self.swing_90d, 2) if self.swing_90d is not None else None,
            "label":            self.label,
        }


def _load_history() -> dict:
    """Load {date_str: {ticker: composite, ...}, ...} from disk."""
    if _HISTORY_FILE.exists():
        try:
            with _HISTORY_FILE.open() as f:
                return json.load(f)
        except Exception as exc:
            log.warning("score_history.json corrupt: %s — starting fresh", exc)
    return {}


def _save_history(history: dict) -> None:
    _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Prune entries older than 120 days
    cutoff = (date.today() - timedelta(days=120)).isoformat()
    pruned = {k: v for k, v in history.items() if k >= cutoff}
    with _HISTORY_FILE.open("w") as f:
        json.dump(pruned, f, indent=2)


def _closest_snapshot(history: dict, target_date: date) -> dict[str, float] | None:
    """Find the closest historical snapshot within ±3 days of target_date."""
    best_key = None
    best_delta = 999
    for k in history:
        try:
            d = date.fromisoformat(k)
            delta = abs((d - target_date).days)
            if delta <= 3 and delta < best_delta:
                best_delta = delta
                best_key = k
        except Exception:
            continue
    return history.get(best_key) if best_key else None


def _swing_penalty(swing: float | None) -> float:
    """Convert a composite-point swing to a penalty (0-100; higher = more penalized)."""
    if swing is None:
        return 0.0
    s = abs(swing)
    if s <= _WARN_SWING:
        return 0.0
    if s >= _MAX_SWING:
        return 100.0
    return (s - _WARN_SWING) / (_MAX_SWING - _WARN_SWING) * 100.0


def compute_factor_stability(
    ticker: str,
    composite_today: float,
    history: dict | None = None,
) -> FactorStabilityResult:
    """
    Return stability score for one ticker.

    Call snapshot_and_save() after computing all tickers to persist today's scores.
    """
    if history is None:
        history = _load_history()

    today = date.today()
    snap_7d  = _closest_snapshot(history, today - timedelta(days=7))
    snap_30d = _closest_snapshot(history, today - timedelta(days=30))
    snap_90d = _closest_snapshot(history, today - timedelta(days=90))

    c7  = snap_7d.get(ticker)  if snap_7d  else None
    c30 = snap_30d.get(ticker) if snap_30d else None
    c90 = snap_90d.get(ticker) if snap_90d else None

    sw7  = abs(composite_today - c7)  if c7  is not None else None
    sw30 = abs(composite_today - c30) if c30 is not None else None
    sw90 = abs(composite_today - c90) if c90 is not None else None

    # Penalties: weight recent period more heavily
    p7  = _swing_penalty(sw7)  * 0.50
    p30 = _swing_penalty(sw30) * 0.30
    p90 = _swing_penalty(sw90) * 0.20

    # Total penalty is the weighted sum; stability = 100 - penalty
    total_penalty = p7 + p30 + p90

    # If no history at all, default to neutral 60 (not perfect, not alarming)
    has_history = any(x is not None for x in [sw7, sw30, sw90])
    if not has_history:
        score = 60.0
    else:
        score = max(0.0, min(100.0, 100.0 - total_penalty))

    if score >= 80:
        label = "Stable"
    elif score >= 55:
        label = "Moderate"
    else:
        label = "Unstable"

    return FactorStabilityResult(
        ticker          = ticker,
        stability_score = score,
        composite_today = composite_today,
        composite_7d    = c7,
        composite_30d   = c30,
        composite_90d   = c90,
        swing_7d        = sw7,
        swing_30d       = sw30,
        swing_90d       = sw90,
        label           = label,
    )


def snapshot_and_save(scores: dict[str, float]) -> None:
    """
    Persist today's composite scores to score_history.json.
    Call once per --refresh-data run after all tickers are scored.
    """
    history = _load_history()
    today_key = date.today().isoformat()
    # Merge — don't overwrite existing today's entry wholesale (could be partial)
    existing = history.get(today_key, {})
    existing.update(scores)
    history[today_key] = existing
    _save_history(history)
    log.info("Score history snapshot saved: %d tickers for %s", len(scores), today_key)
