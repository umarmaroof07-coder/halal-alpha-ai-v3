"""
V6 Phase 8 — Factor Importance Monitor

Monitors factor health from point-in-time snapshots.

Tracks per factor:
  - Information Coefficient (IC): Spearman rank correlation between factor score
    and forward 1-period composite change (as a proxy for predictiveness)
  - Rank IC: same but using ranks instead of raw scores
  - Hit Rate: % of tickers where a high factor score led to composite outperformance
  - Factor Contribution: average weight × score contribution to composite

Flags:
  - Factor Decay: IC trend is declining over trailing snapshots
  - Factor Instability: IC std dev > 0.15 across snapshots
  - Factor Redundancy: two factors with |correlation| > 0.85

Output: data/cache/factor_monitor.json

NOTE: Does NOT automatically change weights. Reports only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_HISTORY_DIR  = Path(__file__).parent.parent / "data" / "history"
_OUTPUT_FILE  = Path(__file__).parent.parent / "data" / "cache" / "factor_monitor.json"

FACTORS = [
    "quality", "momentum", "earnings_revisions", "valuation",
    "earnings_quality", "moat", "capital_allocation", "risk_adjustment",
]

# Thresholds
_IC_DECAY_SLOPE_THRESHOLD   = -0.02   # IC declining at > 2pp per period → decay flag
_IC_INSTABILITY_STD         = 0.15    # IC std dev above this → instability flag
_REDUNDANCY_CORR_THRESHOLD  = 0.85    # |corr| above this between two factors → redundant


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FactorStats:
    factor:           str
    ic:               float | None      # mean Spearman IC across snapshot pairs
    ic_std:           float | None      # standard deviation of IC
    hit_rate:         float | None      # fraction of tickers where high score → outperformance
    avg_contribution: float | None      # mean composite contribution
    n_observations:   int = 0
    flag_decay:       bool = False
    flag_instability: bool = False

    def to_dict(self) -> dict:
        return {
            "factor":           self.factor,
            "ic":               round(self.ic, 4)               if self.ic        is not None else None,
            "ic_std":           round(self.ic_std, 4)           if self.ic_std    is not None else None,
            "hit_rate":         round(self.hit_rate * 100, 1)   if self.hit_rate  is not None else None,
            "avg_contribution": round(self.avg_contribution, 2) if self.avg_contribution is not None else None,
            "n_observations":   self.n_observations,
            "flag_decay":       self.flag_decay,
            "flag_instability": self.flag_instability,
        }


@dataclass
class FactorMonitorResult:
    factors:           list[FactorStats]   = field(default_factory=list)
    redundant_pairs:   list[tuple[str,str,float]] = field(default_factory=list)
    n_snapshots_used:  int = 0
    warnings:          list[str] = field(default_factory=list)
    generated_at:      str = ""

    def to_dict(self) -> dict:
        return {
            "generated_at":    self.generated_at,
            "n_snapshots":     self.n_snapshots_used,
            "warnings":        self.warnings,
            "factors":         [f.to_dict() for f in self.factors],
            "redundant_pairs": [
                {"factor_a": a, "factor_b": b, "correlation": round(c, 3)}
                for a, b, c in self.redundant_pairs
            ],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation between two equal-length lists."""
    n = len(xs)
    if n < 4:
        return None
    try:
        def _ranks(vals: list[float]) -> list[float]:
            sorted_vals = sorted(range(n), key=lambda i: vals[i])
            ranks = [0.0] * n
            for rank, idx in enumerate(sorted_vals, 1):
                ranks[idx] = float(rank)
            return ranks

        rx, ry = _ranks(xs), _ranks(ys)
        mx = sum(rx) / n
        my = sum(ry) / n
        num   = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
        den_x = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
        den_y = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
        if den_x * den_y == 0:
            return None
        return num / (den_x * den_y)
    except Exception:
        return None


def _list_snapshots() -> list[str]:
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


def _load_snapshot(snap_date: str) -> dict | None:
    path = _HISTORY_DIR / f"{snap_date}_snapshot.json"
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return None


def _extract_factor_scores(snapshot: dict) -> dict[str, dict[str, float]]:
    """
    Extract {ticker: {factor: score}} from a snapshot.
    """
    out: dict[str, dict[str, float]] = {}
    for entry in snapshot.get("universe", []):
        t = entry.get("ticker", "")
        if not t:
            continue
        scores: dict[str, float] = {}
        for f in FACTORS:
            v = entry.get(f)
            if v is not None:
                scores[f] = float(v)
        if scores:
            out[t] = scores
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_factor_monitor() -> FactorMonitorResult:
    """
    Load all available snapshots and compute factor health metrics.
    Saves results to data/cache/factor_monitor.json.
    """
    from datetime import datetime as _dt
    result = FactorMonitorResult(generated_at=_dt.now().isoformat())

    snap_dates = _list_snapshots()
    if len(snap_dates) < 2:
        result.warnings.append(
            f"Only {len(snap_dates)} snapshot(s) available — need ≥2 for factor monitoring. "
            "Accumulate snapshot history by running --refresh-data over multiple sessions."
        )
        _save(result)
        return result

    # Build per-period IC series: for each adjacent pair of snapshots,
    # IC(factor) = Spearman(factor_score_t0, composite_t1 - composite_t0)
    ic_series: dict[str, list[float]] = {f: [] for f in FACTORS}
    hit_series: dict[str, list[float]] = {f: [] for f in FACTORS}

    for i in range(len(snap_dates) - 1):
        s0 = _load_snapshot(snap_dates[i])
        s1 = _load_snapshot(snap_dates[i + 1])
        if not s0 or not s1:
            continue

        scores0 = _extract_factor_scores(s0)
        scores1 = _extract_factor_scores(s1)

        # Composite change per ticker
        common = [t for t in scores0 if t in scores1]
        if len(common) < 10:
            continue

        comp0 = {t: s0_entry.get("composite", 0.0)
                 for entry in s0.get("universe", [])
                 if (t := entry.get("ticker", "")) in common
                 for s0_entry in [entry]}
        comp1 = {t: s1_entry.get("composite", 0.0)
                 for entry in s1.get("universe", [])
                 if (t := entry.get("ticker", "")) in common
                 for s1_entry in [entry]}

        comp_change = {t: comp1.get(t, 50.0) - comp0.get(t, 50.0) for t in common}

        result.n_snapshots_used += 1

        for fac in FACTORS:
            fac_scores = [scores0[t].get(fac) for t in common]
            changes    = [comp_change[t] for t in common]

            # Only use tickers where factor score is available
            pairs = [(fs, ch) for fs, ch in zip(fac_scores, changes) if fs is not None]
            if len(pairs) < 10:
                continue

            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]

            ic = _spearman(xs, ys)
            if ic is not None:
                ic_series[fac].append(ic)

            # Hit rate: tickers with above-median factor score AND positive composite change
            med = sorted(xs)[len(xs) // 2]
            above = [(xs[j], ys[j]) for j in range(len(xs)) if xs[j] >= med]
            if above:
                hr = sum(1 for _, ch in above if ch > 0) / len(above)
                hit_series[fac].append(hr)

    # Build FactorStats per factor
    # Also collect cross-factor scores for redundancy check
    all_factor_scores_latest: dict[str, list[float]] = {f: [] for f in FACTORS}
    latest_snap = _load_snapshot(snap_dates[-1])
    if latest_snap:
        for entry in latest_snap.get("universe", []):
            for fac in FACTORS:
                v = entry.get(fac)
                if v is not None:
                    all_factor_scores_latest[fac].append(float(v))

    # Compute IC trend (slope of IC series) for decay detection
    def _slope(series: list[float]) -> float | None:
        n = len(series)
        if n < 3:
            return None
        xs = list(range(n))
        mx = sum(xs) / n
        my = sum(series) / n
        num = sum((xs[i] - mx) * (series[i] - my) for i in range(n))
        den = sum((xs[i] - mx) ** 2 for i in range(n))
        return num / den if den > 0 else None

    import statistics
    for fac in FACTORS:
        ics = ic_series[fac]
        hrs = hit_series[fac]

        mean_ic  = statistics.mean(ics) if ics else None
        std_ic   = statistics.stdev(ics) if len(ics) > 1 else None
        mean_hr  = statistics.mean(hrs) if hrs else None
        slope    = _slope(ics)

        flag_decay       = (slope is not None and slope < _IC_DECAY_SLOPE_THRESHOLD)
        flag_instability = (std_ic is not None and std_ic > _IC_INSTABILITY_STD)

        result.factors.append(FactorStats(
            factor           = fac,
            ic               = mean_ic,
            ic_std           = std_ic,
            hit_rate         = mean_hr,
            avg_contribution = None,   # would need weight×score data; placeholder
            n_observations   = len(ics),
            flag_decay       = flag_decay,
            flag_instability = flag_instability,
        ))

        if flag_decay:
            result.warnings.append(
                f"Factor Decay: '{fac}' IC trend is declining (slope={slope:.3f}). "
                "Review factor data quality."
            )
        if flag_instability:
            result.warnings.append(
                f"Factor Instability: '{fac}' IC std dev = {std_ic:.3f} > {_IC_INSTABILITY_STD}. "
                "Factor scores may be noisy."
            )

    # Redundancy check: cross-factor correlation on latest snapshot
    for i, f1 in enumerate(FACTORS):
        for f2 in FACTORS[i + 1:]:
            xs = all_factor_scores_latest[f1]
            ys = all_factor_scores_latest[f2]
            # Align lengths (some tickers may lack one factor)
            pairs = list(zip(xs, ys))
            if len(pairs) < 20:
                continue
            corr = _spearman([p[0] for p in pairs], [p[1] for p in pairs])
            if corr is not None and abs(corr) >= _REDUNDANCY_CORR_THRESHOLD:
                result.redundant_pairs.append((f1, f2, corr))
                result.warnings.append(
                    f"Factor Redundancy: '{f1}' and '{f2}' are highly correlated "
                    f"(IC corr={corr:.2f}). Consider reviewing overlap."
                )

    _save(result)
    return result


def _save(result: FactorMonitorResult) -> None:
    _OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _OUTPUT_FILE.open("w") as f:
        json.dump(result.to_dict(), f, indent=2)
    log.info("Factor monitor saved: %s", _OUTPUT_FILE.name)
