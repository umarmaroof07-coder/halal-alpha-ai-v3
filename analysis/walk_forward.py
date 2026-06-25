"""
V6 Phase 1 — True Walk-Forward Backtest Engine

Point-in-time constraints:
  - Uses ONLY historical snapshot data available on each rebalance date.
  - No future prices, fundamentals, analyst revisions, or AI scores.
  - Reads from data/history/YYYY-MM-DD_snapshot.json files.
  - Falls back to yfinance for price returns (not factor scores).

Rebalancing: quarterly (March, June, September, December month-ends).

For each rebalance date:
  1. Load the closest prior snapshot (≤ rebalance date).
  2. Rank tickers by composite score from that snapshot.
  3. Select Top 5 compliant tickers.
  4. Hold until next rebalance date; compute actual price returns.
  5. Compare against SPY and QQQ.

Metrics: CAGR, Sharpe, Sortino, Calmar, Max Drawdown, Win Rate,
         Information Ratio, Alpha, Beta, Turnover, Tracking Error.

Output: data/cache/walk_forward_results.json
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
_CACHE_FILE   = Path(__file__).parent.parent / "data" / "cache" / "walk_forward_results.json"
_REBAL_MONTHS = {3, 6, 9, 12}   # quarterly rebalance months


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PeriodReturn:
    start_date: str
    end_date:   str
    portfolio:  list[str]          # tickers held
    port_ret:   float              # gross return for the period
    spy_ret:    float
    qqq_ret:    float
    turnover:   float              # % of portfolio that changed


@dataclass
class WalkForwardMetrics:
    cagr:              float
    sharpe:            float
    sortino:           float
    calmar:            float
    max_drawdown:      float       # negative, e.g. -0.25
    win_rate:          float       # fraction of periods beating SPY
    information_ratio: float
    alpha:             float       # annualised vs SPY
    beta:              float
    turnover:          float       # average per rebalance
    tracking_error:    float
    n_periods:         int

    def to_dict(self) -> dict:
        return {
            "cagr":              round(self.cagr * 100, 2),
            "sharpe":            round(self.sharpe, 3),
            "sortino":           round(self.sortino, 3),
            "calmar":            round(self.calmar, 3),
            "max_drawdown":      round(self.max_drawdown * 100, 2),
            "win_rate":          round(self.win_rate * 100, 1),
            "information_ratio": round(self.information_ratio, 3),
            "alpha":             round(self.alpha * 100, 2),
            "beta":              round(self.beta, 3),
            "turnover":          round(self.turnover * 100, 1),
            "tracking_error":    round(self.tracking_error * 100, 2),
            "n_periods":         self.n_periods,
        }


@dataclass
class WalkForwardResult:
    periods:          list[PeriodReturn] = field(default_factory=list)
    metrics:          WalkForwardMetrics | None = None
    spy_metrics:      WalkForwardMetrics | None = None
    qqq_metrics:      WalkForwardMetrics | None = None
    snapshot_dates:   list[str] = field(default_factory=list)
    warnings:         list[str] = field(default_factory=list)
    generated_at:     str = ""
    n_snapshots_used: int = 0


# ---------------------------------------------------------------------------
# Snapshot loader
# ---------------------------------------------------------------------------

def _list_snapshot_dates() -> list[str]:
    """Return sorted list of available snapshot dates (YYYY-MM-DD)."""
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
    """Load a single snapshot by date string."""
    path = _HISTORY_DIR / f"{snap_date}_snapshot.json"
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except Exception as exc:
        log.warning("Snapshot load failed %s: %s", snap_date, exc)
        return None


def _closest_snapshot_before(snap_dates: list[str], rebal_date: date) -> str | None:
    """Find the most recent snapshot on or before rebal_date."""
    rebal_str = rebal_date.isoformat()
    candidates = [d for d in snap_dates if d <= rebal_str]
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# Rebalance date generator
# ---------------------------------------------------------------------------

def _quarterly_rebal_dates(start: date, end: date) -> list[date]:
    """Generate last-day-of-quarter dates between start and end."""
    dates = []
    y, m = start.year, start.month
    while True:
        # Advance to next rebal month
        while m not in _REBAL_MONTHS:
            m += 1
            if m > 12:
                m = 1
                y += 1
        # Last day of that month
        if m == 12:
            last = date(y + 1, 1, 1) - timedelta(days=1)
        else:
            last = date(y, m + 1, 1) - timedelta(days=1)
        if last > end:
            break
        dates.append(last)
        m += 1
        if m > 12:
            m = 1
            y += 1
    return dates


# ---------------------------------------------------------------------------
# Price fetch (yfinance)
# ---------------------------------------------------------------------------

def _fetch_prices(tickers: list[str], start: str, end: str) -> dict[str, Any]:
    """Fetch daily adjusted close prices from yfinance. Returns {ticker: Series}."""
    try:
        import yfinance as yf
        import pandas as pd
        bench = ["SPY", "QQQ"] + [t for t in tickers if t not in ("SPY", "QQQ")]
        raw = yf.download(bench, start=start, end=end, progress=False,
                          auto_adjust=True, group_by="ticker")
        out: dict[str, Any] = {}
        for t in bench:
            try:
                if len(bench) == 1:
                    out[t] = raw["Close"]
                else:
                    out[t] = raw["Close"][t] if t in raw["Close"].columns else raw[t]["Close"]
            except Exception:
                pass
        return out
    except Exception as exc:
        log.warning("Price fetch failed: %s", exc)
        return {}


def _period_return(prices: Any, start: str, end: str) -> float | None:
    """Compute total return from start to end using daily close prices."""
    try:
        import pandas as pd
        s_dt = pd.Timestamp(start)
        e_dt = pd.Timestamp(end)
        window = prices.loc[s_dt:e_dt].dropna()
        if len(window) < 2:
            return None
        return float(window.iloc[-1] / window.iloc[0]) - 1.0
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _risk_free_rate() -> float:
    return 0.045   # approximate current T-bill rate


def _compute_metrics(
    port_rets:  list[float],
    spy_rets:   list[float],
    periods_per_year: float = 4.0,   # quarterly
    turnovers:  list[float] | None = None,
) -> WalkForwardMetrics:
    import math, statistics

    rf_per_period = _risk_free_rate() / periods_per_year
    n = len(port_rets)
    if n < 2:
        return WalkForwardMetrics(0,0,0,0,0,0,0,0,1.0,0,0,n)

    # CAGR
    total = 1.0
    for r in port_rets:
        total *= (1 + r)
    years = n / periods_per_year
    cagr = total ** (1 / years) - 1.0 if years > 0 else 0.0

    # Excess returns for Sharpe / IR
    excess = [r - rf_per_period for r in port_rets]
    mean_ex = statistics.mean(excess)
    std_ex  = statistics.stdev(excess) if n > 1 else 0.001
    sharpe  = (mean_ex / std_ex * (periods_per_year ** 0.5)) if std_ex > 0 else 0.0

    # Sortino (downside deviation)
    neg = [r for r in excess if r < 0]
    downside = (statistics.mean([r**2 for r in neg]) ** 0.5 * (periods_per_year ** 0.5)) if neg else 0.001
    sortino  = (mean_ex * periods_per_year ** 0.5) / downside if downside > 0 else 0.0

    # Max drawdown
    cum = [1.0]
    for r in port_rets:
        cum.append(cum[-1] * (1 + r))
    peak = cum[0]
    max_dd = 0.0
    for v in cum:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd

    # Calmar
    calmar = (cagr / abs(max_dd)) if max_dd != 0 else 0.0

    # Win rate vs SPY
    wins = sum(1 for p, s in zip(port_rets, spy_rets) if p > s)
    win_rate = wins / n

    # Active returns vs SPY
    active = [p - s for p, s in zip(port_rets, spy_rets)]
    mean_act = statistics.mean(active)
    std_act  = statistics.stdev(active) if n > 1 else 0.001
    ir = (mean_act / std_act * (periods_per_year ** 0.5)) if std_act > 0 else 0.0
    tracking_error = std_act * (periods_per_year ** 0.5)

    # Alpha / Beta vs SPY
    mean_spy = statistics.mean(spy_rets)
    cov_ps   = statistics.mean([(p - statistics.mean(port_rets)) * (s - mean_spy)
                                  for p, s in zip(port_rets, spy_rets)])
    var_spy  = statistics.variance(spy_rets) if n > 1 else 0.001
    beta     = cov_ps / var_spy if var_spy > 0 else 1.0
    alpha    = (statistics.mean(port_rets) - rf_per_period -
                beta * (mean_spy - rf_per_period)) * periods_per_year

    avg_turnover = statistics.mean(turnovers) if turnovers else 0.0

    return WalkForwardMetrics(
        cagr=cagr, sharpe=sharpe, sortino=sortino, calmar=calmar,
        max_drawdown=max_dd, win_rate=win_rate, information_ratio=ir,
        alpha=alpha, beta=beta, turnover=avg_turnover,
        tracking_error=tracking_error, n_periods=n,
    )


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

def run_walk_forward(
    start_date: date | None = None,
    end_date:   date | None = None,
    top_n: int = 5,
) -> WalkForwardResult:
    """
    Run the true walk-forward backtest using point-in-time snapshots.

    Returns WalkForwardResult. Also saves to data/cache/walk_forward_results.json.
    """
    from datetime import datetime as _dt
    result = WalkForwardResult(generated_at=_dt.now().isoformat())

    snap_dates = _list_snapshot_dates()
    if not snap_dates:
        result.warnings.append(
            "No point-in-time snapshots found in data/history/. "
            "Run --refresh-data a few times to accumulate snapshot history before "
            "running a meaningful walk-forward backtest."
        )
        _save_results(result)
        return result

    result.snapshot_dates = snap_dates

    # Default window: from first snapshot to today
    first_snap = date.fromisoformat(snap_dates[0])
    today = date.today()
    start = start_date or first_snap
    end   = end_date   or today

    rebal_dates = _quarterly_rebal_dates(start, end)
    if len(rebal_dates) < 2:
        result.warnings.append(
            f"Only {len(rebal_dates)} rebalance date(s) found between "
            f"{start} and {end}. Need ≥2 periods for backtest metrics. "
            "Accumulate more snapshot history (run --refresh-data across multiple quarters)."
        )
        _save_results(result)
        return result

    log.info("Walk-forward: %d rebalance dates from %s to %s", len(rebal_dates), start, end)

    # Fetch all prices in one call (rebal_dates[0] → today)
    fetch_start = (rebal_dates[0] - timedelta(days=5)).isoformat()
    all_tickers: set[str] = set()
    # Pre-scan snapshots to collect all tickers
    for sd in snap_dates:
        snap = _load_snapshot(sd)
        if snap:
            for entry in snap.get("universe", []):
                all_tickers.add(entry.get("ticker", ""))
    all_tickers.discard("")

    log.info("Fetching prices for %d tickers from %s…", len(all_tickers), fetch_start)
    price_data = _fetch_prices(list(all_tickers), fetch_start, today.isoformat())

    if "SPY" not in price_data or "QQQ" not in price_data:
        result.warnings.append("SPY or QQQ price data unavailable — benchmark comparison will be partial.")

    # Walk forward
    prev_portfolio: list[str] = []
    periods: list[PeriodReturn] = []

    for i, rebal_dt in enumerate(rebal_dates[:-1]):
        next_rebal = rebal_dates[i + 1]

        # Load closest snapshot on or before this rebalance date
        snap_key = _closest_snapshot_before(snap_dates, rebal_dt)
        if not snap_key:
            result.warnings.append(f"No snapshot available at or before {rebal_dt} — skipping period.")
            continue

        snap = _load_snapshot(snap_key)
        if not snap:
            continue
        result.n_snapshots_used += 1

        # Rank by composite score (point-in-time)
        universe = snap.get("universe", [])
        ranked = sorted(
            [e for e in universe if e.get("shariah_status") == "compliant"],
            key=lambda e: e.get("composite", 0),
            reverse=True,
        )
        portfolio = [e["ticker"] for e in ranked[:top_n]]

        if not portfolio:
            result.warnings.append(f"No compliant tickers in snapshot {snap_key} — skipping period.")
            continue

        # Compute equal-weight portfolio return over holding period
        start_str = rebal_dt.isoformat()
        end_str   = next_rebal.isoformat()

        port_returns = []
        for t in portfolio:
            if t in price_data:
                r = _period_return(price_data[t], start_str, end_str)
                if r is not None:
                    port_returns.append(r)

        if not port_returns:
            result.warnings.append(f"No price data for portfolio in period {start_str}→{end_str} — skipping.")
            continue

        port_ret = sum(port_returns) / len(port_returns)   # equal weight
        spy_ret  = _period_return(price_data.get("SPY"), start_str, end_str) or 0.0
        qqq_ret  = _period_return(price_data.get("QQQ"), start_str, end_str) or 0.0

        # Turnover vs previous period
        if prev_portfolio:
            unchanged = set(portfolio) & set(prev_portfolio)
            turnover  = 1.0 - len(unchanged) / max(len(prev_portfolio), len(portfolio))
        else:
            turnover = 1.0

        periods.append(PeriodReturn(
            start_date = start_str,
            end_date   = end_str,
            portfolio  = portfolio,
            port_ret   = port_ret,
            spy_ret    = spy_ret,
            qqq_ret    = qqq_ret,
            turnover   = turnover,
        ))
        prev_portfolio = portfolio

    result.periods = periods

    if len(periods) < 2:
        result.warnings.append(
            f"Only {len(periods)} complete period(s) with price data. "
            "Walk-forward metrics require ≥2 periods. "
            "The system is accumulating snapshot history — re-run in future quarters."
        )
        _save_results(result)
        return result

    # Compute metrics
    port_rets = [p.port_ret for p in periods]
    spy_rets  = [p.spy_ret  for p in periods]
    qqq_rets  = [p.qqq_ret  for p in periods]
    turnovers = [p.turnover for p in periods]

    result.metrics     = _compute_metrics(port_rets, spy_rets, turnovers=turnovers)
    result.spy_metrics = _compute_metrics(spy_rets,  spy_rets, turnovers=None)
    result.qqq_metrics = _compute_metrics(qqq_rets,  spy_rets, turnovers=None)

    _save_results(result)
    log.info("Walk-forward complete: %d periods, CAGR=%.1f%%, Sharpe=%.2f",
             len(periods), result.metrics.cagr * 100, result.metrics.sharpe)
    return result


def _save_results(result: WalkForwardResult) -> None:
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at":     result.generated_at,
        "n_snapshots_used": result.n_snapshots_used,
        "snapshot_dates":   result.snapshot_dates,
        "warnings":         result.warnings,
        "metrics":          result.metrics.to_dict() if result.metrics else None,
        "spy_metrics":      result.spy_metrics.to_dict() if result.spy_metrics else None,
        "qqq_metrics":      result.qqq_metrics.to_dict() if result.qqq_metrics else None,
        "periods": [
            {
                "start_date": p.start_date,
                "end_date":   p.end_date,
                "portfolio":  p.portfolio,
                "port_ret":   round(p.port_ret * 100, 2),
                "spy_ret":    round(p.spy_ret  * 100, 2),
                "qqq_ret":    round(p.qqq_ret  * 100, 2),
                "turnover":   round(p.turnover * 100, 1),
            }
            for p in result.periods
        ],
    }
    with _CACHE_FILE.open("w") as f:
        json.dump(payload, f, indent=2)
