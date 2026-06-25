"""
V6 Phase 7 — Advanced Stress Test Engine

Fetches historical price returns for four crisis periods and computes
a resilience_score (0-100) measuring how well a stock weathered each.

Crisis windows:
  GFC 2008:               2008-09-01 → 2009-03-31
  COVID 2020:             2020-02-01 → 2020-03-31
  Rate hike 2022:         2022-01-01 → 2022-10-31
  Regional Banks 2023:    2023-03-01 → 2023-05-31

Metrics per crisis:
  - Max drawdown
  - Relative performance vs SPY
  - Recovery: did it recover to pre-crisis level within the observation window?
  - Intra-crisis volatility (annualized)

Resilience score components:
  - Drawdown penalty     (smaller drawdown = better)
  - Volatility penalty   (lower vol = better)
  - Relative performance vs SPY bonus/penalty
  - Partial recovery bonus if stock recovered faster than market

Cache: stored in data/cache/stress_test_cache.json (TTL 30 days).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_CACHE_FILE = Path(__file__).parent.parent / "data" / "cache" / "stress_test_cache.json"
_CACHE_TTL_DAYS = 30

CRISIS_WINDOWS = {
    "gfc_2008":         ("2008-09-01", "2009-03-31"),
    "covid_2020":       ("2020-02-01", "2020-03-31"),
    "rates_2022":       ("2022-01-01", "2022-10-31"),
    "regional_banks_2023": ("2023-03-01", "2023-05-31"),
}

# Sector ETF benchmarks for relative stress comparison (used in enhanced output)
SECTOR_ETFS = {
    "Technology":            "XLK",
    "Health Care":           "XLV",
    "Financials":            "XLF",
    "Consumer Discretionary":"XLY",
    "Industrials":           "XLI",
    "Energy":                "XLE",
    "Materials":             "XLB",
    "Real Estate":           "XLRE",
    "Utilities":             "XLU",
    "Communication Services":"XLC",
    "Consumer Staples":      "XLP",
}


@dataclass
class CrisisResult:
    name: str
    max_drawdown:    float | None   # negative number, e.g. -0.45 means -45%
    volatility:      float | None   # annualized
    recovered:       bool  | None   # True if price returned to pre-crisis level
    spy_drawdown:    float | None = None   # SPY drawdown for same window
    relative_vs_spy: float | None = None  # stock_dd - spy_dd (positive = outperformed)


@dataclass
class StressTestResult:
    ticker: str
    resilience_score: float         # 0-100; higher = more resilient
    crises: list[CrisisResult] = field(default_factory=list)
    label: str = ""                 # "Resilient" / "Moderate" / "Vulnerable"
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "resilience_score":      round(self.resilience_score, 1),
            "label":                 self.label,
            "error":                 self.error,
            "n_crises_with_data":    sum(1 for c in self.crises if c.max_drawdown is not None),
            "crises": [
                {
                    "name":            c.name,
                    "max_drawdown":    round(c.max_drawdown * 100, 1) if c.max_drawdown is not None else None,
                    "volatility":      round(c.volatility * 100, 1) if c.volatility is not None else None,
                    "recovered":       c.recovered,
                    "spy_drawdown":    round(c.spy_drawdown * 100, 1) if c.spy_drawdown is not None else None,
                    "relative_vs_spy": round(c.relative_vs_spy * 100, 1) if c.relative_vs_spy is not None else None,
                }
                for c in self.crises
            ],
        }


def _load_cache() -> dict:
    if _CACHE_FILE.exists():
        try:
            with _CACHE_FILE.open() as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_cache(cache: dict) -> None:
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_FILE.open("w") as f:
        json.dump(cache, f, indent=2)


def _analyze_crisis(prices: Any, start: str, end: str) -> CrisisResult | None:
    """Extract crisis-window metrics from a price Series."""
    try:
        import pandas as pd
        start_dt = pd.Timestamp(start)
        end_dt   = pd.Timestamp(end)

        window = prices.loc[start_dt:end_dt].dropna()
        if len(window) < 10:
            return None

        baseline = float(window.iloc[0])
        if baseline <= 0:
            return None

        # Max drawdown
        running_max = window.expanding().max()
        drawdowns   = (window - running_max) / running_max
        max_dd      = float(drawdowns.min())

        # Intra-crisis volatility (annualized)
        rets = window.pct_change().dropna()
        vol  = float(rets.std() * (252 ** 0.5)) if len(rets) >= 5 else None

        # Recovery: did price get back to baseline by end of window?
        final = float(window.iloc[-1])
        recovered = final >= baseline * 0.95   # 5% tolerance

        return CrisisResult(
            name        = "",
            max_drawdown = max_dd,
            volatility   = vol,
            recovered    = recovered,
        )
    except Exception as exc:
        log.debug("Crisis analysis error: %s", exc)
        return None


def _resilience_from_crises(crises: list[CrisisResult]) -> float:
    """
    Convert crisis metrics to 0-100 resilience score.
    Drawdown: -20% or better → 100 pts; -50% or worse → 0 pts
    Volatility: <30% → bonus; >80% → penalty
    """
    if not crises:
        return 50.0

    scores = []
    for c in crises:
        s = 100.0
        if c.max_drawdown is not None:
            dd = abs(c.max_drawdown)
            # Map [0%, 50%] → [100, 0]; clip beyond 50%
            s -= min(100.0, (dd / 0.50) * 100.0)
        if c.volatility is not None:
            if c.volatility > 0.80:
                s -= 15.0
            elif c.volatility > 0.50:
                s -= 8.0
        if c.recovered is True:
            s += 10.0   # partial recovery bonus
        scores.append(max(0.0, min(110.0, s)))

    return max(0.0, min(100.0, sum(scores) / len(scores)))


def compute_stress_test(ticker: str) -> StressTestResult:
    """
    Fetch historical prices and compute resilience score.
    Results are cached for 30 days.
    """
    cache = _load_cache()
    today = date.today().isoformat()

    cached = cache.get(ticker)
    if cached and cached.get("cached_on"):
        try:
            cached_date = date.fromisoformat(cached["cached_on"])
            if (date.today() - cached_date).days < _CACHE_TTL_DAYS:
                r = cached["result"]
                crises = [
                    CrisisResult(
                        name         = c["name"],
                        max_drawdown = c["max_drawdown"] / 100 if c["max_drawdown"] is not None else None,
                        volatility   = c["volatility"] / 100 if c["volatility"] is not None else None,
                        recovered    = c["recovered"],
                    )
                    for c in r.get("crises", [])
                ]
                return StressTestResult(
                    ticker           = ticker,
                    resilience_score = r["resilience_score"],
                    label            = r["label"],
                    crises           = crises,
                )
        except Exception:
            pass

    try:
        import yfinance as yf
        # Fetch ticker + SPY together going back to 2007 to cover all crisis windows
        raw = yf.download([ticker, "SPY"], start="2007-01-01", end=today,
                          progress=False, auto_adjust=True, group_by="ticker")
        try:
            if isinstance(raw.columns, type(raw.columns)) and hasattr(raw.columns, "levels"):
                prices     = raw["Close"][ticker].dropna()
                spy_prices = raw["Close"]["SPY"].dropna()
            else:
                prices     = raw[ticker]["Close"].dropna()
                spy_prices = raw["SPY"]["Close"].dropna()
        except Exception:
            # Single-ticker fallback
            hist = yf.Ticker(ticker).history(start="2007-01-01", end=today, auto_adjust=True)
            if hist.empty or "Close" not in hist.columns:
                raise ValueError("No price history")
            prices = hist["Close"]
            spy_prices = None

        if prices.empty:
            raise ValueError("No price history")

        crises = []
        for name, (start, end) in CRISIS_WINDOWS.items():
            r = _analyze_crisis(prices, start, end)
            if r:
                r.name = name
                # Add SPY comparison
                if spy_prices is not None:
                    spy_r = _analyze_crisis(spy_prices, start, end)
                    if spy_r:
                        r.spy_drawdown    = spy_r.max_drawdown
                        r.relative_vs_spy = (
                            (r.max_drawdown or 0.0) - (spy_r.max_drawdown or 0.0)
                        )   # positive means stock fell less than SPY
                crises.append(r)

        resilience = _resilience_from_crises(crises)
        if resilience >= 75:
            label = "Resilient"
        elif resilience >= 50:
            label = "Moderate"
        else:
            label = "Vulnerable"

        result = StressTestResult(
            ticker           = ticker,
            resilience_score = resilience,
            crises           = crises,
            label            = label,
        )

        cache[ticker] = {"cached_on": today, "result": result.to_dict()}
        _save_cache(cache)
        return result

    except Exception as exc:
        log.warning("%s stress test failed: %s", ticker, exc)
        result = StressTestResult(
            ticker           = ticker,
            resilience_score = 50.0,
            label            = "Moderate",
            error            = str(exc)[:100],
        )
        return result
