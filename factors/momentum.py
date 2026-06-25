"""
Momentum factor — 4 sub-signals, equal-weighted.

Sub-signals:
  1. 6-month return          (P₀ − P₋₁₂₆) / P₋₁₂₆
  2. 12-1 month return       (P₋₂₁ − P₋₂₅₂) / P₋₂₅₂   (skips most recent month)
  3. Price > 50DMA           binary: 100 if true, 0 if false
  4. 50DMA > 200DMA          binary: 100 if true, 0 if false

Raw scores are returned per ticker. Cross-sectional normalization happens
in composite.py so that all factors share the same z-score pipeline.

Missing data for a sub-signal → that sub-signal is excluded from the
ticker's average (not substituted with zero).
If fewer than 2 sub-signals are available → raw score = None (→ neutral 50
in composite).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Trading day approximations
_6M_DAYS   = 126
_1M_DAYS   = 21
_12M_DAYS  = 252
_50D       = 50
_200D      = 200


@dataclass
class MomentumRaw:
    ticker: str
    ret_6m: float | None          = None   # 6-month price return
    ret_12_1m: float | None       = None   # 12-1 month return (skip last month)
    price_above_50dma: bool | None = None  # True/False/None
    golden_cross: bool | None      = None  # 50DMA > 200DMA
    raw_score: float | None        = None  # average of available signals (0–100 binary avg or raw %)
    signals_used: list[str]        = field(default_factory=list)


def compute_momentum(ticker: str, prices: list[float]) -> MomentumRaw:
    """
    Compute momentum sub-signals from a list of daily closing prices.

    Parameters
    ----------
    ticker : str
    prices : list[float]
        Daily closing prices ordered oldest → newest.
        Minimum length needed: 252 trading days (≈1 year).

    Returns
    -------
    MomentumRaw with raw float sub-signals (not yet normalized).
    """
    result = MomentumRaw(ticker=ticker)

    if not prices or len(prices) < 2:
        log.debug("%s: insufficient price history (%d days)", ticker, len(prices))
        return result

    n = len(prices)
    p0 = prices[-1]   # today

    # ── 6-month return ──────────────────────────────────────────────────
    if n >= _6M_DAYS + 1:
        p_6m = prices[-(  _6M_DAYS + 1)]
        if p_6m and p_6m != 0:
            result.ret_6m = (p0 - p_6m) / abs(p_6m)
            result.signals_used.append("ret_6m")

    # ── 12-1 month return (skip most recent month) ──────────────────────
    if n >= _12M_DAYS + 1:
        p_12m = prices[-(_12M_DAYS + 1)]
        p_1m  = prices[-(_1M_DAYS  + 1)]
        if p_12m and p_12m != 0:
            result.ret_12_1m = (p_1m - p_12m) / abs(p_12m)
            result.signals_used.append("ret_12_1m")

    # ── Moving averages ─────────────────────────────────────────────────
    if n >= _50D:
        ma50 = sum(prices[-_50D:]) / _50D
        result.price_above_50dma = p0 > ma50
        result.signals_used.append("price_above_50dma")

        if n >= _200D:
            ma200 = sum(prices[-_200D:]) / _200D
            result.golden_cross = ma50 > ma200
            result.signals_used.append("golden_cross")

    # ── Raw composite: average available signals ─────────────────────────
    # Binary signals: True→1.0, False→0.0 (will be z-scored cross-sectionally)
    values: list[float] = []
    if result.ret_6m    is not None: values.append(result.ret_6m)
    if result.ret_12_1m is not None: values.append(result.ret_12_1m)
    if result.price_above_50dma is not None: values.append(1.0 if result.price_above_50dma else 0.0)
    if result.golden_cross      is not None: values.append(1.0 if result.golden_cross else 0.0)

    if len(values) >= 2:
        result.raw_score = sum(values) / len(values)

    return result


def compute_momentum_batch(ticker_prices: dict[str, list[float]]) -> dict[str, MomentumRaw]:
    """
    Compute momentum for multiple tickers.

    Parameters
    ----------
    ticker_prices : {ticker: [daily_close, ...]}  oldest→newest

    Returns
    -------
    {ticker: MomentumRaw}
    """
    return {t: compute_momentum(t, prices) for t, prices in ticker_prices.items()}
