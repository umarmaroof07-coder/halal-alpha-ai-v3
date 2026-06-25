"""
Valuation factor — 3 sub-signals.

Sub-signals:
  1. Trailing P/E    Price / Trailing EPS         [INVERTED — lower is better]
  2. Forward P/E     Price / FY1 Estimated EPS    [INVERTED — lower is better]
  3. FCF yield       FCF / Market Cap             [higher is better]

Inversion for P/E ratios is applied BEFORE returning raw scores so that
the cross-sectional z-scorer in composite.py treats all signals uniformly
(higher raw = better).

Negative P/E (loss-making) → sub-signal excluded (not zeroed).
Negative FCF → FCF yield is still passed through (negative yield = bad).
Missing data → sub-signal excluded. Fewer than 1 valid signal → raw_score = None.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Cap P/E inversion to prevent division by very small numbers distorting scores
_MIN_PE = 1.0    # ignore P/E below 1 (likely data error)
_MAX_PE = 200.0  # cap extreme P/E ratios before inverting


@dataclass
class ValuationRaw:
    ticker: str
    trailing_pe_inv: float | None  = None   # 1/PE (higher = cheaper)
    forward_pe_inv:  float | None  = None   # 1/forward_PE
    fcf_yield:       float | None  = None   # FCF / Market Cap
    raw_score:       float | None  = None
    signals_used:    list[str]     = field(default_factory=list)


def _invert_pe(pe: float | None) -> float | None:
    """Return 1/PE clamped to valid range, or None if invalid."""
    if pe is None or pe <= 0 or pe < _MIN_PE:
        return None
    capped = min(pe, _MAX_PE)
    return 1.0 / capped


def compute_valuation(
    ticker: str,
    price:              float | None,
    trailing_eps:       float | None,
    forward_eps:        float | None,
    free_cash_flow:     float | None,
    market_cap:         float | None,
    # Allow direct P/E pass-through (avoids recomputing when provider supplies it)
    trailing_pe:        float | None = None,
    forward_pe:         float | None = None,
) -> ValuationRaw:
    """
    Compute valuation sub-signals for a single ticker.

    Prefer direct P/E values if provided; fall back to price/eps computation.
    """
    result = ValuationRaw(ticker=ticker)

    # ── Trailing P/E ────────────────────────────────────────────────────
    pe_trailing = trailing_pe
    if pe_trailing is None and price is not None and trailing_eps is not None and trailing_eps > 0:
        pe_trailing = price / trailing_eps

    inv = _invert_pe(pe_trailing)
    if inv is not None:
        result.trailing_pe_inv = inv
        result.signals_used.append("trailing_pe_inv")

    # ── Forward P/E ─────────────────────────────────────────────────────
    pe_forward = forward_pe
    if pe_forward is None and price is not None and forward_eps is not None and forward_eps > 0:
        pe_forward = price / forward_eps

    inv_fwd = _invert_pe(pe_forward)
    if inv_fwd is not None:
        result.forward_pe_inv = inv_fwd
        result.signals_used.append("forward_pe_inv")

    # ── FCF yield ────────────────────────────────────────────────────────
    if free_cash_flow is not None and market_cap is not None and market_cap > 0:
        result.fcf_yield = free_cash_flow / market_cap
        result.signals_used.append("fcf_yield")

    # ── Raw composite ────────────────────────────────────────────────────
    values: list[float] = [
        v for v in [result.trailing_pe_inv, result.forward_pe_inv, result.fcf_yield]
        if v is not None
    ]
    if len(values) >= 1:
        result.raw_score = sum(values) / len(values)

    return result


def compute_valuation_batch(stocks: list[dict]) -> dict[str, ValuationRaw]:
    """
    Compute valuation for multiple tickers.

    Each dict must contain: ticker, price, marketCap, freeCashFlow,
    peRatio (optional), estimatedEpsAvg (optional), eps (optional).
    """
    results: dict[str, ValuationRaw] = {}
    for s in stocks:
        ticker = s.get("ticker") or s.get("symbol", "")
        results[ticker] = compute_valuation(
            ticker          = ticker,
            price           = s.get("price"),
            trailing_eps    = s.get("eps"),
            forward_eps     = s.get("estimatedEpsAvg"),
            free_cash_flow  = s.get("freeCashFlow"),
            market_cap      = s.get("mktCap") or s.get("marketCap"),
            trailing_pe     = s.get("peRatio"),
            forward_pe      = s.get("forwardPE"),
        )
    return results
