"""
portfolio/limit_prices.py — Disciplined limit-order price suggestions.

Produces three price tiers (conservative / normal / aggressive) for each
BUY NOW stock and recommends one tier based on liquidity and volatility.

SAFETY RULES (never violated):
  - Never suggests market orders.
  - Price data must be fresh (< 5 minutes) or no limit is generated.
  - Bid/ask availability is preferred; ATR/volatility used as fallback.
  - All output carries an explicit fill-not-guaranteed disclaimer.

Do NOT import this module in the factor model, ranking engine, or
portfolio constructor — it is output-only and must not affect scores.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger(__name__)

_STALE_MINUTES    = 5      # refuse to generate limits if price older than this
_ATR_PERIOD       = 14     # standard 14-day ATR
_MA5_PERIOD       = 5
_MA20_PERIOD      = 20
_TIGHT_SPREAD_PCT = 0.10   # spread ≤ 0.10% → liquid
_HIGH_VOL_DAILY   = 0.02   # daily vol ≥ 2% → volatile regime
_FAST_MARKET_PCT  = 0.50   # price > 0.50% above 5-day avg → upward momentum

DISCLAIMER = "Use limit orders only. Fill is not guaranteed. Price may not be reached."


@dataclass
class LimitPriceResult:
    ticker:           str
    live_price:       float
    bid:              Optional[float]
    ask:              Optional[float]
    prev_close:       Optional[float]
    ma5:              Optional[float]
    ma20:             Optional[float]
    atr14:            Optional[float]
    daily_vol:        Optional[float]          # annualised → daily
    spread_pct:       Optional[float]
    conservative_limit: Optional[float]
    normal_limit:       Optional[float]
    aggressive_limit:   Optional[float]
    suggested_limit:    Optional[float]
    suggested_tier:     str                    # "conservative" | "normal" | "aggressive"
    explanation:        str
    warning:            str                    # non-empty when fast-market flag fires
    allocation:         float                  # $ amount to invest
    shares:             Optional[float]        # allocation / suggested_limit
    estimated_fill_cost: Optional[float]       # shares × suggested_limit
    stale:              bool = False
    error:              str  = ""
    disclaimer:         str  = DISCLAIMER

    def to_dict(self) -> dict:
        return {
            "ticker":              self.ticker,
            "live_price":         _r(self.live_price),
            "bid":                _r(self.bid),
            "ask":                _r(self.ask),
            "prev_close":         _r(self.prev_close),
            "ma5":                _r(self.ma5),
            "ma20":               _r(self.ma20),
            "atr14":              _r(self.atr14),
            "daily_vol":          _r(self.daily_vol, 4),
            "spread_pct":         _r(self.spread_pct, 4),
            "conservative_limit": _r(self.conservative_limit),
            "normal_limit":       _r(self.normal_limit),
            "aggressive_limit":   _r(self.aggressive_limit),
            "suggested_limit":    _r(self.suggested_limit),
            "suggested_tier":     self.suggested_tier,
            "explanation":        self.explanation,
            "warning":            self.warning,
            "allocation":         _r(self.allocation),
            "shares":             _r(self.shares, 4) if self.shares else None,
            "estimated_fill_cost": _r(self.estimated_fill_cost),
            "stale":              self.stale,
            "error":              self.error,
            "disclaimer":         self.disclaimer,
        }


def _r(val, decimals: int = 2):
    if val is None:
        return None
    try:
        return round(float(val), decimals)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Price data fetch (yfinance directly — fastest, no FMP rate limits for this)
# ---------------------------------------------------------------------------

def _fetch_price_data(ticker: str) -> dict:
    """
    Returns a dict with keys:
      price, bid, ask, prev_close, history (list of closing prices, newest last),
      timestamp (ISO string of when price was fetched).
    """
    import yfinance as yf

    t = yf.Ticker(ticker)
    info = t.info or {}

    price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("bid")         # last resort
    )
    bid  = info.get("bid") or info.get("bidPrice")
    ask  = info.get("ask") or info.get("askPrice")
    prev = info.get("previousClose") or info.get("regularMarketPreviousClose")

    # 30 calendar days of history for ATR + moving averages
    hist = t.history(period="30d", auto_adjust=True)
    closes = list(hist["Close"].dropna()) if not hist.empty else []
    highs  = list(hist["High"].dropna())  if not hist.empty else []
    lows   = list(hist["Low"].dropna())   if not hist.empty else []

    return {
        "price":      float(price) if price else None,
        "bid":        float(bid)   if bid  else None,
        "ask":        float(ask)   if ask  else None,
        "prev_close": float(prev)  if prev else None,
        "closes":     closes,
        "highs":      highs,
        "lows":       lows,
        "timestamp":  datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Technical computations
# ---------------------------------------------------------------------------

def _ma(closes: list[float], n: int) -> Optional[float]:
    if len(closes) < n:
        return None
    return sum(closes[-n:]) / n


def _atr(highs: list[float], lows: list[float], closes: list[float], n: int = 14) -> Optional[float]:
    """Wilder's ATR over n periods using True Range."""
    if len(closes) < n + 1 or len(highs) < n or len(lows) < n:
        return None
    # highs[i] and lows[i] are bar i; closes[i] is the close of bar i
    # True Range for bar i uses closes[i-1] as prior close
    n_bars = min(len(highs), len(lows), len(closes) - 1)
    trs = []
    for i in range(n_bars):
        h, l, pc = highs[i], lows[i], closes[i]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < n:
        return None
    # Simple average for first ATR, then Wilder smoothing
    atr = sum(trs[:n]) / n
    for tr in trs[n:]:
        atr = (atr * (n - 1) + tr) / n
    return atr


def _daily_vol(closes: list[float], n: int = 20) -> Optional[float]:
    """Daily return std-dev over last n closes."""
    if len(closes) < n + 1:
        return None
    import statistics
    rets = [
        math.log(closes[i] / closes[i - 1])
        for i in range(len(closes) - n, len(closes))
        if closes[i - 1] > 0
    ]
    if len(rets) < 5:
        return None
    return statistics.stdev(rets)


# ---------------------------------------------------------------------------
# Core limit price logic
# ---------------------------------------------------------------------------

def _compute_limits(
    price: float,
    bid:   Optional[float],
    ask:   Optional[float],
    atr:   Optional[float],
) -> tuple[float, float, float]:
    """
    Returns (conservative_limit, normal_limit, aggressive_limit).

    Conservative: price - 0.5%  OR  price - 0.25 × ATR  (whichever is lower)
    Normal:       price - 0.25% OR  bid/ask midpoint      (whichever is lower)
    Aggressive:   ask price      OR  live price
    """
    # Conservative
    pct_conservative = price * 0.995           # price - 0.5%
    atr_conservative = (price - 0.25 * atr) if atr else pct_conservative
    conservative = min(pct_conservative, atr_conservative)

    # Normal
    pct_normal = price * 0.9975                # price - 0.25%
    if bid and ask and bid > 0 and ask > 0:
        midpoint = (bid + ask) / 2.0
        normal = min(pct_normal, midpoint)
    else:
        normal = pct_normal

    # Aggressive
    aggressive = ask if (ask and ask > 0) else price

    # Sanity: enforce conservative ≤ normal ≤ aggressive ≤ price * 1.01
    conservative = min(conservative, normal)
    aggressive   = min(aggressive, price * 1.01)

    return conservative, normal, aggressive


def _choose_tier(
    price:      float,
    bid:        Optional[float],
    ask:        Optional[float],
    ma5:        Optional[float],
    daily_vol:  Optional[float],
    spread_pct: Optional[float],
) -> tuple[str, str, str]:
    """
    Returns (tier, explanation, warning).
    Tier: "conservative" | "normal" | "aggressive"
    """
    warning = ""

    # Fast-market flag: price > 0.5% above 5-day average
    fast_market = (ma5 is not None) and (price > ma5 * (1 + _FAST_MARKET_PCT / 100))

    # Volatility flag
    high_vol = (daily_vol is not None) and (daily_vol >= _HIGH_VOL_DAILY)

    # Liquidity flag: tight spread and bid/ask available
    tight_spread = (
        spread_pct is not None
        and bid is not None
        and ask is not None
        and spread_pct <= _TIGHT_SPREAD_PCT / 100
    )

    if high_vol:
        tier = "conservative"
        explanation = (
            f"Volatile stock (daily σ = {daily_vol*100:.2f}%). "
            "Conservative limit applied to reduce overpaying in choppy conditions."
        )
    elif tight_spread and not fast_market:
        tier = "normal"
        explanation = (
            "Tight bid/ask spread — stock is liquid. "
            "Normal limit (price − 0.25% or midpoint) is appropriate."
        )
    elif fast_market:
        tier = "aggressive"
        explanation = (
            f"Price is above its 5-day average — market moving upward. "
            "Aggressive limit used to improve fill probability."
        )
        warning = (
            "⚠ FAST MARKET: Stock is trading above its recent average. "
            "Aggressive limit may still miss the fill if momentum continues. "
            "Consider waiting for a pullback or accepting the normal limit."
        )
    else:
        tier = "normal"
        explanation = (
            "No strong volatility or liquidity signal. "
            "Normal limit (price − 0.25%) applied."
        )

    return tier, explanation, warning


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_limit_price(
    ticker:     str,
    allocation: float,
    live_price_data: dict | None = None,
) -> LimitPriceResult:
    """
    Compute all three limit price tiers for `ticker`.

    Args:
        ticker:          Stock symbol.
        allocation:      Dollar amount to invest (from portfolio constructor).
        live_price_data: Optional pre-fetched price dict (same shape as _fetch_price_data).
                         If None, fetched fresh from yfinance.

    Returns a LimitPriceResult with all tiers, suggested limit, shares, and fill cost.
    Stale or failed fetches return a result with stale=True or error set.
    """
    raw: dict
    try:
        raw = live_price_data if live_price_data else _fetch_price_data(ticker)
    except Exception as exc:
        log.warning("limit_prices: fetch failed for %s: %s", ticker, exc)
        return LimitPriceResult(
            ticker=ticker, live_price=0.0, bid=None, ask=None,
            prev_close=None, ma5=None, ma20=None, atr14=None,
            daily_vol=None, spread_pct=None, conservative_limit=None,
            normal_limit=None, aggressive_limit=None, suggested_limit=None,
            suggested_tier="none", explanation="Price fetch failed.",
            warning="", allocation=allocation, shares=None,
            estimated_fill_cost=None, error=str(exc),
        )

    price = raw.get("price")
    if not price or price <= 0:
        return LimitPriceResult(
            ticker=ticker, live_price=0.0, bid=None, ask=None,
            prev_close=None, ma5=None, ma20=None, atr14=None,
            daily_vol=None, spread_pct=None, conservative_limit=None,
            normal_limit=None, aggressive_limit=None, suggested_limit=None,
            suggested_tier="none", explanation="No valid live price available.",
            warning="", allocation=allocation, shares=None,
            estimated_fill_cost=None, error="no_price",
        )

    # Staleness check
    stale = False
    ts_str = raw.get("timestamp", "")
    if ts_str:
        try:
            fetched_at = datetime.fromisoformat(ts_str)
            age_min = (datetime.now() - fetched_at).total_seconds() / 60
            if age_min > _STALE_MINUTES:
                stale = True
        except ValueError:
            pass

    if stale:
        return LimitPriceResult(
            ticker=ticker, live_price=float(price),
            bid=None, ask=None, prev_close=None,
            ma5=None, ma20=None, atr14=None, daily_vol=None,
            spread_pct=None, conservative_limit=None, normal_limit=None,
            aggressive_limit=None, suggested_limit=None,
            suggested_tier="none",
            explanation="Price data is stale — limit order price not generated.",
            warning="⚠ STALE DATA: Do not place limit orders until price refreshes.",
            allocation=allocation, shares=None, estimated_fill_cost=None,
            stale=True,
        )

    bid       = raw.get("bid")
    ask       = raw.get("ask")
    prev      = raw.get("prev_close")
    closes    = raw.get("closes", [])
    highs     = raw.get("highs", [])
    lows      = raw.get("lows", [])

    ma5       = _ma(closes, _MA5_PERIOD)
    ma20      = _ma(closes, _MA20_PERIOD)
    atr       = _atr(highs, lows, closes, _ATR_PERIOD)
    dvol      = _daily_vol(closes, 20)

    spread_pct: Optional[float] = None
    if bid and ask and bid > 0 and ask > 0:
        spread_pct = (ask - bid) / ((ask + bid) / 2)

    conservative, normal, aggressive = _compute_limits(float(price), bid, ask, atr)
    tier, explanation, warning = _choose_tier(
        float(price), bid, ask, ma5, dvol, spread_pct
    )

    suggested = {"conservative": conservative, "normal": normal, "aggressive": aggressive}[tier]
    shares = allocation / suggested if suggested and suggested > 0 else None
    fill_cost = shares * suggested if shares else None

    return LimitPriceResult(
        ticker=ticker,
        live_price=float(price),
        bid=bid,
        ask=ask,
        prev_close=prev,
        ma5=ma5,
        ma20=ma20,
        atr14=atr,
        daily_vol=dvol,
        spread_pct=spread_pct,
        conservative_limit=conservative,
        normal_limit=normal,
        aggressive_limit=aggressive,
        suggested_limit=suggested,
        suggested_tier=tier,
        explanation=explanation,
        warning=warning,
        allocation=allocation,
        shares=shares,
        estimated_fill_cost=fill_cost,
    )


def compute_limit_prices_for_picks(
    picks: list[dict],
) -> dict[str, LimitPriceResult]:
    """
    Compute limit prices for a list of Top-5 pick dicts.
    Each dict must have keys: "ticker", "dollar_amount".

    Returns {ticker: LimitPriceResult}.
    """
    results: dict[str, LimitPriceResult] = {}
    for pick in picks:
        ticker     = pick.get("ticker", "")
        allocation = float(pick.get("dollar_amount") or pick.get("allocation") or 0)
        if not ticker:
            continue
        results[ticker] = compute_limit_price(ticker, allocation)
    return results
