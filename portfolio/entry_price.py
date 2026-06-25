"""
portfolio/entry_price.py — Institutional Buy Limit Price Engine (V6)

Produces ATR-anchored entry prices and an entry score for every BUY NOW stock.
No arbitrary fixed discounts. Every number is derived from:
  - live 14-day ATR (volatility anchor)
  - model confidence (price tolerance)
  - valuation score (attractiveness)
  - momentum score (pace of pullback needed)
  - risk adjustment score (margin of safety)

Outputs
-------
  buy_limit        : entry when price dips 0.5 × ATR below current
  strong_buy_limit : entry on a deeper dip of 1.0 × ATR
  entry_score      : 0-100 composite (40% val + 20% mom + 20% conf + 20% risk)
  entry_rating     : "Strong Buy" | "Buy" | "Watch" | "Wait"

Neither output is cached. They are recomputed on every call with fresh ATR.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

# ── Confidence label → ATR multiplier adjustment ──────────────────────────────
# Negative delta = smaller multiplier = smaller deduction = limit CLOSER to market (more willing to pay)
# Positive delta = larger multiplier = bigger deduction = limit FURTHER from market (demand more pullback)
_CONF_MULTIPLIER_DELTA: dict[str, float] = {
    "Very High": -0.10,   # very confident → buy closer to market price
    "High":      -0.06,
    "Medium":    +0.00,
    "Low":       +0.06,   # less confident → wait for a larger pullback
    "Very Low":  +0.10,
}

# ── Entry score thresholds ─────────────────────────────────────────────────────
_RATING_BANDS = [
    (80, "Strong Buy"),
    (60, "Buy"),
    (40, "Watch"),
    (0,  "Wait"),
]

# ── Score weights ──────────────────────────────────────────────────────────────
_W_VAL  = 0.40
_W_MOM  = 0.20
_W_CONF = 0.20
_W_RISK = 0.20


@dataclass
class EntryPriceResult:
    ticker:            str
    current_price:     float
    atr14:             Optional[float]      # None if history unavailable
    buy_limit:         Optional[float]      # current_price − adjusted_0.5_ATR
    strong_buy_limit:  Optional[float]      # current_price − adjusted_1.0_ATR
    entry_score:       float                # 0–100
    entry_rating:      str                  # "Strong Buy" | "Buy" | "Watch" | "Wait"
    pct_above_buy:     Optional[float]      # how far current price sits above buy_limit
    pct_above_strong:  Optional[float]
    valuation_score:   float
    momentum_score:    float
    model_confidence:  float                # 0–100 numeric score
    confidence_label:  str
    risk_score:        float
    composite_score:   float
    explanation:       str
    computed_at:       str = field(default_factory=lambda: datetime.now().isoformat())

    # component sub-scores (0-100 each, for transparency)
    val_component:     float = 0.0
    mom_component:     float = 0.0
    conf_component:    float = 0.0
    risk_component:    float = 0.0

    def to_dict(self) -> dict:
        def _r(v, d=2):
            return round(float(v), d) if v is not None else None

        return {
            "ticker":           self.ticker,
            "current_price":    _r(self.current_price),
            "atr14":            _r(self.atr14),
            "buy_limit":        _r(self.buy_limit),
            "strong_buy_limit": _r(self.strong_buy_limit),
            "entry_score":      _r(self.entry_score, 1),
            "entry_rating":     self.entry_rating,
            "pct_above_buy":    _r(self.pct_above_buy, 2),
            "pct_above_strong": _r(self.pct_above_strong, 2),
            "valuation_score":  _r(self.valuation_score, 1),
            "momentum_score":   _r(self.momentum_score, 1),
            "model_confidence": _r(self.model_confidence, 1),
            "confidence_label": self.confidence_label,
            "risk_score":       _r(self.risk_score, 1),
            "composite_score":  _r(self.composite_score, 1),
            "val_component":    _r(self.val_component, 1),
            "mom_component":    _r(self.mom_component, 1),
            "conf_component":   _r(self.conf_component, 1),
            "risk_component":   _r(self.risk_component, 1),
            "explanation":      self.explanation,
            "computed_at":      self.computed_at,
        }


# ---------------------------------------------------------------------------
# ATR helper (Wilder 14-day)
# ---------------------------------------------------------------------------

def _compute_atr14(ticker: str) -> Optional[float]:
    """Fetch the last 30 days of daily bars and compute Wilder ATR-14."""
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="30d", auto_adjust=True)
        if hist.empty or len(hist) < 15:
            return None
        highs  = hist["High"].values
        lows   = hist["Low"].values
        closes = hist["Close"].values

        n = 14
        # True Range for each bar (bar i uses closes[i-1] as prior close)
        n_bars = min(len(highs), len(lows), len(closes) - 1)
        if n_bars < n:
            return None
        trs = [
            max(highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i]  - closes[i - 1]))
            for i in range(1, n_bars + 1)
        ]
        # Wilder smoothing: seed with simple mean of first n, then smooth
        atr = sum(trs[:n]) / n
        for tr in trs[n:]:
            atr = (atr * (n - 1) + tr) / n
        return float(atr)
    except Exception as exc:
        log.warning("ATR fetch failed for %s: %s", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# Confidence label → numeric score
# ---------------------------------------------------------------------------

def _confidence_label(score: float) -> str:
    if score >= 90:
        return "Very High"
    if score >= 80:
        return "High"
    if score >= 65:
        return "Medium"
    if score >= 50:
        return "Low"
    return "Very Low"


# ---------------------------------------------------------------------------
# Phase 2 — ATR-anchored limit prices
# ---------------------------------------------------------------------------

def compute_buy_limit(
    current_price:    float,
    atr14:            float,
    confidence_label: str,
    valuation_score:  float,
    momentum_score:   float,
    risk_score:       float,
) -> float:
    """
    Buy-limit = current_price − (adjusted_multiplier × ATR_14)

    Base multiplier: 0.5
    Adjustments:
      Confidence  → ±0.10 on the multiplier (more confident = smaller pullback)
      Valuation   → score > 80 raises limit (stock cheap → buy closer to price)
                  → score < 40 lowers limit (expensive → wait for larger dip)
      Momentum    → score > 70 raises limit (trending → smaller pullback acceptable)
                  → score < 40 lowers limit (weak trend → need larger cushion)
      Risk        → risk_score > 75: +0.05 (safe balance sheet → tighter)
                  → risk_score < 40: −0.10 (risky → deeper discount required)
    """
    mult = 0.50

    # Confidence: negative delta → smaller mult → limit closer to market (raise)
    mult += _CONF_MULTIPLIER_DELTA.get(confidence_label, 0.0)

    # Valuation: cheap stock (>80) → buy closer = smaller mult (raise limit)
    if valuation_score > 80:
        mult -= 0.05
    elif valuation_score < 40:
        mult += 0.10    # expensive → bigger deduction → lower limit

    # Momentum: trending (>70) → smaller deduction acceptable (raise limit)
    if momentum_score > 70:
        mult -= 0.05
    elif momentum_score < 40:
        mult += 0.08    # weak trend → bigger cushion needed → lower limit

    # Risk: safe balance sheet (>75) → tighter limit OK (raise)
    if risk_score > 75:
        mult -= 0.05
    elif risk_score < 40:
        mult += 0.10    # risky → deeper margin of safety → lower limit

    # Floor: always subtract at least 0.10 × ATR (limit must be below price)
    mult = max(0.10, mult)

    return current_price - (mult * atr14)


def compute_strong_buy_limit(
    current_price:    float,
    atr14:            float,
    confidence_label: str,
    valuation_score:  float,
    momentum_score:   float,
    risk_score:       float,
) -> float:
    """
    Strong-buy-limit = current_price − (adjusted_multiplier × ATR_14)

    Base multiplier: 1.0 (exactly double the buy-limit base)
    Same adjustment logic as compute_buy_limit but starts from 1.0.
    """
    mult = 1.00

    mult += _CONF_MULTIPLIER_DELTA.get(confidence_label, 0.0)

    if valuation_score > 80:
        mult -= 0.05
    elif valuation_score < 40:
        mult += 0.10

    if momentum_score > 70:
        mult -= 0.05
    elif momentum_score < 40:
        mult += 0.08

    if risk_score > 75:
        mult -= 0.05
    elif risk_score < 40:
        mult += 0.10

    mult = max(0.20, mult)   # strong-buy always deeper than 0.2 × ATR

    return current_price - (mult * atr14)


# ---------------------------------------------------------------------------
# Phase 3 — Entry score
# ---------------------------------------------------------------------------

def compute_entry_score(
    valuation_score:  float,
    momentum_score:   float,
    model_confidence: float,
    risk_score:       float,
) -> tuple[float, float, float, float, float]:
    """
    Returns (entry_score, val_component, mom_component, conf_component, risk_component).

    Components are each scored 0–100 then weighted:
      40% valuation attractiveness
      20% momentum
      20% model confidence
      20% risk

    Valuation attractiveness is the raw valuation_score (higher = cheaper relative to value).
    Momentum is the raw momentum_score.
    Confidence is the raw model_confidence.
    Risk attractiveness: risk_score already scales 0–100 (higher = safer).
    """
    val_c  = float(max(0.0, min(100.0, valuation_score)))
    mom_c  = float(max(0.0, min(100.0, momentum_score)))
    conf_c = float(max(0.0, min(100.0, model_confidence)))
    risk_c = float(max(0.0, min(100.0, risk_score)))

    score = (
        _W_VAL  * val_c +
        _W_MOM  * mom_c +
        _W_CONF * conf_c +
        _W_RISK * risk_c
    )
    return round(score, 1), val_c, mom_c, conf_c, risk_c


def _entry_rating(score: float) -> str:
    for threshold, label in _RATING_BANDS:
        if score >= threshold:
            return label
    return "Wait"


# ---------------------------------------------------------------------------
# Phase 1 — Master function
# ---------------------------------------------------------------------------

def compute_entry_analysis(
    ticker:           str,
    current_price:    float,
    valuation_score:  float,
    composite_score:  float,
    model_confidence: float,   # 0–100 numeric
    risk_score:       float,
    momentum_score:   float,
    atr14:            Optional[float] = None,  # pre-supplied; fetched if None
) -> EntryPriceResult:
    """
    Compute full entry analysis for one stock.

    If atr14 is None, fetches 30 days of yfinance history to compute it.
    No entry prices are generated when ATR is unavailable.
    """
    if atr14 is None:
        atr14 = _compute_atr14(ticker)

    conf_label = _confidence_label(model_confidence)

    buy_limit:       Optional[float] = None
    strong_limit:    Optional[float] = None
    pct_above_buy:   Optional[float] = None
    pct_above_strong: Optional[float] = None
    explanation = ""

    if atr14 and atr14 > 0 and current_price > 0:
        buy_limit    = compute_buy_limit(
            current_price, atr14, conf_label,
            valuation_score, momentum_score, risk_score,
        )
        strong_limit = compute_strong_buy_limit(
            current_price, atr14, conf_label,
            valuation_score, momentum_score, risk_score,
        )
        buy_limit    = round(max(buy_limit,    0.01), 2)
        strong_limit = round(max(strong_limit, 0.01), 2)

        if current_price > buy_limit:
            pct_above_buy = round((current_price - buy_limit) / buy_limit * 100, 2)
        if current_price > strong_limit:
            pct_above_strong = round((current_price - strong_limit) / strong_limit * 100, 2)

        # Human-readable explanation
        if current_price <= buy_limit:
            explanation = (
                f"Currently within Buy Zone "
                f"(price ${current_price:.2f} ≤ buy limit ${buy_limit:.2f}). "
                "Consider placing a limit order at or below the Buy Limit."
            )
        elif current_price <= strong_limit:
            explanation = (
                f"Currently within Strong Buy Zone "
                f"(price ${current_price:.2f} ≤ strong buy ${strong_limit:.2f}). "
                "Consider a full position at or below the Strong Buy Limit."
            )
        else:
            explanation = (
                f"Price is {pct_above_buy:.1f}% above Buy Limit (${buy_limit:.2f}) "
                f"and {pct_above_strong:.1f}% above Strong Buy Limit (${strong_limit:.2f}). "
                f"ATR-14 = ${atr14:.2f}. "
                f"Confidence: {conf_label}."
            )
    else:
        explanation = "ATR unavailable — insufficient price history for entry calculation."

    entry_score, val_c, mom_c, conf_c, risk_c = compute_entry_score(
        valuation_score, momentum_score, model_confidence, risk_score
    )
    rating = _entry_rating(entry_score)

    return EntryPriceResult(
        ticker            = ticker,
        current_price     = current_price,
        atr14             = atr14,
        buy_limit         = buy_limit,
        strong_buy_limit  = strong_limit,
        entry_score       = entry_score,
        entry_rating      = rating,
        pct_above_buy     = pct_above_buy,
        pct_above_strong  = pct_above_strong,
        valuation_score   = valuation_score,
        momentum_score    = momentum_score,
        model_confidence  = model_confidence,
        confidence_label  = conf_label,
        risk_score        = risk_score,
        composite_score   = composite_score,
        explanation       = explanation,
        val_component     = val_c,
        mom_component     = mom_c,
        conf_component    = conf_c,
        risk_component    = risk_c,
    )


def compute_entry_analyses_for_picks(
    picks: list[dict],
) -> dict[str, EntryPriceResult]:
    """
    Compute entry analysis for a list of scored pick dicts.
    Each dict must have: ticker, price, valuation, composite, model_confidence,
    risk_adjustment, momentum (all 0–100 float fields from scored_universe.json).
    Returns {ticker: EntryPriceResult}.
    """
    results: dict[str, EntryPriceResult] = {}
    for pick in picks:
        ticker = pick.get("ticker", "")
        if not ticker:
            continue
        price = float(pick.get("price") or pick.get("current_price") or 0)
        if price <= 0:
            continue
        results[ticker] = compute_entry_analysis(
            ticker           = ticker,
            current_price    = price,
            valuation_score  = float(pick.get("valuation", 50)),
            composite_score  = float(pick.get("composite", 50)),
            model_confidence = float(pick.get("model_confidence", 65)),
            risk_score       = float(pick.get("risk_adjustment", 50)),
            momentum_score   = float(pick.get("momentum", 50)),
        )
    return results
