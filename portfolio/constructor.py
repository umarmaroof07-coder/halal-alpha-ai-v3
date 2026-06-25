"""
Portfolio Constructor v2 — risk-adjusted position sizing.

V4 upgrade: replaces fixed conviction weights with dynamic inverse-volatility
weighting, score-tilted and bounded.

Sizing algorithm:
  1. Compute 1-year daily return volatility (annualized) for each Top-5 ticker.
  2. Start with inverse-volatility weights: w_i = (1/vol_i) / sum(1/vol_j).
  3. Apply score tilt: w_i *= (composite_i / avg_composite)^0.3.
     Small exponent keeps the tilt modest and prevents score domination.
  4. Re-normalize to sum to 1.0.
  5. Clamp each weight to [MIN_WEIGHT, MAX_WEIGHT].
  6. Re-normalize again after clamping.
  7. Convert to dollar amounts: amount_i = w_i × ACCOUNT_SIZE.

Fallback: if volatility fetch fails for any ticker, falls back to equal
weighting with a score tilt (no inverse-vol component).

IMPORTANT: This module never outputs BUY NOW / WATCHLIST / AVOID labels.
Labels are the exclusive responsibility of recommendation_guard.safe_recommendations().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from config.settings import (
    ACCOUNT_SIZE,
    CONVICTION_WEIGHTS,
    CONVICTION_DOLLARS,
    MAX_POSITIONS,
)
from factors.composite import FactorScores
from portfolio.constraints import ConstraintResult

log = logging.getLogger(__name__)

MIN_WEIGHT = 0.05   # 5% floor — no position smaller than $100 on $2k account
MAX_WEIGHT = 0.25   # 25% ceiling — V5: tighter concentration limit


@dataclass
class PortfolioPosition:
    rank: int
    ticker: str
    composite_score: float
    conviction_weight: float   # final weight (risk-adjusted or fallback)
    dollar_amount: float
    price: float
    shares_to_buy: float
    constraint_result: ConstraintResult
    annualized_vol: float | None = None   # for display/audit
    sizing_method: str = "risk_adjusted"  # "risk_adjusted" | "equal_weighted_fallback"

    def to_dict(self) -> dict:
        return {
            "rank":              self.rank,
            "ticker":            self.ticker,
            "composite_score":   round(self.composite_score, 2),
            "conviction_weight": round(self.conviction_weight, 4),
            "dollar_amount":     round(self.dollar_amount, 2),
            "price":             self.price,
            "shares_to_buy":     round(self.shares_to_buy, 4),
            "annualized_vol":    round(self.annualized_vol, 4) if self.annualized_vol else None,
            "sizing_method":     self.sizing_method,
        }


@dataclass
class PortfolioResult:
    positions: list[PortfolioPosition] = field(default_factory=list)
    total_invested: float = 0.0
    cash_remaining: float = ACCOUNT_SIZE
    n_positions: int = 0
    sizing_method: str = "risk_adjusted"

    def to_dict(self) -> dict:
        return {
            "n_positions":    self.n_positions,
            "total_invested": round(self.total_invested, 2),
            "cash_remaining": round(self.cash_remaining, 2),
            "sizing_method":  self.sizing_method,
            "positions":      [p.to_dict() for p in self.positions],
        }


def _fetch_volatility(tickers: list[str]) -> dict[str, float]:
    """Fetch 1-year daily returns and compute annualized volatility."""
    import datetime
    try:
        import yfinance as yf
        end   = datetime.date.today()
        start = end - datetime.timedelta(days=365)
        raw = yf.download(
            tickers, start=str(start), end=str(end),
            progress=False, auto_adjust=True, group_by="ticker",
        )
        vols: dict[str, float] = {}
        for t in tickers:
            try:
                if len(tickers) == 1:
                    close = raw["Close"]
                else:
                    close = raw["Close"][t] if ("Close", t) in raw.columns or t in raw["Close"].columns else raw[t]["Close"]
                rets = close.dropna().pct_change().dropna()
                if len(rets) < 20:
                    continue
                vols[t] = float(rets.std() * (252 ** 0.5))
            except Exception as exc:
                log.debug("Vol fetch failed for %s: %s", t, exc)
        return vols
    except Exception as exc:
        log.warning("Volatility batch fetch failed: %s", exc)
        return {}


def _risk_adjusted_weights(
    tickers: list[str],
    scores:  list[float],
    vols:    dict[str, float],
) -> list[float]:
    """Compute inverse-volatility weights with score tilt, bounded [MIN, MAX]."""
    n = len(tickers)
    if not vols or n == 0:
        return []

    inv_vols = []
    missing  = []
    for i, t in enumerate(tickers):
        v = vols.get(t)
        if v and v > 0:
            inv_vols.append(1.0 / v)
        else:
            inv_vols.append(None)
            missing.append(i)

    if len(missing) == n:
        return []  # all missing — caller falls back to legacy

    # Fill missing with median inverse-vol
    valid = [v for v in inv_vols if v is not None]
    med_inv = sorted(valid)[len(valid) // 2]
    inv_vols = [v if v is not None else med_inv for v in inv_vols]

    # Score tilt: scores → relative multiplier
    avg_score = sum(scores) / n if n > 0 else 50.0
    if avg_score <= 0:
        avg_score = 50.0
    tilt = [(s / avg_score) ** 0.30 for s in scores]

    # Raw weights
    raw = [inv_vols[i] * tilt[i] for i in range(n)]
    total = sum(raw)
    if total <= 0:
        return []
    weights = [r / total for r in raw]

    # Clamp [MIN, MAX] with re-normalization (max 10 iterations)
    for _ in range(10):
        clamped = [max(MIN_WEIGHT, min(MAX_WEIGHT, w)) for w in weights]
        ct = sum(clamped)
        if ct <= 0:
            break
        weights = [c / ct for c in clamped]
        if all(MIN_WEIGHT - 1e-9 <= w <= MAX_WEIGHT + 1e-9 for w in weights):
            break

    return weights


def build_portfolio(
    ranked_scores: list[FactorScores],
    constraint_results: dict[str, ConstraintResult],
    prices: dict[str, float],
) -> PortfolioResult:
    """
    Build a portfolio with risk-adjusted position sizing.

    Falls back to legacy conviction weights if volatility data is unavailable.
    """
    candidates: list[FactorScores] = []
    for score in ranked_scores:
        if len(candidates) >= MAX_POSITIONS:
            break
        ticker = score.ticker
        cr = constraint_results.get(ticker)
        if cr is None or not cr.passed:
            continue
        price = prices.get(ticker)
        if price is None or price <= 0:
            continue
        candidates.append(score)

    if not candidates:
        return PortfolioResult()

    tickers     = [c.ticker for c in candidates]
    comp_scores = [c.composite for c in candidates]

    # Attempt risk-adjusted sizing
    vols    = _fetch_volatility(tickers)
    weights = _risk_adjusted_weights(tickers, comp_scores, vols)
    sizing_method = "risk_adjusted"

    if not weights or len(weights) != len(tickers):
        # Fallback: legacy fixed conviction weights
        weights = list(CONVICTION_WEIGHTS[: len(tickers)])
        total_w = sum(weights)
        weights = [w / total_w for w in weights]
        sizing_method = "equal_weighted_fallback"
        log.warning("Volatility unavailable — using fixed conviction weights")

    positions: list[PortfolioPosition] = []
    for rank, (score, weight) in enumerate(zip(candidates, weights), start=1):
        ticker        = score.ticker
        dollar_amount = round(weight * ACCOUNT_SIZE, 2)
        price         = prices[ticker]
        shares        = dollar_amount / price

        positions.append(PortfolioPosition(
            rank              = rank,
            ticker            = ticker,
            composite_score   = score.composite,
            conviction_weight = round(weight, 4),
            dollar_amount     = dollar_amount,
            price             = price,
            shares_to_buy     = shares,
            constraint_result = constraint_results[ticker],
            annualized_vol    = vols.get(ticker),
            sizing_method     = sizing_method,
        ))

    total_invested = sum(p.dollar_amount for p in positions)
    return PortfolioResult(
        positions      = positions,
        total_invested = total_invested,
        cash_remaining = round(ACCOUNT_SIZE - total_invested, 2),
        n_positions    = len(positions),
        sizing_method  = sizing_method,
    )
