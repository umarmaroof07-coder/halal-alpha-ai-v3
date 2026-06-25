"""
Recommendation Guard — the ONLY place in the codebase that produces
BUY NOW / WATCHLIST / AVOID labels.

Rules:
  BUY NOW   — ticker is in the top-5 portfolio positions.
  WATCHLIST — ticker is Shariah-compliant, passes ALL constraints,
               has a valid composite score, but did not make the top 5.
  AVOID     — ticker fails Shariah or any constraint. Includes rejection reasons.

safe_recommendations() is the only public function here.
Nothing else in the codebase should produce these labels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from portfolio.constructor import PortfolioResult
from portfolio.constraints import ConstraintResult

log = logging.getLogger(__name__)

Action = Literal["BUY NOW", "WATCHLIST", "AVOID"]


@dataclass
class Recommendation:
    ticker: str
    action: Action
    rank: int | None                    # 1–5 for BUY NOW; None otherwise
    composite_score: float
    conviction_weight: float | None     # populated for BUY NOW only
    dollar_amount: float | None         # populated for BUY NOW only
    price: float | None
    shariah_status: str
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ticker":            self.ticker,
            "action":            self.action,
            "rank":              self.rank,
            "composite_score":   round(self.composite_score, 2),
            "conviction_weight": self.conviction_weight,
            "dollar_amount":     self.dollar_amount,
            "price":             self.price,
            "shariah_status":    self.shariah_status,
            "rejection_reasons": self.rejection_reasons,
        }


def safe_recommendations(
    portfolio_result: PortfolioResult,
    all_scores: list,                        # list[FactorScores] for full universe
    constraint_results: dict[str, ConstraintResult],
    prices: dict[str, float],
    shariah_statuses: dict[str, str],
) -> list[Recommendation]:
    """
    Produce the final BUY NOW / WATCHLIST / AVOID recommendation list.

    This is the ONLY function permitted to output these labels.

    Parameters
    ----------
    portfolio_result : PortfolioResult
        Output of build_portfolio() — the top-5 positions already selected.
    all_scores : list[FactorScores]
        Composite scores for every ticker in the screened universe (ranked order).
    constraint_results : dict[str, ConstraintResult]
        Results of check_constraints() for every ticker.
    prices : dict[str, float]
        Current price per ticker.
    shariah_statuses : dict[str, str]
        Shariah status per ticker: "compliant" | "non_compliant" | "unknown".

    Returns
    -------
    list[Recommendation] — one entry per ticker in all_scores,
    sorted: BUY NOW (by rank) → WATCHLIST (by score) → AVOID (by score).
    """
    # Build fast lookup for portfolio positions
    portfolio_tickers: dict[str, object] = {p.ticker: p for p in portfolio_result.positions}

    buy_now:   list[Recommendation] = []
    watchlist: list[Recommendation] = []
    avoid:     list[Recommendation] = []

    for score_obj in all_scores:
        ticker = score_obj.ticker
        composite = score_obj.composite
        price = prices.get(ticker)
        shariah_status = shariah_statuses.get(ticker, "unknown")
        cr = constraint_results.get(ticker)

        if ticker in portfolio_tickers:
            pos = portfolio_tickers[ticker]
            buy_now.append(Recommendation(
                ticker=ticker,
                action="BUY NOW",
                rank=pos.rank,  # type: ignore[attr-defined]
                composite_score=composite,
                conviction_weight=pos.conviction_weight,  # type: ignore[attr-defined]
                dollar_amount=pos.dollar_amount,           # type: ignore[attr-defined]
                price=price,
                shariah_status=shariah_status,
            ))

        elif (
            shariah_status == "compliant"
            and cr is not None
            and cr.passed
            and composite is not None
        ):
            watchlist.append(Recommendation(
                ticker=ticker,
                action="WATCHLIST",
                rank=None,
                composite_score=composite,
                conviction_weight=None,
                dollar_amount=None,
                price=price,
                shariah_status=shariah_status,
            ))

        else:
            reasons: list[str] = []
            if shariah_status != "compliant":
                reasons.append(f"Shariah: {shariah_status}")
            if cr is not None and not cr.passed:
                reasons.extend(cr.failures)
            elif cr is None:
                reasons.append("No constraint data available")

            avoid.append(Recommendation(
                ticker=ticker,
                action="AVOID",
                rank=None,
                composite_score=composite,
                conviction_weight=None,
                dollar_amount=None,
                price=price,
                shariah_status=shariah_status,
                rejection_reasons=reasons,
            ))

    # Sort each bucket
    buy_now.sort(key=lambda r: r.rank or 99)
    watchlist.sort(key=lambda r: r.composite_score, reverse=True)
    avoid.sort(key=lambda r: r.composite_score, reverse=True)

    return buy_now + watchlist + avoid
