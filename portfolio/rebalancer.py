"""
Rebalancer — compares current holdings against a new target portfolio and
produces a list of BUY / SELL / HOLD / CLOSE actions.

Drift threshold: if any position's current weight deviates more than
DRIFT_THRESHOLD from its target weight, needs_rebalance = True.

This module never labels stocks BUY NOW / WATCHLIST / AVOID.
It only produces mechanical trade instructions relative to current holdings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from portfolio.constructor import PortfolioResult, PortfolioPosition

log = logging.getLogger(__name__)

DRIFT_THRESHOLD = 0.05   # 5% — trigger rebalance if any position drifts this far


@dataclass
class CurrentPosition:
    ticker: str
    shares: float
    price: float

    @property
    def value(self) -> float:
        return self.shares * self.price


@dataclass
class RebalanceAction:
    ticker: str
    action: Literal["BUY", "SELL", "HOLD", "CLOSE"]
    current_shares: float
    target_shares: float
    delta_shares: float       # positive = buy more, negative = sell
    delta_dollars: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "ticker":         self.ticker,
            "action":         self.action,
            "current_shares": round(self.current_shares, 4),
            "target_shares":  round(self.target_shares, 4),
            "delta_shares":   round(self.delta_shares, 4),
            "delta_dollars":  round(self.delta_dollars, 2),
            "reason":         self.reason,
        }


@dataclass
class RebalanceReport:
    as_of_date: str
    actions: list[RebalanceAction] = field(default_factory=list)
    drift_pct: float = 0.0          # maximum single-position drift observed
    needs_rebalance: bool = False
    current_value: float = 0.0
    target_value: float = 0.0

    def to_dict(self) -> dict:
        return {
            "as_of_date":      self.as_of_date,
            "needs_rebalance": self.needs_rebalance,
            "drift_pct":       round(self.drift_pct * 100, 2),
            "current_value":   round(self.current_value, 2),
            "target_value":    round(self.target_value, 2),
            "actions":         [a.to_dict() for a in self.actions],
        }


def compute_rebalance(
    current_positions: list[CurrentPosition],
    target_portfolio: PortfolioResult,
    as_of_date: str = "",
) -> RebalanceReport:
    """
    Compute rebalance actions to move from current holdings to target portfolio.

    Parameters
    ----------
    current_positions : list[CurrentPosition]
        Stocks currently held with share counts and current prices.
    target_portfolio : PortfolioResult
        Output of build_portfolio() — the desired end state.
    as_of_date : str
        ISO date label for the report.

    Returns
    -------
    RebalanceReport
    """
    current_map: dict[str, CurrentPosition] = {p.ticker: p for p in current_positions}
    target_map:  dict[str, PortfolioPosition] = {p.ticker: p for p in target_portfolio.positions}

    current_value = sum(p.value for p in current_positions)
    target_value  = target_portfolio.total_invested

    actions: list[RebalanceAction] = []
    max_drift: float = 0.0

    # Stocks in target — BUY / HOLD / SELL
    for ticker, target in target_map.items():
        target_shares = target.shares_to_buy
        current = current_map.get(ticker)
        current_shares = current.shares if current else 0.0

        delta_shares = target_shares - current_shares
        delta_dollars = delta_shares * target.price

        # Drift = how far current dollar value deviates from target dollar amount,
        # expressed as a fraction of the target dollar amount.
        current_dollar_value = current.value if current else 0.0
        if target.dollar_amount > 0:
            drift = abs(current_dollar_value - target.dollar_amount) / target.dollar_amount
        else:
            drift = 0.0
        max_drift = max(max_drift, drift)

        if current_shares == 0:
            action: Literal["BUY", "SELL", "HOLD", "CLOSE"] = "BUY"
            reason = f"New position — target ${target.dollar_amount:.0f} at ${target.price:.2f}"
        elif abs(delta_shares) < 0.0001:
            action = "HOLD"
            reason = "Position within tolerance"
        elif delta_shares > 0:
            action = "BUY"
            reason = f"Add {delta_shares:.4f} shares (drift {drift*100:.1f}%)"
        else:
            action = "SELL"
            reason = f"Trim {abs(delta_shares):.4f} shares (drift {drift*100:.1f}%)"

        actions.append(RebalanceAction(
            ticker=ticker,
            action=action,
            current_shares=current_shares,
            target_shares=target_shares,
            delta_shares=delta_shares,
            delta_dollars=delta_dollars,
            reason=reason,
        ))

    # Stocks held but no longer in target → CLOSE
    for ticker, current in current_map.items():
        if ticker not in target_map:
            actions.append(RebalanceAction(
                ticker=ticker,
                action="CLOSE",
                current_shares=current.shares,
                target_shares=0.0,
                delta_shares=-current.shares,
                delta_dollars=-current.value,
                reason="No longer in target portfolio — exit position",
            ))
            # Closing a position is always a full drift (100% of current value moved)
            max_drift = max(max_drift, 1.0)

    needs_rebalance = max_drift >= DRIFT_THRESHOLD

    return RebalanceReport(
        as_of_date=as_of_date,
        actions=sorted(actions, key=lambda a: a.action),
        drift_pct=max_drift,
        needs_rebalance=needs_rebalance,
        current_value=current_value,
        target_value=target_value,
    )
