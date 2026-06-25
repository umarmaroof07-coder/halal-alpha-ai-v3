"""
Backtest performance metrics.

All metrics operate on a list of monthly decimal returns (e.g. 0.03 = +3%).

Sortino downside deviation — standard denominator (total months, not just negatives):
    downside_i        = min(r_i, 0)   for every month i
    downside_dev      = sqrt( sum(downside_i²) / N ) * sqrt(12)
    sortino           = annualised_mean_return / downside_dev

where N = total number of months (not the count of negative months).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class BacktestMetrics:
    cagr: float                              # annualised return (decimal)
    sharpe: float                            # annualised, risk-free = 0
    sortino: float                           # annualised, standard denominator
    max_drawdown: float                      # peak-to-trough, negative decimal
    win_rate: float                          # fraction of months with r > 0
    total_return: float                      # (end/start) - 1
    calendar_year_returns: dict[int, float]  # {year: annual_return}
    avg_turnover: float                      # average monthly turnover fraction
    best_month: float
    worst_month: float
    n_months: int

    def to_dict(self) -> dict:
        return {
            "cagr":                   round(self.cagr * 100, 2),
            "sharpe":                 round(self.sharpe, 3),
            "sortino":                round(self.sortino, 3),
            "max_drawdown":           round(self.max_drawdown * 100, 2),
            "win_rate":               round(self.win_rate * 100, 2),
            "total_return":           round(self.total_return * 100, 2),
            "avg_turnover":           round(self.avg_turnover * 100, 2),
            "best_month":             round(self.best_month * 100, 2),
            "worst_month":            round(self.worst_month * 100, 2),
            "n_months":               self.n_months,
            "calendar_year_returns":  {
                yr: round(v * 100, 2)
                for yr, v in self.calendar_year_returns.items()
            },
        }


def _cagr(monthly_returns: Sequence[float]) -> float:
    n = len(monthly_returns)
    if n == 0:
        return 0.0
    total = math.prod(1 + r for r in monthly_returns)
    years = n / 12.0
    return total ** (1.0 / years) - 1.0


def _sharpe(monthly_returns: Sequence[float]) -> float:
    n = len(monthly_returns)
    if n < 2:
        return 0.0
    mean = sum(monthly_returns) / n
    variance = sum((r - mean) ** 2 for r in monthly_returns) / (n - 1)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0
    return (mean / std) * math.sqrt(12)


def _sortino(monthly_returns: Sequence[float]) -> float:
    """
    Sortino with standard denominator.

    downside_i   = min(r_i, 0)  for all i in 1..N
    downside_dev = sqrt( sum(downside_i²) / N ) * sqrt(12)
    sortino      = (mean_monthly * 12) / downside_dev
    """
    n = len(monthly_returns)
    if n == 0:
        return 0.0
    mean_monthly = sum(monthly_returns) / n
    sum_sq_down = sum(min(r, 0.0) ** 2 for r in monthly_returns)
    downside_dev = math.sqrt(sum_sq_down / n) * math.sqrt(12)
    if downside_dev == 0:
        return 0.0
    return (mean_monthly * 12) / downside_dev


def _max_drawdown(monthly_returns: Sequence[float]) -> float:
    peak = 1.0
    equity = 1.0
    max_dd = 0.0
    for r in monthly_returns:
        equity *= (1 + r)
        if equity > peak:
            peak = equity
        dd = (equity - peak) / peak
        if dd < max_dd:
            max_dd = dd
    return max_dd


def _calendar_year_returns(
    monthly_returns: Sequence[float],
    years: Sequence[int],
) -> dict[int, float]:
    """
    Compound monthly returns within each calendar year.

    Parameters
    ----------
    monthly_returns : Sequence[float]
        Monthly returns in chronological order.
    years : Sequence[int]
        Calendar year for each monthly return (same length as monthly_returns).
    """
    by_year: dict[int, list[float]] = defaultdict(list)
    for r, yr in zip(monthly_returns, years):
        by_year[yr].append(r)
    return {
        yr: math.prod(1 + r for r in rets) - 1.0
        for yr, rets in sorted(by_year.items())
    }


def compute_metrics(
    monthly_returns: Sequence[float],
    years: Sequence[int],
    turnovers: Sequence[float] | None = None,
) -> BacktestMetrics:
    """
    Compute all backtest metrics from monthly returns.

    Parameters
    ----------
    monthly_returns : Sequence[float]
        Net-of-cost monthly returns in chronological order.
    years : Sequence[int]
        Calendar year for each month (same length as monthly_returns).
    turnovers : Sequence[float] | None
        Monthly turnover fractions. None → avg_turnover = 0.
    """
    n = len(monthly_returns)
    if n == 0:
        return BacktestMetrics(
            cagr=0, sharpe=0, sortino=0, max_drawdown=0,
            win_rate=0, total_return=0, calendar_year_returns={},
            avg_turnover=0, best_month=0, worst_month=0, n_months=0,
        )

    total = math.prod(1 + r for r in monthly_returns) - 1.0
    win_rate = sum(1 for r in monthly_returns if r > 0) / n
    avg_to = sum(turnovers) / len(turnovers) if turnovers else 0.0

    return BacktestMetrics(
        cagr=_cagr(monthly_returns),
        sharpe=_sharpe(monthly_returns),
        sortino=_sortino(monthly_returns),
        max_drawdown=_max_drawdown(monthly_returns),
        win_rate=win_rate,
        total_return=total,
        calendar_year_returns=_calendar_year_returns(monthly_returns, years),
        avg_turnover=avg_to,
        best_month=max(monthly_returns),
        worst_month=min(monthly_returns),
        n_months=n,
    )
