"""
Benchmark comparison — SPY and QQQ vs the portfolio.

All inputs are monthly return sequences of equal length.
Beta is computed via OLS: beta = cov(port, spy) / var(spy).
Alpha = portfolio_cagr - spy_cagr (simple arithmetic alpha, not Jensen's).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from backtester.metrics import _cagr, _sharpe, _max_drawdown


@dataclass
class BenchmarkComparison:
    portfolio_cagr: float
    spy_cagr: float
    qqq_cagr: float
    alpha_vs_spy: float          # portfolio_cagr - spy_cagr
    alpha_vs_qqq: float          # portfolio_cagr - qqq_cagr
    portfolio_sharpe: float
    spy_sharpe: float
    qqq_sharpe: float
    portfolio_max_drawdown: float
    spy_max_drawdown: float
    qqq_max_drawdown: float
    correlation_with_spy: float
    beta_vs_spy: float

    def to_dict(self) -> dict:
        pct = lambda x: round(x * 100, 2)
        r3  = lambda x: round(x, 3)
        return {
            "portfolio_cagr":          pct(self.portfolio_cagr),
            "spy_cagr":                pct(self.spy_cagr),
            "qqq_cagr":                pct(self.qqq_cagr),
            "alpha_vs_spy":            pct(self.alpha_vs_spy),
            "alpha_vs_qqq":            pct(self.alpha_vs_qqq),
            "portfolio_sharpe":        r3(self.portfolio_sharpe),
            "spy_sharpe":              r3(self.spy_sharpe),
            "qqq_sharpe":              r3(self.qqq_sharpe),
            "portfolio_max_drawdown":  pct(self.portfolio_max_drawdown),
            "spy_max_drawdown":        pct(self.spy_max_drawdown),
            "qqq_max_drawdown":        pct(self.qqq_max_drawdown),
            "correlation_with_spy":    r3(self.correlation_with_spy),
            "beta_vs_spy":             r3(self.beta_vs_spy),
        }


def _correlation(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (n - 1)
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x) / (n - 1))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y) / (n - 1))
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def _beta(portfolio: Sequence[float], benchmark: Sequence[float]) -> float:
    n = len(portfolio)
    if n < 2:
        return 0.0
    mb = sum(benchmark) / n
    var_b = sum((b - mb) ** 2 for b in benchmark) / (n - 1)
    if var_b == 0:
        return 0.0
    mp = sum(portfolio) / n
    cov = sum((p - mp) * (b - mb) for p, b in zip(portfolio, benchmark)) / (n - 1)
    return cov / var_b


def compute_benchmark_comparison(
    portfolio_returns: Sequence[float],
    spy_returns: Sequence[float],
    qqq_returns: Sequence[float],
) -> BenchmarkComparison:
    """
    Compare portfolio performance against SPY and QQQ.

    All three sequences must have the same length (aligned by month).
    """
    port_cagr = _cagr(portfolio_returns)
    spy_cagr  = _cagr(spy_returns)
    qqq_cagr  = _cagr(qqq_returns)

    return BenchmarkComparison(
        portfolio_cagr=port_cagr,
        spy_cagr=spy_cagr,
        qqq_cagr=qqq_cagr,
        alpha_vs_spy=port_cagr - spy_cagr,
        alpha_vs_qqq=port_cagr - qqq_cagr,
        portfolio_sharpe=_sharpe(portfolio_returns),
        spy_sharpe=_sharpe(spy_returns),
        qqq_sharpe=_sharpe(qqq_returns),
        portfolio_max_drawdown=_max_drawdown(portfolio_returns),
        spy_max_drawdown=_max_drawdown(spy_returns),
        qqq_max_drawdown=_max_drawdown(qqq_returns),
        correlation_with_spy=_correlation(portfolio_returns, spy_returns),
        beta_vs_spy=_beta(portfolio_returns, spy_returns),
    )
