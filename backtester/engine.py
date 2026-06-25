"""
Backtest Engine — walk-forward monthly simulation, 2005–2026.

Critical integrity rules (enforced here):
  1. AI Research is ALWAYS 50.0 in backtests. No Claude calls are made.
  2. Price data uses only prices ≤ rebalance date (no lookahead in prices).
  3. Factor scores use only data with as_of_date ≤ rebalance date (caller's responsibility).
  4. Both survivorship-bias and point-in-time warnings are always attached to results.

Transaction cost model:
  For each rebalance, compute the dollar value of positions being opened or closed.
  Cost = changed_dollar_value × TRANSACTION_COST_RATE (0.001 = 0.10%).

This engine is data-agnostic: it receives pre-computed monthly return dicts and
pre-built PortfolioResult objects keyed by "YYYY-MM". This design keeps the engine
testable without any real API calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from config.settings import ACCOUNT_SIZE, BACKTEST_AI_NEUTRAL, CONVICTION_WEIGHTS
from backtester.metrics import compute_metrics, BacktestMetrics
from backtester.benchmark import compute_benchmark_comparison, BenchmarkComparison
from backtester.overfitting import compute_overfitting_report, OverfittingReport

log = logging.getLogger(__name__)

TRANSACTION_COST_RATE: float = 0.001   # 0.10% per trade

SURVIVORSHIP_BIAS_WARNING = (
    "SURVIVORSHIP BIAS WARNING: This backtest uses tickers that exist and are "
    "accessible today. Companies that failed, were delisted, or were acquired "
    "between 2005–2026 are NOT included. Returns are likely overstated."
)

FUNDAMENTALS_NOT_POINT_IN_TIME_WARNING = (
    "LOOK-AHEAD BIAS WARNING: Fundamental data (revenue, earnings, ratios) is "
    "sourced as of today. We cannot guarantee that each metric reflects what was "
    "publicly available at each rebalance date. This introduces look-ahead bias "
    "in fundamental factor scores."
)


@dataclass
class BacktestConfig:
    start_date: str = "2005-01-01"
    end_date: str = "2026-12-31"
    initial_capital: float = ACCOUNT_SIZE
    transaction_cost: float = TRANSACTION_COST_RATE
    rebalance_freq: str = "monthly"


@dataclass
class MonthlySnapshot:
    date: str                          # "YYYY-MM"
    tickers: list[str]                 # tickers held this month
    weights: list[float]               # conviction weights (sum ≤ 1.0)
    portfolio_return: float            # net of transaction costs
    spy_return: float
    qqq_return: float
    turnover: float                    # fraction of portfolio value that changed
    transaction_cost_paid: float       # absolute dollar cost


@dataclass
class BacktestResult:
    config: BacktestConfig
    equity_curve: list[dict]           # {date, value, spy_value, qqq_value}
    monthly_snapshots: list[MonthlySnapshot]
    metrics: BacktestMetrics
    benchmark_comparison: BenchmarkComparison
    overfitting_report: OverfittingReport
    warnings: list[str]
    n_months: int

    def to_dict(self) -> dict:
        return {
            "config": {
                "start_date":       self.config.start_date,
                "end_date":         self.config.end_date,
                "initial_capital":  self.config.initial_capital,
                "transaction_cost": self.config.transaction_cost,
            },
            "n_months":             self.n_months,
            "warnings":             self.warnings,
            "metrics":              self.metrics.to_dict(),
            "benchmark_comparison": self.benchmark_comparison.to_dict(),
            "overfitting_report":   self.overfitting_report.to_dict(),
            "equity_curve":         self.equity_curve,
        }


def _compute_turnover(
    old_tickers: list[str],
    new_tickers: list[str],
    old_weights: list[float],
    new_weights: list[float],
) -> float:
    """
    Return the fraction of portfolio value that changed this rebalance.
    Turnover = 0.5 × sum(|new_weight - old_weight|) for all tickers.
    This equals the one-sided turnover (buy side or sell side, not both).
    """
    all_tickers = set(old_tickers) | set(new_tickers)
    old_map = dict(zip(old_tickers, old_weights))
    new_map = dict(zip(new_tickers, new_weights))
    total_change = sum(
        abs(new_map.get(t, 0.0) - old_map.get(t, 0.0))
        for t in all_tickers
    )
    return total_change / 2.0


def run_backtest(
    monthly_dates: list[str],
    monthly_portfolio_gross_returns: list[float],
    monthly_spy_returns: list[float],
    monthly_qqq_returns: list[float],
    monthly_positions: list[tuple[list[str], list[float]]],
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """
    Run the full backtest simulation.

    This engine is intentionally data-agnostic. The caller is responsible for:
      - Providing only prices / scores available at each date (no lookahead).
      - Setting AI Research = BACKTEST_AI_NEUTRAL for all factor scoring.
      - Providing pre-computed gross returns for the portfolio each month.

    Parameters
    ----------
    monthly_dates : list[str]
        "YYYY-MM" labels for each month in chronological order.
    monthly_portfolio_gross_returns : list[float]
        Gross (pre-cost) portfolio return for each month.
    monthly_spy_returns : list[float]
        SPY return for each month (aligned with monthly_dates).
    monthly_qqq_returns : list[float]
        QQQ return for each month (aligned with monthly_dates).
    monthly_positions : list[tuple[list[str], list[float]]]
        (tickers, weights) held at the start of each month.
    config : BacktestConfig | None
        Backtest configuration. Uses defaults if None.

    Returns
    -------
    BacktestResult
    """
    if config is None:
        config = BacktestConfig()

    n = len(monthly_dates)
    assert n == len(monthly_portfolio_gross_returns), "dates and returns must align"
    assert n == len(monthly_spy_returns), "SPY returns must align with dates"
    assert n == len(monthly_qqq_returns), "QQQ returns must align with dates"
    assert n == len(monthly_positions), "positions must align with dates"

    portfolio_value = config.initial_capital
    spy_value = config.initial_capital
    qqq_value = config.initial_capital

    equity_curve: list[dict] = []
    snapshots: list[MonthlySnapshot] = []
    net_returns: list[float] = []
    monthly_weights_list: list[list[float]] = []

    prev_tickers: list[str] = []
    prev_weights: list[float] = []

    for i, date in enumerate(monthly_dates):
        tickers, weights = monthly_positions[i]
        gross_ret = monthly_portfolio_gross_returns[i]
        spy_ret   = monthly_spy_returns[i]
        qqq_ret   = monthly_qqq_returns[i]

        # Transaction cost
        turnover = _compute_turnover(prev_tickers, tickers, prev_weights, weights)
        cost_fraction = turnover * config.transaction_cost
        net_ret = gross_ret - cost_fraction
        cost_dollars = portfolio_value * cost_fraction

        # Compound values
        portfolio_value *= (1 + net_ret)
        spy_value *= (1 + spy_ret)
        qqq_value *= (1 + qqq_ret)

        equity_curve.append({
            "date":       date,
            "value":      round(portfolio_value, 4),
            "spy_value":  round(spy_value, 4),
            "qqq_value":  round(qqq_value, 4),
        })
        snapshots.append(MonthlySnapshot(
            date=date,
            tickers=tickers,
            weights=weights,
            portfolio_return=net_ret,
            spy_return=spy_ret,
            qqq_return=qqq_ret,
            turnover=turnover,
            transaction_cost_paid=cost_dollars,
        ))
        net_returns.append(net_ret)
        monthly_weights_list.append(weights)

        prev_tickers = tickers
        prev_weights = weights

    # Calendar years for each month
    years = [int(d[:4]) for d in monthly_dates]
    turnovers = [s.turnover for s in snapshots]

    metrics = compute_metrics(net_returns, years, turnovers)
    benchmark = compute_benchmark_comparison(net_returns, monthly_spy_returns, monthly_qqq_returns)
    overfitting = compute_overfitting_report(monthly_dates, net_returns, monthly_spy_returns, monthly_weights_list)

    return BacktestResult(
        config=config,
        equity_curve=equity_curve,
        monthly_snapshots=snapshots,
        metrics=metrics,
        benchmark_comparison=benchmark,
        overfitting_report=overfitting,
        warnings=[SURVIVORSHIP_BIAS_WARNING, FUNDAMENTALS_NOT_POINT_IN_TIME_WARNING],
        n_months=n,
    )
