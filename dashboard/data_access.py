"""
Data access helpers for dashboard — thin wrappers around dashboard/state.py.
Kept for backwards compatibility.
"""
from dashboard.state import (
    get_universe,
    get_factor_scores,
    get_constraint_results,
    get_prices,
    get_shariah_statuses,
    get_portfolio,
    get_recommendations,
    get_backtest_result,
    clear_recommendations_cache,
)

__all__ = [
    "get_universe",
    "get_factor_scores",
    "get_constraint_results",
    "get_prices",
    "get_shariah_statuses",
    "get_portfolio",
    "get_recommendations",
    "get_backtest_result",
    "clear_recommendations_cache",
]
