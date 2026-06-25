"""
Dashboard chart helpers — thin wrappers around reports/charts.py.
Kept for backwards compatibility; app.py pages use reports.charts directly.
"""
from reports.charts import (
    plot_equity_curve,
    plot_drawdown,
    plot_calendar_returns,
    plot_rolling_sharpe,
    plot_concentration,
    generate_all_charts,
)

__all__ = [
    "plot_equity_curve",
    "plot_drawdown",
    "plot_calendar_returns",
    "plot_rolling_sharpe",
    "plot_concentration",
    "generate_all_charts",
]
