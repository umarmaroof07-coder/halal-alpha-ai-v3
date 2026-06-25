"""
Charts — Matplotlib-only. No Streamlit, no Plotly, no network calls.

All charts save as PNG to data/reports/charts/.
The survivorship bias warning appears as a figure footnote on every chart.

Charts produced:
  equity_curve.png    — Portfolio vs SPY vs QQQ (log scale)
  drawdown.png        — Drawdown area curve
  calendar_returns.png — Grouped bar chart by year
  rolling_sharpe.png  — 12-month rolling Sharpe
  concentration.png   — Monthly HHI over time
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from backtester.engine import SURVIVORSHIP_BIAS_WARNING

CHARTS_DIR = Path("data/reports/charts")
_WARN_SHORT = "⚠ Survivorship bias: returns likely overstated. See full warnings."

_PORT_COLOR = "#1565C0"   # blue
_SPY_COLOR  = "#2E7D32"   # green
_QQQ_COLOR  = "#6A1B9A"   # purple
_DD_COLOR   = "#C62828"   # red


def _ensure_dir() -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def _add_footnote(fig: plt.Figure) -> None:
    fig.text(
        0.01, 0.005, _WARN_SHORT,
        ha="left", va="bottom",
        fontsize=7, color="#B71C1C",
        style="italic",
    )


def plot_equity_curve(
    dates: Sequence[str],
    portfolio_values: Sequence[float],
    spy_values: Sequence[float],
    qqq_values: Sequence[float],
    output_path: Path | None = None,
) -> Path:
    """Portfolio vs SPY vs QQQ equity curve on a log scale."""
    _ensure_dir()
    if output_path is None:
        output_path = CHARTS_DIR / "equity_curve.png"

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.semilogy(dates, portfolio_values, color=_PORT_COLOR, linewidth=2, label="Portfolio")
    ax.semilogy(dates, spy_values,       color=_SPY_COLOR,  linewidth=1.5, linestyle="--", label="SPY")
    ax.semilogy(dates, qqq_values,       color=_QQQ_COLOR,  linewidth=1.5, linestyle=":",  label="QQQ")

    ax.set_title("Equity Curve — Portfolio vs SPY vs QQQ (log scale)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    _label_xaxis(ax, dates)
    _add_footnote(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _compute_drawdown(returns: Sequence[float]) -> list[float]:
    """Convert monthly returns to drawdown series."""
    equity = 1.0
    peak = 1.0
    dd = []
    for r in returns:
        equity *= (1 + r)
        peak = max(peak, equity)
        dd.append((equity - peak) / peak)
    return dd


def plot_drawdown(
    dates: Sequence[str],
    portfolio_returns: Sequence[float],
    spy_returns: Sequence[float],
    output_path: Path | None = None,
) -> Path:
    """Drawdown area chart for portfolio and SPY."""
    _ensure_dir()
    if output_path is None:
        output_path = CHARTS_DIR / "drawdown.png"

    port_dd = _compute_drawdown(portfolio_returns)
    spy_dd  = _compute_drawdown(spy_returns)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(range(len(dates)), [d * 100 for d in port_dd],
                    alpha=0.5, color=_PORT_COLOR, label="Portfolio")
    ax.fill_between(range(len(dates)), [d * 100 for d in spy_dd],
                    alpha=0.3, color=_SPY_COLOR, label="SPY")
    ax.plot([d * 100 for d in port_dd], color=_PORT_COLOR, linewidth=1)
    ax.plot([d * 100 for d in spy_dd],  color=_SPY_COLOR,  linewidth=1, linestyle="--")

    ax.set_title("Drawdown — Portfolio vs SPY", fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    _label_xaxis(ax, dates)
    _add_footnote(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_calendar_returns(
    calendar_returns_port: dict[int, float],
    calendar_returns_spy: dict[int, float] | None = None,
    calendar_returns_qqq: dict[int, float] | None = None,
    output_path: Path | None = None,
) -> Path:
    """Grouped bar chart of calendar-year returns."""
    _ensure_dir()
    if output_path is None:
        output_path = CHARTS_DIR / "calendar_returns.png"

    years = sorted(calendar_returns_port.keys())
    port_vals = [calendar_returns_port.get(y, 0) * 100 for y in years]
    spy_vals  = [calendar_returns_spy.get(y, 0) * 100 if calendar_returns_spy else 0 for y in years]
    qqq_vals  = [calendar_returns_qqq.get(y, 0) * 100 if calendar_returns_qqq else 0 for y in years]

    x = range(len(years))
    width = 0.28

    fig, ax = plt.subplots(figsize=(max(10, len(years) * 0.6), 6))
    bars_p = ax.bar([i - width for i in x], port_vals, width, label="Portfolio", color=_PORT_COLOR, alpha=0.85)
    bars_s = ax.bar(x,                       spy_vals,  width, label="SPY",       color=_SPY_COLOR,  alpha=0.85)
    bars_q = ax.bar([i + width for i in x],  qqq_vals,  width, label="QQQ",       color=_QQQ_COLOR,  alpha=0.85)

    ax.set_title("Calendar Year Returns", fontsize=13, fontweight="bold")
    ax.set_ylabel("Return (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(y) for y in years], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _add_footnote(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_rolling_sharpe(
    dates: Sequence[str],
    monthly_returns: Sequence[float],
    window: int = 12,
    output_path: Path | None = None,
) -> Path:
    """12-month rolling Sharpe ratio."""
    _ensure_dir()
    if output_path is None:
        output_path = CHARTS_DIR / "rolling_sharpe.png"

    rolling: list[float | None] = [None] * (window - 1)
    rets = list(monthly_returns)
    for i in range(window - 1, len(rets)):
        window_rets = rets[i - window + 1: i + 1]
        mean = sum(window_rets) / window
        variance = sum((r - mean) ** 2 for r in window_rets) / (window - 1) if window > 1 else 0
        std = math.sqrt(variance)
        rolling.append((mean / std * math.sqrt(12)) if std > 0 else 0.0)

    valid_dates  = [d for d, v in zip(dates, rolling) if v is not None]
    valid_values = [v for v in rolling if v is not None]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(valid_dates)), valid_values, color=_PORT_COLOR, linewidth=1.5)
    ax.axhline(0,   color="black", linewidth=0.8)
    ax.axhline(0.5, color=_SPY_COLOR, linewidth=1, linestyle="--", label="0.5 threshold")
    ax.fill_between(range(len(valid_dates)), valid_values, 0,
                    where=[v > 0 for v in valid_values], alpha=0.2, color=_PORT_COLOR)
    ax.fill_between(range(len(valid_dates)), valid_values, 0,
                    where=[v < 0 for v in valid_values], alpha=0.2, color=_DD_COLOR)
    ax.set_title(f"{window}-Month Rolling Sharpe Ratio", fontsize=13, fontweight="bold")
    ax.set_ylabel("Sharpe")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _label_xaxis(ax, valid_dates)
    _add_footnote(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_concentration(
    dates: Sequence[str],
    monthly_hhi: Sequence[float],
    output_path: Path | None = None,
) -> Path:
    """Monthly Herfindahl index (concentration) over time."""
    _ensure_dir()
    if output_path is None:
        output_path = CHARTS_DIR / "concentration.png"

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(range(len(dates)), monthly_hhi, alpha=0.4, color=_PORT_COLOR)
    ax.plot(monthly_hhi, color=_PORT_COLOR, linewidth=1)
    # Reference line: equal-weight 5-stock HHI = 5 × (0.2)² ... but with conviction weights
    # Actual full-portfolio HHI = 0.30²+0.25²+0.20²+0.15²+0.10² = 0.225
    ax.axhline(0.225, color=_SPY_COLOR, linewidth=1, linestyle="--",
               label="Full 5-position HHI (0.225)")
    ax.set_title("Portfolio Concentration (Herfindahl Index)", fontsize=13, fontweight="bold")
    ax.set_ylabel("HHI")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _label_xaxis(ax, dates)
    _add_footnote(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_all_charts(
    result,   # BacktestResult — avoid circular import, typed loosely
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """
    Generate all five charts from a BacktestResult. Returns {name: path}.
    """
    global CHARTS_DIR
    if output_dir is not None:
        CHARTS_DIR = output_dir
    _ensure_dir()

    dates         = [e["date"] for e in result.equity_curve]
    port_values   = [e["value"] for e in result.equity_curve]
    spy_values    = [e.get("spy_value", e["value"]) for e in result.equity_curve]
    qqq_values    = [e.get("qqq_value", e["value"]) for e in result.equity_curve]
    port_rets     = [s.portfolio_return for s in result.monthly_snapshots]
    spy_rets      = [s.spy_return for s in result.monthly_snapshots]
    monthly_hhi   = [
        sum(w ** 2 for w in s.weights) if s.weights else 0.0
        for s in result.monthly_snapshots
    ]

    paths: dict[str, Path] = {}
    paths["equity_curve"]      = plot_equity_curve(dates, port_values, spy_values, qqq_values)
    paths["drawdown"]          = plot_drawdown(dates, port_rets, spy_rets)
    paths["calendar_returns"]  = plot_calendar_returns(result.metrics.calendar_year_returns)
    paths["rolling_sharpe"]    = plot_rolling_sharpe(dates, port_rets)
    paths["concentration"]     = plot_concentration(dates, monthly_hhi)
    return paths


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _label_xaxis(ax: plt.Axes, dates: Sequence[str], max_ticks: int = 12) -> None:
    """Show at most max_ticks evenly spaced date labels on x-axis."""
    n = len(dates)
    if n == 0:
        return
    step = max(1, n // max_ticks)
    tick_pos = list(range(0, n, step))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([dates[i] for i in tick_pos], rotation=45, ha="right", fontsize=8)
