"""Backtest Results page."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def render() -> None:
    st.title("Backtest Results")

    from dashboard.state import get_backtest_result
    from reports.performance_report import PASS_THRESHOLDS

    backtest = get_backtest_result()

    if backtest is None:
        st.error(
            "No backtest found. Run `python3 main.py --backtest` in the terminal first."
        )
        st.code("python3 main.py --backtest")
        return

    # ── Warnings ─────────────────────────────────────────────────────────────
    for w in backtest.get("warnings", []):
        st.warning(w)

    # ── Metric cards ─────────────────────────────────────────────────────────
    st.subheader("Full-Period Performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR",      f"{backtest.get('cagr', 0):.1f}%")
    c2.metric("Sharpe",    f"{backtest.get('sharpe', 0):.2f}")
    c3.metric("Sortino",   f"{backtest.get('sortino', 0):.2f}")
    c4.metric("Max DD",    f"{backtest.get('max_drawdown', 0):.1f}%")
    c5.metric("Win Rate",  f"{backtest.get('win_rate', 0):.1f}%")

    alpha = backtest.get("alpha_vs_spy", 0)
    st.metric("Alpha vs SPY", f"{alpha:+.1f}%")

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    charts_dir = Path("data/reports/charts")
    for chart_name, label in [
        ("equity_curve.png",    "Equity Curve (log scale)"),
        ("drawdown.png",        "Drawdown"),
        ("calendar_returns.png","Calendar Returns"),
        ("rolling_sharpe.png",  "Rolling 12-Month Sharpe"),
        ("concentration.png",   "Portfolio Concentration (HHI)"),
    ]:
        p = charts_dir / chart_name
        if p.exists():
            st.subheader(label)
            st.image(str(p), use_container_width=True)

    st.divider()

    # ── Pass/Fail ─────────────────────────────────────────────────────────────
    st.subheader("Pass / Fail Criteria")
    st.caption(
        "Max Drawdown pass/fail: portfolio must not be worse than SPY by more than 10pp. "
        "Extra warning if portfolio DD < −40%."
    )
    excel_dir = Path("data/reports")
    xlsx_files = sorted(excel_dir.glob("backtest_report_*.xlsx"), reverse=True)
    if xlsx_files:
        st.success(f"Excel report available: `{xlsx_files[0].name}`")
        st.caption("Open the Pass_Fail sheet for full criteria detail.")

    st.info("Run `python3 main.py --backtest` to regenerate with latest data.")
