"""Stress Tests page — crisis drawdown analysis for Top 5 picks."""

from __future__ import annotations

import streamlit as st


_CRISIS_LABELS = {
    "gfc_2008":             "GFC 2008",
    "covid_2020":           "COVID 2020",
    "rates_2022":           "Rates 2022",
    "regional_banks_2023":  "Regional Banks 2023",
}


def render() -> None:
    st.title("Stress Tests")
    st.caption("Historical crisis drawdown analysis vs SPY for current Top 5 picks.")

    from dashboard.state import get_recommendations

    recs   = get_recommendations()
    buy    = [r for r in recs if r.action == "BUY NOW"]
    tickers = [r.ticker for r in buy]

    if not tickers:
        st.warning("No BUY NOW picks. Run --refresh-data first.")
        return

    ticker_choice = st.multiselect("Select tickers to test", tickers, default=tickers)

    if not ticker_choice:
        st.info("Select at least one ticker.")
        return

    if st.button("🧪 Run Stress Tests"):
        from factors.stress_test import compute_stress_test, CRISIS_WINDOWS

        all_results = {}
        progress = st.progress(0)
        for i, t in enumerate(ticker_choice):
            with st.spinner(f"Testing {t}…"):
                try:
                    result = compute_stress_test(t)
                    all_results[t] = result.to_dict()
                except Exception as e:
                    all_results[t] = {"error": str(e)}
            progress.progress((i + 1) / len(ticker_choice))

        st.session_state["stress_results"] = all_results
        progress.empty()

    results = st.session_state.get("stress_results", {})
    if not results:
        st.info("Click **Run Stress Tests** to run crisis analysis.")
        return

    st.divider()

    # ── Summary table ─────────────────────────────────────────────────────────
    st.subheader("Crisis Summary")
    summary_rows = []
    for ticker, data in results.items():
        if "error" in data:
            summary_rows.append({"Ticker": ticker, "Status": f"Error: {data['error']}"})
            continue
        crises = data.get("crises", {})
        row = {"Ticker": ticker}
        for crisis_key, label in _CRISIS_LABELS.items():
            c = crises.get(crisis_key, {})
            if isinstance(c, dict) and c.get("has_data"):
                row[label] = f"{c.get('max_drawdown', 0):.1f}%"
            else:
                row[label] = "n/a"
        summary_rows.append(row)

    if summary_rows:
        st.dataframe(summary_rows, use_container_width=True, hide_index=True)

    st.divider()

    # ── Per-ticker detail ─────────────────────────────────────────────────────
    for ticker, data in results.items():
        if "error" in data:
            st.error(f"{ticker}: {data['error']}")
            continue

        with st.expander(f"{ticker} — Detail"):
            crises = data.get("crises", {})
            detail_rows = []
            for crisis_key, label in _CRISIS_LABELS.items():
                c = crises.get(crisis_key, {})
                if isinstance(c, dict):
                    detail_rows.append({
                        "Crisis":          label,
                        "Has Data":        "✓" if c.get("has_data") else "✗",
                        "Max Drawdown":    f"{c.get('max_drawdown', 0):.1f}%",
                        "Total Return":    f"{c.get('total_return', 0):.1f}%",
                        "SPY Drawdown":    f"{c.get('spy_drawdown', 0):.1f}%",
                        "vs SPY":          f"{c.get('relative_vs_spy', 0):+.1f}%",
                        "Duration (days)": c.get("duration_days", "—"),
                    })

            if detail_rows:
                st.dataframe(detail_rows, use_container_width=True, hide_index=True)

            n_crises = data.get("n_crises_with_data", 0)
            if n_crises >= 3:
                st.success(f"✓ {n_crises}/4 crisis windows covered — stress test reliable.")
            elif n_crises >= 1:
                st.warning(f"⚠ Only {n_crises}/4 crisis windows have data.")
            else:
                st.error("✗ No crisis window data available.")
