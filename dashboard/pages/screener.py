"""Stock Screener page."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("Stock Screener")

    from dashboard.state import get_universe, get_recommendations

    universe = get_universe()
    recs     = get_recommendations()
    rec_map  = {r.ticker: r.action for r in recs}

    # ── Filters ──────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    search  = col1.text_input("Search ticker", "").upper().strip()
    shariah_filter = col2.selectbox(
        "Shariah filter",
        ["All", "compliant", "non_compliant", "unknown"],
    )

    # ── Build filtered table ─────────────────────────────────────────────────
    rows = []
    for s in universe:
        ticker = s["ticker"]
        status = s.get("shariah_status", "unknown")

        if search and search not in ticker:
            continue
        if shariah_filter != "All" and status != shariah_filter:
            continue

        rows.append({
            "Ticker":     ticker,
            "Shariah":    status,
            "Price":      f"${s.get('price', 0):.2f}",
            "Mkt Cap":    f"${s.get('marketCap', 0)/1e9:.1f}B",
            "Avg Volume": f"{s.get('avgVolume', 0)/1e6:.1f}M",
            "Score":      round(s.get("composite", 0), 1),
            "Action":     rec_map.get(ticker, "—"),
        })

    st.write(f"{len(rows)} stocks shown")
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No stocks match the current filters.")
