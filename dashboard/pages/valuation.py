"""Valuation Model page — P/E, FCF yield, simple DCF inputs."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("Valuation Model")

    from dashboard.state import get_universe

    universe = get_universe()
    compliant = [s["ticker"] for s in universe if s.get("shariah_status") == "compliant"]

    if not compliant:
        st.info("No compliant stocks to analyse.")
        return

    ticker = st.selectbox("Select ticker", sorted(compliant))
    stock_data = next((s for s in universe if s["ticker"] == ticker), None)

    # ── Current multiples ─────────────────────────────────────────────────────
    st.subheader("Current Valuation Multiples")
    price = stock_data.get("price", 0) if stock_data else 0

    col1, col2 = st.columns(2)
    col1.metric("Current Price", f"${price:.2f}")
    col2.caption("Additional multiples available once live data layer is connected.")

    st.divider()

    # ── Simple DCF inputs ─────────────────────────────────────────────────────
    st.subheader("Simple DCF Calculator")
    st.caption("Adjust inputs below to estimate intrinsic value.")

    c1, c2, c3 = st.columns(3)
    fcf         = c1.number_input("Annual FCF ($M)", value=5_000, step=100)
    growth_rate = c2.number_input("Growth Rate (%)", value=10.0,  step=0.5,  format="%.1f")
    discount_r  = c3.number_input("Discount Rate (%)", value=10.0, step=0.5, format="%.1f")
    terminal_g  = st.number_input("Terminal Growth Rate (%)", value=3.0, step=0.5, format="%.1f")
    years        = st.slider("Projection Years", 5, 15, 10)

    shares_out = st.number_input("Shares Outstanding (M)", value=15_000, step=100)

    if st.button("Calculate Intrinsic Value"):
        r = discount_r / 100
        g = growth_rate / 100
        tg = terminal_g / 100

        if r <= tg:
            st.error("Discount rate must be greater than terminal growth rate.")
        elif shares_out <= 0:
            st.error("Shares outstanding must be positive.")
        else:
            pv_fcfs = 0.0
            cf = fcf * 1e6
            for yr in range(1, years + 1):
                cf *= (1 + g)
                pv_fcfs += cf / ((1 + r) ** yr)

            terminal_value = (cf * (1 + tg)) / (r - tg)
            pv_terminal    = terminal_value / ((1 + r) ** years)
            enterprise_value = pv_fcfs + pv_terminal
            intrinsic_per_share = enterprise_value / (shares_out * 1e6)

            margin_of_safety = ((intrinsic_per_share - price) / intrinsic_per_share * 100
                                if intrinsic_per_share > 0 else 0)

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Intrinsic Value / Share", f"${intrinsic_per_share:,.2f}")
            col_b.metric("Current Price",           f"${price:.2f}")
            col_c.metric("Margin of Safety",        f"{margin_of_safety:.1f}%")

            if intrinsic_per_share > price:
                st.success(f"{ticker} appears undervalued by {margin_of_safety:.1f}% on these assumptions.")
            else:
                st.warning(f"{ticker} appears overvalued on these assumptions.")

            st.caption(
                "DCF is highly sensitive to growth and discount rate assumptions. "
                "This is a rough estimate, not investment advice."
            )
