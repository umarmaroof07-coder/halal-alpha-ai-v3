"""Final Picks page — 13-gate institutional validation report."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st


_CACHE = Path("data/cache/final_picks_report.json")


def _gate_badge(passed: bool, has_warn: bool) -> str:
    if not passed:
        return "<span class='badge-block'>BLOCK</span>"
    if has_warn:
        return "<span class='badge-warn'>WARN</span>"
    return "<span class='badge-pass'>PASS</span>"


def render() -> None:
    st.title("Final Picks — 13-Gate Institutional Validation")

    from dashboard.state import (
        run_generate_final_picks,
        run_real_money_audit,
        get_recommendations,
    )

    # ── Action buttons ────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🏛 Generate Final Picks", help="Run 13-gate validation"):
            with st.spinner("Running 13-gate validation…"):
                ok, out = run_generate_final_picks()
            if ok:
                st.success("Final picks generated.")
            else:
                st.error("Pipeline error — see output below.")
            with st.expander("Pipeline output"):
                st.code(out)
            st.rerun()

    with col2:
        if st.button("✅ Run Real Money Audit", help="Full 13-gate real-money readiness check"):
            with st.spinner("Running audit…"):
                ok, out = run_real_money_audit()
            if ok:
                st.success("Audit complete.")
            else:
                st.error("Audit reported issues.")
            with st.expander("Audit output"):
                st.code(out)

    st.divider()

    # ── Load report ───────────────────────────────────────────────────────────
    report: dict | None = None
    if _CACHE.exists():
        with _CACHE.open() as f:
            try:
                report = json.load(f)
            except Exception:
                report = None

    if report is None:
        st.info(
            "No final picks report found.  \n"
            "Click **Generate Final Picks** to run the 13-gate validation."
        )
        # Still show live Top 5 from safe_recommendations
        st.subheader("Current Top 5 (from safe_recommendations)")
        recs = get_recommendations()
        buy = [r for r in recs if r.action == "BUY NOW"]
        if buy:
            for r in buy:
                with st.container():
                    st.markdown(f"### #{r.rank} {r.ticker}")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Composite Score", f"{r.composite_score:.1f}")
                    c2.metric("Weight", f"{(r.conviction_weight or 0)*100:.0f}%")
                    c3.metric("$ Allocation", f"${r.dollar_amount or 0:,.0f}")
                    c4.metric("Shariah", r.shariah_status.upper())
                    st.divider()
        else:
            st.warning("No BUY NOW picks — run --refresh-data first.")
        return

    # ── Gate summary ──────────────────────────────────────────────────────────
    issues = report.get("issues", [])
    blocks = [i for i in issues if i.get("severity") == "BLOCK"]
    warns  = [i for i in issues if i.get("severity") == "WARN"]

    ready  = report.get("real_money_ready", False)
    badge  = (
        "<span class='badge-pass' style='font-size:16px;padding:6px 14px'>✅ REAL MONEY READY</span>"
        if ready else
        "<span class='badge-block' style='font-size:16px;padding:6px 14px'>🚫 NOT READY</span>"
    )
    st.markdown(f"**Status:** {badge}", unsafe_allow_html=True)
    st.caption(f"Generated: {report.get('generated_at', '—')}  |  "
               f"Blocks: {len(blocks)}  |  Warnings: {len(warns)}")

    st.divider()

    # ── Gate status table ─────────────────────────────────────────────────────
    st.subheader("Gate Status")
    gate_names = {
        1: "Data Quality",
        2: "Factor Integrity",
        3: "Accounting Quality",
        4: "AI Research",
        5: "Shariah Compliance",
        6: "Position Sizing",
        7: "Liquidity",
        8: "Stress Test",
        9: "Ranking Stability",
        10: "Red Flags",
        11: "Model Confidence",
        12: "Concentration",
        13: "Walk-Forward",
    }

    gate_blocks = {i.get("gate"): True  for i in blocks}
    gate_warns  = {i.get("gate"): True  for i in warns}

    rows = []
    for g, name in gate_names.items():
        blocked  = gate_blocks.get(g, False)
        warned   = gate_warns.get(g, False)
        status   = "BLOCK" if blocked else ("WARN" if warned else "PASS")
        rows.append({"Gate": g, "Name": name, "Status": status})

    st.dataframe(rows, use_container_width=True, hide_index=True)

    if blocks:
        st.error(f"**{len(blocks)} gate(s) blocked.** Resolve before trading.")
        with st.expander("View blocks"):
            for b in blocks:
                st.markdown(f"- Gate {b.get('gate')}: **{b.get('ticker', '—')}** — {b.get('message')}")

    if warns:
        with st.expander(f"View {len(warns)} warning(s)"):
            for w in warns:
                st.markdown(f"- Gate {w.get('gate')}: **{w.get('ticker', '—')}** — {w.get('message')}")

    st.divider()

    # ── Top 5 cards ───────────────────────────────────────────────────────────
    top5 = report.get("top5", [])
    if not top5:
        st.info("No Top 5 in report.")
        return

    # ── Limit order summary table ─────────────────────────────────────────────
    st.subheader("Limit Order Prices")
    st.warning("Use limit orders only. Price may not fill. Never use market orders.")

    limit_results: dict = {}
    if st.button("💰 Compute Limit Prices"):
        with st.spinner("Fetching live prices and computing limits…"):
            from portfolio.limit_prices import compute_limit_price
            for pick in top5:
                t   = pick.get("ticker", "")
                amt = float(pick.get("dollar_amount") or 0)
                if t:
                    limit_results[t] = compute_limit_price(t, amt).to_dict()
            st.session_state["limit_results"] = limit_results
        st.rerun()

    cached_limits = st.session_state.get("limit_results", {})
    if cached_limits:
        limit_rows = []
        for pick in top5:
            t   = pick.get("ticker", "")
            lp  = cached_limits.get(t, {})
            if not lp or lp.get("error") or lp.get("stale"):
                limit_rows.append({
                    "Ticker":          t,
                    "Live Price":      "—",
                    "Suggested Limit": "stale/error",
                    "Tier":            "—",
                    "Shares":          "—",
                    "Est. Fill Cost":  "—",
                })
                continue
            limit_rows.append({
                "Ticker":          t,
                "Live Price":      f"${lp.get('live_price', 0):.2f}",
                "Bid":             f"${lp.get('bid', 0):.2f}" if lp.get("bid") else "—",
                "Ask":             f"${lp.get('ask', 0):.2f}" if lp.get("ask") else "—",
                "Suggested Limit": f"${lp.get('suggested_limit', 0):.2f}",
                "Tier":            lp.get("suggested_tier", "—"),
                "Cons. Limit":     f"${lp.get('conservative_limit', 0):.2f}",
                "Normal Limit":    f"${lp.get('normal_limit', 0):.2f}",
                "Aggr. Limit":     f"${lp.get('aggressive_limit', 0):.2f}",
                "Shares":          f"{lp.get('shares', 0):.3f}" if lp.get("shares") else "—",
                "Est. Fill Cost":  f"${lp.get('estimated_fill_cost', 0):,.2f}" if lp.get("estimated_fill_cost") else "—",
            })
        st.dataframe(limit_rows, use_container_width=True, hide_index=True)

        for pick in top5:
            t  = pick.get("ticker", "")
            lp = cached_limits.get(t, {})
            if lp.get("warning"):
                st.warning(f"**{t}:** {lp['warning']}")
            if lp.get("explanation"):
                st.caption(f"{t}: {lp['explanation']}")

    st.divider()

    # ── Top 5 cards ───────────────────────────────────────────────────────────
    st.subheader("Top 5 Final Picks")
    for pick in top5:
        ticker    = pick.get("ticker", "?")
        score     = pick.get("composite", 0)
        weight    = pick.get("weight", 0) * 100
        alloc     = pick.get("dollar_amount", 0)
        conf      = pick.get("model_confidence", {})
        conf_lbl  = conf.get("label", "—") if isinstance(conf, dict) else "—"
        conf_scr  = conf.get("score", 0)   if isinstance(conf, dict) else 0
        ai_score  = pick.get("ai_research", 0)
        shariah   = pick.get("shariah_status", "unknown").upper()
        strengths = pick.get("strengths", [])
        risks     = pick.get("risks", [])
        thesis    = pick.get("thesis_fail_condition", "")
        lp_data   = st.session_state.get("limit_results", {}).get(ticker, {})
        sug_limit = lp_data.get("suggested_limit")

        with st.expander(
            f"#{pick.get('rank', '?')} {ticker}  —  Score {score:.1f}  |  {weight:.0f}%  |  ${alloc:,.0f}",
            expanded=True,
        ):
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Composite",  f"{score:.1f}")
            c2.metric("Weight",     f"{weight:.0f}%")
            c3.metric("Allocation", f"${alloc:,.0f}")
            c4.metric("AI Score",   f"{ai_score:.0f}")
            c5.metric("Confidence", f"{conf_lbl} ({conf_scr:.0f})")

            if sug_limit:
                shares = alloc / sug_limit if sug_limit > 0 else None
                ca, cb, cc = st.columns(3)
                ca.metric("Live Price",     f"${lp_data.get('live_price', 0):.2f}")
                cb.metric("Suggested Limit", f"${sug_limit:.2f}")
                cc.metric("Shares",          f"{shares:.3f}" if shares else "—")

            c6, c7 = st.columns(2)
            with c6:
                sh_color = "green" if shariah == "COMPLIANT" else ("red" if shariah == "NON_COMPLIANT" else "orange")
                st.markdown(
                    f"**Shariah:** <span style='color:{sh_color}'>{shariah}</span>",
                    unsafe_allow_html=True,
                )
            with c7:
                st.markdown(f"**Risk:** {pick.get('risk_label', '—')}")

            if strengths:
                st.markdown("**Strengths:**")
                for s in strengths:
                    st.markdown(f"  - {s}")

            if risks:
                st.markdown("**Risks:**")
                for r in risks:
                    st.markdown(f"  - {r}")

            if thesis:
                st.warning(f"**Thesis fails if:** {thesis}")

    st.divider()

    # ── Stress test summary ───────────────────────────────────────────────────
    stress = report.get("stress_results", {})
    if stress:
        st.subheader("Gate 8 — Stress Test Results")
        for ticker, crises in stress.items():
            if not isinstance(crises, dict):
                continue
            st.markdown(f"**{ticker}**")
            crisis_rows = []
            for crisis, data in crises.items():
                if not isinstance(data, dict):
                    continue
                crisis_rows.append({
                    "Crisis":         crisis,
                    "Max Drawdown":   f"{data.get('max_drawdown', 0):.1f}%",
                    "Total Return":   f"{data.get('total_return', 0):.1f}%",
                    "SPY Drawdown":   f"{data.get('spy_drawdown', 0):.1f}%",
                    "vs SPY":         f"{data.get('relative_vs_spy', 0):+.1f}%",
                    "Has Data":       "✓" if data.get("has_data") else "✗",
                })
            if crisis_rows:
                st.dataframe(crisis_rows, use_container_width=True, hide_index=True)
