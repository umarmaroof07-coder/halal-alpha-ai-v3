"""Factor Monitor page — IC, hit rate, decay/instability flags across snapshots."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("Factor Monitor")
    st.caption(
        "Tracks Information Coefficient (IC), hit rate, and stability across "
        "point-in-time snapshots. For reporting only — never changes live weights."
    )

    from dashboard.state import get_factor_monitor_result

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🔬 Run Factor Monitor"):
            with st.spinner("Analyzing factor health across snapshots…"):
                try:
                    from analysis.factor_monitor import run_factor_monitor
                    result = run_factor_monitor()
                    d = result.to_dict()
                    import json
                    from pathlib import Path
                    Path("data/cache/factor_monitor.json").write_text(json.dumps(d, indent=2))
                    st.session_state["_factor_monitor_cache"] = d
                    st.success(f"Analysis complete — {d.get('n_snapshots_used', 0)} snapshot(s) used.")
                except Exception as e:
                    st.error(f"Error: {e}")
            st.rerun()

    data = st.session_state.get("_factor_monitor_cache") or get_factor_monitor_result()

    if data is None:
        st.info(
            "No factor monitor results found.  \n"
            "Run the monitor above (requires ≥2 snapshots from --refresh-data runs on different dates)."
        )
        return

    st.divider()
    st.caption(
        f"Generated: {data.get('generated_at', '—')}  |  "
        f"Snapshots used: {data.get('n_snapshots_used', 0)}"
    )

    # ── Warnings ──────────────────────────────────────────────────────────────
    warnings = data.get("warnings", [])
    if warnings:
        for w in warnings:
            st.warning(w)

    st.divider()

    # ── Factor stats table ────────────────────────────────────────────────────
    st.subheader("Factor Health")
    factors = data.get("factors", [])
    if not factors:
        st.info("No factor stats computed — need ≥2 snapshots.")
    else:
        rows = []
        for f in factors:
            rows.append({
                "Factor":        f.get("factor", "?"),
                "IC":            f"{f.get('ic', 0):.3f}" if f.get("ic") is not None else "—",
                "IC Std":        f"{f.get('ic_std', 0):.3f}" if f.get("ic_std") is not None else "—",
                "Hit Rate":      f"{f.get('hit_rate', 0):.1%}" if f.get("hit_rate") is not None else "—",
                "Avg Contrib":   f"{f.get('avg_contribution', 0):.2f}" if f.get("avg_contribution") is not None else "—",
                "N Obs":         f.get("n_observations", 0),
                "Decay Flag":    "⚠ DECAY"       if f.get("flag_decay")       else "✓",
                "Instab. Flag":  "⚠ UNSTABLE"    if f.get("flag_instability") else "✓",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

        # Flag summary
        decaying   = [f["factor"] for f in factors if f.get("flag_decay")]
        unstable   = [f["factor"] for f in factors if f.get("flag_instability")]
        if decaying:
            st.warning(f"IC decay detected in: {', '.join(decaying)}")
        if unstable:
            st.warning(f"IC instability detected in: {', '.join(unstable)}")
        if not decaying and not unstable:
            st.success("No factor health issues detected.")

    st.divider()

    # ── Redundant pairs ───────────────────────────────────────────────────────
    redundant = data.get("redundant_pairs", [])
    if redundant:
        st.subheader("Redundant Factor Pairs (correlation > 0.85)")
        st.warning(
            "These factor pairs are highly correlated. They provide overlapping signal. "
            "Review but do NOT change weights based solely on this — consult the investment committee."
        )
        for pair in redundant:
            st.markdown(f"- **{pair[0]}** ↔ **{pair[1]}**  (ρ = {pair[2]:.3f})")
    else:
        st.info("No redundant factor pairs detected.")

    with st.expander("Raw JSON"):
        st.json(data)
