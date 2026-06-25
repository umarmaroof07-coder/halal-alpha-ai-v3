"""Reusable Streamlit UI components."""

from __future__ import annotations

import streamlit as st


def bias_warnings() -> None:
    """Render the two mandatory bias warnings."""
    st.warning(
        "⚠ SURVIVORSHIP BIAS: Backtest uses tickers that exist today. "
        "Returns are likely overstated."
    )
    st.warning(
        "⚠ LOOK-AHEAD BIAS: Fundamental data is not point-in-time."
    )


def shariah_badge(status: str) -> str:
    """Return a short label for Shariah status."""
    return {"compliant": "✓ Halal", "non_compliant": "✗ Haram", "unknown": "? Unknown"}.get(
        status, "? Unknown"
    )
