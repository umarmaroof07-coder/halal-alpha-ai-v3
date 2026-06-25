"""
Risk Engine — quantifies company-specific downside risk factors.

Output: risk_adjustment_score (0–100)
  100 = very low risk (maximum safety reward)
   50 = average risk (neutral contribution)
    0 = very high risk (maximum penalty)

The score is used as a 5% weight in the composite. Higher score = safer company
= small boost. Lower score = riskier company = small penalty.

Sub-signals (each → [0,1] where 1 = lower risk):
  1. Leverage safety    — equity_ratio (high equity, low debt = safer)
  2. FCF safety         — positive FCF margin = cash-generating, not cash-burning
  3. Earnings quality   — high EQ raw score = clean accrual-free earnings (safer)
  4. SBC burden         — low SBC/FCF = less earnings distortion risk
  5. Sector cyclicality — defensive sectors less prone to macro drawdown

Risk labels (based on average signal score):
  Low     ≥ 0.65
  Medium  ≥ 0.35
  High    <  0.35

Warnings:
  - Negative FCF margin: "Company is cash-consuming — operating burn risk"
  - High leverage (equity_ratio < 0.30): "High debt burden"
  - SBC > 40% of FCF: "SBC distorts reported FCF — overstated cash generation"
  - High cyclicality sector: "Sector subject to significant macro cyclicality"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Sectors classified as high/medium/low cyclicality.
# Remaining sectors default to medium.
_HIGH_CYCLICALITY = {
    "Energy", "Metals & Mining", "Basic Materials", "Mining",
    "Airlines", "Road & Rail", "Automobiles", "Semiconductors",
    "Chemicals", "Construction", "Building",
}
_LOW_CYCLICALITY = {
    "Health Care", "Healthcare", "Biotechnology", "Pharmaceuticals",
    "Life Sciences Tools & Services", "Food Products", "Beverages",
    "Consumer Defensive", "Utilities",
}


@dataclass
class RiskRaw:
    ticker: str

    # Sub-signal values
    leverage_safety:  float | None = None   # 0-1
    fcf_safety:       float | None = None   # 0-1
    eq_safety:        float | None = None   # 0-1 from EQ raw_score
    sbc_safety:       float | None = None   # 0-1
    cyclicality_safety: float | None = None # 0-1

    # Output
    raw_score:    float | None = None  # 0-1 (higher = lower risk)
    risk_label:   str          = "Medium"
    signals_used: list[str]    = field(default_factory=list)
    warnings:     list[str]    = field(default_factory=list)


def compute_risk(
    ticker:         str,
    equity_ratio:   float | None = None,   # equity / (equity + debt) → [0,1]
    fcf_margin:     float | None = None,   # FCF / Revenue
    eq_raw_score:   float | None = None,   # from EarningsQualityRaw.raw_score [0,1]
    sbc_ratio:      float | None = None,   # SBC / FCF from EarningsQualityRaw
    sector:         str   = "",
) -> RiskRaw:
    result = RiskRaw(ticker=ticker)
    signals: list[float] = []

    # ── Signal 1: Leverage safety ────────────────────────────────────────────
    # equity_ratio already [0,1]: 1 = all equity (safest), 0 = all debt (riskiest)
    if equity_ratio is not None:
        result.leverage_safety = float(equity_ratio)
        signals.append(result.leverage_safety)
        result.signals_used.append("leverage_safety")
        if equity_ratio < 0.30:
            result.warnings.append(
                f"High debt burden — equity ratio {equity_ratio*100:.0f}% "
                f"(debt = {(1-equity_ratio)*100:.0f}% of capital structure)"
            )

    # ── Signal 2: FCF safety ─────────────────────────────────────────────────
    # FCF margin: <0 = dangerous, 0-5% = marginal, >15% = strong
    if fcf_margin is not None:
        if fcf_margin < 0:
            result.fcf_safety = 0.0
            result.warnings.append(
                f"Negative FCF margin ({fcf_margin*100:.1f}%) — "
                f"company is cash-consuming at current scale"
            )
        elif fcf_margin >= 0.20:
            result.fcf_safety = 1.0
        else:
            result.fcf_safety = fcf_margin / 0.20
        signals.append(result.fcf_safety)
        result.signals_used.append("fcf_safety")

    # ── Signal 3: Earnings quality as risk proxy ─────────────────────────────
    # High EQ score = cash-backed earnings = lower accounting risk
    if eq_raw_score is not None:
        result.eq_safety = float(eq_raw_score)   # already [0,1]
        signals.append(result.eq_safety)
        result.signals_used.append("eq_safety")

    # ── Signal 4: SBC burden ─────────────────────────────────────────────────
    # High SBC/FCF = FCF overstated = financial distortion risk
    if sbc_ratio is not None:
        if sbc_ratio <= 0.10:
            result.sbc_safety = 1.0
        elif sbc_ratio <= 0.50:
            result.sbc_safety = 1.0 - (sbc_ratio - 0.10) / 0.40
        else:
            result.sbc_safety = 0.0
        signals.append(result.sbc_safety)
        result.signals_used.append("sbc_burden")
        if sbc_ratio > 0.40:
            result.warnings.append(
                f"SBC = {sbc_ratio*100:.0f}% of FCF — "
                f"reported FCF significantly overstates true owner earnings"
            )

    # ── Signal 5: Sector cyclicality ────────────────────────────────────────
    sector_clean = (sector or "").strip()
    if sector_clean in _HIGH_CYCLICALITY:
        result.cyclicality_safety = 0.35
        result.warnings.append(
            f"Sector '{sector_clean}' has high cyclicality — "
            f"earnings sensitive to macro/commodity cycles"
        )
    elif sector_clean in _LOW_CYCLICALITY:
        result.cyclicality_safety = 0.80
    else:
        result.cyclicality_safety = 0.55   # medium default

    signals.append(result.cyclicality_safety)
    result.signals_used.append("sector_cyclicality")

    if len(signals) < 2:
        return result

    avg = sum(signals) / len(signals)
    result.raw_score = max(0.0, min(1.0, avg))

    if avg >= 0.65:
        result.risk_label = "Low"
    elif avg >= 0.35:
        result.risk_label = "Medium"
    else:
        result.risk_label = "High"

    return result
