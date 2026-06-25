"""
Earnings Quality 2.0 — measures the reliability and sustainability of reported earnings.

Sub-signals (all normalized to [0,1] internally):
  1. FCF Conversion     FCF / Net Income  (higher = cash-backed earnings)
  2. Accrual Ratio      (NI − FCF) / Revenue  (lower = better)
  3. Share Dilution     (shares_cur − shares_pri) / shares_pri  (buybacks = good)
  4. Debt Trend         (debt_cur − debt_pri) / equity_cur  (deleveraging = good)
  5. ROIC Trend         roic_cur − roic_pri  (improvement = good)
  6. Margin Stability   abs(op_margin_cur − op_margin_pri)  (stable = good)
  7. Asset Turnover     revenue / total_assets trend  (improving efficiency = good)

SBC Penalty (applied directly to raw score before z-scoring):
  SBC/FCF > 50% → large  penalty (−0.20)
  SBC/FCF > 30% → medium penalty (−0.10)
  SBC/FCF > 20% → small  penalty (−0.05)

Accounting Distortion Flags:
  Revenue growth > 50% → possible merger/acquisition artifact
  ROIC > 100%          → possible one-time tax benefit or distortion

Two or more valid signals are required; otherwise raw_score = None → neutral 50.
Raw floats enter the cross-sectional z-scorer in composite.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

_SBC_PENALTY_LARGE  = 0.20   # SBC/FCF > 50%
_SBC_PENALTY_MEDIUM = 0.10   # SBC/FCF > 30%
_SBC_PENALTY_SMALL  = 0.05   # SBC/FCF > 20%


@dataclass
class EarningsQualityRaw:
    ticker: str

    # Sub-signal values (pre-normalized)
    fcf_conversion:   float | None = None   # FCF / NI
    accrual_ratio:    float | None = None   # (NI − FCF) / Rev
    sbc_ratio:        float | None = None   # SBC / FCF
    share_dilution:   float | None = None   # share count change rate
    debt_trend:       float | None = None   # debt change / equity
    roic_trend:       float | None = None   # ROIC year-on-year change
    margin_stability: float | None = None   # abs(op_margin_cur - op_margin_prior) [EQ2]
    asset_turnover_trend: float | None = None  # change in rev/assets [EQ2]

    # Distortion flags
    distortion_flags: list[str]    = field(default_factory=list)

    # SBC penalty applied to raw score
    sbc_penalty:      float = 0.0
    sbc_penalty_tier: str   = "none"        # "none" | "small" | "medium" | "large"

    # Result
    raw_score:        float | None = None
    signals_used:     list[str]    = field(default_factory=list)
    warnings:         list[str]    = field(default_factory=list)


def compute_earnings_quality(
    ticker:               str,
    net_income:           float | None = None,
    free_cash_flow:       float | None = None,
    revenue:              float | None = None,
    revenue_prior:        float | None = None,   # EQ2: for asset turnover trend
    sbc:                  float | None = None,
    shares_current:       float | None = None,
    shares_prior:         float | None = None,
    total_debt_current:   float | None = None,
    total_debt_prior:     float | None = None,
    total_equity_current: float | None = None,
    roic_current:         float | None = None,
    roic_prior:           float | None = None,
    operating_margin:     float | None = None,   # EQ2: current op margin
    operating_margin_prior: float | None = None, # EQ2: prior op margin
    total_assets_current: float | None = None,   # EQ2: for asset turnover
    total_assets_prior:   float | None = None,   # EQ2: for asset turnover trend
) -> EarningsQualityRaw:
    result = EarningsQualityRaw(ticker=ticker)
    signals: list[float] = []

    # ── Signal 1: FCF Conversion = FCF / NI ────────────────────────────────
    if (net_income is not None and free_cash_flow is not None
            and net_income > 0):
        result.fcf_conversion = free_cash_flow / net_income
        # Clamp to [0, 1.5] → [0, 1]
        conv_score = min(max(result.fcf_conversion, 0.0) / 1.5, 1.0)
        signals.append(conv_score)
        result.signals_used.append("fcf_conversion")
        if result.fcf_conversion < 0.7:
            result.warnings.append(
                f"Weak FCF conversion {result.fcf_conversion:.2f}× "
                f"— NI may not be fully backed by cash"
            )

    # ── Signal 2: Accrual Ratio = (NI − FCF) / Revenue ─────────────────────
    # Lower = better quality earnings (FCF exceeds NI = good)
    if (net_income is not None and free_cash_flow is not None
            and revenue is not None and revenue > 0):
        result.accrual_ratio = (net_income - free_cash_flow) / revenue
        ar = max(-0.30, min(0.30, result.accrual_ratio))
        accrual_score = (0.30 - ar) / 0.60        # –0.3 → 1.0, +0.3 → 0.0
        signals.append(accrual_score)
        result.signals_used.append("accrual_ratio")

    # ── SBC penalty (not a signal — direct deduction) ───────────────────────
    if (sbc is not None and free_cash_flow is not None and free_cash_flow > 0):
        result.sbc_ratio = sbc / free_cash_flow
        if result.sbc_ratio > 0.50:
            result.sbc_penalty      = _SBC_PENALTY_LARGE
            result.sbc_penalty_tier = "large"
            result.warnings.append(
                f"SBC = {result.sbc_ratio*100:.0f}% of FCF "
                f"(large penalty −{_SBC_PENALTY_LARGE:.0%} applied)"
            )
        elif result.sbc_ratio > 0.30:
            result.sbc_penalty      = _SBC_PENALTY_MEDIUM
            result.sbc_penalty_tier = "medium"
            result.warnings.append(
                f"SBC = {result.sbc_ratio*100:.0f}% of FCF "
                f"(medium penalty −{_SBC_PENALTY_MEDIUM:.0%} applied)"
            )
        elif result.sbc_ratio > 0.20:
            result.sbc_penalty      = _SBC_PENALTY_SMALL
            result.sbc_penalty_tier = "small"
            result.warnings.append(
                f"SBC = {result.sbc_ratio*100:.0f}% of FCF "
                f"(small penalty −{_SBC_PENALTY_SMALL:.0%} applied)"
            )

    # ── Signal 3: Share Dilution ─────────────────────────────────────────────
    if (shares_current is not None and shares_prior is not None
            and shares_prior > 0):
        result.share_dilution = (shares_current - shares_prior) / shares_prior
        dil = max(-0.10, min(0.10, result.share_dilution))
        dilution_score = (0.10 - dil) / 0.20      # −10% → 1.0, +10% → 0.0
        signals.append(dilution_score)
        result.signals_used.append("share_dilution")
        if result.share_dilution > 0.02:
            result.warnings.append(
                f"Share count up {result.share_dilution*100:.1f}% "
                f"— shareholder dilution"
            )

    # ── Signal 4: Debt Trend ─────────────────────────────────────────────────
    if (total_debt_current is not None and total_debt_prior is not None
            and total_equity_current is not None and total_equity_current > 0):
        debt_change         = total_debt_current - total_debt_prior
        result.debt_trend   = debt_change / total_equity_current
        dt = max(-0.20, min(0.20, result.debt_trend))
        debt_score = (0.20 - dt) / 0.40           # −20% → 1.0, +20% → 0.0
        signals.append(debt_score)
        result.signals_used.append("debt_trend")
        if result.debt_trend > 0.10:
            result.warnings.append(
                f"Debt rising {result.debt_trend*100:.0f}% vs equity "
                f"— leverage increasing"
            )

    # ── Signal 5: ROIC Trend ─────────────────────────────────────────────────
    if roic_current is not None and roic_prior is not None:
        result.roic_trend = roic_current - roic_prior
        rt = max(-0.10, min(0.10, result.roic_trend))
        roic_trend_score = (rt + 0.10) / 0.20     # +10pp → 1.0, −10pp → 0.0
        signals.append(roic_trend_score)
        result.signals_used.append("roic_trend")
        if result.roic_trend < -0.02:
            result.warnings.append(
                f"ROIC declining {abs(result.roic_trend)*100:.1f}pp year-over-year"
            )

    # ── Signal 6: Margin Stability (EQ 2.0) ──────────────────────────────────
    # Stable operating margins = predictable, reliable business
    # Volatile margins = lower quality earnings
    if operating_margin is not None and operating_margin_prior is not None:
        margin_chg = abs(operating_margin - operating_margin_prior)
        result.margin_stability = margin_chg
        # 0pp change → 1.0 (perfect stability), 10pp change → 0.0 (volatile)
        stability_score = max(0.0, 1.0 - margin_chg / 0.10)
        signals.append(stability_score)
        result.signals_used.append("margin_stability")
        if margin_chg > 0.05:
            result.warnings.append(
                f"Operating margin volatile: {margin_chg*100:.1f}pp swing "
                f"year-over-year (earnings predictability reduced)"
            )

    # ── Signal 7: Asset Turnover Trend (EQ 2.0) ──────────────────────────────
    # Improving revenue per dollar of assets = capital deployed more efficiently
    if (revenue is not None and revenue_prior is not None
            and total_assets_current is not None and total_assets_prior is not None
            and total_assets_current > 0 and total_assets_prior > 0):
        at_current = revenue / total_assets_current
        at_prior   = revenue_prior / total_assets_prior
        if at_prior > 0:
            at_change = (at_current - at_prior) / at_prior
            result.asset_turnover_trend = at_change
            # [−30%, +30%] improvement → [0, 1]
            at_score = max(0.0, min(1.0, (at_change + 0.30) / 0.60))
            signals.append(at_score)
            result.signals_used.append("asset_turnover")

    # ── Accounting Distortion Detection ──────────────────────────────────────
    # These are flags — they don't modify the raw score directly but are
    # stored for the integrity gate and final report.
    if (revenue is not None and revenue_prior is not None
            and revenue_prior > 0):
        rev_growth = (revenue - revenue_prior) / revenue_prior
        if rev_growth > 0.50:
            result.distortion_flags.append(
                f"Revenue growth {rev_growth*100:.0f}% — "
                f"possible merger/acquisition-driven (not organic growth)"
            )

    if roic_current is not None and roic_current > 1.0:
        result.distortion_flags.append(
            f"ROIC = {roic_current*100:.0f}% (>100%) — "
            f"possible one-time tax benefit, asset sale, or accounting distortion"
        )

    # ── Need ≥2 valid signals ────────────────────────────────────────────────
    if len(signals) < 2:
        return result

    raw = sum(signals) / len(signals)
    raw -= result.sbc_penalty
    result.raw_score = max(0.0, min(1.0, raw))
    return result
