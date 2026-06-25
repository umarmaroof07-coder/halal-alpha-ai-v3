"""
Capital Allocation Quality — evaluates how wisely management deploys capital.

Measures whether management creates per-share value through intelligent
capital decisions, not just earnings growth.

Sub-signals (all normalized to [0,1] internally):
  1. Buyback quality   — net share reduction (negative dilution = reward)
  2. Debt paydown      — net debt reduction as % of equity (deleveraging = reward)
  3. ROIC level        — high sustained ROIC = capital deployed at good returns
  4. ROIC improvement  — year-on-year improvement in capital returns
  5. FCF per share     — FCF / share growing = per-share value creation

Penalties / Warnings:
  - Share dilution > 3% → "moderate dilution" warning
  - Share dilution > 7% → "significant dilution" warning + score penalty
  - Debt increase > 15% of equity → "leverage increasing" warning
  - ROIC declining → "returns on capital weakening" warning

Requires ≥2 valid signals; otherwise raw_score = None → neutral 50 in composite.
Raw float is returned. Cross-sectional z-scoring happens in composite.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

_DILUTION_WARN_MODERATE = 0.03   # 3% share count increase → moderate warning
_DILUTION_WARN_SEVERE   = 0.07   # 7% share count increase → severe warning + penalty
_SEVERE_DILUTION_PENALTY = 0.10  # raw score penalty for >7% dilution


@dataclass
class CapitalAllocationRaw:
    ticker: str

    # Sub-signal values
    buyback_rate:      float | None = None   # -(share_dilution): positive = buyback
    debt_paydown_rate: float | None = None   # -(debt_change/equity): positive = paydown
    roic_level:        float | None = None   # current ROIC [0,1]
    roic_improvement:  float | None = None   # ROIC current - prior
    fcf_per_share_growth: float | None = None  # (FCF/share current − prior) / |prior|

    # Penalty for severe dilution
    dilution_penalty:  float = 0.0

    # Result
    raw_score:    float | None = None
    signals_used: list[str]    = field(default_factory=list)
    warnings:     list[str]    = field(default_factory=list)


def _roic_to_score(roic: float) -> float:
    """ROIC level → [0,1]. >30% = 1.0, <0% = 0."""
    if roic >= 0.30: return 1.0
    if roic >= 0.20: return 0.8 + (roic - 0.20) / 0.10 * 0.2
    if roic >= 0.10: return 0.5 + (roic - 0.10) / 0.10 * 0.3
    if roic >= 0.05: return 0.2 + (roic - 0.05) / 0.05 * 0.3
    if roic >= 0.0:  return roic / 0.05 * 0.2
    return 0.0


def _clamp_map(value: float, low: float, high: float) -> float:
    if value >= high: return 1.0
    if value <= low:  return 0.0
    return (value - low) / (high - low)


def compute_capital_allocation(
    ticker:              str,
    shares_current:      float | None = None,
    shares_prior:        float | None = None,
    total_debt_current:  float | None = None,
    total_debt_prior:    float | None = None,
    total_equity_current: float | None = None,
    roic_current:        float | None = None,
    roic_prior:          float | None = None,
    fcf_current:         float | None = None,
    fcf_prior:           float | None = None,
) -> CapitalAllocationRaw:
    result = CapitalAllocationRaw(ticker=ticker)
    signals: list[float] = []

    # ── Signal 1: Buyback quality ────────────────────────────────────────────
    # Negative share dilution (buybacks) = positive signal.
    # Range: buyback_rate in [−10%, +10%] → [0,1]
    if shares_current is not None and shares_prior is not None and shares_prior > 0:
        dilution = (shares_current - shares_prior) / shares_prior
        result.buyback_rate = -dilution   # positive = buyback

        # Clamp to [−10%, +10%] and map: buyback+10% → 1.0, dilution+10% → 0.0
        buyback_score = _clamp_map(-dilution, -0.10, 0.10)
        signals.append(buyback_score)
        result.signals_used.append("buyback_rate")

        if dilution > _DILUTION_WARN_SEVERE:
            result.dilution_penalty = _SEVERE_DILUTION_PENALTY
            result.warnings.append(
                f"Significant share dilution: +{dilution*100:.1f}% shares issued "
                f"(penalty −{_SEVERE_DILUTION_PENALTY:.0%} applied)"
            )
        elif dilution > _DILUTION_WARN_MODERATE:
            result.warnings.append(
                f"Moderate share dilution: +{dilution*100:.1f}% — "
                f"shareholder ownership being reduced"
            )
        elif dilution < -0.01:
            # Buyback
            result.warnings = [w for w in result.warnings if "dilution" not in w.lower()]

    # ── Signal 2: Debt paydown ───────────────────────────────────────────────
    # Debt reduction as % of equity = positive signal.
    # Range: paydown_rate in [−20%, +20%] of equity → [0,1]
    if (total_debt_current is not None and total_debt_prior is not None
            and total_equity_current is not None and total_equity_current > 0):
        debt_change = total_debt_current - total_debt_prior
        result.debt_paydown_rate = -debt_change / total_equity_current  # positive = paydown

        paydown_score = _clamp_map(-debt_change / total_equity_current, -0.20, 0.20)
        signals.append(paydown_score)
        result.signals_used.append("debt_paydown")

        if debt_change / total_equity_current > 0.15:
            result.warnings.append(
                f"Leverage increasing: debt up "
                f"{(debt_change/total_equity_current)*100:.0f}% vs equity"
            )

    # ── Signal 3: ROIC level ─────────────────────────────────────────────────
    if roic_current is not None:
        result.roic_level = roic_current
        signals.append(_roic_to_score(roic_current))
        result.signals_used.append("roic_level")

    # ── Signal 4: ROIC improvement ───────────────────────────────────────────
    if roic_current is not None and roic_prior is not None:
        roic_chg = roic_current - roic_prior
        result.roic_improvement = roic_chg
        # [−10pp, +10pp] → [0,1]
        roic_impr_score = _clamp_map(roic_chg, -0.10, 0.10)
        signals.append(roic_impr_score)
        result.signals_used.append("roic_improvement")
        if roic_chg < -0.03:
            result.warnings.append(
                f"ROIC declining {abs(roic_chg)*100:.1f}pp — "
                f"returns on capital weakening"
            )

    # ── Signal 5: FCF per share growth ──────────────────────────────────────
    if (fcf_current is not None and fcf_prior is not None
            and shares_current is not None and shares_prior is not None
            and shares_current > 0 and shares_prior > 0):
        fcf_ps_current = fcf_current / shares_current
        fcf_ps_prior   = fcf_prior   / shares_prior
        if fcf_ps_prior != 0 and abs(fcf_ps_prior) >= abs(fcf_ps_current) * 0.05:
            fcf_ps_growth = (fcf_ps_current - fcf_ps_prior) / abs(fcf_ps_prior)
            result.fcf_per_share_growth = fcf_ps_growth
            # Growth of [−50%, +100%] → [0, 1]
            fcf_ps_score = _clamp_map(fcf_ps_growth, -0.50, 1.00)
            signals.append(fcf_ps_score)
            result.signals_used.append("fcf_per_share_growth")

    if len(signals) < 2:
        return result

    raw = sum(signals) / len(signals)
    raw -= result.dilution_penalty
    result.raw_score = max(0.0, min(1.0, raw))
    return result
