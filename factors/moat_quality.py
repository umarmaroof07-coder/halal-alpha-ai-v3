"""
Quantitative Moat Quality 2.0 — non-AI moat signals from financial fundamentals.

Signals:
  1. ROIC Level          >20% = strong competitive advantage
  2. ROIC Stability      low year-on-year variance = durable moat
  3. Operating Margin    pricing power and operating leverage
  4. Margin Stability    consistent margins = predictable business
  5. FCF Margin          cash generation quality
  6. Debt Level          equity_ratio (low debt = financial flexibility)
  7. Revenue Consistency positive growth = sustained demand
  8. Gross Margin        pricing power and product economics

Final blended_moat_score = 70% quant_moat + 30% AI moat (with confidence scaling).
AI moat = (raw_ai_moat − 50) × confidence + 50 before blending.
When AI moat is unavailable, blended = 100% quant.

blended_moat_score is used as a raw input to the composite factor pipeline
(z-scored cross-sectionally, weighted at 10% in composite).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class MoatQualityRaw:
    ticker: str

    # Sub-signal scores (each 0–1 internally)
    roic_score:            float | None = None
    roic_stability_score:  float | None = None
    op_margin_score:       float | None = None
    margin_stability_score:float | None = None
    fcf_margin_score:      float | None = None
    debt_level_score:      float | None = None
    rev_consistency_score: float | None = None

    gross_margin_score:    float | None = None   # signal 8

    # Outputs
    quant_moat_score:  float | None = None   # 0–100, pure quantitative
    ai_moat_score:     float | None = None   # 0–100, from AI research (raw)
    blended_moat_score:float | None = None   # 70% quant + 30% conf-scaled AI

    signals_used: list[str] = field(default_factory=list)


def _roic_to_score(roic: float) -> float:
    """ROIC → quality score [0, 1]. >30% = excellent, <0% = 0."""
    if roic >= 0.30:  return 1.0
    if roic >= 0.20:  return 0.8 + (roic - 0.20) / 0.10 * 0.2
    if roic >= 0.10:  return 0.5 + (roic - 0.10) / 0.10 * 0.3
    if roic >= 0.05:  return 0.2 + (roic - 0.05) / 0.05 * 0.3
    if roic >= 0.0:   return roic / 0.05 * 0.2
    return 0.0


def _clamp_map(value: float, low: float, high: float) -> float:
    """Linear map from [low, high] → [0, 1], clamped at boundaries."""
    if value >= high: return 1.0
    if value <= low:  return 0.0
    return (value - low) / (high - low)


def compute_moat_quality(
    ticker:              str,
    roic_current:        float | None = None,
    roic_prior:          float | None = None,
    operating_margin:    float | None = None,
    operating_margin_prior: float | None = None,
    fcf_margin:          float | None = None,
    equity_ratio:        float | None = None,   # equity / (equity + debt), [0,1]
    revenue_growth:      float | None = None,   # post-winsorization value or None
    gross_margin:        float | None = None,   # gross_profit / revenue [NEW]
    ai_moat_score:       float | None = None,   # from ai_detail["moat_score"], 0–100
    ai_confidence:       float = 0.0,           # AI confidence 0–1 [NEW]
) -> MoatQualityRaw:
    result = MoatQualityRaw(ticker=ticker)
    signals: list[float] = []

    # Signal 1: ROIC Level
    if roic_current is not None:
        result.roic_score = _roic_to_score(roic_current)
        signals.append(result.roic_score)
        result.signals_used.append("roic_level")

    # Signal 2: ROIC Stability
    if roic_current is not None and roic_prior is not None:
        change = abs(roic_current - roic_prior)
        # <2pp → 1.0, >10pp → 0.0
        result.roic_stability_score = max(0.0, 1.0 - change / 0.10)
        signals.append(result.roic_stability_score)
        result.signals_used.append("roic_stability")

    # Signal 3: Operating Margin
    if operating_margin is not None:
        # 5% → 0.0, 30% → 1.0
        result.op_margin_score = _clamp_map(operating_margin, 0.05, 0.30)
        signals.append(result.op_margin_score)
        result.signals_used.append("op_margin")

    # Signal 4: Margin Stability (operating margin year-on-year)
    if operating_margin is not None and operating_margin_prior is not None:
        change = abs(operating_margin - operating_margin_prior)
        # <1pp → 1.0, >5pp → 0.0
        result.margin_stability_score = max(0.0, 1.0 - change / 0.05)
        signals.append(result.margin_stability_score)
        result.signals_used.append("margin_stability")

    # Signal 5: FCF Margin
    if fcf_margin is not None:
        # 0% → 0.0, 20% → 1.0
        result.fcf_margin_score = _clamp_map(fcf_margin, 0.0, 0.20)
        signals.append(result.fcf_margin_score)
        result.signals_used.append("fcf_margin")

    # Signal 6: Debt Level
    if equity_ratio is not None:
        result.debt_level_score = float(equity_ratio)   # already [0, 1]
        signals.append(result.debt_level_score)
        result.signals_used.append("debt_level")

    # Signal 7: Revenue Consistency
    if revenue_growth is not None:
        # –10% → 0.0, 0% → 0.25, +30% → 1.0
        result.rev_consistency_score = _clamp_map(revenue_growth, -0.10, 0.30)
        signals.append(result.rev_consistency_score)
        result.signals_used.append("rev_consistency")

    # Signal 8: Gross Margin (pricing power + product economics)
    # Low-margin businesses rarely have durable moats.
    # 20% → 0.0, 70% → 1.0
    if gross_margin is not None:
        gross_margin_score = _clamp_map(gross_margin, 0.20, 0.70)
        signals.append(gross_margin_score)
        result.signals_used.append("gross_margin")

    if len(signals) < 2:
        return result

    quant = sum(signals) / len(signals) * 100.0
    result.quant_moat_score = round(quant, 2)

    result.ai_moat_score = ai_moat_score
    if ai_moat_score is not None:
        from ai_research._base import effective_score as _eff_score
        # confidence < 0.30 → clamp to neutral 50 (no signal); otherwise fade continuously
        eff_ai = _eff_score(ai_moat_score, ai_confidence)
        # One-way dampening: AI can only reduce moat, never boost it.
        # A stock must earn its moat through quantitative signals alone.
        eff_ai = min(50.0, eff_ai)
        result.blended_moat_score = round(0.70 * quant + 0.30 * eff_ai, 2)
    else:
        result.blended_moat_score = round(quant, 2)

    return result
