"""
V6 Phase 3 — Model Confidence 2.0

Produces a single overall_confidence_score (0-100) from five inputs:

  data_quality       30%  — how complete and fresh the data is
  ai_confidence      20%  — AI sub-analyzer confidence score
  factor_stability   20%  — how stable scores have been over time
  analyst_coverage   15%  — number of sell-side analysts covering the stock
  stress_reliability 15%  — stress test data coverage across crisis periods

Label thresholds: 90+=Very High, 80+=High, 65+=Medium, 50+=Low, <50=Very Low
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfidenceResult:
    ticker: str
    overall_confidence_score: float   # 0-100
    label: str                        # Very High / High / Medium / Low / Very Low
    data_quality_input: float
    analyst_coverage_input: float
    history_depth_input: float        # kept for backwards compatibility
    ai_confidence_input: float
    factor_stability_input: float
    accounting_quality_input: float   # kept for backwards compatibility
    stress_reliability_input: float = 50.0   # V6 new component

    def to_dict(self) -> dict:
        return {
            "overall_confidence_score":  round(self.overall_confidence_score, 1),
            "label":                     self.label,
            "data_quality_input":        round(self.data_quality_input, 1),
            "analyst_coverage_input":    round(self.analyst_coverage_input, 1),
            "ai_confidence_input":       round(self.ai_confidence_input, 1),
            "factor_stability_input":    round(self.factor_stability_input, 1),
            "stress_reliability_input":  round(self.stress_reliability_input, 1),
            # legacy fields kept for test compatibility
            "history_depth_input":       round(self.history_depth_input, 1),
            "accounting_quality_input":  round(self.accounting_quality_input, 1),
        }


def _analyst_coverage_score(num_analysts: int | None) -> float:
    """Map analyst count to 0-100."""
    if num_analysts is None:
        return 30.0
    if num_analysts >= 20:
        return 100.0
    if num_analysts >= 10:
        return 80.0
    if num_analysts >= 5:
        return 60.0
    if num_analysts >= 2:
        return 40.0
    return 20.0


def _history_depth_score(years_of_data: int | None) -> float:
    """Map years of historical data to 0-100."""
    if years_of_data is None:
        return 30.0
    if years_of_data >= 5:
        return 100.0
    if years_of_data >= 3:
        return 70.0
    if years_of_data >= 2:
        return 50.0
    return 25.0


def _accounting_quality_score(distortion_flags: list[str] | None, sbc_fcf_ratio: float | None) -> float:
    """Higher score = cleaner accounting."""
    score = 100.0
    if distortion_flags:
        score -= min(50.0, len(distortion_flags) * 15.0)
    if sbc_fcf_ratio is not None and sbc_fcf_ratio > 0.20:
        score -= 10.0
    if sbc_fcf_ratio is not None and sbc_fcf_ratio > 0.50:
        score -= 15.0
    return max(0.0, score)


def _stress_reliability_score(n_crisis_periods_with_data: int | None) -> float:
    """
    Map number of crisis periods with real data to 0-100.
    0 periods → 30 (neutral/unknown); 1→50; 2→75; 3+ → 100.
    """
    if n_crisis_periods_with_data is None:
        return 30.0
    if n_crisis_periods_with_data >= 4:
        return 100.0
    if n_crisis_periods_with_data >= 3:
        return 85.0
    if n_crisis_periods_with_data >= 2:
        return 65.0
    if n_crisis_periods_with_data >= 1:
        return 45.0
    return 30.0


def compute_model_confidence(
    ticker: str,
    data_quality_score: float,           # 0-100 from DataQualityScore.overall
    num_analysts: int | None = None,
    years_of_data: int | None = None,    # kept for API compatibility
    ai_confidence: float | None = None,  # 0-1 from ai_detail["confidence"]
    factor_stability_score: float | None = None,  # 0-100 from FactorStability
    distortion_flags: list[str] | None = None,    # kept for API compatibility
    sbc_fcf_ratio: float | None = None,           # kept for API compatibility
    n_stress_periods: int | None = None,          # V6: crisis periods with real data
) -> ModelConfidenceResult:
    """
    V6 Model Confidence 2.0 weighting:
      30% data_quality
      20% ai_confidence
      20% factor_stability
      15% analyst_coverage
      15% stress_reliability
    """

    # Convert ai confidence 0-1 → 0-100
    ai_score = (ai_confidence or 0.0) * 100.0
    ai_score = max(0.0, min(100.0, ai_score))

    # Factor stability defaults to neutral if no history yet
    stab_score = factor_stability_score if factor_stability_score is not None else 50.0

    cov_score    = _analyst_coverage_score(num_analysts)
    hist_score   = _history_depth_score(years_of_data)        # legacy, not in V6 formula
    acct_score   = _accounting_quality_score(distortion_flags, sbc_fcf_ratio)  # legacy
    stress_score = _stress_reliability_score(n_stress_periods)

    # V6 formula
    overall = (
        0.30 * data_quality_score +
        0.20 * ai_score           +
        0.20 * stab_score         +
        0.15 * cov_score          +
        0.15 * stress_score
    )
    overall = max(0.0, min(100.0, overall))

    if overall >= 90:
        label = "Very High"
    elif overall >= 80:
        label = "High"
    elif overall >= 65:
        label = "Medium"
    elif overall >= 50:
        label = "Low"
    else:
        label = "Very Low"

    return ModelConfidenceResult(
        ticker                   = ticker,
        overall_confidence_score = overall,
        label                    = label,
        data_quality_input       = data_quality_score,
        analyst_coverage_input   = cov_score,
        history_depth_input      = hist_score,
        ai_confidence_input      = ai_score,
        factor_stability_input   = stab_score,
        accounting_quality_input = acct_score,
        stress_reliability_input = stress_score,
    )
