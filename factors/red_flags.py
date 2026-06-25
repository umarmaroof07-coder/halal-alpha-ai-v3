"""
V5 Phase 6 — Red Flag Engine

Scores 0-100 where higher = fewer red flags (cleaner stock).
Only deductions — no score inflation from positive signals.

Checks:
  - Excessive SBC (stock-based compensation as % of FCF)
  - High financial leverage (equity ratio < thresholds)
  - Rapid revenue growth that may indicate M&A distortion
  - AI-detected red flags (from ai_detail)
  - Earnings distortion flags
  - Capital allocation warnings
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RedFlagResult:
    ticker: str
    red_flag_score: float        # 0-100; higher = fewer red flags
    flags: list[str] = field(default_factory=list)
    label: str = ""              # "Clean" / "Caution" / "Warning" / "Red Flag"

    def to_dict(self) -> dict:
        return {
            "red_flag_score": round(self.red_flag_score, 1),
            "flags":          self.flags,
            "label":          self.label,
        }


def compute_red_flags(
    ticker: str,
    sbc_fcf_ratio: float | None = None,
    equity_ratio: float | None = None,
    revenue_growth: float | None = None,
    ai_red_flags: list[str] | None = None,
    distortion_flags: list[str] | None = None,
    capital_allocation_warnings: list[str] | None = None,
    net_upgrades_90d: int | None = None,
    risk_level: str | None = None,
) -> RedFlagResult:
    """
    Compute red flag score for a single ticker.

    All inputs are optional; missing data is treated conservatively.
    """
    score = 100.0
    flags: list[str] = []

    # SBC checks
    if sbc_fcf_ratio is not None:
        if sbc_fcf_ratio > 0.50:
            score -= 20.0
            flags.append(f"SBC {sbc_fcf_ratio*100:.0f}% of FCF (extreme)")
        elif sbc_fcf_ratio > 0.20:
            score -= 10.0
            flags.append(f"SBC {sbc_fcf_ratio*100:.0f}% of FCF (elevated)")

    # Leverage
    if equity_ratio is not None:
        if equity_ratio < 0.20:
            score -= 15.0
            flags.append(f"Equity ratio {equity_ratio*100:.0f}% (high leverage)")
        elif equity_ratio < 0.40:
            score -= 7.0
            flags.append(f"Equity ratio {equity_ratio*100:.0f}% (elevated leverage)")

    # Revenue growth distortion
    if revenue_growth is not None and revenue_growth > 0.50:
        score -= 8.0
        flags.append(f"Revenue growth {revenue_growth*100:.0f}% (possible M&A distortion)")

    # AI red flags
    if ai_red_flags:
        n = len(ai_red_flags)
        if n >= 6:
            deduction = 20.0
        elif n >= 4:
            deduction = 12.0
        elif n >= 2:
            deduction = 6.0
        else:
            deduction = 2.0
        score -= deduction
        flags.append(f"AI flagged {n} red flag(s): {', '.join(ai_red_flags[:3])}")

    # Earnings distortion flags
    if distortion_flags:
        n = len(distortion_flags)
        score -= min(15.0, n * 5.0)
        flags.append(f"Earnings distortion: {'; '.join(distortion_flags[:2])}")

    # Capital allocation warnings
    if capital_allocation_warnings:
        n = len(capital_allocation_warnings)
        score -= min(10.0, n * 4.0)
        flags.append(f"Cap alloc warnings: {'; '.join(capital_allocation_warnings[:2])}")

    # Analyst downgrade pressure
    if net_upgrades_90d is not None and net_upgrades_90d < -3:
        score -= 8.0
        flags.append(f"Net analyst downgrades 90d: {net_upgrades_90d}")

    # Risk engine
    if risk_level == "High":
        score -= 10.0
        flags.append("Risk engine: HIGH")

    score = max(0.0, min(100.0, score))

    if score >= 85:
        label = "Clean"
    elif score >= 70:
        label = "Caution"
    elif score >= 50:
        label = "Warning"
    else:
        label = "Red Flag"

    return RedFlagResult(
        ticker         = ticker,
        red_flag_score = score,
        flags          = flags,
        label          = label,
    )
