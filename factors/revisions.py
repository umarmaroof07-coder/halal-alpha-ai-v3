"""
Analyst Revision Engine v2 — institutional-grade earnings revision factor.

Seven signals, equal-weighted with confidence gating:

  Signal 1 — EPS Revision Direction (current year)
    Measures: pct change in consensus EPS estimate over 7d, 30d, 90d.
    Blended: 50% × 30d + 30% × 90d + 20% × 7d.
    Interpretation: positive = analysts lifting estimates = positive momentum.

  Signal 2 — EPS Revision Acceleration
    Measures: whether the 7d rate of change is faster than 30d/90d average.
    Positive acceleration = analysts becoming more bullish recently.

  Signal 3 — Revision Breadth
    Measures: (upgrades − downgrades) / total across 7d and 30d windows.
    High breadth = broad consensus shift, not just one analyst.

  Signal 4 — Revenue Estimate Trend
    Measures: analyst revenue estimate growth vs. prior period.
    Rising revenue estimates = top-line expansion signal.

  Signal 5 — Price Target Momentum
    Measures: median analyst PT vs. current price (upside %).
    Upside > 20% = significant analyst conviction on value.

  Signal 6 — Upgrade/Downgrade Momentum (90-day)
    Measures: net upgrades (upgrade_count − downgrade_count) over 90d.
    Net positive = analysts changing views to more bullish.

  Signal 7 — Analyst Coverage Quality
    Measures: number of analysts covering; penalizes thin coverage.
    <3 analysts = low confidence; 10+ = high confidence.

Confidence gating:
  0 signals → neutral 50.
  1–2 signals → confidence "low", contribution downweighted.
  3–4 signals → confidence "medium".
  5–7 signals → confidence "high".

Output:
  raw_score: 0–1 (cross-sectionally z-scored in composite.py)
  revision_confidence: "high" | "medium" | "low" | "none"
  revision_reason: human-readable breakdown
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

_SMALL_FLOAT = 1e-9


@dataclass
class RevisionsRaw:
    ticker: str

    # Signal sub-scores (each 0–1 internally)
    eps_direction_score:     float | None = None
    eps_acceleration_score:  float | None = None
    breadth_score:           float | None = None
    revenue_trend_score:     float | None = None
    pt_upside_score:         float | None = None
    upgrade_momentum_score:  float | None = None
    coverage_score:          float | None = None

    # Underlying values (for audit/display)
    eps_7d_change:    float | None = None   # % change in EPS est last 7d
    eps_30d_change:   float | None = None   # % change in EPS est last 30d
    eps_90d_change:   float | None = None   # % change in EPS est last 90d
    rev_breadth_30d:  float | None = None   # (up−down)/total last 30d
    price_target_upside: float | None = None
    net_upgrades_90d:    int | None = None
    total_analysts:      int = 0

    raw_score:           float | None = None
    signals_used:        list[str] = field(default_factory=list)
    revisions_confidence: str = "none"  # "high" | "medium" | "low" | "none"
    revisions_reason:    str = ""

    # Legacy compatibility fields (used by existing serializers)
    weighted_score:   float | None = None
    buy_ratio:        float | None = None
    consensus_inv:    float | None = None


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _pct_change(current: float | None, prior: float | None) -> float | None:
    """Safe percentage change, returns None if either input is missing or prior≈0."""
    if current is None or prior is None:
        return None
    if abs(prior) < _SMALL_FLOAT:
        return None
    return (current - prior) / abs(prior)


def _map_pct_to_score(pct: float, low: float = -0.10, high: float = 0.10) -> float:
    """Map a percentage change [low, high] → [0, 1], clipped at boundaries."""
    if pct >= high:
        return 1.0
    if pct <= low:
        return 0.0
    return (pct - low) / (high - low)


def compute_revisions(
    ticker: str,
    # Legacy inputs (recommendation counts)
    strong_buy:  int | None = None,
    buy:         int | None = None,
    hold:        int | None = None,
    sell:        int | None = None,
    strong_sell: int | None = None,
    consensus_rating: float | None = None,
    price_target_upside: float | None = None,
    # New V2 inputs — EPS trend
    eps_current:    float | None = None,   # current consensus EPS estimate
    eps_7d_ago:     float | None = None,   # estimate 7 days ago
    eps_30d_ago:    float | None = None,   # estimate 30 days ago
    eps_90d_ago:    float | None = None,   # estimate 90 days ago
    # Next year EPS revision
    eps_ny_current: float | None = None,
    eps_ny_30d_ago: float | None = None,
    # Revision breadth
    rev_up_7d:  int | None = None,    # analysts who raised estimates last 7d
    rev_up_30d: int | None = None,    # analysts who raised estimates last 30d
    rev_dn_30d: int | None = None,    # analysts who cut estimates last 30d
    rev_dn_7d:  int | None = None,    # analysts who cut estimates last 7d
    # Revenue estimates
    rev_est_growth: float | None = None,   # revenue estimate growth (yoy)
    # Upgrade/downgrade momentum
    upgrades_90d:   int | None = None,
    downgrades_90d: int | None = None,
    # Number of analysts covering
    num_analysts:   int | None = None,
    # Price target (absolute)
    pt_mean:    float | None = None,
    pt_median:  float | None = None,
    current_price: float | None = None,
) -> RevisionsRaw:
    result = RevisionsRaw(ticker=ticker)
    signals: list[float] = []
    reasons: list[str] = []

    # ── Signal 1: EPS Revision Direction ─────────────────────────────────
    chg_7d  = _pct_change(eps_current, eps_7d_ago)
    chg_30d = _pct_change(eps_current, eps_30d_ago)
    chg_90d = _pct_change(eps_current, eps_90d_ago)

    result.eps_7d_change  = chg_7d
    result.eps_30d_change = chg_30d
    result.eps_90d_change = chg_90d

    eps_dir_inputs = [(chg_30d, 0.50), (chg_90d, 0.30), (chg_7d, 0.20)]
    eps_dir_score_num = eps_dir_score_den = 0.0
    for chg, w in eps_dir_inputs:
        if chg is not None:
            eps_dir_score_num += w * _map_pct_to_score(chg, -0.10, 0.10)
            eps_dir_score_den += w
    if eps_dir_score_den > 0:
        s = eps_dir_score_num / eps_dir_score_den
        result.eps_direction_score = s
        signals.append(s)
        result.signals_used.append("eps_direction")
        if chg_30d is not None:
            reasons.append(f"EPS est: {chg_30d*100:+.1f}% (30d), {(chg_90d or 0)*100:+.1f}% (90d)")

    # ── Signal 2: EPS Revision Acceleration ──────────────────────────────
    if chg_7d is not None and chg_30d is not None:
        # Is the 7-day rate faster/slower than the 30-day trend?
        baseline = chg_90d if chg_90d is not None else chg_30d
        accel = chg_7d - (baseline / (90 / 7) if chg_90d is not None else chg_30d / (30 / 7))
        # Map acceleration [-5%, +5%] → [0, 1]
        s = _map_pct_to_score(accel, -0.05, 0.05)
        result.eps_acceleration_score = s
        signals.append(s)
        result.signals_used.append("eps_acceleration")
        if accel > 0.002:
            reasons.append("Revision momentum accelerating")
        elif accel < -0.002:
            reasons.append("Revision momentum decelerating")

    # ── Signal 3: Revision Breadth ────────────────────────────────────────
    if rev_up_30d is not None or rev_dn_30d is not None:
        up   = (rev_up_30d  or 0)
        dn   = (rev_dn_30d  or 0)
        total = up + dn
        if total > 0:
            breadth = (up - dn) / total  # [-1, +1]
            result.rev_breadth_30d = breadth
            s = (breadth + 1.0) / 2.0   # → [0, 1]
            result.breadth_score = s
            signals.append(s)
            result.signals_used.append("breadth")
            pct_up = up / total * 100
            reasons.append(f"Revision breadth: {pct_up:.0f}% raised vs {100-pct_up:.0f}% cut (30d)")

    # ── Signal 4: Revenue Estimate Trend ─────────────────────────────────
    if rev_est_growth is not None:
        s = _map_pct_to_score(rev_est_growth, -0.05, 0.20)
        result.revenue_trend_score = s
        signals.append(s)
        result.signals_used.append("revenue_trend")
        reasons.append(f"Rev est growth: {rev_est_growth*100:+.1f}%")

    # ── Signal 5: Price Target Upside ────────────────────────────────────
    pt_upside = price_target_upside
    if pt_upside is None and pt_median is not None and current_price and current_price > 0:
        pt_upside = (pt_median - current_price) / current_price
    if pt_upside is None and pt_mean is not None and current_price and current_price > 0:
        pt_upside = (pt_mean - current_price) / current_price

    if pt_upside is not None:
        clipped = max(-0.50, min(0.80, pt_upside))
        result.price_target_upside = clipped
        s = _map_pct_to_score(clipped, -0.20, 0.30)
        result.pt_upside_score = s
        signals.append(s)
        result.signals_used.append("pt_upside")
        reasons.append(f"Analyst PT: {pt_upside*100:+.0f}% upside")

    # ── Signal 6: Upgrade/Downgrade Momentum (90d) ───────────────────────
    if upgrades_90d is not None or downgrades_90d is not None:
        up90 = upgrades_90d   or 0
        dn90 = downgrades_90d or 0
        result.net_upgrades_90d = up90 - dn90
        total90 = up90 + dn90
        if total90 > 0:
            net_ratio = (up90 - dn90) / total90  # [-1, +1]
            s = (net_ratio + 1.0) / 2.0
        else:
            s = 0.50
        result.upgrade_momentum_score = s
        signals.append(s)
        result.signals_used.append("upgrade_momentum")
        if up90 > 0 or dn90 > 0:
            reasons.append(f"90d: {up90}↑ upgrades, {dn90}↓ downgrades")

    # ── Signal 7: Analyst Coverage Quality ───────────────────────────────
    total_a = num_analysts
    if total_a is None:
        # Infer from recommendation counts
        sb = strong_buy or 0; b = buy or 0; h = hold or 0
        s_ = sell or 0; ss = strong_sell or 0
        rec_total = sb + b + h + s_ + ss
        if rec_total > 0:
            total_a = rec_total

    if total_a is not None and total_a > 0:
        result.total_analysts = total_a
        # <3 = 0.3, 5 = 0.6, 10+ = 1.0
        if total_a >= 10:
            s = 1.0
        elif total_a >= 5:
            s = 0.6 + (total_a - 5) / 5.0 * 0.4
        elif total_a >= 3:
            s = 0.3 + (total_a - 3) / 2.0 * 0.3
        else:
            s = 0.2
        result.coverage_score = s
        signals.append(s)
        result.signals_used.append("coverage")

    # ── Legacy: recommendation counts as fallback if no EPS trend data ───
    if "eps_direction" not in result.signals_used and "breadth" not in result.signals_used:
        sb = strong_buy or 0; b = buy or 0; h = hold or 0
        s_ = sell or 0; ss = strong_sell or 0
        total_rec = sb + b + h + s_ + ss
        if total_rec > 0:
            weighted = (sb * 2 + b * 1 - s_ * 1 - ss * 2) / total_rec / 2.0
            buy_r = (sb + b) / total_rec
            result.weighted_score = weighted
            result.buy_ratio      = buy_r
            # Map [-1,+1] → [0,1]
            s_w = (weighted + 1.0) / 2.0
            s_b = buy_r
            signals.extend([s_w, s_b])
            result.signals_used.extend(["weighted_rec", "buy_ratio"])
            pct_buy = buy_r * 100
            reasons.append(f"{pct_buy:.0f}% buy/strong-buy ({total_rec} analysts)")

        if consensus_rating is not None and 1.0 <= consensus_rating <= 5.0:
            inv = (5.0 - consensus_rating) / 4.0
            result.consensus_inv = inv
            signals.append(inv)
            result.signals_used.append("consensus_inv")

    # ── Aggregate ─────────────────────────────────────────────────────────
    if not signals:
        result.revisions_confidence = "none"
        result.revisions_reason = "No analyst revision data"
        return result

    result.raw_score = sum(signals) / len(signals)

    n = len(signals)
    if n >= 5:
        result.revisions_confidence = "high"
    elif n >= 3:
        result.revisions_confidence = "medium"
    else:
        result.revisions_confidence = "low"

    result.revisions_reason = " | ".join(reasons) if reasons else "Analyst data available"
    return result


def compute_revisions_batch(recommendations: list[dict]) -> dict[str, RevisionsRaw]:
    """Legacy batch helper for backward compatibility."""
    results: dict[str, RevisionsRaw] = {}
    for r in recommendations:
        ticker = r.get("ticker") or r.get("symbol", "")
        results[ticker] = compute_revisions(
            ticker               = ticker,
            strong_buy           = r.get("strongBuy"),
            buy                  = r.get("buy"),
            hold                 = r.get("hold"),
            sell                 = r.get("sell"),
            strong_sell          = r.get("strongSell"),
            consensus_rating     = r.get("consensusRating"),
            price_target_upside  = r.get("priceTargetUpside"),
        )
    return results
