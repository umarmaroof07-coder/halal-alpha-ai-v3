"""
Value Trap Guard — caps the valuation score for cheap-but-declining cyclicals.

A stock looks cheap (high valuation score) but is cheap because its revenue
is shrinking and it operates in a sector with no pricing power. Without this
guard, the model ranks value traps highly because the valuation factor
dominates the composite at 10% weight.

Rule:
    IF revenue_growth < -5%
    AND valuation raw_score > VALUE_TRAP_VAL_THRESHOLD (mapped to ~85 scored)
    AND sector is cyclical
    THEN cap valuation raw_score at VALUE_TRAP_CAP

The raw_score lives on [0, 1]. The cap of 0.60 maps to roughly a scored
valuation of 60 after cross-sectional z-scoring (neutral-ish territory).
The threshold of 0.85 maps to the top ~15% of the valuation distribution.

Only the valuation raw_score is modified. All other factors are untouched.
The stock is NOT removed from the universe — the warning is surfaced instead.
"""

from __future__ import annotations

CYCLICAL_SECTORS = {
    "Energy",
    "Materials",
    "Basic Materials",
    "Metals & Mining",
    "Mining",
    "Chemicals",
    "Fertilizers & Agricultural Chemicals",
    "Oil & Gas",
    "Oil, Gas & Consumable Fuels",
}

REVENUE_GROWTH_THRESHOLD = -0.05   # < -5%
VAL_SCORED_THRESHOLD     = 85.0    # top ~15% after z-scoring (0-100 scale)
VAL_SCORED_CAP           = 40.0    # cap valuation to below-neutral (was 60 — tightened)
COMPOSITE_PENALTY        = 5.0     # additional composite deduction on top of valuation cap
WARNING_TEXT = (
    "Possible value trap: cheap valuation + declining revenue in cyclical sector. "
    "Valuation capped at 40 and composite penalised -5 pts."
)


def apply_value_trap_guard(
    ticker:          str,
    valuation_score: float | None,   # z-scored 0-100
    revenue_growth:  float | None,
    sector:          str,
) -> tuple[float | None, str | None]:
    """
    Check value-trap conditions on the z-scored valuation (0-100 scale).
    Returns (adjusted_score, warning_or_None).
    """
    if valuation_score is None:
        return valuation_score, None
    if revenue_growth is None:
        return valuation_score, None

    is_cyclical    = any(cs.lower() in sector.lower() for cs in CYCLICAL_SECTORS)
    revenue_shrink = revenue_growth < REVENUE_GROWTH_THRESHOLD
    very_cheap     = valuation_score > VAL_SCORED_THRESHOLD

    if is_cyclical and revenue_shrink and very_cheap:
        return VAL_SCORED_CAP, WARNING_TEXT

    return valuation_score, None


def apply_value_trap_guard_batch(
    tickers:        list[str],
    factor_scores:  list,            # list[FactorScores] — mutated in-place
    revenue_growth: dict[str, float | None],
    sector_map:     dict[str, str],
    factor_weights: dict[str, float],
) -> dict[str, str]:
    """
    Apply the guard to all FactorScores in-place, capping .valuation and
    recomputing .composite for affected tickers.
    Returns {ticker: warning_text} for any tickers where the guard fired.
    """
    warnings: dict[str, str] = {}
    for fs in factor_scores:
        new_score, warning = apply_value_trap_guard(
            ticker          = fs.ticker,
            valuation_score = fs.valuation,
            revenue_growth  = revenue_growth.get(fs.ticker),
            sector          = sector_map.get(fs.ticker, ""),
        )
        if warning:
            old_val       = fs.valuation
            fs.valuation  = new_score
            # Recompute composite: subtract old valuation contribution, add new
            val_weight     = factor_weights.get("valuation", 0.10)
            fs.composite  += (new_score - old_val) * val_weight
            # Additional flat penalty — accounts for debt risk + institutional selling
            fs.composite  -= COMPOSITE_PENALTY
            fs.composite   = round(fs.composite, 2)
            warnings[fs.ticker] = warning

    return warnings
