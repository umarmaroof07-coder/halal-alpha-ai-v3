from config.settings import ACCOUNT_SIZE, CONVICTION_WEIGHTS, CONVICTION_DOLLARS

# ---------------------------------------------------------------------------
# Factor weights — must sum to 1.0
# ---------------------------------------------------------------------------
# V5 design principles (balanced model — valuation, earnings quality, capital
#   allocation, and risk discipline kept strong):
#   1. AI Research is NOT a direct composite factor.
#      AI influences only the Moat factor (30% blend × 10% weight = 3% indirect).
#      With average confidence ~0.25, actual AI impact ~0.75% of composite.
#   2. Revisions at 20%: institutional EPS trend data (7d/30d/90d) is
#      the strongest near-term signal from the sell side.
#   3. Quality at 20%: 5-yr persistence model is higher quality but
#      the universe is large enough that cross-sectional spread is well captured.
#   4. Capital Allocation at 10%: long-run compounding is determined
#      by how management deploys capital — a primary driver of excess returns.
#   5. Momentum at 15%: still useful for timing but lower conviction.
#   6. Valuation at 10%: keeps discipline against overpaying.
#   7. Earnings Quality at 10%: accounting-quality guard.
#   8. Risk Adjustment at 5%: safety guard for leverage and cyclicality.

FACTOR_WEIGHTS: dict[str, float] = {
    "quality":            0.20,   # 5-yr: CAGR, ROIC avg, stability (60% multiyear + 40% single-yr)
    "momentum":           0.15,   # price momentum (6m, 12-1m, MA crossovers)
    "earnings_revisions": 0.20,   # EPS trend 7/30/90d, breadth, PT momentum, 90d upgrades
    "valuation":          0.10,   # P/E, FCF yield, EV multiples
    "earnings_quality":   0.10,   # FCF conversion, accruals, SBC penalty, distortion flags
    "moat":               0.10,   # quant moat (8 signals) + 30% AI (conf-scaled)
    "capital_allocation": 0.10,   # buybacks, debt paydown, ROIC improvement, FCF/share growth
    "risk_adjustment":    0.05,   # leverage, FCF safety, cyclicality (higher = safer)
}

# ---------------------------------------------------------------------------
# AI Research sub-weights — must sum to 1.0
# ---------------------------------------------------------------------------

AI_RESEARCH_WEIGHTS: dict[str, float] = {
    "transcript":   0.30,
    "sec_filing":   0.30,
    "moat":         0.25,
    "management":   0.15,
}

# ---------------------------------------------------------------------------
# Ranking: industry → sector → market (3-tier blending)
# ---------------------------------------------------------------------------

INDUSTRY_BLEND_ALPHA: float = 0.50   # 50% global
SECTOR_BLEND_ALPHA:   float = 0.30   # 30% sector-relative
INDUSTRY_BLEND_BETA:  float = 0.20   # 20% industry-relative
SECTOR_MIN_MEMBERS:   int   = 5      # minimum sector size for sector z-score
INDUSTRY_MIN_MEMBERS: int   = 3      # minimum industry size for industry z-score

# ---------------------------------------------------------------------------
# Asserts — fire at import time so misconfiguration is caught immediately
# ---------------------------------------------------------------------------

assert round(sum(FACTOR_WEIGHTS.values()), 10) == 1.0, (
    f"FACTOR_WEIGHTS must sum to 1.0, got {sum(FACTOR_WEIGHTS.values())}"
)

assert round(sum(AI_RESEARCH_WEIGHTS.values()), 10) == 1.0, (
    f"AI_RESEARCH_WEIGHTS must sum to 1.0, got {sum(AI_RESEARCH_WEIGHTS.values())}"
)

assert round(sum(CONVICTION_WEIGHTS), 10) == 1.0, (
    f"CONVICTION_WEIGHTS must sum to 1.0, got {sum(CONVICTION_WEIGHTS)}"
)

assert round(sum(CONVICTION_DOLLARS), 2) == ACCOUNT_SIZE, (
    f"CONVICTION_DOLLARS must sum to {ACCOUNT_SIZE}, got {sum(CONVICTION_DOLLARS)}"
)
