"""
Business Reality Audit — Gate 14.

Computes BusinessHealthScore (0-100) for the Top 20 composite-ranked stocks.
This is a VALIDATION LAYER only — it may penalise and restrict Top-5 eligibility
but never changes factor weights or forces stocks in/out by identity.

Components
──────────
  Growth Health          25%   revenue/EPS/FCF CAGR trends
  Balance Sheet Health   25%   equity ratio, debt trend, ROIC coverage proxy
  Capital Allocation     20%   SBC, share count trend, debt paydown
  Business Durability    20%   sector/industry quality (pricing power, disruption)
  Market Confirmation    10%   analyst revision direction, momentum

Top-5 eligibility gate: BusinessHealthScore >= 60.
Severe Decline override: rev CAGR 5yr < 0 AND fcf CAGR 5yr < 0 AND rising debt → score capped at 55.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BusinessHealthResult:
    ticker: str
    score: float            # 0-100 composite
    grade: str              # Exceptional / Strong / Good / Watch / High Risk

    # Component scores (0-100)
    growth_score:              float = 50.0
    balance_sheet_score:       float = 50.0
    capital_allocation_score:  float = 50.0
    durability_score:          float = 50.0
    market_confirmation_score: float = 50.0

    # Risk flags
    value_trap_risk:     str | None = None
    severe_decline_risk: bool       = False
    debt_risk:           str | None = None
    dilution_risk:       str | None = None
    durability_risk:     str | None = None

    # Top-5 eligibility
    eligible_for_top5:    bool      = True
    ineligibility_reason: str | None = None

    # Thesis narrative
    top_strengths:  list[str] = field(default_factory=list)
    top_risks:      list[str] = field(default_factory=list)
    thesis_breakers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score":                     round(self.score, 1),
            "grade":                     self.grade,
            "growth_score":              round(self.growth_score, 1),
            "balance_sheet_score":       round(self.balance_sheet_score, 1),
            "capital_allocation_score":  round(self.capital_allocation_score, 1),
            "durability_score":          round(self.durability_score, 1),
            "market_confirmation_score": round(self.market_confirmation_score, 1),
            "value_trap_risk":           self.value_trap_risk,
            "severe_decline_risk":       self.severe_decline_risk,
            "debt_risk":                 self.debt_risk,
            "dilution_risk":             self.dilution_risk,
            "durability_risk":           self.durability_risk,
            "eligible_for_top5":         self.eligible_for_top5,
            "ineligibility_reason":      self.ineligibility_reason,
            "top_strengths":             self.top_strengths,
            "top_risks":                 self.top_risks,
            "thesis_breakers":           self.thesis_breakers,
        }


# ---------------------------------------------------------------------------
# Internal scoring helpers
# ---------------------------------------------------------------------------

_CAGR_CAP = 5.0  # quality.py caps CAGR at max(-0.90, min(5.0, cagr))
                  # A value of exactly 5.0 means the cap was hit — treat as missing.


def _score_cagr(cagr: float | None) -> float:
    """Map a growth rate / CAGR to 0-100. None or capped (5.0) → neutral 50."""
    if cagr is None or cagr >= _CAGR_CAP:   # 5.0 = hit cap = data unreliable
        return 50.0
    if cagr >= 0.20: return 100.0
    if cagr >= 0.15: return  90.0
    if cagr >= 0.10: return  80.0
    if cagr >= 0.05: return  65.0
    if cagr >= 0.00: return  50.0
    if cagr >= -0.05: return 30.0
    if cagr >= -0.10: return 15.0
    return 5.0


def _compute_growth_score(qr, rev_growth_1yr: float | None) -> float:
    """Growth Health (25%).  Rewards consistent positive revenue/EPS/FCF CAGR."""
    scores = []

    # 1-year revenue growth
    scores.append(_score_cagr(rev_growth_1yr))

    # 5-year CAGRs
    scores.append(_score_cagr(getattr(qr, "revenue_cagr_5yr", None)))
    scores.append(_score_cagr(getattr(qr, "eps_cagr_5yr", None)))
    scores.append(_score_cagr(getattr(qr, "fcf_cagr_5yr", None)))

    available = [s for s in scores if s != 50.0]  # 50 = neutral-missing
    if not available:
        return 50.0
    # Weight available signals; missing ones count 50
    return sum(scores) / len(scores)


def _compute_balance_sheet_score(qr) -> float:
    """Balance Sheet Health (25%).  Rewards low leverage and declining debt."""
    scores = []

    eq = getattr(qr, "equity_ratio", None)
    if eq is not None:
        if   eq >= 0.60: scores.append(95.0)
        elif eq >= 0.40: scores.append(80.0)
        elif eq >= 0.25: scores.append(65.0)
        elif eq >= 0.10: scores.append(45.0)
        else:            scores.append(15.0)

    # debt_trend: 1.0 = declining (good), 0.0 = rising fast (bad)
    dt = getattr(qr, "debt_trend", None)
    if dt is not None:
        scores.append(dt * 100.0)

    # ROIC as interest-coverage proxy
    roic = getattr(qr, "roic", None)
    if roic is not None:
        if   roic >= 0.20: scores.append(90.0)
        elif roic >= 0.15: scores.append(80.0)
        elif roic >= 0.10: scores.append(65.0)
        elif roic >= 0.05: scores.append(50.0)
        elif roic >= 0.00: scores.append(30.0)
        else:              scores.append(10.0)

    return sum(scores) / len(scores) if scores else 50.0


def _compute_capital_allocation_score(qr, car, sbc_fcf_ratio: float | None) -> float:
    """Capital Allocation Health (20%).  SBC penalties, share reduction rewards."""
    # Start from share count trend (0=rising fast, 1.0=declining) if available
    sct = getattr(qr, "share_count_trend", None)
    score = (sct * 100.0) if sct is not None else 60.0

    # Buyback rate (positive = buying back shares)
    if car is not None:
        br = getattr(car, "buyback_rate", None)
        if br is not None:
            if   br >= 0.05: score += 10.0
            elif br >= 0.02: score +=  5.0

        # Debt paydown (positive = reducing debt)
        dpr = getattr(car, "debt_paydown_rate", None)
        if dpr is not None and dpr > 0.03:
            score += 5.0

    # SBC tiers — spec: >20% = penalty, >30% = bigger, >40% = severe
    if sbc_fcf_ratio is not None:
        if   sbc_fcf_ratio > 0.40: score -= 20.0
        elif sbc_fcf_ratio > 0.30: score -= 12.0
        elif sbc_fcf_ratio > 0.20: score -=  6.0

    return max(0.0, min(100.0, score))


# Sector / industry keywords → durability tier
_HIGH_DURABILITY_KEYWORDS = {
    "software", "information technology", "payments", "fintech",
    "medical device", "medical technology", "healthcare technology",
    "health care technology", "semiconductor", "data processing",
    "healthcare equipment", "diagnostics", "biotechnology", "biopharmaceutical",
    "pharmaceutical", "application software", "systems software",
    "health care equipment", "life science",
}

_LOW_DURABILITY_KEYWORDS = {
    "energy", "oil & gas", "oil, gas", "coal", "mining", "metals & mining",
    "basic materials", "chemicals", "fertilizer", "steel", "aluminum", "copper",
    "wireless telecommunication", "diversified telecommunication", "tobacco",
}

# Medium-low: secular structural decline but not commodity. Score 42.
# Linear TV / cable / broadcast face cord-cutting; print / publishing face digital shift.
_MEDIUM_LOW_DURABILITY_KEYWORDS = {
    "media", "broadcasting", "cable television", "television", "publishing",
}


def _compute_durability_score(sector: str, industry: str = "") -> tuple[float, str | None]:
    """Business Durability (20%).  Returns (score 0-100, risk_text | None)."""
    combined = (sector + " " + industry).lower()

    for kw in _HIGH_DURABILITY_KEYWORDS:
        if kw in combined:
            return 83.0, None

    for kw in _LOW_DURABILITY_KEYWORDS:
        if kw in combined:
            return 33.0, (
                f"{sector} faces structural commodity/disruption risk — "
                "pricing power is limited and revenues can be cyclical."
            )

    # Linear media / broadcast: cord-cutting erodes affiliate fees and ad revenue
    for kw in _MEDIUM_LOW_DURABILITY_KEYWORDS:
        if kw in combined:
            return 42.0, (
                f"{sector} faces secular cord-cutting and linear TV decline. "
                "Cable affiliate fees and linear advertising face structural headwinds."
            )

    return 62.0, None   # medium durability


def _compute_market_confirmation_score(rr, mr) -> float:
    """Market Confirmation (10%).  Analyst revisions + price momentum."""
    scores = []

    if rr is not None:
        nu = getattr(rr, "net_upgrades_90d", None)
        if nu is not None:
            if   nu >=  5: scores.append(90.0)
            elif nu >=  2: scores.append(75.0)
            elif nu >=  0: scores.append(60.0)
            elif nu >= -2: scores.append(40.0)
            else:          scores.append(15.0)

        e30 = getattr(rr, "eps_30d_change", None)
        if e30 is not None:
            if   e30 >=  0.03: scores.append(85.0)
            elif e30 >=  0.01: scores.append(70.0)
            elif e30 >= -0.01: scores.append(55.0)
            elif e30 >= -0.03: scores.append(35.0)
            else:              scores.append(15.0)

    if mr is not None:
        r6 = getattr(mr, "ret_6m", None)
        if r6 is not None:
            if   r6 >=  0.15: scores.append(90.0)
            elif r6 >=  0.05: scores.append(70.0)
            elif r6 >= -0.05: scores.append(55.0)
            elif r6 >= -0.15: scores.append(35.0)
            else:             scores.append(15.0)

    return sum(scores) / len(scores) if scores else 50.0


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def compute_business_health(
    ticker:             str,
    quality_raw,                    # QualityRaw | None
    capital_alloc_raw,              # CapitalAllocationRaw | None
    revisions_raw,                  # RevisionRaw | None
    momentum_raw,                   # MomentumRaw | None
    sbc_fcf_ratio:      float | None,
    sector:             str,
    industry:           str         = "",
    revenue_growth_1yr: float | None = None,
    value_trap_warning: str | None  = None,
) -> BusinessHealthResult:
    """
    Compute the BusinessHealthScore for a single ticker.

    All inputs are raw objects already computed during --refresh-data.
    No new API calls are made.
    """
    qr  = quality_raw
    car = capital_alloc_raw
    rr  = revisions_raw
    mr  = momentum_raw

    # ── Components ──────────────────────────────────────────────────────────
    growth = _compute_growth_score(qr, revenue_growth_1yr) if qr else 50.0
    bs     = _compute_balance_sheet_score(qr)              if qr else 50.0
    ca     = _compute_capital_allocation_score(qr, car, sbc_fcf_ratio)
    dur, dur_risk = _compute_durability_score(sector, industry)
    mc     = _compute_market_confirmation_score(rr, mr)

    raw = (
        growth * 0.25 +
        bs     * 0.25 +
        ca     * 0.20 +
        dur    * 0.20 +
        mc     * 0.10
    )

    # ── Severe Decline override (historical: rev+FCF CAGR both negative) ─────
    severe = False
    if qr:
        rev5 = getattr(qr, "revenue_cagr_5yr", None)
        fcf5 = getattr(qr, "fcf_cagr_5yr",     None)
        dt   = getattr(qr, "debt_trend",        None)  # 0=rising, 1=declining
        if (rev5 is not None and rev5 < 0.0 and
                fcf5 is not None and fcf5 < 0.0 and
                dt   is not None and dt  < 0.3):
            severe = True
            raw = min(raw, 55.0)

    # ── Commodity Cycle Peak Value Trap ──────────────────────────────────────
    # Cyclical/commodity sector (dur ≤ 40) + analysts actively cutting estimates
    # (eps_30d_change < -2%) + stock is very cheap (valuation would be high) =
    # market is pricing in earnings decline the trailing data doesn't show yet.
    # Cap BHS at 57 to block from Top 5.
    commodity_peak = False
    if rr is not None and dur <= 40.0:
        eps_trend = getattr(rr, "eps_30d_change", None)
        if eps_trend is not None and eps_trend < -0.02:
            commodity_peak = True
            raw = min(raw, 57.0)

    score = round(max(0.0, min(100.0, raw)), 1)

    # ── Grade ────────────────────────────────────────────────────────────────
    if   score >= 90: grade = "Exceptional"
    elif score >= 80: grade = "Strong"
    elif score >= 70: grade = "Good"
    elif score >= 60: grade = "Watch"
    else:             grade = "High Risk"

    # ── Top-5 eligibility ────────────────────────────────────────────────────
    eligible      = score >= 60.0
    if not eligible:
        if commodity_peak:
            inelig_reason = (
                f"Commodity Cycle Peak: analysts cutting EPS estimates in {sector} "
                f"(cyclical sector). BHS capped at {score:.0f} — trailing growth overstates trend."
            )
        else:
            inelig_reason = f"BusinessHealthScore {score:.0f}/100 is below the 60-point minimum."
    else:
        inelig_reason = None

    # ── Risk flags ───────────────────────────────────────────────────────────
    debt_risk = None
    eq = getattr(qr, "equity_ratio", None) if qr else None
    if eq is not None and eq < 0.15:
        debt_risk = f"High leverage — equity ratio only {eq*100:.0f}%."

    dilution_risk = None
    if sbc_fcf_ratio is not None and sbc_fcf_ratio > 0.20:
        tier = "extreme" if sbc_fcf_ratio > 0.40 else ("high" if sbc_fcf_ratio > 0.30 else "elevated")
        dilution_risk = f"SBC is {sbc_fcf_ratio*100:.0f}% of FCF ({tier}) — reported FCF overstates owner earnings."

    # ── Narrative ────────────────────────────────────────────────────────────
    strengths: list[str] = []
    risks:     list[str] = []

    if growth >= 75:
        strengths.append(f"Strong multi-year revenue/earnings/FCF growth (growth score {growth:.0f}/100)")
    elif growth >= 60:
        strengths.append("Positive but moderate revenue growth")

    if bs >= 75:
        strengths.append(f"Conservative balance sheet — low leverage, declining debt")
    if ca >= 75:
        strengths.append("Shareholder-friendly capital allocation — buybacks or low SBC")
    if dur >= 75:
        strengths.append(f"High-durability business model in {sector}")
    if mc >= 75:
        strengths.append("Analyst upgrades and positive earnings revisions")

    if growth < 50:
        risks.append("Weak or declining revenue, earnings, or FCF trends")
    if bs < 50:
        risks.append("High or rising leverage — balance sheet deteriorating")
    if ca < 50:
        risks.append("Poor capital allocation — dilution or increasing debt")
    if dur < 50:
        risks.append(f"Low-durability sector ({sector}) — commodity/cyclical risk")
    if mc < 45:
        risks.append("Persistent analyst downgrades or negative earnings revisions")
    if value_trap_warning:
        risks.append("Value trap: stock is cheap because the business is declining")
    if severe:
        risks.append("Structural decline confirmed: revenue AND FCF shrinking over 5 years")
    if commodity_peak:
        risks.append(
            f"Commodity cycle peak risk: analysts cutting EPS estimates in a low-durability "
            f"({sector}) sector. Trailing growth likely overstates structural trend."
        )

    # ── Thesis Breakers ──────────────────────────────────────────────────────
    breakers: list[str] = ["BusinessHealthScore falls below 60"]

    if qr:
        rev5 = getattr(qr, "revenue_cagr_5yr", None)
        if rev5 is None or rev5 >= 0.0:
            breakers.append("Revenue CAGR turns negative")
        dt = getattr(qr, "debt_trend", None)
        if dt is None or dt >= 0.3:
            breakers.append("Net Debt/EBITDA exceeds 4×")

    if rr is not None:
        breakers.append("Analyst revisions turn sharply negative (net downgrades > 5 in 90 days)")

    return BusinessHealthResult(
        ticker                   = ticker,
        score                    = score,
        grade                    = grade,
        growth_score             = round(growth, 1),
        balance_sheet_score      = round(bs,     1),
        capital_allocation_score = round(ca,     1),
        durability_score         = round(dur,    1),
        market_confirmation_score= round(mc,     1),
        value_trap_risk          = value_trap_warning,
        severe_decline_risk      = severe,
        debt_risk                = debt_risk,
        dilution_risk            = dilution_risk,
        durability_risk          = dur_risk,
        eligible_for_top5        = eligible,
        ineligibility_reason     = inelig_reason,
        top_strengths            = strengths,
        top_risks                = risks,
        thesis_breakers          = breakers,
    )


def compute_business_health_batch(
    top20_tickers:       list[str],
    quality_raw_map:     dict,
    capital_alloc_map:   dict,
    revisions_raw_map:   dict,
    momentum_raw_map:    dict,
    sbc_map:             dict,
    sector_map:          dict,
    industry_map:        dict,
    revenue_growth_map:  dict,
    value_trap_warnings: dict,
) -> dict[str, BusinessHealthResult]:
    """
    Compute BusinessHealthResult for every ticker in top20_tickers.
    Returns {ticker: BusinessHealthResult}.
    """
    results: dict[str, BusinessHealthResult] = {}
    for ticker in top20_tickers:
        sbc_data      = sbc_map.get(ticker) or {}
        sbc_fcf_ratio = sbc_data.get("sbc_fcf_ratio")
        results[ticker] = compute_business_health(
            ticker             = ticker,
            quality_raw        = quality_raw_map.get(ticker),
            capital_alloc_raw  = capital_alloc_map.get(ticker),
            revisions_raw      = revisions_raw_map.get(ticker),
            momentum_raw       = momentum_raw_map.get(ticker),
            sbc_fcf_ratio      = sbc_fcf_ratio,
            sector             = sector_map.get(ticker, ""),
            industry           = industry_map.get(ticker, ""),
            revenue_growth_1yr = revenue_growth_map.get(ticker),
            value_trap_warning = value_trap_warnings.get(ticker),
        )
    return results
