"""
Quality factor v2 — 5-year persistence model.

Philosophy: single-year quality scores are dominated by boom/bust cycles.
Institutional investors care about PERSISTENCE — can the company sustain
high returns over a full business cycle? This model rewards durability.

Multi-year signals (primary, 60% weight):
  1. 5-Year Revenue CAGR          sustained top-line growth
  2. 5-Year EPS CAGR              earnings power compounding
  3. 5-Year FCF CAGR              cash generation compounding
  4. 5-Year Average ROIC          capital efficiency over cycle
  5. ROIC Stability               low variance = durable moat
  6. FCF Margin Stability         predictable cash flows
  7. Share Count Trend            dilution destroys value
  8. Debt Trend                   rising debt = rising risk

Single-year signals (current period, 40% weight):
  9.  ROIC (current)              current capital efficiency
  10. Operating Margin (current)  current pricing power
  11. Net Margin (current)        bottom-line profitability
  12. FCF Margin (current)        current cash conversion
  13. Equity Ratio (current)      current leverage safety
  14. Revenue Growth (1yr)        most recent growth signal

Protections:
  - All growth winsorized at ±500%.
  - Growth excluded when prior period is negative.
  - Prior must be ≥5% of current absolute value.
  - Contribution cap 25% per signal (prevents any one signal dominating).
  - Fewer than 3 signals → raw_score = None → neutral 50 in composite.

Missing multi-year data → falls back gracefully to single-year only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

_DEFAULT_TAX_RATE    = 0.21
_GROWTH_CAP          = 5.0    # ±500%
_BASE_PCT_MIN        = 0.05
_MAX_CONTRIBUTION    = 0.25


@dataclass
class GrowthSignal:
    raw:   float | None
    value: float | None
    flag:  str  # "ok" | "capped" | "prior_negative" | "prior_too_small" | "missing"


@dataclass
class QualityRaw:
    ticker: str

    # Single-year fundamentals
    roic:             float | None = None
    operating_margin: float | None = None
    net_margin:       float | None = None
    fcf_margin:       float | None = None
    equity_ratio:     float | None = None
    revenue_growth:   float | None = None
    earnings_growth:  float | None = None
    fcf_growth:       float | None = None

    # Multi-year (5yr) metrics
    revenue_cagr_5yr:     float | None = None
    eps_cagr_5yr:         float | None = None
    fcf_cagr_5yr:         float | None = None
    roic_avg_5yr:         float | None = None
    roic_stability:       float | None = None  # 1 - CoV (higher = more stable)
    fcf_margin_stability: float | None = None  # 1 - CoV of FCF margin
    share_count_trend:    float | None = None  # 1.0 if declining, 0.0 if rising fast
    debt_trend:           float | None = None  # 1.0 if declining, 0.0 if rising fast

    # Growth signal audit trail
    revenue_growth_sig:  GrowthSignal | None = None
    earnings_growth_sig: GrowthSignal | None = None
    fcf_growth_sig:      GrowthSignal | None = None

    raw_score:        float | None = None
    signals_used:     list[str] = field(default_factory=list)
    contributions:    dict[str, float] = field(default_factory=dict)
    years_of_data:    int = 1   # how many years of historical data used


def _safe_growth(current, prior) -> GrowthSignal:
    if current is None or prior is None or prior == 0:
        return GrowthSignal(raw=None, value=None, flag="missing")
    if prior < 0:
        return GrowthSignal(raw=(current - prior)/abs(prior), value=None, flag="prior_negative")
    if current != 0 and abs(prior) < _BASE_PCT_MIN * abs(current):
        return GrowthSignal(raw=(current - prior)/abs(prior), value=None, flag="prior_too_small")
    raw = (current - prior) / abs(prior)
    if abs(raw) > _GROWTH_CAP:
        return GrowthSignal(raw=raw, value=max(-_GROWTH_CAP, min(_GROWTH_CAP, raw)), flag="capped")
    return GrowthSignal(raw=raw, value=raw, flag="ok")


def _cagr(start: float | None, end: float | None, years: int) -> float | None:
    """Compute CAGR. Returns None if inputs invalid or result is implausible."""
    if start is None or end is None or years <= 0:
        return None
    if start <= 0 or end <= 0:
        return None
    cagr = (end / start) ** (1.0 / years) - 1.0
    return max(-0.90, min(5.0, cagr))  # cap at ±90%/+500%


def _coeff_of_variation(values: list[float]) -> float | None:
    """Standard deviation / mean. Returns None if mean≈0 or fewer than 2 values."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    if abs(mean) < 1e-9:
        return None
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return (var ** 0.5) / abs(mean)


def _apply_contribution_cap(signal_values: dict, max_pct: float = _MAX_CONTRIBUTION) -> dict:
    result = dict(signal_values)
    for _ in range(20):
        total_abs = sum(abs(v) for v in result.values())
        if total_abs == 0:
            break
        cap = total_abs * max_pct
        new = {k: max(-cap, min(cap, v)) for k, v in result.items()}
        if all(abs(new[k] - result[k]) < 1e-12 for k in result):
            break
        result = new
    return result


def compute_quality(
    ticker: str,
    # Current period
    revenue:          float | None = None,
    net_income:       float | None = None,
    free_cash_flow:   float | None = None,
    operating_income: float | None = None,
    total_equity:     float | None = None,
    total_debt:       float | None = None,
    cash:             float | None = None,
    # Prior period (1 year ago)
    revenue_prior:        float | None = None,
    net_income_prior:     float | None = None,
    free_cash_flow_prior: float | None = None,
    effective_tax_rate:   float | None = None,
    # Historical series (oldest first, including current as last element)
    # List of 5 values covering 5 years; fewer is fine — we use what's available
    revenue_series:    list[float | None] | None = None,  # [yr-4, yr-3, yr-2, yr-1, current]
    eps_series:        list[float | None] | None = None,
    fcf_series:        list[float | None] | None = None,
    roic_series:       list[float | None] | None = None,
    fcf_margin_series: list[float | None] | None = None,
    shares_series:     list[float | None] | None = None,
    total_debt_series: list[float | None] | None = None,
) -> QualityRaw:
    result = QualityRaw(ticker=ticker)
    candidate: dict[str, float] = {}

    # ── Single-year fundamentals ──────────────────────────────────────────
    if operating_income is not None and total_equity is not None and total_debt is not None:
        cash_v = cash or 0.0
        equity_s = max(0.0, total_equity)
        debt_s   = max(0.0, total_debt)
        ic = equity_s + debt_s - cash_v
        if ic > 0:
            tax = effective_tax_rate if effective_tax_rate is not None else _DEFAULT_TAX_RATE
            result.roic = operating_income * (1 - tax) / ic

    if operating_income is not None and revenue:
        result.operating_margin = operating_income / revenue
    if net_income is not None and revenue:
        result.net_margin = net_income / revenue
    if free_cash_flow is not None and revenue:
        result.fcf_margin = free_cash_flow / revenue
    if total_equity is not None and total_debt is not None:
        eq_s = max(0.0, total_equity); dt_s = max(0.0, total_debt)
        denom = eq_s + dt_s
        if denom > 0:
            result.equity_ratio = eq_s / denom

    rev_sig = _safe_growth(revenue, revenue_prior)
    ni_sig  = _safe_growth(net_income, net_income_prior)
    fcf_sig = _safe_growth(free_cash_flow, free_cash_flow_prior)
    result.revenue_growth_sig  = rev_sig
    result.earnings_growth_sig = ni_sig
    result.fcf_growth_sig      = fcf_sig
    result.revenue_growth  = rev_sig.value
    result.earnings_growth = ni_sig.value
    result.fcf_growth      = fcf_sig.value

    # Single-year signals → 40% weight bucket
    sy_signals: dict[str, float] = {}
    for name, val in [
        ("roic",             result.roic),
        ("operating_margin", result.operating_margin),
        ("net_margin",       result.net_margin),
        ("fcf_margin",       result.fcf_margin),
        ("equity_ratio",     result.equity_ratio),
        ("revenue_growth",   result.revenue_growth),
        ("earnings_growth",  result.earnings_growth),
        ("fcf_growth",       result.fcf_growth),
    ]:
        if val is not None:
            sy_signals[name] = val

    # ── Multi-year signals (5-year persistence) ───────────────────────────
    my_signals: dict[str, float] = {}

    def _clean_series(s: list | None) -> list[float]:
        if not s:
            return []
        return [v for v in s if v is not None]

    rev_s  = _clean_series(revenue_series)
    eps_s  = _clean_series(eps_series)
    fcf_s  = _clean_series(fcf_series)
    roic_s = _clean_series(roic_series)
    fcfm_s = _clean_series(fcf_margin_series)
    shr_s  = _clean_series(shares_series)
    dbt_s  = _clean_series(total_debt_series)

    years_available = max(len(rev_s), len(eps_s), len(fcf_s), len(roic_s), 0)
    result.years_of_data = max(1, years_available)

    if len(rev_s) >= 2:
        n_yrs = len(rev_s) - 1
        cagr = _cagr(rev_s[0], rev_s[-1], n_yrs)
        if cagr is not None:
            result.revenue_cagr_5yr = cagr
            my_signals["revenue_cagr"] = cagr

    if len(eps_s) >= 2:
        # EPS CAGR — only valid if both endpoints positive
        n_yrs = len(eps_s) - 1
        cagr = _cagr(eps_s[0], eps_s[-1], n_yrs)
        if cagr is not None:
            result.eps_cagr_5yr = cagr
            my_signals["eps_cagr"] = cagr

    if len(fcf_s) >= 2:
        n_yrs = len(fcf_s) - 1
        cagr = _cagr(fcf_s[0], fcf_s[-1], n_yrs)
        if cagr is not None:
            result.fcf_cagr_5yr = cagr
            my_signals["fcf_cagr"] = cagr

    if len(roic_s) >= 2:
        result.roic_avg_5yr = sum(roic_s) / len(roic_s)
        my_signals["roic_avg"] = result.roic_avg_5yr
        cov = _coeff_of_variation(roic_s)
        if cov is not None:
            result.roic_stability = max(0.0, 1.0 - cov)
            my_signals["roic_stability"] = result.roic_stability

    if len(fcfm_s) >= 3:
        cov = _coeff_of_variation(fcfm_s)
        if cov is not None:
            result.fcf_margin_stability = max(0.0, 1.0 - cov)
            my_signals["fcf_margin_stability"] = result.fcf_margin_stability

    if len(shr_s) >= 2:
        # Share count trend: declining shares = buybacks = 1.0
        # Rising shares = dilution = 0.0
        shr_chg = (shr_s[-1] - shr_s[0]) / abs(shr_s[0]) if shr_s[0] != 0 else 0
        # [-10%, 0%] → [1.0, 0.5]; [0%, +10%] → [0.5, 0.0]
        shr_trend = max(0.0, min(1.0, 0.5 - shr_chg * 5.0))
        result.share_count_trend = shr_trend
        my_signals["share_count_trend"] = shr_trend

    if len(dbt_s) >= 2 and dbt_s[0] is not None and dbt_s[0] > 0:
        dbt_chg = (dbt_s[-1] - dbt_s[0]) / abs(dbt_s[0])
        dbt_trend = max(0.0, min(1.0, 0.5 - dbt_chg * 2.5))
        result.debt_trend = dbt_trend
        my_signals["debt_trend"] = dbt_trend

    # ── Blend: 60% multi-year + 40% single-year ───────────────────────────
    if not sy_signals and not my_signals:
        return result
    if len(sy_signals) + len(my_signals) < 3:
        return result

    result.signals_used = list(sy_signals.keys()) + list(my_signals.keys())

    # Apply contribution cap within each bucket
    sy_capped = _apply_contribution_cap(sy_signals) if sy_signals else {}
    my_capped = _apply_contribution_cap(my_signals) if my_signals else {}

    sy_avg = sum(sy_capped.values()) / len(sy_capped) if sy_capped else None
    my_avg = sum(my_capped.values()) / len(my_capped) if my_capped else None

    if my_avg is not None and sy_avg is not None:
        raw = 0.60 * my_avg + 0.40 * sy_avg
    elif my_avg is not None:
        raw = my_avg
    else:
        raw = sy_avg  # type: ignore[assignment]

    result.raw_score = raw

    all_capped = {**{f"sy_{k}": v for k, v in sy_capped.items()},
                  **{f"my_{k}": v for k, v in my_capped.items()}}
    total_abs = sum(abs(v) for v in all_capped.values()) or 1.0
    result.contributions = {k: round(abs(v) / total_abs * 100, 1) for k, v in all_capped.items()}

    return result


def compute_quality_batch(stocks: list[dict]) -> dict[str, "QualityRaw"]:
    results: dict[str, QualityRaw] = {}
    for s in stocks:
        ticker = s.get("ticker") or s.get("symbol", "")
        results[ticker] = compute_quality(
            ticker            = ticker,
            revenue           = s.get("revenue"),
            net_income        = s.get("netIncome"),
            free_cash_flow    = s.get("freeCashFlow"),
            operating_income  = s.get("operatingIncome"),
            total_equity      = s.get("totalEquity"),
            total_debt        = s.get("totalDebt"),
            cash              = s.get("cashAndEquivalents"),
            revenue_prior     = s.get("revenue_prior"),
            net_income_prior  = s.get("netIncome_prior"),
            free_cash_flow_prior = s.get("freeCashFlow_prior"),
            effective_tax_rate = s.get("effectiveTaxRate"),
        )
    return results
