"""
Shariah filter — two-stage screen.

Stage 1 — Industry/activity screen (CANNOT be overridden)
  Any match on the haram keyword list = non_compliant, permanently.

Stage 2 — Financial ratio screen (CAN be overridden by manual CSV)
  Checks AAOIFI-based thresholds.
  Missing data → UNKNOWN (never assumed compliant).

Manual CSV (data/manual/shariah_overrides.csv):
  Columns: ticker, override_status, reason
  override_status: "compliant" | "non_compliant"
  Effect: may only change a ratio-stage result.
          A ticker that failed Stage 1 stays non_compliant regardless.

Final statuses:
  "compliant"     — passed both stages (or ratio override applied)
  "non_compliant" — failed Stage 1 (permanent) or failed Stage 2 (no override)
  "unknown"       — missing data prevented ratio check; not compliant
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from config.shariah_rules import (
    MAX_DEBT_TO_ASSETS,
    MAX_INTEREST_INCOME_RATIO,
    MAX_ACCOUNTS_RECEIVABLE_RATIO,
    MANUAL_EXCLUSIONS,
)
from screener.exclusion_list import is_haram_industry, is_spac

log = logging.getLogger(__name__)

Status = Literal["compliant", "non_compliant", "unknown"]

MANUAL_CSV_PATH = Path("data/manual/shariah_overrides.csv")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ShariahResult:
    ticker: str
    status: Status                = "unknown"
    industry_pass: bool | None    = None   # None = not checked (missing data)
    ratio_pass: bool | None       = None
    manual_override: bool         = False
    reasons: list[str]            = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.status == "compliant"


# ---------------------------------------------------------------------------
# Manual CSV loader
# ---------------------------------------------------------------------------

def _load_manual_overrides() -> dict[str, str]:
    """
    Return {ticker: override_status} from the manual CSV.
    Silently returns empty dict if file does not exist.
    """
    if not MANUAL_CSV_PATH.exists():
        return {}
    overrides: dict[str, str] = {}
    try:
        with MANUAL_CSV_PATH.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                ticker = (row.get("ticker") or "").strip().upper()
                status = (row.get("override_status") or "").strip().lower()
                if ticker and status in ("compliant", "non_compliant"):
                    overrides[ticker] = status
    except Exception as exc:
        log.warning("Could not load shariah_overrides.csv: %s", exc)
    return overrides


# ---------------------------------------------------------------------------
# Core filter
# ---------------------------------------------------------------------------

def screen_ticker(
    ticker: str,
    industry: str,
    sector: str,
    company_name: str,
    # Financial ratios — pass None if data is unavailable
    total_debt: float | None,
    total_assets: float | None,
    interest_income: float | None,
    total_revenue: float | None,
    accounts_receivable: float | None,
    # Manual override lookup (pass pre-loaded dict for batch efficiency)
    manual_overrides: dict[str, str] | None = None,
) -> ShariahResult:
    """
    Screen a single ticker through both Shariah stages.

    Parameters match the normalized field names returned by fmp_provider /
    yfinance_provider so callers do not need to remap.
    """
    result = ShariahResult(ticker=ticker)

    # ── Hardcoded manual exclusions (config/shariah_rules.py) ────────────
    if ticker.upper() in MANUAL_EXCLUSIONS:
        result.status = "non_compliant"
        result.industry_pass = False
        result.reasons.append("In manual exclusion list (config)")
        return result

    # ── Stage 1: Industry / activity screen ──────────────────────────────
    if not industry and not sector:
        result.status = "unknown"
        result.industry_pass = None
        result.reasons.append("Missing industry and sector data")
        return result

    if is_spac(company_name, industry):
        result.status = "non_compliant"
        result.industry_pass = False
        result.reasons.append("SPAC / blank-check company")
        return result

    if is_haram_industry(industry, sector):
        result.status = "non_compliant"
        result.industry_pass = False
        result.reasons.append(f"Haram industry/sector: '{industry}' / '{sector}'")
        return result

    result.industry_pass = True

    # ── Stage 2: Financial ratio screen ──────────────────────────────────
    ratio_reasons: list[str] = []
    data_missing = False

    # Debt-to-assets
    if total_debt is None or total_assets is None or total_assets == 0:
        ratio_reasons.append("Missing debt/assets data")
        data_missing = True
    elif (total_debt / total_assets) > MAX_DEBT_TO_ASSETS:
        ratio_reasons.append(
            f"Debt/assets {total_debt/total_assets:.2%} > {MAX_DEBT_TO_ASSETS:.0%}"
        )

    # Interest income ratio
    if interest_income is None or total_revenue is None or total_revenue == 0:
        ratio_reasons.append("Missing interest income / revenue data")
        data_missing = True
    elif interest_income > 0 and (interest_income / total_revenue) > MAX_INTEREST_INCOME_RATIO:
        ratio_reasons.append(
            f"Interest income {interest_income/total_revenue:.2%} > {MAX_INTEREST_INCOME_RATIO:.0%}"
        )

    # Accounts receivable ratio
    if accounts_receivable is None or total_assets is None or total_assets == 0:
        ratio_reasons.append("Missing accounts receivable / assets data")
        data_missing = True
    elif accounts_receivable > 0 and (accounts_receivable / total_assets) > MAX_ACCOUNTS_RECEIVABLE_RATIO:
        ratio_reasons.append(
            f"Receivables/assets {accounts_receivable/total_assets:.2%} > {MAX_ACCOUNTS_RECEIVABLE_RATIO:.0%}"
        )

    ratio_failed = len([r for r in ratio_reasons if "Missing" not in r]) > 0

    if data_missing and not ratio_failed:
        # Some data missing but no explicit failures — conservative: unknown
        result.status = "unknown"
        result.ratio_pass = None
        result.reasons.extend(ratio_reasons)
        return result

    if ratio_failed:
        result.ratio_pass = False
        result.reasons.extend([r for r in ratio_reasons if "Missing" not in r])

        # Check manual override (ratio failures only)
        overrides = manual_overrides if manual_overrides is not None else _load_manual_overrides()
        if overrides.get(ticker.upper()) == "compliant":
            result.status = "compliant"
            result.ratio_pass = True
            result.manual_override = True
            result.reasons.append("Manual CSV override: marked compliant")
        else:
            result.status = "non_compliant"
        return result

    # All ratio checks passed (ignoring missing-data warnings)
    result.ratio_pass = True
    result.status = "compliant"
    if ratio_reasons:
        result.reasons.extend(ratio_reasons)   # include data-missing warnings
    return result


# ---------------------------------------------------------------------------
# Batch screen
# ---------------------------------------------------------------------------

def screen_batch(stocks: list[dict], manual_overrides: dict[str, str] | None = None) -> list[ShariahResult]:
    """
    Screen a list of stock dicts.

    Each dict must contain at minimum:
      ticker, industry, sector, companyName,
      totalDebt, totalAssets, interestIncome, revenue, accountsReceivable

    Field names match the normalized output of fmp_provider / yfinance_provider.
    """
    if manual_overrides is None:
        manual_overrides = _load_manual_overrides()

    results = []
    for s in stocks:
        ticker = s.get("ticker") or s.get("symbol", "")
        try:
            r = screen_ticker(
                ticker           = ticker,
                industry         = s.get("industry", ""),
                sector           = s.get("sector", ""),
                company_name     = s.get("companyName", s.get("name", "")),
                total_debt       = s.get("totalDebt"),
                total_assets     = s.get("totalAssets"),
                interest_income  = s.get("interestIncome"),
                total_revenue    = s.get("revenue"),
                accounts_receivable = s.get("accountsReceivable"),
                manual_overrides = manual_overrides,
            )
        except Exception as exc:
            log.error("screen_ticker failed for %s: %s", ticker, exc)
            r = ShariahResult(
                ticker  = ticker,
                status  = "unknown",
                reasons = [f"Screening error: {exc}"],
            )
        results.append(r)
    return results
