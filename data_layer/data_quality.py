"""
Data quality validation.

Validates that provider responses have the minimum fields needed before
passing data downstream.  Returns a typed result so callers always check.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class QualityResult:
    passed: bool
    missing_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed


# Minimum fields required from each endpoint type
_REQUIRED: dict[str, list[str]] = {
    "profile": ["symbol", "companyName", "sector", "industry", "mktCap", "price"],
    "quote": ["symbol", "price"],
    "income_statement": ["revenue", "netIncome"],
    "balance_sheet": ["totalAssets", "totalDebt"],
    "cash_flow": ["freeCashFlow"],
    "key_metrics": ["roe", "debtToEquity"],
    "analyst_estimates": ["estimatedEpsAvg"],
}


def check(data: Any, endpoint: str) -> QualityResult:
    """
    Validate *data* for *endpoint*.

    *data* may be a dict (single record) or list (take first element).
    Returns QualityResult — always check .passed before using the data.
    """
    required = _REQUIRED.get(endpoint)
    if required is None:
        return QualityResult(passed=True, warnings=[f"No quality rules defined for '{endpoint}'"])

    if not data:
        return QualityResult(passed=False, missing_fields=["<empty response>"])

    record = data[0] if isinstance(data, list) else data

    if not isinstance(record, dict):
        return QualityResult(passed=False, missing_fields=["<not a dict>"])

    missing = [f for f in required if record.get(f) is None]

    if missing:
        log.debug("Quality check failed for %s — missing: %s", endpoint, missing)
        return QualityResult(passed=False, missing_fields=missing)

    return QualityResult(passed=True)


def check_shariah_fields(balance_sheet: dict, income_stmt: dict) -> QualityResult:
    """
    Verify that the fields needed for Shariah ratio checks are present.
    Missing = conservative exclude.
    """
    bs_required = ["totalAssets", "totalDebt", "shortTermInvestments"]
    is_required = ["revenue", "interestIncome"]

    missing = []
    for f in bs_required:
        if balance_sheet.get(f) is None:
            missing.append(f"balance_sheet.{f}")
    for f in is_required:
        if income_stmt.get(f) is None:
            missing.append(f"income_stmt.{f}")

    return QualityResult(passed=len(missing) == 0, missing_fields=missing)
