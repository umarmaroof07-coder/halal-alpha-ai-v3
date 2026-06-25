"""
Universe builder — assembles the Shariah-compliant stock universe.

⚠  SURVIVORSHIP BIAS WARNING
   The default universe is built from currently listed S&P 500 / Russell 1000
   constituents fetched at runtime.  These are TODAY'S members, not the
   historical membership at any past date.  Using this universe for historical
   backtests will introduce survivorship bias: companies that went bankrupt,
   were acquired, or were delisted before today are absent.

   For backtests:
     - Pass a pre-built, date-stamped ticker list via `tickers` parameter.
     - Do NOT call build_universe() without a ticker list for historical dates.
     - Never assume current constituents represent historical membership.

Validation report structure:
    {
        "as_of_date": "YYYY-MM-DD",
        "total_screened": int,
        "compliant": [{"ticker": ..., "reasons": [...]}, ...],
        "non_compliant": [{"ticker": ..., "reasons": [...]}, ...],
        "unknown": [{"ticker": ..., "reasons": [...]}, ...],
        "survivorship_bias_warning": str,
    }
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from data_layer import live_data_provider as ldp
from screener.shariah_filter import ShariahResult, screen_batch, _load_manual_overrides

log = logging.getLogger(__name__)

SURVIVORSHIP_BIAS_WARNING = (
    "Universe built from current exchange listings. "
    "Historical backtests using this universe will suffer survivorship bias: "
    "companies delisted, acquired, or bankrupt before today are excluded. "
    "For historical accuracy, supply a date-stamped ticker list."
)


def _fetch_stock_data(ticker: str) -> dict[str, Any]:
    """Fetch profile + latest financials for one ticker. Returns a flat dict."""
    data: dict[str, Any] = {"ticker": ticker, "symbol": ticker}
    try:
        profile = ldp.get_profile(ticker)
        data.update({
            "companyName": profile.get("companyName", ""),
            "sector":      profile.get("sector", ""),
            "industry":    profile.get("industry", ""),
            "mktCap":      profile.get("mktCap"),
            "price":       profile.get("price"),
            "exchange":    profile.get("exchangeShortName", profile.get("exchange", "")),
        })
    except Exception as exc:
        log.warning("Could not fetch profile for %s: %s", ticker, exc)
        return data  # return partial — screener will mark unknown

    try:
        bs = ldp.get_balance_sheet(ticker, limit=1)
        if bs:
            data["totalAssets"] = bs[0].get("totalAssets")
            data["totalDebt"]   = bs[0].get("totalDebt")
            data["accountsReceivable"] = bs[0].get("accountsReceivable")
    except Exception as exc:
        log.warning("Could not fetch balance sheet for %s: %s", ticker, exc)

    try:
        inc = ldp.get_income_statement(ticker, limit=1)
        if inc:
            data["revenue"]        = inc[0].get("revenue")
            data["interestIncome"] = inc[0].get("interestIncome")
    except Exception as exc:
        log.warning("Could not fetch income statement for %s: %s", ticker, exc)

    return data


def build_universe(
    tickers: list[str] | None = None,
    as_of_date: str | None = None,
    is_backtest: bool = False,
) -> dict:
    """
    Build and screen the Shariah-compliant universe.

    Parameters
    ----------
    tickers : list[str] | None
        Explicit ticker list. If None, fetches live from FMP stock screener.
        For backtests, ALWAYS supply an explicit date-stamped list.
    as_of_date : str | None
        ISO date string "YYYY-MM-DD". Defaults to today. Used in the report only
        — does NOT gate data fetching (supply pre-fetched data for true backtests).
    is_backtest : bool
        If True, emits an extra survivorship bias warning in the report and logs.

    Returns
    -------
    dict with keys: as_of_date, total_screened, compliant, non_compliant,
                    unknown, survivorship_bias_warning
    """
    report_date = as_of_date or str(date.today())

    if is_backtest:
        log.warning(SURVIVORSHIP_BIAS_WARNING)
        if tickers is None:
            raise ValueError(
                "is_backtest=True but no ticker list supplied. "
                "Never use live universe for historical backtests."
            )

    if tickers is None:
        log.info("Fetching live universe from FMP stock screener...")
        try:
            raw = ldp.get_stock_screener() if hasattr(ldp, "get_stock_screener") else []
        except Exception as exc:
            log.error("Stock screener failed: %s", exc)
            raw = []
        tickers = [s.get("symbol", "") for s in raw if s.get("symbol")]

    log.info("Building universe for %d tickers as of %s", len(tickers), report_date)

    manual_overrides = _load_manual_overrides()

    # Fetch data and screen each ticker
    stock_data_list: list[dict] = []
    for ticker in tickers:
        stock_data_list.append(_fetch_stock_data(ticker))

    results: list[ShariahResult] = screen_batch(stock_data_list, manual_overrides)

    # Build report
    compliant     = []
    non_compliant = []
    unknown       = []

    for r in results:
        entry = {
            "ticker":          r.ticker,
            "industry_pass":   r.industry_pass,
            "ratio_pass":      r.ratio_pass,
            "manual_override": r.manual_override,
            "reasons":         r.reasons,
        }
        if r.status == "compliant":
            compliant.append(entry)
        elif r.status == "non_compliant":
            non_compliant.append(entry)
        else:
            unknown.append(entry)

    report = {
        "as_of_date":    report_date,
        "total_screened": len(results),
        "compliant":     compliant,
        "non_compliant": non_compliant,
        "unknown":       unknown,
        "survivorship_bias_warning": SURVIVORSHIP_BIAS_WARNING,
    }

    log.info(
        "Universe complete: %d compliant, %d non-compliant, %d unknown",
        len(compliant), len(non_compliant), len(unknown),
    )
    return report


def get_compliant_tickers(universe_report: dict) -> list[str]:
    """Extract just the compliant ticker symbols from a universe report."""
    return [e["ticker"] for e in universe_report.get("compliant", [])]
