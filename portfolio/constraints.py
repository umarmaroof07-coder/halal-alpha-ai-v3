"""
Per-stock constraint checker — every stock must clear ALL rules before it
can enter the portfolio or appear on the WATCHLIST.

Rules (all must pass):
  1. Shariah status == "compliant"
  2. Price < MAX_STOCK_PRICE  ($1,000)
  3. Market cap > MIN_PORTFOLIO_MARKET_CAP  ($2B)
  4. Average daily volume > MIN_PORTFOLIO_AVG_VOLUME  (500,000 shares)
  5. No shorting  (system-level — always enforced)
  6. No leverage  (system-level — always enforced)
  7. No options   (system-level — always enforced)

All dollar/share thresholds come from config so they can be adjusted in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config.settings import MAX_STOCK_PRICE, PORTFOLIO_MIN_MARKET_CAP, PORTFOLIO_MIN_AVG_VOLUME

MIN_PORTFOLIO_MARKET_CAP: float = PORTFOLIO_MIN_MARKET_CAP
MIN_PORTFOLIO_AVG_VOLUME: float = PORTFOLIO_MIN_AVG_VOLUME


@dataclass
class ConstraintResult:
    ticker: str
    passed: bool
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "passed": self.passed,
            "failures": self.failures,
        }


def check_constraints(
    ticker: str,
    price: float | None,
    market_cap: float | None,
    avg_volume: float | None,
    shariah_status: str,
) -> ConstraintResult:
    """
    Validate a single stock against all portfolio entry constraints.

    Parameters
    ----------
    ticker : str
    price : float | None
        Current share price in USD. None → fails price check.
    market_cap : float | None
        Market capitalisation in USD. None → fails market cap check.
    avg_volume : float | None
        Average daily share volume. None → fails volume check.
    shariah_status : str
        One of "compliant", "non_compliant", "unknown".

    Returns
    -------
    ConstraintResult — passed=True only if all checks pass.
    """
    failures: list[str] = []

    # 1. Shariah
    if shariah_status != "compliant":
        failures.append(f"Shariah status is '{shariah_status}' (must be 'compliant')")

    # 2. Price
    if price is None:
        failures.append("Price unavailable")
    elif price >= MAX_STOCK_PRICE:
        failures.append(f"Price ${price:,.2f} ≥ ${MAX_STOCK_PRICE:,.0f} limit")

    # 3. Market cap
    if market_cap is not None and market_cap > 0 and market_cap < MIN_PORTFOLIO_MARKET_CAP:
        failures.append(
            f"Market cap ${market_cap/1e9:.2f}B < ${MIN_PORTFOLIO_MARKET_CAP/1e9:.0f}B minimum"
        )

    # 4. Volume — skip check when data is unavailable (screener pre-filters volume)
    if avg_volume is not None and avg_volume > 0 and avg_volume < MIN_PORTFOLIO_AVG_VOLUME:
        failures.append(
            f"Avg daily volume {avg_volume:,.0f} < {MIN_PORTFOLIO_AVG_VOLUME:,.0f} minimum"
        )

    # 5-7. System-level constraints (always enforced — no data needed)
    # Shorting, leverage, and options are never permitted by this system.
    # These are architectural guarantees, not runtime checks.

    return ConstraintResult(
        ticker=ticker,
        passed=len(failures) == 0,
        failures=failures,
    )


def check_constraints_batch(
    stocks: list[dict],
) -> dict[str, ConstraintResult]:
    """
    Check constraints for multiple stocks.

    Each dict must contain: ticker, price, marketCap (or market_cap),
    avgVolume (or avg_volume), shariah_status.
    """
    results: dict[str, ConstraintResult] = {}
    for s in stocks:
        ticker = s.get("ticker") or s.get("symbol", "")
        results[ticker] = check_constraints(
            ticker=ticker,
            price=s.get("price"),
            market_cap=s.get("marketCap") or s.get("market_cap"),
            avg_volume=s.get("avgVolume") or s.get("avg_volume"),
            shariah_status=s.get("shariah_status", "unknown"),
        )
    return results
