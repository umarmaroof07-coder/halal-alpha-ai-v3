"""
Live universe builder.

Data flow
---------
1. Fetch candidates: S&P 500 → Russell 1000 → FMP stock screener (broad fallback)
   If FMP is unreachable, load from data/manual/watchlist.csv.
2. De-duplicate by ticker symbol.
3. Pre-filter: exchange, type, name keywords, market cap, price, avg volume.
4. Fetch per-ticker fundamentals (balance sheet + income statement) for Shariah ratios.
5. Run Shariah screening: unknown = excluded.
6. Save three CSVs to data/cache/:
     universe_raw.csv          — all candidates after step 3 (pre-Shariah)
     universe_screened.csv     — Shariah-compliant only
     shariah_validation_report.csv — all tickers with full Shariah verdict

Saved CSVs are the sole source of truth for downstream modules.  Callers
(dashboard/state.py, main.py) load from CSVs; they never call this module
during normal operation.
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from config.settings import CACHE_DIR
from data_layer import fmp_provider
from data_layer.fmp_provider import ProviderError
from screener.shariah_filter import ShariahResult, screen_batch, _load_manual_overrides

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

UNIVERSE_RAW_CSV       = CACHE_DIR / "universe_raw.csv"
UNIVERSE_SCREENED_CSV  = CACHE_DIR / "universe_screened.csv"
SHARIAH_REPORT_CSV     = CACHE_DIR / "shariah_validation_report.csv"
WATCHLIST_CSV          = Path("data/manual/watchlist.csv")

# ---------------------------------------------------------------------------
# Pre-filter constants
# ---------------------------------------------------------------------------

EXCHANGE_ALLOWLIST = {"NYSE", "NASDAQ", "AMEX", "NYSE ARCA", "NYSE MKT", "BATS", "CBOE BZX"}

# Name-based exclusion: lower-case substrings that identify non-common-stock securities
_NAME_EXCLUSIONS: list[str] = [
    # SPACs
    "acquisition corp", "blank check", "special purpose acquisition",
    "merger corp", "holdings acquisition",
    # Preferred / depositary
    "preferred", " pfd", " pref ",
    "depositary share", "depositary receipt",
    # Derivatives / structured
    "warrant", " wt ", "rights ",
    # Pooled vehicles
    " etf", "exchange traded fund", "exchange-traded fund",
    "index fund", "money market",
    # Partnerships / trusts
    "limited partnership", " l.p.", " lp trust",
    "unit trust", "royalty trust",
    # REITs — caught by Shariah later but filter early for speed
    "real estate investment trust",
]

# type field from FMP stock list (lower-case) that are not common stock
_TYPE_EXCLUSIONS: set[str] = {"etf", "fund", "trust", "right", "warrant", "unit"}

# Universe filter thresholds — must stay in sync with config/settings.py
from config.settings import UNIVERSE_MIN_MARKET_CAP, UNIVERSE_MIN_AVG_VOLUME, MAX_STOCK_PRICE as _CFG_MAX_PRICE
_MIN_MARKET_CAP = UNIVERSE_MIN_MARKET_CAP   # $2B
_MAX_PRICE       = _CFG_MAX_PRICE           # $1,000
_MIN_AVG_VOLUME  = UNIVERSE_MIN_AVG_VOLUME  # 500k shares/day

# Inter-ticker delay when fetching fundamentals (seconds)
_FETCH_DELAY = 0.15


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RawStock:
    ticker: str
    name: str = ""
    exchange: str = ""
    sector: str = ""
    industry: str = ""
    mkt_cap: float | None = None
    price: float | None = None
    avg_volume: float | None = None
    stock_type: str = ""          # "stock", "etf", etc.
    source: str = ""              # "sp500", "russell1000", "screener", "watchlist"


@dataclass
class ScreenedStock:
    ticker: str
    name: str = ""
    exchange: str = ""
    sector: str = ""
    industry: str = ""
    mkt_cap: float | None = None
    price: float | None = None
    avg_volume: float | None = None
    shariah_status: str = "unknown"


@dataclass
class UniverseResult:
    as_of_date: str
    raw_count: int
    screened_count: int          # after pre-filter
    compliant_count: int
    non_compliant_count: int
    unknown_count: int
    compliant_tickers: list[str] = field(default_factory=list)
    sources_used: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Fetch candidates
# ---------------------------------------------------------------------------

def _fetch_sp500() -> list[RawStock]:
    log.info("Fetching S&P 500 constituents from FMP...")
    try:
        rows = fmp_provider.get_sp500_constituents()
        stocks = []
        for r in rows:
            sym = r.get("symbol", "").strip().upper()
            if sym:
                stocks.append(RawStock(
                    ticker=sym,
                    name=r.get("name", ""),
                    sector=r.get("sector", ""),
                    source="sp500",
                ))
        if stocks:
            log.info("  S&P 500 (FMP): %d symbols", len(stocks))
            return stocks
        raise ProviderError("Empty response")
    except ProviderError as exc:
        log.warning("S&P 500 FMP fetch failed (%s) — trying Wikipedia...", exc)
        return _fetch_sp500_wikipedia()


def _fetch_sp500_wikipedia() -> list[RawStock]:
    """
    Scrape S&P 500 constituents from Wikipedia.
    Used as fallback when FMP constituent endpoint is unavailable.
    Returns list of RawStock with ticker + name + sector only
    (price/mkt_cap filled later from FMP per-ticker profile).
    """
    import requests
    from bs4 import BeautifulSoup

    log.info("  Falling back to Wikipedia S&P 500 list...")
    try:
        headers = {"User-Agent": "HalalAlphaAI/1.0 (financial research tool)"}
        r = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers, timeout=20,
        )
        if r.status_code != 200:
            log.warning("Wikipedia returned HTTP %s", r.status_code)
            return []

        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table", {"id": "constituents"})
        if table is None:
            log.warning("Wikipedia S&P 500 table not found in page.")
            return []

        stocks = []
        for row in table.find_all("tr")[1:]:
            cells = row.find_all("td")
            if len(cells) < 4:
                continue
            ticker  = cells[0].text.strip().replace(".", "-")
            name    = cells[1].text.strip()
            sector  = cells[3].text.strip() if len(cells) > 3 else ""
            if ticker:
                stocks.append(RawStock(
                    ticker=ticker, name=name, sector=sector, source="sp500_wiki",
                ))
        log.info("  Wikipedia S&P 500: %d symbols", len(stocks))
        return stocks
    except Exception as exc:
        log.warning("Wikipedia S&P 500 fetch failed: %s", exc)
        return []


def _fetch_russell1000() -> list[RawStock]:
    log.info("Fetching Russell 1000 constituents from FMP...")
    try:
        rows = fmp_provider.get_russell1000_constituents()
        stocks = []
        for r in rows:
            sym = r.get("symbol", "").strip().upper()
            if sym:
                stocks.append(RawStock(
                    ticker=sym,
                    name=r.get("name", ""),
                    sector=r.get("sector", ""),
                    source="russell1000",
                ))
        log.info("  Russell 1000: %d symbols", len(stocks))
        return stocks
    except Exception as exc:
        log.warning("Russell 1000 fetch failed: %s", exc)
        return []


def _fetch_screener_broad() -> list[RawStock]:
    """Broad FMP stock screener: market cap > $2B, price < $1000, volume > 500k, US only."""
    log.info("Fetching broad FMP stock screener universe...")
    try:
        rows = fmp_provider.get_stock_screener(
            market_cap_more_than=_MIN_MARKET_CAP,
            price_lower_than=_MAX_PRICE,
            volume_more_than=_MIN_AVG_VOLUME,
            exchange="nasdaq,nyse,amex",
            country="US",
            is_etf=False,
            is_fund=False,
            limit=2000,
        )
        stocks = []
        for r in rows:
            sym = r.get("symbol", "").strip().upper()
            if sym:
                stocks.append(RawStock(
                    ticker=sym,
                    name=r.get("companyName", ""),
                    exchange=r.get("exchangeShortName", r.get("exchange", "")),
                    sector=r.get("sector", ""),
                    industry=r.get("industry", ""),
                    mkt_cap=_to_float(r.get("marketCap")),
                    price=_to_float(r.get("price")),
                    avg_volume=_to_float(r.get("volume")),
                    stock_type="" if not r.get("isEtf") else "etf",
                    source="screener",
                ))
        log.info("  Screener: %d symbols", len(stocks))
        return stocks
    except ProviderError as exc:
        log.warning("Broad screener fetch failed: %s", exc)
        return []


def _fetch_watchlist_csv(path: Path) -> list[RawStock]:
    """Load tickers from a one-ticker-per-line CSV (manual fallback)."""
    stocks = []
    if not path.exists():
        return stocks
    with path.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            sym = row[0].strip().upper()
            if sym and not sym.startswith("#"):
                # Handle optional columns: ticker,name,sector
                stocks.append(RawStock(
                    ticker=sym,
                    name=row[1].strip() if len(row) > 1 else "",
                    sector=row[2].strip() if len(row) > 2 else "",
                    source="watchlist",
                ))
    log.info("Watchlist CSV: %d symbols from %s", len(stocks), path)
    return stocks


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _dedup(stocks: list[RawStock]) -> list[RawStock]:
    """Keep first occurrence per ticker (S&P 500 wins over screener fallback)."""
    seen: set[str] = set()
    result: list[RawStock] = []
    for s in stocks:
        if s.ticker not in seen:
            seen.add(s.ticker)
            result.append(s)
    return result


# ---------------------------------------------------------------------------
# Pre-filter
# ---------------------------------------------------------------------------

def _name_contains_exclusion(name: str) -> bool:
    n = name.lower()
    return any(kw in n for kw in _NAME_EXCLUSIONS)


def _type_excluded(stock_type: str) -> bool:
    return stock_type.lower() in _TYPE_EXCLUSIONS


def _exchange_allowed(exchange: str) -> bool:
    return exchange.upper() in EXCHANGE_ALLOWLIST


def pre_filter(
    stocks: list[RawStock],
    min_market_cap: float = _MIN_MARKET_CAP,
    max_price: float = _MAX_PRICE,
    min_avg_volume: float = _MIN_AVG_VOLUME,
) -> tuple[list[RawStock], list[tuple[str, str]]]:
    """
    Apply pre-filters. Returns (passed, rejected) where rejected is a list of
    (ticker, reason) tuples.  Filters that need fundamentals (market cap / price /
    volume) are skipped when the field is None — those are resolved after profile fetch.
    """
    passed: list[RawStock] = []
    rejected: list[tuple[str, str]] = []

    for s in stocks:
        # --- Name-based ---
        if _name_contains_exclusion(s.name):
            rejected.append((s.ticker, f"name exclusion: {s.name[:60]}"))
            continue

        # --- Type-based ---
        if _type_excluded(s.stock_type):
            rejected.append((s.ticker, f"type excluded: {s.stock_type}"))
            continue

        # --- Exchange (skip if unknown — resolved after profile fetch) ---
        if s.exchange and not _exchange_allowed(s.exchange):
            rejected.append((s.ticker, f"exchange excluded: {s.exchange}"))
            continue

        # --- Market cap, price, volume — only filter when data is present ---
        if s.mkt_cap is not None and s.mkt_cap < min_market_cap:
            rejected.append((s.ticker, f"market cap too small: ${s.mkt_cap/1e9:.1f}B"))
            continue
        if s.price is not None and s.price >= max_price:
            rejected.append((s.ticker, f"price too high: ${s.price:.0f}"))
            continue
        if s.avg_volume is not None and s.avg_volume < min_avg_volume:
            rejected.append((s.ticker, f"avg volume too low: {s.avg_volume:,.0f}"))
            continue

        passed.append(s)

    return passed, rejected


# ---------------------------------------------------------------------------
# Fetch fundamentals for Shariah screening
# ---------------------------------------------------------------------------

def _fetch_fundamentals(ticker: str) -> dict[str, Any]:
    """
    Fetch profile + balance sheet + income statement for one ticker.
    Returns a flat dict suitable for screen_batch().
    Never raises — returns partial data on error.
    """
    data: dict[str, Any] = {"ticker": ticker, "symbol": ticker}
    try:
        from data_layer import live_data_provider as ldp
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
        log.debug("Profile fetch failed for %s: %s", ticker, type(exc).__name__)
        return data

    try:
        from data_layer import live_data_provider as ldp
        bs = ldp.get_balance_sheet(ticker, limit=1)
        if bs:
            data["totalAssets"]        = bs[0].get("totalAssets")
            data["totalDebt"]          = bs[0].get("totalDebt")
            data["accountsReceivable"] = bs[0].get("accountsReceivable")
    except Exception as exc:
        log.debug("Balance sheet fetch failed for %s: %s", ticker, type(exc).__name__)

    try:
        from data_layer import live_data_provider as ldp
        inc = ldp.get_income_statement(ticker, limit=1)
        if inc:
            data["revenue"]        = inc[0].get("revenue")
            data["interestIncome"] = inc[0].get("interestIncome")
    except Exception as exc:
        log.debug("Income statement fetch failed for %s: %s", ticker, type(exc).__name__)

    return data


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def _save_raw_csv(stocks: list[RawStock]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with UNIVERSE_RAW_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ticker", "name", "exchange", "sector", "industry",
            "mkt_cap", "price", "avg_volume", "stock_type", "source",
        ])
        w.writeheader()
        for s in stocks:
            w.writerow({
                "ticker": s.ticker, "name": s.name, "exchange": s.exchange,
                "sector": s.sector, "industry": s.industry,
                "mkt_cap": s.mkt_cap or "", "price": s.price or "",
                "avg_volume": s.avg_volume or "", "stock_type": s.stock_type,
                "source": s.source,
            })


def _save_screened_csv(stocks: list[ScreenedStock]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with UNIVERSE_SCREENED_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ticker", "name", "exchange", "sector", "industry",
            "mkt_cap", "price", "avg_volume", "shariah_status",
        ])
        w.writeheader()
        for s in stocks:
            w.writerow({
                "ticker": s.ticker, "name": s.name, "exchange": s.exchange,
                "sector": s.sector, "industry": s.industry,
                "mkt_cap": s.mkt_cap or "", "price": s.price or "",
                "avg_volume": s.avg_volume or "", "shariah_status": s.shariah_status,
            })


def _save_shariah_report_csv(results: list[ShariahResult]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with SHARIAH_REPORT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ticker", "status", "industry_pass", "ratio_pass",
            "manual_override", "reasons",
        ])
        w.writeheader()
        for r in results:
            w.writerow({
                "ticker":          r.ticker,
                "status":          r.status,
                "industry_pass":   r.industry_pass,
                "ratio_pass":      r.ratio_pass,
                "manual_override": r.manual_override,
                "reasons":         "; ".join(r.reasons),
            })


def load_screened_csv() -> list[dict]:
    """Load universe_screened.csv and return list of dicts. Returns [] if file absent."""
    if not UNIVERSE_SCREENED_CSV.exists():
        return []
    with UNIVERSE_SCREENED_CSV.open() as f:
        return list(csv.DictReader(f))


def load_raw_csv() -> list[dict]:
    """Load universe_raw.csv. Returns [] if file absent."""
    if not UNIVERSE_RAW_CSV.exists():
        return []
    with UNIVERSE_RAW_CSV.open() as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_live_universe(
    watchlist_path: Path | None = None,
    fetch_delay: float = _FETCH_DELAY,
    skip_fundamentals: bool = False,
) -> UniverseResult:
    """
    Build the full live Shariah-screened universe and save CSVs.

    Parameters
    ----------
    watchlist_path : Path | None
        Manual CSV fallback (one ticker per line). Used when FMP fails completely.
    fetch_delay : float
        Seconds to sleep between per-ticker FMP calls (rate-limit protection).
    skip_fundamentals : bool
        If True, skip balance sheet / income statement fetches and run Shariah
        with only profile data (much faster; ratio scores will be "unknown").

    Returns
    -------
    UniverseResult with counts and compliant_tickers list.
    """
    as_of = str(date.today())
    warnings: list[str] = []
    sources_used: list[str] = []

    # ── 1. Fetch candidates ─────────────────────────────────────────────────
    candidates: list[RawStock] = []

    sp500 = _fetch_sp500()
    if sp500:
        candidates.extend(sp500)
        sources_used.append("sp500")

    russell = _fetch_russell1000()
    if russell:
        candidates.extend(russell)
        sources_used.append("russell1000")

    # Always add broad screener to catch large-caps not in indices
    screener = _fetch_screener_broad()
    if screener:
        candidates.extend(screener)
        sources_used.append("screener")

    # Watchlist fallback
    if not candidates:
        wl_path = watchlist_path or WATCHLIST_CSV
        wl = _fetch_watchlist_csv(wl_path)
        if wl:
            candidates.extend(wl)
            sources_used.append("watchlist")
            warnings.append(
                "FMP unavailable — loaded tickers from watchlist CSV. "
                "Fundamental data and Shariah ratio screening may be incomplete."
            )
        else:
            warnings.append(
                "FMP unreachable and no watchlist CSV found at "
                f"{WATCHLIST_CSV}. Universe is empty."
            )

    raw_count = len(candidates)
    log.info("Candidates before dedup: %d", raw_count)

    # ── 2. Dedup ─────────────────────────────────────────────────────────────
    candidates = _dedup(candidates)
    log.info("After dedup: %d unique symbols", len(candidates))

    # ── 3. Pre-filter ────────────────────────────────────────────────────────
    pre_filtered, rejected = pre_filter(candidates)
    log.info(
        "Pre-filter: %d passed, %d rejected",
        len(pre_filtered), len(rejected),
    )
    if rejected:
        log.debug("Pre-filter rejections sample: %s", rejected[:5])

    # Save raw CSV (pre-Shariah, post-pre-filter)
    _save_raw_csv(pre_filtered)

    # ── 4. Fetch fundamentals for Shariah screening ──────────────────────────
    stock_data_list: list[dict] = []
    n = len(pre_filtered)
    log.info("Fetching fundamentals for %d tickers (this may take a few minutes)…", n)

    for i, stock in enumerate(pre_filtered):
        if i > 0 and i % 50 == 0:
            log.info("  Progress: %d / %d", i, n)

        if skip_fundamentals:
            # Build minimal data dict from what we already have
            data: dict = {
                "ticker":      stock.ticker,
                "symbol":      stock.ticker,
                "companyName": stock.name,
                "sector":      stock.sector,
                "industry":    stock.industry,
                "mktCap":      stock.mkt_cap,
                "price":       stock.price,
                "exchange":    stock.exchange,
            }
        else:
            data = _fetch_fundamentals(stock.ticker)
            # Backfill metadata from screener if profile fetch failed
            if not data.get("companyName"):
                data["companyName"] = stock.name
            if not data.get("sector"):
                data["sector"] = stock.sector
            if not data.get("industry"):
                data["industry"] = stock.industry
            if not data.get("mktCap") and stock.mkt_cap:
                data["mktCap"] = stock.mkt_cap
            if not data.get("price") and stock.price:
                data["price"] = stock.price
            if not data.get("exchange") and stock.exchange:
                data["exchange"] = stock.exchange

            if fetch_delay > 0:
                time.sleep(fetch_delay)

        stock_data_list.append(data)

    # ── 4b. Re-filter using fundamental data (fills in mkt_cap / price / volume) ─
    # Wikipedia-sourced stocks have no price/mkt_cap until profile is fetched.
    refined: list[dict] = []
    rejected_post: list[tuple[str, str]] = []
    for d in stock_data_list:
        mkt_cap = _to_float(d.get("mktCap"))
        price   = _to_float(d.get("price"))
        if mkt_cap is not None and mkt_cap < _MIN_MARKET_CAP:
            rejected_post.append((d["ticker"], f"market cap too small after profile: ${mkt_cap/1e9:.1f}B"))
            continue
        if price is not None and price >= _MAX_PRICE:
            rejected_post.append((d["ticker"], f"price too high after profile: ${price:.0f}"))
            continue
        refined.append(d)

    if rejected_post:
        log.info("Post-fundamental filter removed %d tickers", len(rejected_post))
    stock_data_list = refined

    # ── 5. Shariah screen ─────────────────────────────────────────────────────
    log.info("Running Shariah screening on %d tickers…", len(stock_data_list))
    manual_overrides = _load_manual_overrides()
    shariah_results: list[ShariahResult] = screen_batch(stock_data_list, manual_overrides)
    _save_shariah_report_csv(shariah_results)

    # ── 6. Build screened stocks list (compliant only) ────────────────────────
    # Build a fast lookup: ticker → stock data dict
    data_by_ticker: dict[str, dict] = {d["ticker"]: d for d in stock_data_list}
    # And original RawStock metadata
    raw_by_ticker: dict[str, RawStock] = {s.ticker: s for s in pre_filtered}

    compliant_stocks: list[ScreenedStock] = []
    compliant_count   = 0
    non_compliant_count = 0
    unknown_count     = 0

    for r in shariah_results:
        d  = data_by_ticker.get(r.ticker, {})
        rs = raw_by_ticker.get(r.ticker)
        screened = ScreenedStock(
            ticker=r.ticker,
            name=d.get("companyName", rs.name if rs else ""),
            exchange=d.get("exchange", rs.exchange if rs else ""),
            sector=d.get("sector", rs.sector if rs else ""),
            industry=d.get("industry", rs.industry if rs else ""),
            mkt_cap=_to_float(d.get("mktCap")) or (rs.mkt_cap if rs else None),
            price=_to_float(d.get("price")) or (rs.price if rs else None),
            avg_volume=rs.avg_volume if rs else None,
            shariah_status=r.status,
        )
        if r.status == "compliant":
            compliant_count += 1
            compliant_stocks.append(screened)
        elif r.status == "non_compliant":
            non_compliant_count += 1
        else:
            unknown_count += 1

    _save_screened_csv(compliant_stocks)

    log.info(
        "Universe built: %d compliant, %d non-compliant, %d unknown  (saved to %s)",
        compliant_count, non_compliant_count, unknown_count, CACHE_DIR,
    )

    return UniverseResult(
        as_of_date=as_of,
        raw_count=len(pre_filtered),
        screened_count=len(shariah_results),
        compliant_count=compliant_count,
        non_compliant_count=non_compliant_count,
        unknown_count=unknown_count,
        compliant_tickers=[s.ticker for s in compliant_stocks],
        sources_used=sources_used,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _to_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None
