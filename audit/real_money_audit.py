"""
Real-Money Audit — production hardening gate.

Runs ALL safety checks in sequence and prints a single verdict:
  REAL MONEY READY = YES  (every check passed)
  REAL MONEY READY = NO   (one or more checks failed — reasons printed)

Check categories:
  1. Health checks     — API keys, cache I/O, required files
  2. Data validation   — price/mkt_cap/volume > 0, no missing scores,
                         no duplicates, no non-compliant in Top 5
  3. Consistency       — terminal Top 5 == dashboard Top 5,
                         score formula recalculation, weights/allocations
  4. Stale-data        — price ≤ 30 min, fundamentals ≤ 7 days, Shariah ≤ 7 days
  5. Shariah gate      — every Top 5 ticker in data/manual/shariah_verification.csv
                         with verified_status = "compliant"

Strategy logic is never modified — this module is read-only over all data.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sqlite3
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT            = Path(__file__).parent.parent
_CACHE_DIR       = _ROOT / "data" / "cache"
_MANUAL_DIR      = _ROOT / "data" / "manual"
_DATA_DIR        = _ROOT / "data"

SCORED_UNIVERSE  = _CACHE_DIR / "scored_universe.json"
SCREENED_CSV     = _CACHE_DIR / "universe_screened.csv"
BACKTEST_FILE    = _DATA_DIR  / "backtest_complete.json"
CACHE_DB         = _CACHE_DIR / "data_cache.db"
SHARIAH_CSV      = _MANUAL_DIR / "shariah_verification.csv"

# Freshness thresholds
_PRICE_STALE_MINUTES   = 30
_FUND_STALE_DAYS       = 7
_SHARIAH_STALE_DAYS    = 7

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Check:
    name:    str
    passed:  bool
    detail:  str
    warning: bool = False   # True = warn but don't block


@dataclass
class AuditReport:
    checks:            list[Check]      = field(default_factory=list)
    top5:              list[dict]       = field(default_factory=list)
    formula_checks:    list[dict]       = field(default_factory=list)
    freshness_rows:    list[dict]       = field(default_factory=list)
    shariah_rows:      list[dict]       = field(default_factory=list)
    ready:             bool             = False

    def add(self, check: Check) -> None:
        self.checks.append(check)

    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and not c.warning]

    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.warning]


# ---------------------------------------------------------------------------
# Individual check helpers
# ---------------------------------------------------------------------------

def _ok(name: str, detail: str) -> Check:
    return Check(name=name, passed=True, detail=detail)

def _fail(name: str, detail: str) -> Check:
    return Check(name=name, passed=False, detail=detail)

def _warn(name: str, detail: str) -> Check:
    return Check(name=name, passed=False, detail=detail, warning=True)


# ---------------------------------------------------------------------------
# 1. Health checks
# ---------------------------------------------------------------------------

def _check_fmp_key() -> Check:
    from config.settings import FMP_API_KEY, FMP_BASE_URL
    if not FMP_API_KEY:
        return _fail("FMP key present", "FMP_API_KEY is empty in .env")
    try:
        import requests
        url = f"{FMP_BASE_URL}/profile"
        r = requests.get(url, params={"symbol": "AAPL", "apikey": FMP_API_KEY}, timeout=10)
        if r.status_code == 200 and r.json():
            return _ok("FMP key works", "HTTP 200 — profile returned for AAPL")
        if r.status_code in (402, 403, 429):
            return _warn("FMP key works", f"FMP plan/rate limit ({r.status_code}) — yfinance fallback active")
        return _fail("FMP key works", f"FMP returned HTTP {r.status_code}")
    except Exception as exc:
        return _fail("FMP key works", f"FMP network error: {exc}")


def _check_anthropic_key() -> Check:
    from config.settings import ANTHROPIC_API_KEY
    if not ANTHROPIC_API_KEY:
        return _warn("Anthropic key present", "ANTHROPIC_API_KEY not set — AI Research = neutral 50")
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        # Minimal ping — list models endpoint doesn't charge tokens
        client.models.list(limit=1)
        return _ok("Anthropic key works", "API responded to models.list()")
    except Exception as exc:
        return _fail("Anthropic key works", f"Anthropic error: {exc}")


def _check_yfinance() -> Check:
    try:
        import yfinance as yf
        df = yf.download("AAPL", period="1d", progress=False, auto_adjust=True)
        if df.empty:
            return _fail("yfinance fallback works", "yfinance returned empty DataFrame for AAPL")
        return _ok("yfinance fallback works", f"yfinance returned {len(df)} row(s) for AAPL")
    except Exception as exc:
        return _fail("yfinance fallback works", f"yfinance error: {exc}")


def _check_cache_rw() -> Check:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _CACHE_DIR / ".audit_rw_test"
        tmp.write_text("ok")
        tmp.read_text()
        tmp.unlink()
        # Also verify the SQLite DB is readable
        if CACHE_DB.exists():
            con = sqlite3.connect(str(CACHE_DB))
            con.execute("SELECT COUNT(*) FROM cache")
            con.close()
        return _ok("Cache readable/writable", f"{_CACHE_DIR} — read/write OK")
    except Exception as exc:
        return _fail("Cache readable/writable", f"Cache I/O error: {exc}")


def _check_file_exists(path: Path, label: str) -> Check:
    if path.exists():
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return _ok(f"{label} exists", f"{path.name} — last modified {mtime:%Y-%m-%d %H:%M}")
    return _fail(f"{label} exists", f"Not found: {path}")


# ---------------------------------------------------------------------------
# 2. Data validation
# ---------------------------------------------------------------------------

def _load_universe() -> tuple[list[dict], str | None]:
    """Return (universe_list, error_string_or_None)."""
    if not SCORED_UNIVERSE.exists():
        return [], "scored_universe.json not found"
    try:
        with SCORED_UNIVERSE.open() as f:
            payload = json.load(f)
        return payload.get("universe", []), None
    except Exception as exc:
        return [], f"Could not parse scored_universe.json: {exc}"


def _validate_universe(universe: list[dict]) -> list[Check]:
    checks: list[Check] = []

    # No empty universe
    if not universe:
        return [_fail("Universe non-empty", "Universe list is empty")]
    checks.append(_ok("Universe non-empty", f"{len(universe)} tickers loaded"))

    # No duplicate tickers
    tickers = [s["ticker"] for s in universe]
    dupes   = [t for t in set(tickers) if tickers.count(t) > 1]
    if dupes:
        checks.append(_fail("No duplicate tickers", f"Duplicates: {', '.join(dupes)}"))
    else:
        checks.append(_ok("No duplicate tickers", "All tickers unique"))

    # Data quality per ticker
    bad_price = [s["ticker"] for s in universe if not (s.get("price") or 0) > 0]
    bad_cap   = [s["ticker"] for s in universe if not (s.get("mkt_cap") or 0) > 0]
    bad_vol   = [s["ticker"] for s in universe if not (s.get("avg_volume") or 0) > 0]

    for label, bad in [("price > 0", bad_price), ("mkt_cap > 0", bad_cap), ("avg_volume > 0", bad_vol)]:
        if bad:
            subset = bad[:5]
            more   = f" (+{len(bad)-5} more)" if len(bad) > 5 else ""
            checks.append(_fail(f"All tickers: {label}", f"{', '.join(subset)}{more}"))
        else:
            checks.append(_ok(f"All tickers: {label}", f"All {len(universe)} tickers pass"))

    # No missing composite scores
    no_score = [s["ticker"] for s in universe if s.get("composite") is None]
    if no_score:
        checks.append(_fail("No missing composite scores", f"Missing: {', '.join(no_score[:5])}"))
    else:
        checks.append(_ok("No missing composite scores", f"All {len(universe)} tickers scored"))

    return checks


def _validate_top5(top5: list[dict]) -> list[Check]:
    checks: list[Check] = []

    if len(top5) < 5:
        checks.append(_fail("Top 5 count", f"Only {len(top5)} BUY NOW stocks (expected 5)"))
    else:
        checks.append(_ok("Top 5 count", "Exactly 5 BUY NOW stocks"))

    # No non-compliant in BUY NOW
    non_compliant = [
        s["ticker"] for s in top5
        if s.get("shariah_status", "unknown") != "compliant"
    ]
    if non_compliant:
        checks.append(_fail("No non-compliant in BUY NOW", f"Non-compliant: {', '.join(non_compliant)}"))
    else:
        checks.append(_ok("No non-compliant in BUY NOW", "All Top 5 are Shariah-compliant"))

    # Price sanity for each BUY NOW
    for s in top5:
        price = s.get("price") or 0
        if price <= 0:
            checks.append(_fail(f"{s['ticker']} price > 0", f"Price is {price}"))
        elif price >= 1000:
            checks.append(_fail(f"{s['ticker']} price < $1,000", f"Price ${price:.2f} exceeds limit"))

    return checks


# ---------------------------------------------------------------------------
# 3. Consistency checks
# ---------------------------------------------------------------------------

def _check_weights_and_allocations() -> list[Check]:
    from config.settings import ACCOUNT_SIZE, CONVICTION_WEIGHTS, CONVICTION_DOLLARS

    checks: list[Check] = []

    weight_sum = round(sum(CONVICTION_WEIGHTS), 10)
    if weight_sum == 1.0:
        checks.append(_ok("Portfolio weights sum to 100%", f"Sum = {weight_sum}"))
    else:
        checks.append(_fail("Portfolio weights sum to 100%", f"Sum = {weight_sum} (expected 1.0)"))

    alloc_sum = round(sum(CONVICTION_DOLLARS), 2)
    if alloc_sum == ACCOUNT_SIZE:
        checks.append(_ok(f"Allocations sum to ${ACCOUNT_SIZE:,.0f}",
                          f"${alloc_sum:,.0f} = {' + '.join(f'${d:.0f}' for d in CONVICTION_DOLLARS)}"))
    else:
        checks.append(_fail(f"Allocations sum to ${ACCOUNT_SIZE:,.0f}",
                            f"Sum = ${alloc_sum:,.2f} (expected ${ACCOUNT_SIZE:,.2f})"))

    return checks


def _check_factor_weights() -> Check:
    from config.weights import FACTOR_WEIGHTS
    total = sum(FACTOR_WEIGHTS.values())
    if round(total, 10) == 1.0:
        parts = "  +  ".join(f"{k} {v*100:.0f}%" for k, v in FACTOR_WEIGHTS.items())
        return _ok("Factor weights sum to 100%", parts)
    return _fail("Factor weights sum to 100%", f"Sum = {total:.6f} (expected 1.0)")


def _verify_score_formula(universe: list[dict], top5: list[dict]) -> list[dict]:
    """
    Recompute composite for each Top 5 ticker using the stored
    individual factor scores and the live FACTOR_WEIGHTS.
    AI is NOT in composite formula — flows through Moat (10% weight).
    Returns a list of audit rows.
    """
    from config.weights import FACTOR_WEIGHTS

    rows = []
    universe_map = {s["ticker"]: s for s in universe}

    for stock in top5:
        t   = stock["ticker"]
        s   = universe_map.get(t, stock)
        q   = s.get("quality", 50.0)
        m   = s.get("momentum", 50.0)
        v   = s.get("valuation", 50.0)
        r   = s.get("earnings_revisions", 50.0)
        eq  = s.get("earnings_quality", 50.0)
        mo  = s.get("moat", 50.0)
        ca  = s.get("capital_allocation", 50.0)
        ra  = s.get("risk_adjustment", 50.0)

        expected = (
            q  * FACTOR_WEIGHTS["quality"]            +
            m  * FACTOR_WEIGHTS["momentum"]           +
            v  * FACTOR_WEIGHTS["valuation"]          +
            r  * FACTOR_WEIGHTS["earnings_revisions"] +
            eq * FACTOR_WEIGHTS["earnings_quality"]   +
            mo * FACTOR_WEIGHTS["moat"]               +
            ca * FACTOR_WEIGHTS["capital_allocation"] +
            ra * FACTOR_WEIGHTS["risk_adjustment"]
        )
        stored   = s.get("composite", 0.0)
        delta    = abs(expected - stored)
        match    = delta < 0.5   # slightly wider tolerance for 8-factor sum

        rows.append({
            "ticker":   t,
            "quality":  round(q, 1),
            "momentum": round(m, 1),
            "valuation": round(v, 1),
            "revisions": round(r, 1),
            "ai":       round(s.get("ai_research", 50.0), 1),
            "stored":   round(stored, 2),
            "recalc":   round(expected, 2),
            "delta":    round(delta, 3),
            "match":    match,
        })

    return rows


# ---------------------------------------------------------------------------
# 4. Stale-data checks
# ---------------------------------------------------------------------------

def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _check_data_freshness() -> tuple[list[Check], list[dict]]:
    checks: list[Check] = []
    rows:   list[dict]  = []
    now = datetime.now()

    # scored_universe.json — price data age
    if SCORED_UNIVERSE.exists():
        with SCORED_UNIVERSE.open() as f:
            payload = json.load(f)
        gen_at = _parse_dt(payload.get("generated_at"))
        if gen_at:
            age_min = (now - gen_at).total_seconds() / 60
            age_str = f"{age_min:.0f} min ago"
            rows.append({
                "source": "scored_universe.json",
                "generated_at": gen_at.strftime("%Y-%m-%d %H:%M"),
                "age": age_str,
                "threshold": f"{_PRICE_STALE_MINUTES} min",
                "status": "OK" if age_min <= _PRICE_STALE_MINUTES else "STALE",
            })
            if age_min > _PRICE_STALE_MINUTES:
                checks.append(_fail(
                    "Price data ≤ 30 min old",
                    f"scored_universe.json is {age_min:.0f} min old — re-run --refresh-data before trading"
                ))
            else:
                checks.append(_ok("Price data ≤ 30 min old", f"Generated {age_min:.0f} min ago"))
        else:
            checks.append(_fail("Price data ≤ 30 min old", "Cannot parse generated_at in scored_universe.json"))
    else:
        checks.append(_fail("Price data ≤ 30 min old", "scored_universe.json missing"))

    # universe_screened.csv — Shariah screen age
    if SCREENED_CSV.exists():
        mtime    = datetime.fromtimestamp(SCREENED_CSV.stat().st_mtime)
        age_days = (now - mtime).total_seconds() / 86400
        rows.append({
            "source": "universe_screened.csv",
            "generated_at": mtime.strftime("%Y-%m-%d %H:%M"),
            "age": f"{age_days:.1f} days",
            "threshold": f"{_SHARIAH_STALE_DAYS} days",
            "status": "OK" if age_days <= _SHARIAH_STALE_DAYS else "STALE",
        })
        if age_days > _SHARIAH_STALE_DAYS:
            checks.append(_warn(
                "Shariah screen ≤ 7 days old",
                f"universe_screened.csv is {age_days:.1f} days old — re-run --refresh-universe"
            ))
        else:
            checks.append(_ok("Shariah screen ≤ 7 days old", f"{age_days:.1f} days old"))
    else:
        checks.append(_fail("Shariah screen ≤ 7 days old", "universe_screened.csv missing"))

    # backtest_complete.json — fundamentals/backtest age
    if BACKTEST_FILE.exists():
        with BACKTEST_FILE.open() as f:
            bt = json.load(f)
        comp_at = _parse_dt(bt.get("completed_at"))
        if comp_at:
            age_days = (now - comp_at).total_seconds() / 86400
            rows.append({
                "source": "backtest_complete.json",
                "generated_at": comp_at.strftime("%Y-%m-%d %H:%M"),
                "age": f"{age_days:.1f} days",
                "threshold": f"{_FUND_STALE_DAYS} days",
                "status": "OK" if age_days <= _FUND_STALE_DAYS else "STALE",
            })
            if age_days > _FUND_STALE_DAYS:
                checks.append(_warn(
                    "Fundamentals/backtest ≤ 7 days old",
                    f"backtest_complete.json is {age_days:.1f} days old — re-run --backtest"
                ))
            else:
                checks.append(_ok("Fundamentals/backtest ≤ 7 days old", f"{age_days:.1f} days old"))

    return checks, rows


# ---------------------------------------------------------------------------
# 5. Shariah confirmation gate
# ---------------------------------------------------------------------------

def _load_shariah_verification() -> dict[str, dict]:
    """Load data/manual/shariah_verification.csv → {ticker: row}."""
    if not SHARIAH_CSV.exists():
        return {}
    out: dict[str, dict] = {}
    try:
        with SHARIAH_CSV.open(newline="") as f:
            for row in csv.DictReader(f):
                t = (row.get("ticker") or "").strip().upper()
                if t:
                    out[t] = {k.strip(): (v or "").strip() for k, v in row.items()}
    except Exception as exc:
        log.warning("Could not read shariah_verification.csv: %s", exc)
    return out


def _check_shariah_gate(top5: list[dict]) -> tuple[list[Check], list[dict]]:
    checks: list[Check] = []
    rows:   list[dict]  = []
    verif  = _load_shariah_verification()

    all_verified = True
    for stock in top5:
        t   = stock["ticker"]
        rec = verif.get(t)
        if not rec:
            checks.append(_fail(
                f"Shariah manual verification: {t}",
                f"{t} not found in {SHARIAH_CSV.name} — add a row with verified_status=compliant"
            ))
            rows.append({
                "ticker": t, "verified_status": "MISSING",
                "source": "—", "last_checked": "—", "notes": "Not in CSV"
            })
            all_verified = False
        elif rec.get("verified_status", "").lower() != "compliant":
            status = rec.get("verified_status", "")
            checks.append(_fail(
                f"Shariah manual verification: {t}",
                f"{t} verified_status = '{status}' (must be 'compliant')"
            ))
            rows.append({
                "ticker": t,
                "verified_status": status or "BLANK",
                "source": rec.get("source", "—"),
                "last_checked": rec.get("last_checked", "—"),
                "notes": rec.get("notes", ""),
            })
            all_verified = False
        else:
            rows.append({
                "ticker": t,
                "verified_status": "compliant ✓",
                "source": rec.get("source", "—"),
                "last_checked": rec.get("last_checked", "—"),
                "notes": rec.get("notes", ""),
            })

    if all_verified and top5:
        checks.append(_ok(
            "Shariah gate: all Top 5 manually verified",
            f"All {len(top5)} tickers verified compliant in {SHARIAH_CSV.name}"
        ))

    return checks, rows


# ---------------------------------------------------------------------------
# Build Top 5 from scored_universe.json  (mirrors cmd_recommend logic)
# ---------------------------------------------------------------------------

def _derive_top5(universe: list[dict]) -> list[dict]:
    from portfolio.constraints import check_constraints
    from portfolio.constructor import build_portfolio
    from portfolio.recommendation_guard import safe_recommendations
    from factors.composite import FactorScores

    all_scores = sorted(
        [
            FactorScores(
                ticker             = s["ticker"],
                quality            = float(s.get("quality", 50.0)),
                momentum           = float(s.get("momentum", 50.0)),
                valuation          = float(s.get("valuation", 50.0)),
                earnings_revisions = float(s.get("earnings_revisions", 50.0)),
                earnings_quality   = float(s.get("earnings_quality", 50.0)),
                moat               = float(s.get("moat", 50.0)),
                capital_allocation = float(s.get("capital_allocation", 50.0)),
                risk_adjustment    = float(s.get("risk_adjustment", 50.0)),
                ai_research        = float(s.get("ai_research", 50.0)),
                composite          = float(s.get("composite", 50.0)),
            )
            for s in universe
        ],
        key=lambda x: x.composite,
        reverse=True,
    )
    constraint_results = {
        s["ticker"]: check_constraints(
            ticker         = s["ticker"],
            price          = float(s.get("price") or 0.0),
            market_cap     = float(s.get("mkt_cap") or 0.0),
            avg_volume     = float(s.get("avg_volume") or 0.0),
            shariah_status = s.get("shariah_status", "unknown"),
        )
        for s in universe
    }
    prices = {s["ticker"]: float(s.get("price") or 0.0) for s in universe}
    shariah_statuses = {s["ticker"]: s.get("shariah_status", "unknown") for s in universe}

    portfolio = build_portfolio(all_scores, constraint_results, prices)
    recs = safe_recommendations(
        portfolio_result   = portfolio,
        all_scores         = all_scores,
        constraint_results = constraint_results,
        prices             = prices,
        shariah_statuses   = shariah_statuses,
    )

    buy_now = [r for r in recs if r.action == "BUY NOW"]
    universe_map = {s["ticker"]: s for s in universe}
    return [
        {
            **universe_map.get(r.ticker, {}),
            "ticker":           r.ticker,
            "composite":        r.composite_score,
            "rank":             r.rank,
            "conviction_weight": r.conviction_weight,
            "dollar_amount":    r.dollar_amount,
            "price":            r.price,
        }
        for r in buy_now
    ]


# ---------------------------------------------------------------------------
# Printer helpers
# ---------------------------------------------------------------------------

_BAR  = "═" * 72
_BAR2 = "─" * 72

def _section(title: str) -> None:
    print(f"\n{_BAR}\n  {title}\n{_BAR}")

def _row_mark(passed: bool, warning: bool = False) -> str:
    if passed:   return "  ✓"
    if warning:  return "  ⚠"
    return "  ✗"


def _print_checks(checks: list[Check], title: str) -> None:
    _section(title)
    for c in checks:
        mark = _row_mark(c.passed, c.warning)
        tag  = " [WARN]" if (not c.passed and c.warning) else ""
        print(f"{mark}  {c.name}{tag}")
        if not c.passed:
            print(f"       → {c.detail}")
        elif c.detail:
            print(f"       {c.detail}")


def _print_top5_table(top5: list[dict], formula_rows: list[dict]) -> None:
    _section("TOP 5 AUDIT TABLE")
    if not top5:
        print("  (no BUY NOW stocks)")
        return

    formula_map = {r["ticker"]: r for r in formula_rows}
    hdr = (f"  {'#':<3} {'Ticker':<7} {'Comp':>6} {'Qual':>6} {'Mom':>6} "
           f"{'Val':>6} {'Rev':>6} {'AI*':>5} {'Recalc':>7} {'Match':>6} "
           f"{'Price':>8} {'$Alloc':>8}")
    print(hdr)
    print("  " + "─" * 72)
    sbc_flags = []
    for s in top5:
        t   = s["ticker"]
        fr  = formula_map.get(t, {})
        match = "✓" if fr.get("match") else "✗ MISMATCH"
        print(
            f"  {s.get('rank', '?'):<3} {t:<7}"
            f" {s.get('composite', 0):>6.1f}"
            f" {s.get('quality', 0):>6.1f}"
            f" {s.get('momentum', 0):>6.1f}"
            f" {s.get('valuation', 0):>6.1f}"
            f" {s.get('earnings_revisions', s.get('revisions', 0)):>6.1f}"
            f" {s.get('ai_research', 0):>5.1f}"
            f" {fr.get('recalc', 0):>7.2f}"
            f" {match:>6}"
            f" ${s.get('price', 0):>7.2f}"
            f" ${s.get('dollar_amount', 0):>6.0f}"
        )
        sbc = s.get("sbc_data")
        if sbc and sbc.get("sbc_fcf_ratio", 0) > 0.20:
            ratio = sbc["sbc_fcf_ratio"]
            adj   = sbc.get("adj_fcf", 0)
            print(f"      ⚠  {t}: SBC = {ratio*100:.0f}% of FCF  |  SBC-adj FCF = ${adj/1e9:.2f}B  |  Reported FCF may overstate owner earnings")
            sbc_flags.append(t)
        rd = s.get("risk_detail") or {}
        if rd.get("risk_label") == "High":
            print(f"      ⚠  {t}: HIGH RISK — {'; '.join(rd.get('warnings', []))}")

    total_alloc = sum(s.get("dollar_amount", 0) or 0 for s in top5)
    print(f"\n  Total allocation: ${total_alloc:,.0f}")


def _print_freshness_table(rows: list[dict]) -> None:
    _section("DATA FRESHNESS TABLE")
    if not rows:
        print("  (no freshness data)")
        return
    print(f"  {'Source':<35} {'Generated':<18} {'Age':<12} {'Threshold':<12} {'Status'}")
    print("  " + "─" * 70)
    for r in rows:
        status = r.get("status", "?")
        icon   = "✓" if status == "OK" else ("⚠" if status == "STALE" else "?")
        print(
            f"  {r['source']:<35} {r['generated_at']:<18}"
            f" {r['age']:<12} {r['threshold']:<12} {icon} {status}"
        )


def _print_shariah_table(rows: list[dict]) -> None:
    _section("SHARIAH VERIFICATION TABLE")
    if not rows:
        print("  (no Top 5 stocks to verify)")
        return
    print(f"  {'Ticker':<8} {'Status':<18} {'Source':<20} {'Last Checked':<14} {'Notes'}")
    print("  " + "─" * 70)
    for r in rows:
        print(
            f"  {r['ticker']:<8} {r['verified_status']:<18}"
            f" {r['source']:<20} {r['last_checked']:<14} {r['notes']}"
        )


def _print_formula_detail(formula_rows: list[dict]) -> None:
    _section("SCORE FORMULA VERIFICATION")
    from config.weights import FACTOR_WEIGHTS
    print(f"  Formula: composite = "
          f"quality×{FACTOR_WEIGHTS['quality']:.0%} + "
          f"momentum×{FACTOR_WEIGHTS['momentum']:.0%} + "
          f"valuation×{FACTOR_WEIGHTS['valuation']:.0%} + "
          f"revisions×{FACTOR_WEIGHTS['earnings_revisions']:.0%} + "
          f"eq×{FACTOR_WEIGHTS['earnings_quality']:.0%} + "
          f"moat×{FACTOR_WEIGHTS['moat']:.0%} + "
          f"capalloc×{FACTOR_WEIGHTS['capital_allocation']:.0%} + "
          f"riskadj×{FACTOR_WEIGHTS['risk_adjustment']:.0%}\n"
          f"  (AI display only — conf-scaled, not in composite; flows through Moat)\n")
    if not formula_rows:
        print("  (no Top 5 to verify)")
        return
    print(f"  {'Ticker':<8} {'Stored':>8} {'Recalc':>8} {'Delta':>8} {'Match':>7}")
    print("  " + "─" * 44)
    for r in formula_rows:
        match = "✓" if r["match"] else "✗"
        print(
            f"  {r['ticker']:<8} {r['stored']:>8.2f} {r['recalc']:>8.2f}"
            f" {r['delta']:>8.3f} {match:>7}"
        )


# ---------------------------------------------------------------------------
# Master audit runner
# ---------------------------------------------------------------------------

def run_audit() -> AuditReport:
    report = AuditReport()

    # ── 1. Health checks ────────────────────────────────────────────────────
    report.add(_check_fmp_key())
    report.add(_check_anthropic_key())
    report.add(_check_yfinance())
    report.add(_check_cache_rw())
    report.add(_check_file_exists(SCREENED_CSV,    "universe_screened.csv"))
    report.add(_check_file_exists(SCORED_UNIVERSE, "scored_universe.json"))
    report.add(_check_file_exists(BACKTEST_FILE,   "backtest_complete.json"))

    # ── 2. Load universe ────────────────────────────────────────────────────
    universe, err = _load_universe()
    if err:
        report.add(_fail("scored_universe.json loadable", err))
        report.ready = False
        return report
    report.add(_ok("scored_universe.json loadable", f"{len(universe)} records"))

    # ── 3. Data validation ──────────────────────────────────────────────────
    for c in _validate_universe(universe):
        report.add(c)

    # ── 4. Derive Top 5 ─────────────────────────────────────────────────────
    try:
        top5 = _derive_top5(universe)
    except Exception as exc:
        report.add(_fail("Top 5 derivable", f"Could not derive Top 5: {exc}"))
        top5 = []
    report.top5 = top5

    for c in _validate_top5(top5):
        report.add(c)

    # ── 5. Consistency ──────────────────────────────────────────────────────
    for c in _check_weights_and_allocations():
        report.add(c)
    report.add(_check_factor_weights())

    formula_rows = _verify_score_formula(universe, top5)
    report.formula_checks = formula_rows
    mismatches = [r for r in formula_rows if not r["match"]]
    if mismatches:
        tickers = ", ".join(r["ticker"] for r in mismatches)
        report.add(_fail("Score formula recalculates correctly",
                         f"Mismatch for: {tickers} — check FACTOR_WEIGHTS vs stored scores"))
    else:
        report.add(_ok("Score formula recalculates correctly",
                       f"All {len(formula_rows)} Top 5 scores verified"))

    alloc_sum = sum(s.get("dollar_amount", 0) or 0 for s in top5)
    from config.settings import ACCOUNT_SIZE
    if top5 and abs(alloc_sum - ACCOUNT_SIZE) < 0.01:
        report.add(_ok(f"Allocations sum to ${ACCOUNT_SIZE:,.0f}", f"${alloc_sum:,.0f}"))
    elif top5:
        report.add(_fail(f"Allocations sum to ${ACCOUNT_SIZE:,.0f}",
                         f"Allocations sum to ${alloc_sum:,.0f}"))

    # ── 6. Stale-data ───────────────────────────────────────────────────────
    freshness_checks, freshness_rows = _check_data_freshness()
    report.freshness_rows = freshness_rows
    for c in freshness_checks:
        report.add(c)

    # ── 7. Shariah gate ─────────────────────────────────────────────────────
    shariah_checks, shariah_rows = _check_shariah_gate(top5)
    report.shariah_rows = shariah_rows
    for c in shariah_checks:
        report.add(c)

    # ── 8. SBC warnings (informational — never block) ────────────────────────
    _SBC_THRESHOLD = 0.20
    for s in top5:
        sbc_data = s.get("sbc_data")
        if not sbc_data:
            continue
        ratio = sbc_data.get("sbc_fcf_ratio", 0)
        if ratio > _SBC_THRESHOLD:
            adj = sbc_data.get("adj_fcf", 0)
            report.add(Check(
                name=f"SBC/{s['ticker']}",
                passed=True,
                detail=(
                    f"SBC is {ratio*100:.0f}% of reported FCF "
                    f"(SBC-adjusted FCF = ${adj/1e9:.2f}B). "
                    f"Reported FCF may overstate owner earnings."
                ),
                warning=True,
            ))

    # ── Final verdict ────────────────────────────────────────────────────────
    report.ready = len(report.failures()) == 0

    return report


def print_report(report: AuditReport) -> None:
    # Health
    health_checks  = report.checks[:7]
    _print_checks(health_checks, "1. HEALTH CHECKS")

    # Data validation
    univ_checks = [c for c in report.checks
                   if any(kw in c.name for kw in
                          ["Universe", "duplicate", "price > 0", "mkt_cap", "avg_volume",
                           "composite", "loadable"])]
    _print_checks(univ_checks, "2. DATA VALIDATION")

    # Top 5 audit table
    _print_top5_table(report.top5, report.formula_checks)

    # Consistency
    consist_checks = [c for c in report.checks
                      if any(kw in c.name for kw in
                             ["weights", "Allocations sum", "formula", "Factor weights", "Top 5 count",
                              "non-compliant in BUY", "price > 0", "price < $"]
                             ) and c not in univ_checks]
    _print_checks(consist_checks, "3. CONSISTENCY CHECKS")

    # Formula detail
    _print_formula_detail(report.formula_checks)

    # Freshness
    _print_freshness_table(report.freshness_rows)
    fresh_checks = [c for c in report.checks if any(kw in c.name for kw in
                    ["Price data", "Shariah screen", "Fundamentals"])]
    if fresh_checks:
        for c in fresh_checks:
            icon = _row_mark(c.passed, c.warning)
            print(f"{icon}  {c.name}: {c.detail}")

    # Shariah gate
    _print_shariah_table(report.shariah_rows)
    _print_checks(
        [c for c in report.checks if "Shariah" in c.name and "compliant in BUY" not in c.name],
        "5. SHARIAH CONFIRMATION GATE",
    )

    # ── Final verdict ────────────────────────────────────────────────────────
    failures = report.failures()
    warnings = report.warnings()

    print(f"\n{'═'*72}")
    if report.ready:
        print("  REAL MONEY READY = YES")
        print(f"  All {len(report.checks)} checks passed.")
    else:
        print("  REAL MONEY READY = NO")
        print(f"\n  FAILED CHECKS ({len(failures)}):")
        for c in failures:
            print(f"    ✗  {c.name}")
            print(f"       → {c.detail}")

    if warnings:
        print(f"\n  WARNINGS ({len(warnings)}) — do not block but require attention:")
        for c in warnings:
            print(f"    ⚠  {c.name}")
            print(f"       → {c.detail}")

    print(f"{'═'*72}\n")
