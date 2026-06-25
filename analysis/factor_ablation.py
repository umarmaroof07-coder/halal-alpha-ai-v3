"""
Factor Ablation & Version Comparison Backtest
=============================================

METHODOLOGY
-----------
This study uses STATIC CROSS-SECTIONAL FACTOR SCORES (today's scores from
scored_universe.json) applied to 5 years of ACTUAL monthly price returns
fetched from yfinance (2021-06 → 2026-05, ~60 months).

CRITICAL LIMITATIONS (printed with every result):
  1. LOOK-AHEAD BIAS: Factor scores are computed from TODAY's fundamentals.
     The stock that ranks #1 on quality today may not have ranked #1 five
     years ago. This overstates predictive power.
  2. SURVIVORSHIP BIAS: Only tickers in today's 202-stock universe are tested.
     Companies that failed or were acquired 2021-2026 are excluded.
  3. STATIC RANKS: We hold the same relative ordering for all 60 months.
     Real performance would change as fundamentals change quarterly.

Despite these limitations, this study is still useful for:
  - Identifying which factor DIRECTIONS are cross-sectionally predictive
    (i.e., does sorting on quality actually produce better price returns?)
  - Detecting factors that HURT performance (negative IC or active return)
  - Comparing factor weight configurations on the same universe/period
  - Measuring marginal contribution of each factor to composite alpha

CONFIGURATIONS TESTED
---------------------
  SPY / QQQ           — passive benchmarks
  Equal Weight Top-20 — naive baseline (no factor preference)
  V1                  — Quality 50% + Momentum 50% (original 2-factor model)
  V3                  — Q20% M20% Val15% Rev15% EQ10% Moat10% CA10%
  V4 equal            — V4 weights, equal-sized positions (no inv-vol)
  V4 inv-vol          — V4 weights + inverse-volatility sizing
  V5 inv-vol          — V4 weights + inv-vol, max 25% per position

FACTOR ABLATION (leave-one-out from V4)
  V4 − Quality        — remove quality, redistribute weight equally
  V4 − Momentum       — etc.
  V4 − Revisions
  V4 − Valuation
  V4 − EarningsQ
  V4 − Moat
  V4 − CapAlloc
  V4 − Risk

SINGLE FACTOR (top-5 sorted by one factor only, equal weight)
  Quality only
  Momentum only
  Revisions only
  Valuation only
  EarningsQ only
  Moat only
  CapAlloc only
  Risk only
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Config ───────────────────────────────────────────────────────────────────

STUDY_YEARS  = 5          # how many years of history to fetch
N_POSITIONS  = 5          # top-N stocks to hold each period
N_TOP_POOL   = 20         # restrict to top-20 by composite (matches live system)
REBALANCE    = "monthly"  # only monthly supported
COST_RATE    = 0.001      # 0.10% per trade (one-way)
RISK_FREE    = 0.0        # annualised risk-free rate

FACTORS = ["quality","momentum","valuation","earnings_revisions",
           "earnings_quality","moat","capital_allocation","risk_adjustment"]

# Version weight configurations (must sum to 1.0; AI Research excluded — display only)
CONFIGS: dict[str, dict[str, float]] = {
    "V1": {
        "quality": 0.50,
        "momentum": 0.50,
    },
    "V3": {
        "quality":            0.20,
        "momentum":           0.20,
        "valuation":          0.15,
        "earnings_revisions": 0.15,
        "earnings_quality":   0.10,
        "moat":               0.10,
        "capital_allocation": 0.10,
    },
    "V4": {
        "quality":            0.20,
        "momentum":           0.15,
        "earnings_revisions": 0.20,
        "valuation":          0.10,
        "earnings_quality":   0.10,
        "moat":               0.10,
        "capital_allocation": 0.10,
        "risk_adjustment":    0.05,
    },
}
CONFIGS["V5"] = CONFIGS["V4"].copy()   # same weights, different position sizing


# ── Data loading ─────────────────────────────────────────────────────────────

_PRICE_CACHE = ROOT / "data" / "cache" / "ablation_price_cache.json"


def load_universe() -> list[dict]:
    path = ROOT / "data" / "cache" / "scored_universe.json"
    with path.open() as f:
        return json.load(f)["universe"]


def fetch_monthly_returns(tickers: list[str], years: int) -> dict[str, dict[str, float]]:
    """
    Download daily price data for ALL tickers in ONE bulk yfinance call,
    then resample to monthly returns.  One HTTP request = far less rate-limit
    pressure than hundreds of individual calls.
    Returns {ticker: {YYYY-MM: monthly_return}}.
    """
    import yfinance as yf
    import time

    end   = date.today()
    start = end - timedelta(days=365 * years + 45)   # extra buffer for month-end alignment

    print(f"  Bulk-downloading {len(tickers)} tickers (daily → monthly) …", flush=True)

    for attempt in range(5):
        try:
            raw = yf.download(
                tickers, start=str(start), end=str(end),
                interval="1d", progress=False, auto_adjust=True,
                group_by="ticker",
            )
            break
        except Exception as exc:
            wait = 15 * (attempt + 1)
            print(f"  Attempt {attempt+1} failed ({exc}). Waiting {wait}s …", flush=True)
            time.sleep(wait)
    else:
        print("  ERROR: all download attempts failed.")
        return {}

    result: dict[str, dict[str, float]] = {}
    n = len(tickers)
    for t in tickers:
        try:
            if n == 1:
                close = raw["Close"]
            else:
                try:
                    close = raw["Close"][t]
                except Exception:
                    close = raw[t]["Close"]
            # Resample daily → last trading day of month, then pct_change
            monthly = close.dropna().resample("ME").last()
            rets    = monthly.pct_change().dropna()
            if len(rets) >= 6:
                result[t] = {str(d.date())[:7]: float(r) for d, r in rets.items()}
        except Exception:
            pass

    good = sum(1 for v in result.values() if v)
    print(f"  Got monthly returns for {good}/{n} tickers.", flush=True)
    return result


def fetch_benchmark_returns(years: int) -> dict[str, dict[str, float]]:
    """Returns {symbol: {YYYY-MM: monthly_return}} for SPY and QQQ."""
    return fetch_monthly_returns(["SPY", "QQQ"], years)


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation between two lists."""
    n = len(xs)
    if n < 5:
        return 0.0
    def _ranks(vals: list[float]) -> list[float]:
        indexed = sorted(enumerate(vals), key=lambda x: x[1])
        ranks = [0.0] * n
        for rank, (orig_idx, _) in enumerate(indexed):
            ranks[orig_idx] = float(rank)
        return ranks
    rx = _ranks(xs)
    ry = _ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = math.sqrt(
        sum((rx[i] - mx) ** 2 for i in range(n)) *
        sum((ry[i] - my) ** 2 for i in range(n))
    )
    return num / den if den > 1e-12 else 0.0


# ── Composite scoring ─────────────────────────────────────────────────────────

def composite_score(stock: dict, weights: dict[str, float]) -> float:
    """Weighted sum of factor scores. Missing factors treated as neutral 50."""
    total = 0.0
    w_sum = 0.0
    for fac, w in weights.items():
        v = stock.get(fac, 50.0) or 50.0
        total += v * w
        w_sum += w
    return total / w_sum if w_sum > 0 else 50.0


def rank_universe(universe: list[dict], weights: dict[str, float]) -> list[dict]:
    """Return universe sorted by composite score descending."""
    scored = [(composite_score(s, weights), s) for s in universe]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored]


# ── Volatility-based sizing ───────────────────────────────────────────────────

def inv_vol_weights(
    tickers: list[str],
    scores: list[float],
    price_history: dict[str, dict[str, float]],
    max_w: float = 0.35,
) -> list[float]:
    """Inverse-volatility weights with score tilt and max cap."""
    # Compute 12-month annualised vol for each ticker
    vols: list[float | None] = []
    for t in tickers:
        hist = list((price_history.get(t) or {}).values())
        if len(hist) >= 12:
            last12 = hist[-12:]
            mean_ = sum(last12) / len(last12)
            var_ = sum((r - mean_) ** 2 for r in last12) / (len(last12) - 1)
            vols.append(math.sqrt(var_ * 12))
        else:
            vols.append(None)

    valid = [v for v in vols if v is not None and v > 0]
    if not valid:
        n = len(tickers)
        return [1 / n] * n
    med_inv = 1.0 / sorted(valid)[len(valid) // 2]
    inv = [1.0 / v if (v and v > 0) else med_inv for v in vols]

    avg_s = sum(scores) / len(scores) if scores else 50
    if avg_s <= 0:
        avg_s = 50
    tilt = [(s / avg_s) ** 0.30 for s in scores]

    raw = [inv[i] * tilt[i] for i in range(len(tickers))]
    total = sum(raw)
    if total <= 0:
        n = len(tickers)
        return [1 / n] * n
    w = [r / total for r in raw]

    # Clamp and renormalise
    min_w = 1 / (2 * len(tickers))
    for _ in range(10):
        w = [max(min_w, min(max_w, x)) for x in w]
        t = sum(w)
        w = [x / t for x in w]
        if all(min_w - 1e-9 <= x <= max_w + 1e-9 for x in w):
            break

    return w


# ── Portfolio return calculation ──────────────────────────────────────────────

def portfolio_monthly_return(
    tickers: list[str],
    weights: list[float],
    month: str,
    price_history: dict[str, dict[str, float]],
) -> float:
    """Weighted return for one month. Missing tickers held at 0."""
    r = 0.0
    for t, w in zip(tickers, weights):
        ret = (price_history.get(t) or {}).get(month, 0.0)
        r += w * ret
    return r


def benchmark_monthly_return(
    month: str,
    bench_hist: dict[str, dict[str, float]],
    symbol: str = "SPY",
) -> float:
    return (bench_hist.get(symbol) or {}).get(month, 0.0)


# ── Single backtest simulation ────────────────────────────────────────────────

@dataclass
class BacktestResult:
    label: str
    monthly_returns: list[float]
    monthly_turnovers: list[float]
    months: list[str]
    top5_history: list[list[str]]

    @property
    def cagr(self) -> float:
        n = len(self.monthly_returns)
        if n == 0: return 0.0
        total = math.prod(1 + r for r in self.monthly_returns)
        return total ** (12 / n) - 1.0

    @property
    def sharpe(self) -> float:
        rs = self.monthly_returns
        n = len(rs)
        if n < 2: return 0.0
        mean = sum(rs) / n
        std = math.sqrt(sum((r - mean) ** 2 for r in rs) / (n - 1))
        if std == 0: return 0.0
        return (mean - RISK_FREE / 12) / std * math.sqrt(12)

    @property
    def sortino(self) -> float:
        rs = self.monthly_returns
        n = len(rs)
        if n == 0: return 0.0
        mean = sum(rs) / n
        sq_down = sum(min(r, 0) ** 2 for r in rs)
        dd_dev = math.sqrt(sq_down / n) * math.sqrt(12)
        if dd_dev == 0: return 0.0
        return (mean * 12) / dd_dev

    @property
    def max_drawdown(self) -> float:
        peak = eq = 1.0
        mdd = 0.0
        for r in self.monthly_returns:
            eq *= (1 + r)
            if eq > peak: peak = eq
            dd = (eq - peak) / peak
            if dd < mdd: mdd = dd
        return mdd

    @property
    def win_rate(self) -> float:
        n = len(self.monthly_returns)
        if n == 0: return 0.0
        return sum(1 for r in self.monthly_returns if r > 0) / n

    @property
    def avg_turnover(self) -> float:
        t = self.monthly_turnovers
        return sum(t) / len(t) if t else 0.0

    @property
    def total_return(self) -> float:
        return math.prod(1 + r for r in self.monthly_returns) - 1.0

    def hit_rate_vs(self, bench: list[float]) -> float:
        """Fraction of months where this portfolio beats the benchmark."""
        pairs = [(r, b) for r, b in zip(self.monthly_returns, bench)]
        if not pairs: return 0.0
        return sum(1 for r, b in pairs if r > b) / len(pairs)


def run_simulation(
    label: str,
    universe: list[dict],
    weights_cfg: dict[str, float],
    all_months: list[str],
    price_history: dict[str, dict[str, float]],
    sizing: str = "equal",          # "equal" | "invvol" | "invvol25"
    n_pos: int = N_POSITIONS,
    top_pool: int = N_TOP_POOL,
) -> BacktestResult:
    """Run one backtest configuration across all months."""
    ranked    = rank_universe(universe, weights_cfg)
    pool      = ranked[:top_pool]
    top_ticks = [s["ticker"] for s in pool[:n_pos]]
    top_scores = [composite_score(s, weights_cfg) for s in pool[:n_pos]]

    # Compute position weights once (static — same for all months)
    if sizing == "equal":
        pos_weights = [1 / n_pos] * n_pos
    elif sizing == "invvol":
        pos_weights = inv_vol_weights(top_ticks, top_scores, price_history, max_w=0.35)
    elif sizing == "invvol25":
        pos_weights = inv_vol_weights(top_ticks, top_scores, price_history, max_w=0.25)
    else:
        pos_weights = [1 / n_pos] * n_pos

    monthly_rets: list[float] = []
    monthly_to:   list[float] = []
    prev_ticks = []
    prev_w     = []

    for month in all_months:
        gross_r = portfolio_monthly_return(top_ticks, pos_weights, month, price_history)

        # Turnover: first month is 100%, then 0% (static portfolio)
        if not prev_ticks:
            to = 1.0
        else:
            old_map = dict(zip(prev_ticks, prev_w))
            new_map = dict(zip(top_ticks, pos_weights))
            all_t   = set(prev_ticks) | set(top_ticks)
            to      = sum(abs(new_map.get(t, 0) - old_map.get(t, 0)) for t in all_t) / 2.0
        cost   = to * COST_RATE
        net_r  = gross_r - cost

        monthly_rets.append(net_r)
        monthly_to.append(to)
        prev_ticks = top_ticks
        prev_w     = pos_weights

    return BacktestResult(
        label            = label,
        monthly_returns  = monthly_rets,
        monthly_turnovers = monthly_to,
        months           = all_months,
        top5_history     = [top_ticks] * len(all_months),
    )


def run_benchmark(
    label: str,
    symbol: str,
    all_months: list[str],
    bench_hist: dict[str, dict[str, float]],
) -> BacktestResult:
    rets = [(bench_hist.get(symbol) or {}).get(m, 0.0) for m in all_months]
    return BacktestResult(
        label             = label,
        monthly_returns   = rets,
        monthly_turnovers = [0.0] * len(rets),
        months            = all_months,
        top5_history      = [],
    )


def run_eq_top20(
    label: str,
    universe: list[dict],
    all_months: list[str],
    price_history: dict[str, dict[str, float]],
    n: int = 20,
) -> BacktestResult:
    """Equal-weight top-N by V4 composite — naive baseline."""
    # Use V4 weights for ranking but equal position sizing across 20 stocks
    ranked = rank_universe(universe, CONFIGS["V4"])
    pool   = [s["ticker"] for s in ranked[:n]]
    w      = [1 / n] * n
    rets   = [portfolio_monthly_return(pool, w, m, price_history) - (1.0 if i == 0 else 0.0) * COST_RATE
              for i, m in enumerate(all_months)]
    # cleaner implementation:
    monthly_rets = []
    for i, m in enumerate(all_months):
        r = portfolio_monthly_return(pool, w, m, price_history)
        cost = COST_RATE if i == 0 else 0.0
        monthly_rets.append(r - cost)
    return BacktestResult(
        label             = label,
        monthly_returns   = monthly_rets,
        monthly_turnovers = [1.0 if i == 0 else 0.0 for i in range(len(all_months))],
        months            = all_months,
        top5_history      = [pool[:5]] * len(all_months),
    )


# ── Information Coefficient ───────────────────────────────────────────────────

def compute_ic(
    universe: list[dict],
    factor: str,
    price_history: dict[str, dict[str, float]],
    all_months: list[str],
    lookahead: int = 1,
) -> float:
    """
    Rank IC: Spearman correlation between today's factor score and
    next-month actual return, averaged across all months.
    """
    ics: list[float] = []
    for i in range(len(all_months) - lookahead):
        next_month = all_months[i + lookahead]
        fac_vals: list[float] = []
        ret_vals: list[float] = []
        for s in universe:
            t   = s["ticker"]
            ret = (price_history.get(t) or {}).get(next_month)
            if ret is not None:
                fac_vals.append(s.get(factor, 50.0) or 50.0)
                ret_vals.append(ret)
        if len(fac_vals) < 10:
            continue
        ic = _spearman(fac_vals, ret_vals)
        ics.append(ic)
    return sum(ics) / len(ics) if ics else 0.0


def compute_leave_one_out(
    base_weights: dict[str, float],
    removed_factor: str,
) -> dict[str, float]:
    """Remove one factor from weights and redistribute proportionally."""
    remaining = {k: v for k, v in base_weights.items() if k != removed_factor}
    total = sum(remaining.values())
    return {k: v / total for k, v in remaining.items()}


# ── Printing ─────────────────────────────────────────────────────────────────

BAR = "═" * 96
DIV = "─" * 96

def _pct(v: float) -> str:
    return f"{v * 100:+.1f}%"

def _pp(v: float) -> str:
    return f"{v * 100:.1f}%"

def _f2(v: float) -> str:
    return f"{v:.2f}"


def print_results_table(
    results: list[BacktestResult],
    spy_rets: list[float],
    label: str = "",
) -> None:
    print(f"\n{BAR}")
    if label:
        print(f"  {label}")
        print(BAR)
    print(f"  {'Strategy':<30}  {'CAGR':>7}  {'Sharpe':>7}  {'Sortino':>7}  "
          f"{'MaxDD':>7}  {'WinRate':>7}  {'HitVsSPY':>8}  {'Turnover':>8}  Top-5")
    print("  " + DIV)
    spy_bench = spy_rets
    for r in results:
        hit = r.hit_rate_vs(spy_bench)
        tops = ", ".join(r.top5_history[0][:5]) if r.top5_history else "—"
        print(
            f"  {r.label:<30}  {_pp(r.cagr):>7}  {_f2(r.sharpe):>7}  "
            f"{_f2(r.sortino):>7}  {_pp(r.max_drawdown):>7}  {_pp(r.win_rate):>7}  "
            f"{_pp(hit):>8}  {_pp(r.avg_turnover):>8}  {tops}"
        )
    print("  " + DIV)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{BAR}")
    print("  HALAL ALPHA AI — FACTOR ABLATION & VERSION COMPARISON")
    print(f"  Study window: {STUDY_YEARS} years  |  Top-{N_POSITIONS} positions  |  Monthly rebalance")
    print(f"  Transaction cost: {COST_RATE*100:.2f}% per trade  |  Risk-free: {RISK_FREE*100:.1f}%")
    print(BAR)

    print("""
  CRITICAL LIMITATIONS
  ─────────────────────────────────────────────────────────────────────────────
  1. LOOK-AHEAD BIAS:  Factor scores are TODAY's values, not historical.
     Rankings are static across all 60 months — an optimistic simplification.
  2. SURVIVORSHIP BIAS: Only 202 stocks alive today are tested.
     Failed or acquired companies 2021-2026 are excluded, overstating returns.
  3. STATIC PORTFOLIO: Positions never change within a version (zero turnover
     after month 1). This understates real-world turnover and cost drag.
  4. INTERPRETATION LIMIT: These results show cross-sectional predictiveness
     of today's factor values on 5yr historical returns. They are NOT a
     true walk-forward backtest of any version's live factor scores.
  ─────────────────────────────────────────────────────────────────────────────
""")

    # Load universe
    universe = load_universe()
    print(f"  Universe loaded: {len(universe)} tickers")

    # Fetch price data (use cache if available and < 24 hours old)
    tickers = [s["ticker"] for s in universe]
    if _PRICE_CACHE.exists():
        cache_age = (date.today().toordinal() -
                     date.fromisoformat(json.loads(_PRICE_CACHE.read_text()).get("date","2000-01-01")).toordinal())
        if cache_age == 0:
            print("  Using cached price data (fetched today).", flush=True)
            cached = json.loads(_PRICE_CACHE.read_text())
            price_history = cached["data"]
        else:
            price_history = None
    else:
        price_history = None

    if price_history is None:
        all_tickers = tickers + ["SPY", "QQQ"]
        price_history = fetch_monthly_returns(all_tickers, STUDY_YEARS)
        _PRICE_CACHE.write_text(json.dumps({"date": str(date.today()), "data": price_history}))

    bench_hist = {sym: price_history.get(sym, {}) for sym in ["SPY", "QQQ"]}
    # Merge benchmark into price_history for convenience
    price_history.update(bench_hist)

    # Use SPY months as the reference (SPY has full history; missing tickers → 0 that month)
    spy_months = sorted(price_history.get("SPY", {}).keys())
    cutoff = str(date.today().replace(year=date.today().year - STUDY_YEARS))[:7]
    common_months = [m for m in spy_months if m >= cutoff]
    if not common_months:
        print("ERROR: No SPY price data available.")
        return
    print(f"  Reference months (SPY): {common_months[0]} → {common_months[-1]} ({len(common_months)} months)")
    # Report coverage
    covered = sum(1 for t in [s["ticker"] for s in universe]
                  if len([m for m in common_months if price_history.get(t, {}).get(m) is not None]) >= len(common_months) // 2)
    print(f"  Tickers with ≥50% month coverage: {covered}/{len(universe)}")

    spy_rets = [(bench_hist.get("SPY") or {}).get(m, 0.0) for m in common_months]
    qqq_rets = [(bench_hist.get("QQQ") or {}).get(m, 0.0) for m in common_months]

    # ── 1. VERSION COMPARISON ────────────────────────────────────────────────

    print(f"\n{BAR}")
    print("  SECTION 1 — VERSION COMPARISON")
    print(BAR)

    version_results: list[BacktestResult] = []
    version_results.append(run_benchmark("SPY (passive)", "SPY", common_months, bench_hist))
    version_results.append(run_benchmark("QQQ (passive)", "QQQ", common_months, bench_hist))
    version_results.append(run_eq_top20("Equal-Weight Top-20 (baseline)", universe, common_months, price_history))
    version_results.append(run_simulation("V1  (Quality 50% + Mom 50%)",   universe, CONFIGS["V1"], common_months, price_history, sizing="equal"))
    version_results.append(run_simulation("V3  (7 factors, old weights)",   universe, CONFIGS["V3"], common_months, price_history, sizing="equal"))
    version_results.append(run_simulation("V4  (8 factors, equal weight)",  universe, CONFIGS["V4"], common_months, price_history, sizing="equal"))
    version_results.append(run_simulation("V4  (8 factors, inv-vol 35%)",   universe, CONFIGS["V4"], common_months, price_history, sizing="invvol"))
    version_results.append(run_simulation("V5  (8 factors, inv-vol 25%)",   universe, CONFIGS["V5"], common_months, price_history, sizing="invvol25"))

    print_results_table(version_results, spy_rets, "VERSION COMPARISON — 5-Year Performance (2021–2026)")

    # Annual breakdown
    print(f"\n  ANNUAL RETURNS BY VERSION")
    print(f"  {'Strategy':<30}  " + "  ".join(f"{y:>6}" for y in range(date.today().year - STUDY_YEARS, date.today().year)))
    print("  " + DIV)
    year_range = list(range(date.today().year - STUDY_YEARS, date.today().year))
    for r in version_results:
        by_year: dict[int, list[float]] = {}
        for m, ret in zip(r.months, r.monthly_returns):
            yr = int(m[:4])
            by_year.setdefault(yr, []).append(ret)
        year_rets = []
        for yr in year_range:
            rets_y = by_year.get(yr, [])
            ann = (math.prod(1 + x for x in rets_y) - 1) if rets_y else 0.0
            year_rets.append(ann)
        row = "  ".join(f"{_pp(x):>6}" for x in year_rets)
        print(f"  {r.label:<30}  {row}")
    print("  " + DIV)

    # ── 2. SINGLE-FACTOR ANALYSIS ─────────────────────────────────────────────

    print(f"\n{BAR}")
    print("  SECTION 2 — SINGLE-FACTOR PERFORMANCE")
    print("  (Top-5 sorted by one factor only, equal weight, no composite)")
    print(BAR)

    single_results: list[BacktestResult] = [
        run_benchmark("SPY (passive)", "SPY", common_months, bench_hist),
    ]
    for fac in FACTORS:
        r = run_simulation(f"Single: {fac}", universe, {fac: 1.0}, common_months, price_history, sizing="equal")
        single_results.append(r)

    print_results_table(single_results, spy_rets, "SINGLE FACTOR vs SPY")

    # IC table
    print(f"\n  INFORMATION COEFFICIENT (IC) — Spearman rank correlation of factor vs next-month return")
    print(f"  Positive IC = factor rank correctly predicts next-month return direction")
    print(f"\n  {'Factor':<25}  {'IC':>8}  {'|IC|':>8}  Interpretation")
    print("  " + DIV)
    ic_results: list[tuple[str, float]] = []
    for fac in FACTORS:
        ic = compute_ic(universe, fac, price_history, common_months)
        ic_results.append((fac, ic))
    ic_results.sort(key=lambda x: x[1], reverse=True)
    for fac, ic in ic_results:
        interp = "✓ positive predictive" if ic > 0.02 else ("✗ negative" if ic < -0.02 else "~ noise")
        print(f"  {fac:<25}  {ic:>8.4f}  {abs(ic):>8.4f}  {interp}")
    print("  " + DIV)

    # ── 3. LEAVE-ONE-OUT ABLATION ─────────────────────────────────────────────

    print(f"\n{BAR}")
    print("  SECTION 3 — LEAVE-ONE-OUT FACTOR ABLATION")
    print("  (Remove one factor from V4, redistribute weight evenly, equal position sizing)")
    print(BAR)

    base = run_simulation("V4 FULL (baseline)", universe, CONFIGS["V4"], common_months, price_history, sizing="equal")
    ablation_results: list[BacktestResult] = [base]

    for fac in FACTORS:
        if fac not in CONFIGS["V4"]:
            continue
        loo_weights = compute_leave_one_out(CONFIGS["V4"], fac)
        r = run_simulation(f"V4 − {fac}", universe, loo_weights, common_months, price_history, sizing="equal")
        ablation_results.append(r)

    print_results_table(ablation_results, spy_rets, "LEAVE-ONE-OUT: removing each factor from V4")

    # Marginal alpha table
    print(f"\n  MARGINAL ALPHA CONTRIBUTION (removing factor vs V4 full)")
    print(f"  Positive delta CAGR = removing this factor HELPS → factor is HURTING performance")
    print(f"  Negative delta CAGR = removing this factor HURTS → factor is ADDING alpha\n")
    print(f"  {'Factor Removed':<25}  {'ΔCAGR':>8}  {'ΔSharpe':>8}  {'ΔMaxDD':>8}  Verdict")
    print("  " + DIV)
    base_cagr   = base.cagr
    base_sharpe = base.sharpe
    base_mdd    = base.max_drawdown
    for r in ablation_results[1:]:
        removed = r.label.replace("V4 − ", "")
        d_cagr   = r.cagr - base_cagr
        d_sharpe = r.sharpe - base_sharpe
        d_mdd    = r.max_drawdown - base_mdd   # negative = worse drawdown
        if d_cagr > 0.005 or d_sharpe > 0.05:
            verdict = "✗ REMOVE — hurts performance"
        elif d_cagr < -0.005 or d_sharpe < -0.05:
            verdict = "✓ KEEP — adds alpha"
        else:
            verdict = "~ MARGINAL — inconclusive"
        print(f"  {removed:<25}  {_pct(d_cagr):>8}  {d_sharpe:>+8.3f}  {_pct(d_mdd):>8}  {verdict}")
    print("  " + DIV)

    # ── 4. RECOMMENDED CONFIG ─────────────────────────────────────────────────

    print(f"\n{BAR}")
    print("  SECTION 4 — RECOMMENDED CONFIGURATION")
    print(BAR)

    # Identify factors to remove (those that hurt performance in both IC and ablation)
    bad_factors: list[str] = []
    ic_map = dict(ic_results)
    for r in ablation_results[1:]:
        removed = r.label.replace("V4 − ", "")
        d_cagr  = r.cagr - base_cagr
        d_sharpe = r.sharpe - base_sharpe
        ic_val = ic_map.get(removed, 0)
        if (d_cagr > 0.005 or d_sharpe > 0.05) and ic_val < 0.01:
            bad_factors.append(removed)

    keep_factors = [f for f in FACTORS if f in CONFIGS["V4"] and f not in bad_factors]

    print(f"\n  Factors recommended for REMOVAL (hurt performance AND negative IC):")
    if bad_factors:
        for f in bad_factors:
            ic_val = ic_map.get(f, 0)
            print(f"    ✗  {f:<25}  IC={ic_val:+.4f}")
    else:
        print(f"    (none — all factors pass the evidence threshold)")

    print(f"\n  Factors recommended to KEEP:")
    for f in keep_factors:
        ic_val = ic_map.get(f, 0)
        print(f"    ✓  {f:<25}  IC={ic_val:+.4f}")

    if bad_factors:
        # Build recommended weights
        v4_keep = {k: v for k, v in CONFIGS["V4"].items() if k in keep_factors}
        total = sum(v4_keep.values())
        v4_rec = {k: v / total for k, v in v4_keep.items()}

        print(f"\n  Recommended weight configuration (V4 weights renormalised over kept factors):")
        for k, v in sorted(v4_rec.items(), key=lambda x: x[1], reverse=True):
            print(f"    {k:<25}  {v*100:.1f}%")

        rec = run_simulation(
            "RECOMMENDED CONFIG (optimal)", universe, v4_rec, common_months, price_history, sizing="invvol25"
        )
        print(f"\n  Recommended config performance vs V4:")
        print(f"    CAGR:        {_pp(rec.cagr):>8}  vs  {_pp(base.cagr):>8}  (Δ {_pct(rec.cagr - base.cagr)})")
        print(f"    Sharpe:      {rec.sharpe:>8.3f}  vs  {base.sharpe:>8.3f}  (Δ {rec.sharpe-base.sharpe:+.3f})")
        print(f"    Max DD:      {_pp(rec.max_drawdown):>8}  vs  {_pp(base.max_drawdown):>8}")
    else:
        print(f"\n  V4 configuration is optimal — no factors meet the removal threshold.")

    # ── 5. SUMMARY TABLE ──────────────────────────────────────────────────────

    print(f"\n{BAR}")
    print("  SECTION 5 — FINAL SUMMARY")
    print(BAR)
    print(f"""
  Factor quality hierarchy (by IC + ablation evidence):
  {'Rank':<5}  {'Factor':<25}  {'IC':>8}  {'ΔCAGR (remove)':>16}  {'ΔSharpe':>8}  Decision
  {DIV}""")
    decision_rows = []
    for r in ablation_results[1:]:
        removed = r.label.replace("V4 − ", "")
        d_cagr   = r.cagr - base_cagr
        d_sharpe = r.sharpe - base_sharpe
        ic_val   = ic_map.get(removed, 0)
        decision_rows.append((removed, ic_val, d_cagr, d_sharpe))
    # Sort by IC desc then by -delta_cagr (negative delta = hurts when removed = good factor)
    decision_rows.sort(key=lambda x: (x[1], -x[2]), reverse=True)
    for rank, (fac, ic, d_cagr, d_sharpe) in enumerate(decision_rows, 1):
        if d_cagr < -0.005 and ic > 0.02:
            dec = "✓✓ Strong keep"
        elif d_cagr < 0 or ic > 0:
            dec = "✓  Keep"
        elif d_cagr > 0.005 or ic < -0.01:
            dec = "✗  Remove"
        else:
            dec = "~  Marginal"
        print(f"  {rank:<5}  {fac:<25}  {ic:>8.4f}  {_pct(d_cagr):>16}  {d_sharpe:>+8.3f}  {dec}")
    print(f"  {DIV}")
    print(f"""
  Warnings (always present):
  • Results use TODAY's factor scores → look-ahead bias inflates all returns
  • Only today's 202-stock universe → survivorship bias
  • True V1/V3/V4 walk-forward backtests require point-in-time fundamental data
  • IC values over 60 months on 202 stocks have low statistical power (SE ≈ 0.13)
  • Do NOT remove factors based solely on this study without additional evidence
""")
    print(BAR)


if __name__ == "__main__":
    main()
