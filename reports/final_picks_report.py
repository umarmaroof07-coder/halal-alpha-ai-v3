"""
Final Picks Integrity Report — V6 thirteen-gate institutional validation system.

Thirteen gates must pass (or produce only warnings) before Top 5 is output.
Critical failures (BLOCK severity) prevent final picks from being shown.

Gates
─────
  1.  Data Quality          completeness, freshness, duplicates, obvious bad values
  2.  Factor Integrity      score transparency, per-signal contribution, no dominance
  3.  Accounting Quality    SBC ratio, base-period artifacts, merger growth, leverage
  4.  AI Research           Top-20 coverage, confidence, source availability
  5.  Shariah               automated screen + manual CSV + age check for Top 5
  6.  Recommendation Guard  safe_recommendations() is sole Top-5 source
  7.  Final Ranking         explainable Top-20 table before Top-5 output
  8.  Stress Test           historical resilience in 2008/2020/2022/2023 crises
  9.  Ranking Stability     bootstrap 3 independent runs, Top-5 overlap ≥80%
 10.  Red Flags             structural issues (SBC, leverage, AI flags, distortions)
 11.  Model Confidence      per-stock confidence score (Data Quality+AI+Stability+Coverage+Stress)
 12.  Sector Concentration  max 2 per industry, max 35% per sector, max 25% single position
 13.  Walk-Forward Check    snapshot availability check for true walk-forward backtest

Severity
────────
  BLOCK  — prevents Top-5 output; must be resolved before picks are shown
  WARN   — shown prominently; does not prevent output
  OK     — silent pass
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

_ROOT        = Path(__file__).parent.parent
_CACHE_DIR   = _ROOT / "data" / "cache"
_MANUAL_DIR  = _ROOT / "data" / "manual"

SCORED_UNIVERSE = _CACHE_DIR / "scored_universe.json"
SCREENED_CSV    = _CACHE_DIR / "universe_screened.csv"
SHARIAH_CSV     = _MANUAL_DIR / "shariah_verification.csv"

# ── Thresholds ───────────────────────────────────────────────────────────────
_STALE_MINUTES          = 30      # data freshness block threshold
_SBC_WARN_RATIO         = 0.20    # SBC / FCF > 20% → warn
_MERGER_REV_GROWTH      = 0.50    # revenue growth > 50% → possible merger warn
_HIGH_LEVERAGE_EQ_RATIO = 0.40    # equity_ratio < 40% (D/E > 1.5×) → warn
_AI_LOW_CONFIDENCE      = 0.20    # AI confidence < 0.20 → warn
_TOP_N                  = 20      # gate 2–4 cover the pre-AI Top 20


# ---------------------------------------------------------------------------
# Issue model
# ---------------------------------------------------------------------------

class Sev(str, Enum):
    BLOCK = "BLOCK"
    WARN  = "WARN"
    OK    = "OK"


@dataclass
class Issue:
    severity: Sev
    gate:     int            # 1–13
    ticker:   str | None     # None for universe-level issues
    message:  str


@dataclass
class FinalPicksReport:
    issues:   list[Issue]  = field(default_factory=list)
    top20:    list[dict]   = field(default_factory=list)
    top5:     list[dict]   = field(default_factory=list)
    blocked:  bool         = False
    universe_size: int     = 0
    generated_at:  str     = ""
    ranking_stability_pct: float = 100.0  # bootstrap overlap %

    # ── Derived helpers ──────────────────────────────────────────────────────
    def blocks(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == Sev.BLOCK]

    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == Sev.WARN]

    def blocks_for(self, gate: int) -> list[Issue]:
        return [i for i in self.blocks() if i.gate == gate]

    def warnings_for(self, gate: int) -> list[Issue]:
        return [i for i in self.warnings() if i.gate == gate]


def _issue(sev: Sev, gate: int, ticker: str | None, msg: str) -> Issue:
    return Issue(severity=sev, gate=gate, ticker=ticker, message=msg)

def _block(gate: int, ticker: str | None, msg: str) -> Issue:
    return _issue(Sev.BLOCK, gate, ticker, msg)

def _warn(gate: int, ticker: str | None, msg: str) -> Issue:
    return _issue(Sev.WARN, gate, ticker, msg)


# ---------------------------------------------------------------------------
# Gate 1 — Data Quality
# ---------------------------------------------------------------------------

def _gate1_data_quality(universe: list[dict], generated_at: str) -> list[Issue]:
    issues: list[Issue] = []

    # Freshness — use file mtime (avoids timezone confusion with naive timestamps)
    try:
        mtime = SCORED_UNIVERSE.stat().st_mtime
        age   = datetime.now() - datetime.fromtimestamp(mtime)
        if age > timedelta(minutes=_STALE_MINUTES):
            issues.append(_block(1, None,
                f"scored_universe.json is {int(age.total_seconds()/60)} min old "
                f"(limit {_STALE_MINUTES} min) — run --refresh-data first"))
    except Exception:
        issues.append(_block(1, None, "Cannot stat scored_universe.json for freshness check"))

    # Duplicates
    seen: set[str] = set()
    for item in universe:
        t = item.get("ticker", "")
        if t in seen:
            issues.append(_block(1, t, f"Duplicate ticker {t} in scored_universe.json"))
        seen.add(t)

    # Per-ticker completeness — only flag if in Top 20 (others are lower priority)
    top20_tickers = {item["ticker"] for item in universe[:20]}
    for item in universe:
        t = item.get("ticker", "")
        in_top20 = t in top20_tickers

        price = item.get("price")
        mkt   = item.get("mkt_cap")
        vol   = item.get("avg_volume")
        comp  = item.get("composite")

        if in_top20:
            if not price or price <= 0:
                issues.append(_block(1, t, "Price missing or zero"))
            if not mkt or mkt <= 0:
                issues.append(_block(1, t, "Market cap missing or zero"))
            if not vol or vol <= 0:
                issues.append(_block(1, t, "Avg volume missing or zero"))
            if comp is None:
                issues.append(_block(1, t, "Composite score missing"))
        else:
            # Outside Top 20 — only warn for missing composite (affects ranking)
            if comp is None:
                issues.append(_warn(1, t, "Composite score missing (outside Top 20)"))

    return issues


# ---------------------------------------------------------------------------
# Gate 2 — Factor Integrity (Top 20)
# ---------------------------------------------------------------------------

def _gate2_factor_integrity(top20: list[dict]) -> list[Issue]:
    """
    Check per-signal contributions for every Top-20 stock.
    Warns if any quality sub-signal drives > 25% of the quality raw score
    (which would indicate the iterative cap failed or new data slipped through).
    """
    issues: list[Issue] = []

    for item in top20:
        t  = item["ticker"]
        qd = item.get("quality_detail") or {}
        contribs = qd.get("contributions") or {}

        if not contribs:
            issues.append(_warn(2, t, "No quality contribution breakdown available — re-run --refresh-data"))
            continue

        for signal, pct in contribs.items():
            if pct > 25.5:   # 0.5pp tolerance for floating point
                issues.append(_warn(2, t,
                    f"Quality sub-signal '{signal}' contributes {pct:.1f}% "
                    f"(cap is 25%) — possible cap convergence issue"))

        # Warn if fewer than 4 signals (low confidence raw score)
        n_sigs = len(qd.get("signals_used") or [])
        if n_sigs < 4:
            issues.append(_warn(2, t,
                f"Only {n_sigs} quality signals available (need ≥4 for reliable score)"))

    return issues


# ---------------------------------------------------------------------------
# Gate 3 — Accounting Quality (Top 20)
# ---------------------------------------------------------------------------

def _gate3_accounting(top20: list[dict]) -> list[Issue]:
    issues: list[Issue] = []

    for item in top20:
        t   = item["ticker"]
        qd  = item.get("quality_detail") or {}
        sbc = item.get("sbc_data") or {}

        # SBC / FCF ratio + penalty tier
        ratio = sbc.get("sbc_fcf_ratio", 0)
        if ratio > _SBC_WARN_RATIO:
            adj  = sbc.get("adj_fcf", 0)
            tier = "large" if ratio > 0.50 else ("medium" if ratio > 0.30 else "small")
            eqd  = item.get("earnings_quality_detail") or {}
            pen  = eqd.get("sbc_penalty", 0)
            issues.append(_warn(3, t,
                f"SBC is {ratio*100:.0f}% of reported FCF "
                f"(SBC-adj FCF = ${adj/1e9:.2f}B) — {tier} penalty applied "
                f"(−{pen:.0%} to earnings quality raw score)"))

        # Growth exclusions — base-period artifacts
        for name, label in [
            ("earnings_growth", "Earnings growth"),
            ("fcf_growth",      "FCF growth"),
        ]:
            sig = qd.get(name) or {}
            flag = sig.get("flag", "ok")
            raw  = sig.get("raw")
            if flag == "prior_negative":
                raw_str = f"{raw*100:.0f}%" if raw is not None else "n/a"
                issues.append(_warn(3, t,
                    f"{label} excluded — prior year was negative (raw={raw_str}). "
                    f"Score does not reflect this apparent improvement."))
            elif flag == "prior_too_small":
                raw_str = f"{raw:.0f}x" if raw is not None else "n/a"
                issues.append(_warn(3, t,
                    f"{label} excluded — prior-year base < 5% of current "
                    f"(raw growth={raw_str}). Base-period artifact."))

        # Revenue growth — possible merger/acquisition
        rev_sig = qd.get("revenue_growth") or {}
        rev_val = rev_sig.get("value")
        if rev_val is not None and rev_val > _MERGER_REV_GROWTH:
            issues.append(_warn(3, t,
                f"Revenue growth {rev_val*100:.0f}% — may include "
                f"merger/acquisition revenues (not organic growth)"))

        # High leverage
        eq_ratio = qd.get("equity_ratio")
        if eq_ratio is not None and eq_ratio < _HIGH_LEVERAGE_EQ_RATIO:
            de_approx = (1 - eq_ratio) / eq_ratio if eq_ratio > 0 else float("inf")
            issues.append(_warn(3, t,
                f"High leverage — equity ratio {eq_ratio*100:.0f}% "
                f"(≈ D/E {de_approx:.1f}×)"))

        # Negative FCF margin
        fcf_margin = qd.get("fcf_margin")
        if fcf_margin is not None and fcf_margin < 0:
            issues.append(_warn(3, t,
                f"Negative FCF margin ({fcf_margin*100:.1f}%) — "
                f"company is cash-consuming at current scale"))

        # Accounting distortion flags (EQ 2.0)
        eqd = item.get("earnings_quality_detail") or {}
        for flag in (eqd.get("distortion_flags") or []):
            issues.append(_warn(3, t, f"Distortion detected: {flag}"))

        # Capital allocation warnings
        cad = item.get("capital_allocation_detail") or {}
        for w in (cad.get("warnings") or []):
            issues.append(_warn(3, t, f"Capital allocation: {w}"))

        # Risk warnings
        rd = item.get("risk_detail") or {}
        for w in (rd.get("warnings") or []):
            rl = rd.get("risk_label", "")
            if rl == "High":
                issues.append(_warn(3, t, f"Risk[{rl}]: {w}"))

    return issues


# ---------------------------------------------------------------------------
# Gate 4 — AI Research (Top 20)
# ---------------------------------------------------------------------------

def _gate4_ai_research(top20: list[dict]) -> list[Issue]:
    issues: list[Issue] = []

    for item in top20:
        t   = item["ticker"]
        ai  = item.get("ai_research", 50.0)
        aid = item.get("ai_detail") or {}

        if not aid:
            # No AI detail at all — fell back to neutral
            if abs(ai - 50.0) < 0.01:
                issues.append(_warn(4, t,
                    "AI score = neutral 50 (no source text available — "
                    "EDGAR filing and company profile both missing)"))
            else:
                issues.append(_warn(4, t,
                    f"AI score = {ai:.1f} but no detail record found — "
                    f"audit trail missing"))
            continue

        conf = aid.get("confidence", 0)
        if conf < _AI_LOW_CONFIDENCE:
            issues.append(_warn(4, t,
                f"AI confidence = {conf:.2f} (below {_AI_LOW_CONFIDENCE}) — "
                f"score based on limited source text"))

        # Neutral score with low confidence = effectively no signal
        if abs(ai - 50.0) < 0.5 and conf < _AI_LOW_CONFIDENCE:
            issues.append(_warn(4, t,
                "AI score is effectively neutral — "
                "no reliable qualitative signal for this ticker"))

        # Flag heavy red-flag count
        n_red = len(aid.get("all_red_flags") or [])
        n_pos = len(aid.get("all_positives") or [])
        if n_red > n_pos and n_red >= 5:
            issues.append(_warn(4, t,
                f"AI found {n_red} red flags vs {n_pos} positives — "
                f"review AI detail before investing"))

    return issues


# ---------------------------------------------------------------------------
# Gate 5 — Shariah
# ---------------------------------------------------------------------------

_SHARIAH_WARN_DAYS  = 30   # Phase 9: verification older than this → warn
_SHARIAH_BLOCK_DAYS = 90   # Phase 9: verification older than this → block


def _gate5_shariah(top5: list[dict]) -> list[Issue]:
    issues: list[Issue] = []

    # Load manual verification CSV (columns: ticker, verified_status, source, date, notes)
    manual: dict[str, dict] = {}
    if SHARIAH_CSV.exists():
        with SHARIAH_CSV.open() as f:
            for row in csv.DictReader(f):
                t = row.get("ticker", "").strip().upper()
                s = row.get("verified_status", "").strip().lower()
                # Support both column names: "date" and "last_checked"
                d = (row.get("last_checked") or row.get("date") or "").strip()
                if t:
                    manual[t] = {"status": s, "date": d}
    else:
        issues.append(_block(5, None,
            f"{SHARIAH_CSV.name} not found — create it with manual Shariah verifications"))

    today = datetime.now().date()

    for item in top5:
        t = item["ticker"]
        auto = item.get("shariah_status", "unknown")

        if auto == "non_compliant":
            issues.append(_block(5, t,
                "Automated Shariah screen = non_compliant — "
                "must not appear in BUY NOW"))
        elif auto == "unknown":
            issues.append(_block(5, t,
                "Shariah status unknown — excluded by system rules"))

        entry = manual.get(t)
        if not entry:
            issues.append(_block(5, t,
                f"Not found in {SHARIAH_CSV.name} — "
                f"add row: {t},compliant,<source>,<date>,<notes>"))
            continue

        verified = entry.get("status", "")
        if verified != "compliant":
            issues.append(_block(5, t,
                f"Manual verification status = '{verified}' (must be 'compliant')"))

        # V5 Phase 9 — Shariah age check
        date_str = entry.get("date", "")
        if date_str:
            try:
                verified_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                age_days = (today - verified_date).days
                if age_days > _SHARIAH_BLOCK_DAYS:
                    issues.append(_block(5, t,
                        f"Shariah verification is {age_days}d old (>{_SHARIAH_BLOCK_DAYS}d limit) — "
                        f"re-verify on Zoya/Islamicly before investing"))
                elif age_days > _SHARIAH_WARN_DAYS:
                    issues.append(_warn(5, t,
                        f"Shariah verification is {age_days}d old (>{_SHARIAH_WARN_DAYS}d) — "
                        f"consider refreshing"))
            except ValueError:
                issues.append(_warn(5, t,
                    f"Shariah verification date '{date_str}' could not be parsed (expected YYYY-MM-DD)"))
        else:
            issues.append(_warn(5, t, "Shariah verification has no date — cannot check staleness"))

    return issues


# ---------------------------------------------------------------------------
# Gate 8 — Stress Test  (Top 5)
# ---------------------------------------------------------------------------

def _gate8_stress_test(top5: list[dict]) -> tuple[list[Issue], dict[str, Any]]:
    """Fetch historical crisis performance for Top-5 tickers."""
    issues: list[Issue] = []
    resilience_map: dict[str, Any] = {}

    try:
        from factors.stress_test import compute_stress_test
    except ImportError:
        issues.append(_warn(8, None, "stress_test module unavailable — skipping"))
        return issues, resilience_map

    for item in top5:
        t = item["ticker"]
        try:
            result = compute_stress_test(t)
            resilience_map[t] = result.to_dict()
            if result.resilience_score < 30:
                issues.append(_warn(8, t,
                    f"Low resilience score ({result.resilience_score:.0f}/100) — "
                    f"high historical drawdowns in crisis periods"))
        except Exception as exc:
            issues.append(_warn(8, t, f"Stress test failed: {exc}"))

    return issues, resilience_map


# ---------------------------------------------------------------------------
# Gate 9 — Ranking Stability  (Phase 5 multi-run validation)
# ---------------------------------------------------------------------------

def _gate9_ranking_stability(universe: list[dict]) -> tuple[list[Issue], float]:
    """
    Bootstrap 3 independent composite runs with ±1% score noise.
    Top-5 overlap < 80% → MODEL_INSTABILITY warning.
    Returns (issues, overlap_pct).
    """
    issues: list[Issue] = []
    import random

    top5_tickers = set()
    all_runs: list[set[str]] = []

    # Use the pre-scored composites; add small noise to simulate re-scoring
    for run_idx in range(3):
        rng = random.Random(run_idx * 42)
        noisy = []
        for s in universe:
            noise = rng.uniform(-1.0, 1.0)   # ±1 composite point
            noisy.append((s["ticker"], (s.get("composite") or 0) + noise))

        noisy.sort(key=lambda x: x[1], reverse=True)
        top5_set = {t for t, _ in noisy[:5]}
        all_runs.append(top5_set)
        if run_idx == 0:
            top5_tickers = top5_set

    # Compute average pairwise overlap
    pairs = [(all_runs[0], all_runs[1]), (all_runs[0], all_runs[2]), (all_runs[1], all_runs[2])]
    overlaps = [len(a & b) / 5.0 for a, b in pairs]
    avg_overlap = sum(overlaps) / len(overlaps)
    pct = avg_overlap * 100.0

    if pct < 80:
        issues.append(_warn(9, None,
            f"MODEL_INSTABILITY: Top-5 bootstrap overlap = {pct:.0f}% (threshold 80%) — "
            f"scores are sensitive to small perturbations; review factor weights"))
    return issues, pct


# ---------------------------------------------------------------------------
# Gate 10 — Red Flags  (Top 5)
# ---------------------------------------------------------------------------

def _gate10_red_flags(top5: list[dict]) -> list[Issue]:
    issues: list[Issue] = []
    for item in top5:
        t  = item["ticker"]
        rf = item.get("red_flags") or {}
        label = rf.get("label", "")
        score = rf.get("red_flag_score", 100)
        flags = rf.get("flags") or []
        if label == "Red Flag":
            issues.append(_block(10, t,
                f"Red flag score {score:.0f}/100 — critical structural issues: "
                f"{'; '.join(flags[:2])}"))
        elif label == "Warning":
            issues.append(_warn(10, t,
                f"Red flag score {score:.0f}/100 — structural warnings: "
                f"{'; '.join(flags[:2])}"))
    return issues


# ---------------------------------------------------------------------------
# Top-5 derivation (mirrors safe_recommendations logic)
# ---------------------------------------------------------------------------

def _derive_top5_and_top20(universe: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Reproduce the portfolio construction pipeline to get Top-5 BUY NOW
    and Top-20 candidates. Mirrors cmd_recommend() logic exactly.
    """
    from config.settings import (
        MAX_STOCK_PRICE, PORTFOLIO_MIN_MARKET_CAP,
        PORTFOLIO_MIN_AVG_VOLUME, CONVICTION_WEIGHTS, CONVICTION_DOLLARS,
    )
    from portfolio.recommendation_guard import safe_recommendations
    from portfolio.constructor import build_portfolio
    from portfolio.constraints import check_constraints
    from factors.composite import FactorScores

    # Build FactorScores list
    all_scores = []
    for s in universe:
        all_scores.append(FactorScores(
            ticker             = s["ticker"],
            quality            = s.get("quality", 50.0) or 50.0,
            momentum           = s.get("momentum", 50.0) or 50.0,
            valuation          = s.get("valuation", 50.0) or 50.0,
            earnings_revisions = s.get("earnings_revisions", 50.0) or 50.0,
            earnings_quality   = s.get("earnings_quality", 50.0) or 50.0,
            moat               = s.get("moat", 50.0) or 50.0,
            capital_allocation = s.get("capital_allocation", 50.0) or 50.0,
            risk_adjustment    = s.get("risk_adjustment", 50.0) or 50.0,
            ai_research        = s.get("ai_research", 50.0) or 50.0,
            composite          = s.get("composite", 50.0) or 50.0,
        ))

    all_scores.sort(key=lambda s: s.composite, reverse=True)

    prices           = {s["ticker"]: float(s.get("price") or 0.0) for s in universe}
    shariah_statuses = {s["ticker"]: s.get("shariah_status", "unknown") for s in universe}
    univ_map         = {s["ticker"]: s for s in universe}

    constraint_results = {}
    for fs in all_scores:
        t   = fs.ticker
        s   = univ_map.get(t, {})
        constraint_results[t] = check_constraints(
            ticker=t,
            price=s.get("price"),
            market_cap=s.get("mkt_cap"),
            avg_volume=s.get("avg_volume"),
            shariah_status=s.get("shariah_status", "unknown"),
        )

    portfolio = build_portfolio(all_scores, constraint_results, prices)
    recs = safe_recommendations(
        portfolio_result=portfolio,
        all_scores=all_scores,
        constraint_results=constraint_results,
        prices=prices,
        shariah_statuses=shariah_statuses,
    )

    buy_now_tickers = {r.ticker for r in recs if r.action == "BUY NOW"}
    buy_now_ranked  = [r for r in recs if r.action == "BUY NOW"]

    # Enrich top5 with rank + dollar_amount from portfolio
    rank_map   = {r.ticker: r.rank          for r in buy_now_ranked}
    dollar_map = {r.ticker: r.dollar_amount for r in buy_now_ranked}
    weight_map = {r.ticker: r.conviction_weight for r in buy_now_ranked}

    top5 = []
    for t in sorted(buy_now_tickers, key=lambda x: rank_map.get(x, 99)):
        s = dict(univ_map.get(t, {}))
        s["rank"]             = rank_map.get(t, "?")
        s["dollar_amount"]    = dollar_map.get(t, 0)
        s["conviction_weight"]= weight_map.get(t, 0)
        top5.append(s)

    # Top 20 by composite (pre-constraint)
    top20 = [univ_map[fs.ticker] for fs in all_scores[:_TOP_N] if fs.ticker in univ_map]

    return top5, top20


# ---------------------------------------------------------------------------
# Gate 11 — Model Confidence 2.0
# ---------------------------------------------------------------------------

_CONFIDENCE_WARN_THRESHOLD = 50.0   # below this → warn


def _gate11_model_confidence(top5: list[dict]) -> list[Issue]:
    issues: list[Issue] = []
    from factors.model_confidence import compute_model_confidence

    for item in top5:
        t   = item["ticker"]
        dq  = item.get("factor_stability") or {}   # V5 stored as factor_stability
        aid = item.get("ai_detail") or {}
        fst = item.get("factor_stability") or {}
        st  = item.get("stress_test") or {}

        # Data quality proxy from coverage of fields
        dq_fields = ["quality", "momentum", "valuation", "earnings_revisions",
                     "earnings_quality", "moat", "capital_allocation", "risk_adjustment"]
        populated = sum(1 for f in dq_fields if item.get(f) is not None)
        dq_score  = populated / len(dq_fields) * 100.0

        conf = compute_model_confidence(
            ticker               = t,
            data_quality_score   = dq_score,
            num_analysts         = None,
            ai_confidence        = aid.get("confidence"),
            factor_stability_score = fst.get("stability_score"),
            n_stress_periods     = st.get("n_crises_with_data"),
        )

        # Store confidence score back on the item for display
        item["model_confidence"] = conf.to_dict()

        if conf.overall_confidence_score < _CONFIDENCE_WARN_THRESHOLD:
            issues.append(_warn(11, t,
                f"Model confidence {conf.overall_confidence_score:.0f}/100 ({conf.label}) — "
                "review data quality and AI coverage before investing"))

    return issues


# ---------------------------------------------------------------------------
# Gate 12 — Sector / Industry Concentration
# ---------------------------------------------------------------------------

def _gate12_concentration(top5: list[dict], universe: list[dict]) -> list[Issue]:
    from portfolio.concentration import check_concentration

    tickers = [s["ticker"] for s in top5]
    weight_map   = {s["ticker"]: s.get("conviction_weight", 0.20) for s in top5}
    sector_map   = {s["ticker"]: s.get("sector", "Unknown") for s in top5}
    industry_map = {s["ticker"]: s.get("industry", "Unknown") for s in top5}

    result = check_concentration(tickers, weight_map, sector_map, industry_map)
    issues: list[Issue] = []
    for ci in result.issues:
        sev = Sev.BLOCK if ci.severity == "BLOCK" else Sev.WARN
        issues.append(Issue(severity=sev, gate=12, ticker=None, message=ci.message))
    return issues


# ---------------------------------------------------------------------------
# Gate 13 — Walk-Forward Snapshot Availability
# ---------------------------------------------------------------------------

def _gate13_walk_forward() -> list[Issue]:
    from data_layer.snapshot import get_snapshot_summary

    summary = get_snapshot_summary()
    n = summary["count"]
    issues: list[Issue] = []

    if n == 0:
        issues.append(_warn(13, None,
            "No point-in-time snapshots in data/history/ yet. "
            "Run --refresh-data daily/weekly to accumulate history for the "
            "true walk-forward backtest (--walk-forward)."))
    elif n < 4:
        issues.append(_warn(13, None,
            f"Only {n} point-in-time snapshot(s) available ({summary['first_date']} → "
            f"{summary['last_date']}). Walk-forward backtest needs ≥8 quarterly periods. "
            "Keep running --refresh-data to accumulate history."))
    return issues


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_final_picks() -> FinalPicksReport:
    report = FinalPicksReport()

    # ── Load universe ─────────────────────────────────────────────────────────
    if not SCORED_UNIVERSE.exists():
        report.issues.append(_block(1, None,
            "scored_universe.json not found — run --refresh-data first"))
        report.blocked = True
        return report

    try:
        raw = json.loads(SCORED_UNIVERSE.read_text())
    except Exception as exc:
        report.issues.append(_block(1, None, f"Cannot parse scored_universe.json: {exc}"))
        report.blocked = True
        return report

    universe = raw.get("universe", [])
    generated_at = raw.get("generated_at", "")
    report.universe_size = len(universe)
    report.generated_at  = generated_at

    # Sort by composite descending (file should already be sorted)
    universe.sort(key=lambda s: s.get("composite", 0) or 0, reverse=True)

    # ── Gate 1 — Data Quality ────────────────────────────────────────────────
    report.issues.extend(_gate1_data_quality(universe, generated_at))

    # ── Derive Top 20 and Top 5 ───────────────────────────────────────────────
    try:
        top5, top20 = _derive_top5_and_top20(universe)
    except Exception as exc:
        report.issues.append(_block(6, None, f"Portfolio construction failed: {exc}"))
        report.blocked = len(report.blocks()) > 0
        return report

    report.top5  = top5
    report.top20 = top20

    # ── Gate 2 — Factor Integrity ────────────────────────────────────────────
    report.issues.extend(_gate2_factor_integrity(top20))

    # ── Gate 3 — Accounting Quality ──────────────────────────────────────────
    report.issues.extend(_gate3_accounting(top20))

    # ── Gate 4 — AI Research ─────────────────────────────────────────────────
    report.issues.extend(_gate4_ai_research(top20))

    # ── Gate 5 — Shariah ─────────────────────────────────────────────────────
    report.issues.extend(_gate5_shariah(top5))

    # ── Gate 6 — Recommendation guard is enforced structurally ───────────────
    # safe_recommendations() is the sole source — no additional check needed.

    # ── Gate 7 — Final Ranking (structural; produces no issues) ──────────────

    # ── Gate 8 — Stress Test ─────────────────────────────────────────────────
    g8_issues, resilience_map = _gate8_stress_test(top5)
    report.issues.extend(g8_issues)
    # Attach resilience data to top5 entries
    for s in report.top5:
        if s["ticker"] in resilience_map:
            s["stress_test"] = resilience_map[s["ticker"]]

    # ── Gate 9 — Ranking Stability ────────────────────────────────────────────
    g9_issues, overlap_pct = _gate9_ranking_stability(universe)
    report.issues.extend(g9_issues)
    report.ranking_stability_pct = overlap_pct

    # ── Gate 10 — Red Flags ───────────────────────────────────────────────────
    report.issues.extend(_gate10_red_flags(top5))

    # ── Gate 11 — Model Confidence 2.0 ────────────────────────────────────────
    report.issues.extend(_gate11_model_confidence(top5))

    # ── Gate 12 — Sector / Industry Concentration ─────────────────────────────
    report.issues.extend(_gate12_concentration(top5, universe))

    # ── Gate 13 — Walk-Forward Snapshot Availability ──────────────────────────
    report.issues.extend(_gate13_walk_forward())

    report.blocked = len(report.blocks()) > 0
    return report


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

_BAR = "═" * 72
_DIV = "─" * 72

def _sec(title: str) -> None:
    print(f"\n{_BAR}\n  {title}\n{_BAR}")

def _row(icon: str, msg: str, indent: int = 0) -> None:
    print(f"{'  ' * indent}{icon}  {msg}")


def print_final_report(report: FinalPicksReport) -> None:
    # ── Header ───────────────────────────────────────────────────────────────
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _sec(f"HALAL ALPHA AI V6 — FINAL PICKS INTEGRITY REPORT\n  {now}")
    print(f"\n  Universe: {report.universe_size} compliant tickers")
    try:
        mtime = SCORED_UNIVERSE.stat().st_mtime
        age   = datetime.now() - datetime.fromtimestamp(mtime)
        print(f"  Data age: {int(age.total_seconds()/60)} min")
    except Exception:
        print(f"  Data age: unknown")

    # ── Gate status summary ───────────────────────────────────────────────────
    _sec("GATE STATUS SUMMARY")
    gate_names = {
        1:  "Data Quality",
        2:  "Factor Integrity",
        3:  "Accounting Quality",
        4:  "AI Research",
        5:  "Shariah Verification",
        6:  "Recommendation Guard",
        7:  "Final Ranking",
        8:  "Stress Test",
        9:  "Ranking Stability",
        10: "Red Flags",
        11: "Model Confidence",
        12: "Sector Concentration",
        13: "Walk-Forward Check",
    }
    for g, name in gate_names.items():
        blocks = report.blocks_for(g)
        warns  = report.warnings_for(g)
        if g == 6:
            print(f"  ✓  Gate {g:>2}: {name}  —  safe_recommendations() enforced")
        elif g == 7:
            print(f"  ✓  Gate {g:>2}: {name}  —  Top-20 table generated below")
        elif g == 9:
            pct = report.ranking_stability_pct
            if blocks:
                print(f"  ✗  Gate {g:>2}: {name}  —  {len(blocks)} BLOCK(S)")
            elif warns:
                print(f"  ⚠  Gate {g:>2}: {name}  —  bootstrap overlap {pct:.0f}% (threshold 80%)")
            else:
                print(f"  ✓  Gate {g:>2}: {name}  —  bootstrap overlap {pct:.0f}% ✓")
        elif blocks:
            print(f"  ✗  Gate {g:>2}: {name}  —  {len(blocks)} BLOCK(S)")
        elif warns:
            print(f"  ⚠  Gate {g:>2}: {name}  —  {len(warns)} warning(s)")
        else:
            print(f"  ✓  Gate {g:>2}: {name}  —  clean")

    # ── Gate 1: Data Quality detail ──────────────────────────────────────────
    g1 = [i for i in report.issues if i.gate == 1]
    if g1:
        _sec("GATE 1 — DATA QUALITY ISSUES")
        for i in g1:
            icon = "✗" if i.severity == Sev.BLOCK else "⚠"
            loc  = f"[{i.ticker}] " if i.ticker else ""
            print(f"  {icon}  {loc}{i.message}")

    # ── Gate 2: Factor Integrity ─────────────────────────────────────────────
    _sec("GATE 2 — FACTOR INTEGRITY  (Top 20 quality signal breakdown)")
    print(f"  {'Ticker':<6}  {'Q':>5}  {'ROIC%':>5}  {'OpM%':>5}  {'NIM%':>5}  "
          f"{'FCFM%':>5}  {'EqR%':>5}  {'RevG':>5}  {'NIG':>4}  {'FCFG':>4}  {'Sigs':>4}  Flags")
    print("  " + _DIV)
    for item in report.top20:
        t  = item["ticker"]
        qd = item.get("quality_detail") or {}

        def _pct(key: str) -> str:
            v = qd.get(key)
            return f"{v*100:>5.1f}" if v is not None else "  n/a"

        def _gsig(key: str) -> str:
            sig = qd.get(key) or {}
            v   = sig.get("value")
            flg = sig.get("flag", "ok")
            if v is None:
                short = {"prior_negative": " neg", "prior_too_small": "tiny",
                         "missing": "miss"}.get(flg, flg[:4])
                return f"{short:>4}"
            return f"{v*100:>4.0f}"

        contribs    = qd.get("contributions") or {}
        max_contrib = max(contribs.values(), default=0)
        n_sigs      = len(qd.get("signals_used") or [])
        flags       = []
        for sig_name in ["earnings_growth", "fcf_growth"]:
            sig = qd.get(sig_name) or {}
            if sig.get("flag") in ("prior_negative", "prior_too_small"):
                flags.append(f"{sig_name[:3]}={sig['flag'][:3]}")
        if max_contrib > 25.5:
            flags.append(f"cap_breach")
        flag_str = ", ".join(flags)

        q_score = item.get("quality", 0) or 0
        print(
            f"  {t:<6}  {q_score:>5.1f}  {_pct('roic')}  {_pct('operating_margin')}  "
            f"{_pct('net_margin')}  {_pct('fcf_margin')}  {_pct('equity_ratio')}  "
            f"{_gsig('revenue_growth')}  {_gsig('earnings_growth')}  {_gsig('fcf_growth')}  "
            f"{n_sigs:>4}  {flag_str}"
        )
    print("  " + _DIV)
    g2_warns = report.warnings_for(2)
    if g2_warns:
        for i in g2_warns:
            print(f"  ⚠  [{i.ticker}] {i.message}")

    # ── Gate 3: Accounting Quality ────────────────────────────────────────────
    g3 = report.warnings_for(3)
    _sec("GATE 3 — ACCOUNTING QUALITY")
    if not g3:
        print("  ✓  No accounting quality issues found in Top 20.")
    else:
        for i in g3:
            print(f"  ⚠  [{i.ticker}] {i.message}")

    # ── Gate 4: AI Research ───────────────────────────────────────────────────
    _sec("GATE 4 — AI RESEARCH  (Top 20 coverage)")
    print(f"  {'Ticker':<6}  {'AI Score':>8}  {'Conf':>5}  {'Source':>8}  "
          f"{'Reds':>5}  {'Pos':>4}  Status")
    print("  " + _DIV)
    for item in report.top20:
        t   = item["ticker"]
        ai  = item.get("ai_research", 50.0) or 50.0
        aid = item.get("ai_detail") or {}

        if not aid:
            src  = "none"
            conf = 0.0
            n_r  = 0
            n_p  = 0
            status = "neutral — no source"
        else:
            conf = aid.get("confidence", 0)
            n_r  = len(aid.get("all_red_flags") or [])
            n_p  = len(aid.get("all_positives") or [])
            has_filing  = bool(aid.get("sec_filing_score", 0))
            has_profile = bool(aid.get("transcript_score"))
            src = "EDGAR" if has_filing else ("profile" if has_profile else "none")
            if abs(ai - 50.0) < 0.01:
                status = "neutral 50"
            elif conf < _AI_LOW_CONFIDENCE:
                status = f"low conf={conf:.2f}"
            else:
                status = "scored"

        print(f"  {t:<6}  {ai:>8.1f}  {conf:>5.2f}  {src:>8}  {n_r:>5}  {n_p:>4}  {status}")

    g4_warns = report.warnings_for(4)
    if g4_warns:
        print()
        for i in g4_warns:
            print(f"  ⚠  [{i.ticker}] {i.message}")

    # ── Gate 5: Shariah ───────────────────────────────────────────────────────
    g5 = [i for i in report.issues if i.gate == 5]
    _sec("GATE 5 — SHARIAH VERIFICATION  (Top 5 only)")
    if not g5:
        print("  ✓  All Top-5 tickers verified compliant.")
    else:
        for i in g5:
            icon = "✗" if i.severity == Sev.BLOCK else "⚠"
            print(f"  {icon}  [{i.ticker}] {i.message}")

    # ── Gate 6: Recommendation Guard ──────────────────────────────────────────
    _sec("GATE 6 — RECOMMENDATION GUARD")
    print("  ✓  All Top-5 picks sourced exclusively from safe_recommendations().")
    print("  ✓  Never from raw ranking or direct score comparison.")

    # ── Gate 8: Stress Test ───────────────────────────────────────────────────
    g8 = [i for i in report.issues if i.gate == 8]
    _sec("GATE 8 — STRESS TEST  (Top 5 historical crisis resilience)")
    print(f"  {'Ticker':<6}  {'Score':>7}  {'Label':<12}  {'GFC08':>7}  {'COVID20':>7}  {'Rates22':>7}  {'Banks23':>8}  {'vsSPY08':>8}  Error")
    print("  " + _DIV)
    for s in report.top5:
        t   = s["ticker"]
        st  = s.get("stress_test") or {}
        sc  = st.get("resilience_score", "n/a")
        lbl = st.get("label", "n/a")
        err = st.get("error", "")
        sc_str = f"{sc:.0f}/100" if isinstance(sc, (int, float)) else str(sc)
        crises_map = {c["name"]: c for c in (st.get("crises") or [])}
        def _dd(name: str) -> str:
            c = crises_map.get(name)
            if not c:
                return "    n/a"
            dd = c.get("max_drawdown")
            return f"{dd:>6.0f}%" if dd is not None else "    n/a"
        def _vs(name: str) -> str:
            c = crises_map.get(name)
            if not c:
                return "    n/a"
            rv = c.get("relative_vs_spy")
            sym = "+" if rv and rv > 0 else ""
            return f"{sym}{rv:>6.0f}%" if rv is not None else "    n/a"
        print(f"  {t:<6}  {sc_str:>7}  {lbl:<12}  {_dd('gfc_2008')}  {_dd('covid_2020')}  "
              f"{_dd('rates_2022')}  {_dd('regional_banks_2023')}  {_vs('gfc_2008')}  {(err or '')[:30]}")
    if g8:
        print()
        for i in g8:
            icon = "✗" if i.severity == Sev.BLOCK else "⚠"
            loc  = f"[{i.ticker}] " if i.ticker else ""
            print(f"  {icon}  {loc}{i.message}")

    # ── Gate 9: Ranking Stability ──────────────────────────────────────────────
    _sec("GATE 9 — RANKING STABILITY  (bootstrap 3 independent runs)")
    print(f"  Bootstrap overlap (3 runs, ±1pt noise): {report.ranking_stability_pct:.0f}%  "
          f"(threshold: 80%)")
    g9 = [i for i in report.issues if i.gate == 9]
    if not g9:
        print("  ✓  Top-5 is stable across perturbations.")
    else:
        for i in g9:
            print(f"  ⚠  {i.message}")

    # ── Gate 10: Red Flags ────────────────────────────────────────────────────
    g10 = [i for i in report.issues if i.gate == 10]
    _sec("GATE 10 — RED FLAGS  (Top 5)")
    print(f"  {'Ticker':<6}  {'Score':>6}  {'Label':<12}  Flags")
    print("  " + _DIV)
    for s in report.top5:
        t  = s["ticker"]
        rf = s.get("red_flags") or {}
        sc = rf.get("red_flag_score", 100)
        lb = rf.get("label", "n/a")
        fl = rf.get("flags") or []
        flags_str = " | ".join(fl[:2]) if fl else "none"
        print(f"  {t:<6}  {sc:>6.0f}  {lb:<12}  {flags_str}")
    if g10:
        print()
        for i in g10:
            icon = "✗" if i.severity == Sev.BLOCK else "⚠"
            loc  = f"[{i.ticker}] " if i.ticker else ""
            print(f"  {icon}  {loc}{i.message}")

    # ── Gate 11: Model Confidence ─────────────────────────────────────────────
    g11 = [i for i in report.issues if i.gate == 11]
    _sec("GATE 11 — MODEL CONFIDENCE 2.0  (Top 5)")
    print(f"  {'Ticker':<6}  {'Score':>7}  {'Label':<12}  {'DataQ':>6}  {'AI':>6}  {'Stab':>6}  {'Stress':>7}")
    print("  " + _DIV)
    for s in report.top5:
        t  = s["ticker"]
        mc = s.get("model_confidence") or {}
        score = mc.get("overall_confidence_score", "n/a")
        lbl   = mc.get("label", "n/a")
        dq    = mc.get("data_quality_input", "n/a")
        ai_c  = mc.get("ai_confidence_input", "n/a")
        stab  = mc.get("factor_stability_input", "n/a")
        stress= mc.get("stress_reliability_input", "n/a")
        def _fmt(v) -> str:
            return f"{v:>6.1f}" if isinstance(v, (int, float)) else "   n/a"
        sc_str = f"{score:.0f}/100" if isinstance(score, (int, float)) else str(score)
        print(f"  {t:<6}  {sc_str:>7}  {lbl:<12}  {_fmt(dq)}  {_fmt(ai_c)}  {_fmt(stab)}  {_fmt(stress)}")
    if g11:
        print()
        for i in g11:
            print(f"  ⚠  [{i.ticker}] {i.message}")

    # ── Gate 12: Sector Concentration ────────────────────────────────────────
    g12 = [i for i in report.issues if i.gate == 12]
    _sec("GATE 12 — SECTOR CONCENTRATION  (Top 5)")
    if not g12:
        # Show composition summary
        sec_count: dict[str, list[str]] = {}
        ind_count: dict[str, list[str]] = {}
        for s in report.top5:
            sec = s.get("sector", "Unknown")
            ind = s.get("industry", "Unknown")
            sec_count.setdefault(sec, []).append(s["ticker"])
            ind_count.setdefault(ind, []).append(s["ticker"])
        print("  ✓  No concentration violations.")
        for sec, tks in sec_count.items():
            print(f"     Sector '{sec}': {', '.join(tks)}")
    else:
        for i in g12:
            icon = "✗" if i.severity == Sev.BLOCK else "⚠"
            print(f"  {icon}  {i.message}")

    # ── Gate 13: Walk-Forward Check ───────────────────────────────────────────
    g13 = [i for i in report.issues if i.gate == 13]
    from data_layer.snapshot import get_snapshot_summary
    snap_summary = get_snapshot_summary()
    _sec("GATE 13 — WALK-FORWARD CHECK  (point-in-time snapshot history)")
    if not g13:
        print(f"  ✓  {snap_summary['count']} snapshot(s) available "
              f"({snap_summary['first_date']} → {snap_summary['last_date']}).")
        print("  ✓  Run --walk-forward to execute true walk-forward backtest.")
    else:
        for i in g13:
            print(f"  ⚠  {i.message}")

    # ── Gate 7 / Final Ranking: Top 20 table ──────────────────────────────────
    _sec("GATE 7 — TOP 20 RANKING TABLE")
    print(f"  {'#':<3}  {'Ticker':<6}  {'Comp':>6}  {'Q':>5}  {'Mom':>5}  "
          f"{'Val':>5}  {'Rev':>5}  {'EQ':>5}  {'Moat':>5}  {'CA':>5}  {'Risk':>5}  "
          f"{'AI*':>5}  {'Price':>8}  Notes")
    print("  " + _DIV)

    top5_tickers = {s["ticker"] for s in report.top5}
    for rank, item in enumerate(report.top20, 1):
        t     = item["ticker"]
        comp  = item.get("composite", 0) or 0
        q     = item.get("quality", 0) or 0
        m     = item.get("momentum", 0) or 0
        v     = item.get("valuation", 0) or 0
        r     = item.get("earnings_revisions", 0) or 0
        eq    = item.get("earnings_quality", 50) or 50
        mo    = item.get("moat", 50) or 50
        ca    = item.get("capital_allocation", 50) or 50
        ra    = item.get("risk_adjustment", 50) or 50
        ai    = item.get("ai_research", 50) or 50
        price = item.get("price", 0) or 0

        # Risk label badge
        rd = item.get("risk_detail") or {}
        rl = rd.get("risk_label", "?")[:1]   # L/M/H

        # Inline accounting notes
        notes = []
        sbc = item.get("sbc_data") or {}
        if sbc.get("sbc_fcf_ratio", 0) > _SBC_WARN_RATIO:
            tier  = "L" if sbc["sbc_fcf_ratio"] > 0.50 else ("M" if sbc["sbc_fcf_ratio"] > 0.30 else "S")
            notes.append(f"SBC={sbc['sbc_fcf_ratio']*100:.0f}%({tier})")
        qd = item.get("quality_detail") or {}
        for sig in ["earnings_growth", "fcf_growth"]:
            s = qd.get(sig) or {}
            if s.get("flag") in ("prior_negative", "prior_too_small"):
                notes.append(f"{sig[:3]}_excl")
        if qd.get("equity_ratio") and qd["equity_ratio"] < _HIGH_LEVERAGE_EQ_RATIO:
            notes.append("high_lev")
        eqd = item.get("earnings_quality_detail") or {}
        if eqd.get("fcf_conversion") is not None and eqd["fcf_conversion"] < 0.7:
            notes.append(f"fcf_conv={eqd['fcf_conversion']:.2f}")
        if eqd.get("distortion_flags"):
            notes.append("distortion!")
        note_str = ", ".join(notes)

        buy_flag = " ★" if t in top5_tickers else ""
        print(
            f"  {rank:<3}  {t:<6}  {comp:>6.1f}  {q:>5.1f}  {m:>5.1f}  "
            f"{v:>5.1f}  {r:>5.1f}  {eq:>5.1f}  {mo:>5.1f}  {ca:>5.1f}  "
            f"{ra:>5.1f}  {ai:>5.1f}  ${price:>7.2f}  Risk:{rl} {note_str}{buy_flag}"
        )
    print("  " + _DIV)
    print("  ★ = BUY NOW  |  SBC tiers: S=small(>20%) M=medium(>30%) L=large(>50%)")
    print("  Weights: Q20% M15% Rev20% Val10% EQ10% Moat10% CA10% Risk5%")
    print("  *AI display only (conf-scaled) — not in composite; flows through Moat (10% weight)")

    # ── Final verdict ──────────────────────────────────────────────────────────
    blocks = report.blocks()
    warns  = report.warnings()

    if blocks:
        _sec("INTEGRITY STATUS: ✗  BLOCKED")
        print(f"\n  {len(blocks)} critical issue(s) must be resolved before picks are shown:\n")
        for i in blocks:
            loc = f"[{i.ticker}] " if i.ticker else ""
            print(f"    ✗  Gate {i.gate}: {loc}{i.message}")
        print(f"\n  ✗  Top 5 withheld until all BLOCK issues are resolved.\n")
        print(f"{_BAR}\n")
        return

    # ── Final Top 5 ──────────────────────────────────────────────────────────
    _sec("FINAL TOP 5 — BUY NOW")
    if not report.top5:
        print("  No BUY NOW stocks found — check Shariah verification and constraints.")
        print(f"{_BAR}\n")
        return

    print(f"\n  {'#':<3}  {'Ticker':<6}  {'Score':>6}  {'Q':>5}  {'Mom':>5}  "
          f"{'Rev':>5}  {'Val':>5}  {'EQ':>5}  {'Moat':>5}  {'CA':>5}  "
          f"{'Risk':>5}  {'AI*':>5}  {'Weight':>7}  {'$Amount':>8}  {'Price':>8}  Shariah\n")
    for s in report.top5:
        t    = s["ticker"]
        comp = s.get("composite", 0) or 0
        q    = s.get("quality", 50) or 50
        m    = s.get("momentum", 50) or 50
        rv   = s.get("earnings_revisions", 50) or 50
        v    = s.get("valuation", 50) or 50
        eq   = s.get("earnings_quality", 50) or 50
        mo   = s.get("moat", 50) or 50
        ca   = s.get("capital_allocation", 50) or 50
        ra   = s.get("risk_adjustment", 50) or 50
        ai   = s.get("ai_research", 50) or 50
        w    = s.get("conviction_weight", 0) or 0
        d    = s.get("dollar_amount", 0) or 0
        p    = s.get("price", 0) or 0
        rank = s.get("rank", "?")
        shar = s.get("shariah_status", "?").upper()
        rd   = s.get("risk_detail") or {}
        rl   = rd.get("risk_label", "?")
        print(
            f"  {rank:<3}  {t:<6}  {comp:>6.1f}  {q:>5.1f}  {m:>5.1f}  "
            f"{rv:>5.1f}  {v:>5.1f}  {eq:>5.1f}  {mo:>5.1f}  {ca:>5.1f}  "
            f"{ra:>5.1f}  {ai:>5.1f}  {w*100:>6.0f}%  ${d:>7.0f}  ${p:>7.2f}  {shar}"
        )

        # ── V5 Phase 12 — Final Thesis Engine ────────────────────────────────
        qd   = s.get("quality_detail") or {}
        aid  = s.get("ai_detail") or {}
        sbc  = s.get("sbc_data") or {}
        cad  = s.get("capital_allocation_detail") or {}
        eqd  = s.get("earnings_quality_detail") or {}
        rev  = s.get("revisions_detail") or {}
        rf   = s.get("red_flags") or {}
        st   = s.get("stress_test") or {}
        fst  = s.get("factor_stability") or {}

        # Key drivers (top 3 factor scores above 60)
        factor_vals = [
            ("Quality",     s.get("quality", 50)),
            ("Momentum",    s.get("momentum", 50)),
            ("Revisions",   s.get("earnings_revisions", 50)),
            ("Valuation",   s.get("valuation", 50)),
            ("EarningsQ",   s.get("earnings_quality", 50)),
            ("Moat",        s.get("moat", 50)),
            ("CapAlloc",    s.get("capital_allocation", 50)),
            ("Risk",        s.get("risk_adjustment", 50)),
        ]
        drivers = [f"{n}={v:.0f}" for n, v in sorted(factor_vals, key=lambda x: x[1], reverse=True)[:3] if v > 60]

        strengths = []
        if qd.get("roic") and qd["roic"] > 0.20:
            strengths.append(f"ROIC={qd['roic']*100:.0f}%")
        if qd.get("operating_margin") and qd["operating_margin"] > 0.20:
            strengths.append(f"OpMargin={qd['operating_margin']*100:.0f}%")
        if qd.get("equity_ratio") and qd["equity_ratio"] > 0.80:
            strengths.append(f"low-debt(EqR={qd['equity_ratio']*100:.0f}%)")
        if s.get("momentum", 50) > 70:
            strengths.append(f"strong-momentum({s['momentum']:.0f})")
        if s.get("valuation", 50) > 70:
            strengths.append(f"cheap-valuation({s['valuation']:.0f})")
        if cad.get("buyback_rate") and cad["buyback_rate"] > 0.01:
            strengths.append(f"buyback({cad['buyback_rate']*100:.1f}%)")
        if rl == "Low":
            strengths.append("low-risk")
        if qd.get("revenue_cagr_5yr") and qd["revenue_cagr_5yr"] > 0.10:
            strengths.append(f"5yr-RevCAGR={qd['revenue_cagr_5yr']*100:.0f}%")
        if qd.get("roic_avg_5yr") and qd["roic_avg_5yr"] > 0.15:
            strengths.append(f"5yr-ROIC={qd['roic_avg_5yr']*100:.0f}%")
        net_up = rev.get("net_upgrades_90d")
        if net_up is not None and net_up > 2:
            strengths.append(f"analyst-upgrades(net+{net_up})")

        risks = []
        if sbc.get("sbc_fcf_ratio", 0) > _SBC_WARN_RATIO:
            risks.append(f"SBC={sbc['sbc_fcf_ratio']*100:.0f}%FCF")
        for sig in ["earnings_growth", "fcf_growth"]:
            gs = qd.get(sig) or {}
            if gs.get("flag") in ("prior_negative", "prior_too_small"):
                risks.append(f"{sig[:3]}_base_artifact")
        rev_gs = qd.get("revenue_growth") or {}
        if rev_gs.get("value") and rev_gs["value"] > _MERGER_REV_GROWTH:
            risks.append(f"rev_growth_{rev_gs['value']*100:.0f}%_possible_merger")
        n_reds = len(aid.get("all_red_flags") or [])
        if n_reds >= 4:
            risks.append(f"AI_flagged_{n_reds}_red_flags")
        if ai < 48:
            risks.append(f"AI_bearish(conf-adj={ai:.0f})")
        if rl == "High":
            risks.append("HIGH_RISK")
        for df in (eqd.get("distortion_flags") or []):
            risks.append(f"distortion: {df[:50]}")
        for w_str in (cad.get("warnings") or []):
            risks.append(f"cap_alloc: {w_str[:40]}")

        # What would invalidate the thesis
        invalidators = []
        if s.get("momentum", 50) > 65:
            invalidators.append("momentum reversal")
        if rev.get("eps_direction_score") and rev["eps_direction_score"] > 0.6:
            invalidators.append("analyst downgrade cycle")
        if qd.get("fcf_margin") and qd["fcf_margin"] > 0:
            invalidators.append("FCF margin compression")
        if not invalidators:
            invalidators.append("fundamental deterioration")

        mc_score = (s.get("model_confidence") or {}).get("overall_confidence_score", "?")
        mc_label = (s.get("model_confidence") or {}).get("label", "?")
        mc_str   = f"{mc_score:.0f}" if isinstance(mc_score, (int, float)) else "?"
        print(f"       Risk: {rl}  |  Moat: {mo:.1f}  |  Stability: {fst.get('label','?')}  |  "
              f"RedFlags: {rf.get('label','?')} ({rf.get('red_flag_score',100):.0f}/100)  |  "
              f"Confidence: {mc_label} ({mc_str}/100)")
        if drivers:
            print(f"       Key drivers:  {' | '.join(drivers)}")
        if strengths:
            print(f"       Strengths:    {' | '.join(strengths)}")
        if risks:
            print(f"       ⚠ Risks:      {' | '.join(risks)}")
        resilience = st.get("resilience_score")
        if resilience is not None:
            print(f"       Resilience:   {resilience:.0f}/100 ({st.get('label','?')})")
        print(f"       Thesis fails if: {' | '.join(invalidators)}")
        print()

    if warns:
        print(f"\n  {len(warns)} active warning(s) — does not block, but review before investing:")
        for i in warns:
            loc = f"[{i.ticker}] " if i.ticker else ""
            print(f"    ⚠  Gate {i.gate}: {loc}{i.message}")

    status = "CLEAN" if not warns else f"WARNINGS ({len(warns)})"
    _sec(f"INTEGRITY STATUS: ✓  {status}")
    print(f"  All gates passed. Top 5 output is authentic and explainable.\n")
    print(f"{_BAR}\n")
