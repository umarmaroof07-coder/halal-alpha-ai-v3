"""
Composite scorer — normalizes all factor raw scores and combines them.

Pipeline per factor:
  1. Collect raw_score from each factor module for all tickers.
  2. Three-tier cross-sectional normalization:
       a. Global z-score (50% weight)
       b. Sector-relative z-score (30% weight, min 5 members)
       c. Industry-relative z-score (20% weight, min 3 members)
     Each component clipped [−3,+3], mapped [0,100].
     Tiers with insufficient members fall back to global.
  3. Missing raw_score (None) → neutral 50.

Final composite V4 (8 factors, no standalone AI):
  Quality           20%  ← 5-year persistence model
  Momentum          15%
  Earnings Rev.     20%  ← institutional EPS trend, breadth, PT, upgrades
  Valuation         10%
  Earnings Quality  10%
  Moat              10%  ← 70% quant + 30% AI (conf-scaled)
  Capital Alloc.    10%  ← raised from 5%; key long-run value driver
  Risk Adjustment    5%  ← higher = safer

AI embedded in Moat (~3% effective weight). Never a standalone factor.
ai_research field on FactorScores is display-only / transparency.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from config.settings import BACKTEST_AI_NEUTRAL
from config.weights import (
    FACTOR_WEIGHTS,
    INDUSTRY_BLEND_ALPHA, SECTOR_BLEND_ALPHA, INDUSTRY_BLEND_BETA,
    SECTOR_MIN_MEMBERS, INDUSTRY_MIN_MEMBERS,
)
from factors.momentum import MomentumRaw
from factors.quality import QualityRaw
from factors.valuation import ValuationRaw
from factors.revisions import RevisionsRaw
from factors.earnings_quality import EarningsQualityRaw
from factors.moat_quality import MoatQualityRaw
from factors.capital_allocation import CapitalAllocationRaw
from factors.risk_engine import RiskRaw

log = logging.getLogger(__name__)

NEUTRAL = 50.0


# ---------------------------------------------------------------------------
# Scored result
# ---------------------------------------------------------------------------

@dataclass
class FactorScores:
    ticker: str
    quality:            float = NEUTRAL
    momentum:           float = NEUTRAL
    valuation:          float = NEUTRAL
    earnings_revisions: float = NEUTRAL
    earnings_quality:   float = NEUTRAL
    moat:               float = NEUTRAL   # z-scored blended moat
    capital_allocation: float = NEUTRAL   # z-scored capital allocation
    risk_adjustment:    float = NEUTRAL   # z-scored risk (higher = safer)
    ai_research:        float = NEUTRAL   # DISPLAY ONLY — not in composite
    composite:          float = NEUTRAL

    def to_dict(self) -> dict:
        return {
            "ticker":             self.ticker,
            "quality":            round(self.quality, 2),
            "momentum":           round(self.momentum, 2),
            "valuation":          round(self.valuation, 2),
            "earnings_revisions": round(self.earnings_revisions, 2),
            "earnings_quality":   round(self.earnings_quality, 2),
            "moat":               round(self.moat, 2),
            "capital_allocation": round(self.capital_allocation, 2),
            "risk_adjustment":    round(self.risk_adjustment, 2),
            "ai_research":        round(self.ai_research, 2),
            "composite":          round(self.composite, 2),
        }


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _zscore_to_0_100(values: list[float | None]) -> list[float]:
    indices = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(indices) < 2:
        return [NEUTRAL] * len(values)
    raw  = np.array([v for _, v in indices], dtype=float)
    mean = raw.mean()
    std  = raw.std()
    result = [NEUTRAL] * len(values)
    if std == 0:
        return result
    for i, v in indices:
        z = max(-3.0, min(3.0, (v - mean) / std))
        result[i] = (z + 3.0) / 6.0 * 100.0
    return result


def _group_scores(
    raw_values: list[float | None],
    tickers:    list[str],
    group_map:  dict[str, str],
    min_members: int,
) -> list[float]:
    """Z-score within groups (sector or industry). Groups below min_members use global."""
    global_scores = _zscore_to_0_100(raw_values)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(tickers):
        groups[group_map.get(t, "__unknown__")].append(i)
    group_scores = list(global_scores)
    for grp, indices in groups.items():
        if len(indices) < min_members:
            continue
        grp_raws = [raw_values[i] for i in indices]
        grp_norm = _zscore_to_0_100(grp_raws)
        for k, i in enumerate(indices):
            group_scores[i] = grp_norm[k]
    return group_scores


def _three_tier_scores(
    raw_values:   list[float | None],
    tickers:      list[str],
    sector_map:   dict[str, str],
    industry_map: dict[str, str],
) -> list[float]:
    """
    Three-tier blend: 50% global + 30% sector + 20% industry.
    Falls back to global when a tier has insufficient members.
    """
    global_scores   = _zscore_to_0_100(raw_values)
    sector_scores   = _group_scores(raw_values, tickers, sector_map,   SECTOR_MIN_MEMBERS)
    industry_scores = _group_scores(raw_values, tickers, industry_map, INDUSTRY_MIN_MEMBERS)

    return [
        INDUSTRY_BLEND_ALPHA * global_scores[i] +
        SECTOR_BLEND_ALPHA   * sector_scores[i]  +
        INDUSTRY_BLEND_BETA  * industry_scores[i]
        for i in range(len(tickers))
    ]


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

def compute_composite(
    tickers: list[str],
    momentum_raw:           dict[str, MomentumRaw],
    quality_raw:            dict[str, QualityRaw],
    valuation_raw:          dict[str, ValuationRaw],
    revisions_raw:          dict[str, RevisionsRaw],
    earnings_quality_raw:   dict[str, EarningsQualityRaw] | None = None,
    moat_raw:               dict[str, MoatQualityRaw]     | None = None,
    capital_allocation_raw: dict[str, CapitalAllocationRaw] | None = None,
    risk_raw:               dict[str, RiskRaw]             | None = None,
    ai_research_scores:     dict[str, float]               | None = None,
    ai_confidence_scores:   dict[str, float]               | None = None,
    sector_map:             dict[str, str]                 | None = None,
    industry_map:           dict[str, str]                 | None = None,
    is_backtest: bool = False,
) -> list[FactorScores]:
    n = len(tickers)
    if n == 0:
        return []

    eq_raw   = earnings_quality_raw   or {}
    moat_map = moat_raw               or {}
    ca_map   = capital_allocation_raw or {}
    risk_map = risk_raw               or {}
    smap     = sector_map             or {}
    imap     = industry_map           or {}

    def _raw(d, key, attr="raw_score"):
        obj = d.get(key)
        if obj is None:
            return None
        return getattr(obj, attr, None)

    mom_raws  = [_raw(momentum_raw,  t) for t in tickers]
    qua_raws  = [_raw(quality_raw,   t) for t in tickers]
    val_raws  = [_raw(valuation_raw, t) for t in tickers]
    rev_raws  = [_raw(revisions_raw, t) for t in tickers]
    eq_raws   = [_raw(eq_raw,        t) for t in tickers]
    ca_raws   = [_raw(ca_map,        t) for t in tickers]
    risk_raws = [_raw(risk_map,      t) for t in tickers]

    moat_raws = []
    for t in tickers:
        mq = moat_map.get(t)
        if mq is not None and mq.blended_moat_score is not None:
            moat_raws.append(mq.blended_moat_score / 100.0)
        else:
            moat_raws.append(None)

    # ── Normalize with 3-tier blending ──────────────────────────────────────
    if smap or imap:
        def score(raws):
            return _three_tier_scores(raws, tickers, smap, imap)
        mom_scores  = score(mom_raws)
        qua_scores  = score(qua_raws)
        val_scores  = score(val_raws)
        rev_scores  = score(rev_raws)
        eq_scores   = score(eq_raws)
        moat_scores = score(moat_raws)
        ca_scores   = score(ca_raws)
        risk_scores = score(risk_raws)
    else:
        mom_scores  = _zscore_to_0_100(mom_raws)
        qua_scores  = _zscore_to_0_100(qua_raws)
        val_scores  = _zscore_to_0_100(val_raws)
        rev_scores  = _zscore_to_0_100(rev_raws)
        eq_scores   = _zscore_to_0_100(eq_raws)
        moat_scores = _zscore_to_0_100(moat_raws)
        ca_scores   = _zscore_to_0_100(ca_raws)
        risk_scores = _zscore_to_0_100(risk_raws)

    # ── AI Research (display only, NOT in composite) ─────────────────────────
    if is_backtest or ai_research_scores is None:
        ai_display = [BACKTEST_AI_NEUTRAL] * n
    else:
        from ai_research._base import effective_score as _eff_score
        conf_map = ai_confidence_scores or {}
        ai_display = []
        for t in tickers:
            raw_ai = float(ai_research_scores.get(t, BACKTEST_AI_NEUTRAL))
            conf   = float(conf_map.get(t, 0.0))
            # conf < 0.30 → exactly neutral 50; otherwise continuous fade
            eff_ai = _eff_score(raw_ai, conf)
            ai_display.append(round(eff_ai, 2))

    # ── Build FactorScores and composite ────────────────────────────────────
    results: list[FactorScores] = []
    for i, ticker in enumerate(tickers):
        q  = qua_scores[i];  m  = mom_scores[i]
        v  = val_scores[i];  r  = rev_scores[i]
        eq = eq_scores[i];   mo = moat_scores[i]
        ca = ca_scores[i];   ra = risk_scores[i]
        ai = ai_display[i]

        composite = (
            q  * FACTOR_WEIGHTS["quality"]            +
            m  * FACTOR_WEIGHTS["momentum"]           +
            v  * FACTOR_WEIGHTS["valuation"]          +
            r  * FACTOR_WEIGHTS["earnings_revisions"] +
            eq * FACTOR_WEIGHTS["earnings_quality"]   +
            mo * FACTOR_WEIGHTS["moat"]               +
            ca * FACTOR_WEIGHTS["capital_allocation"] +
            ra * FACTOR_WEIGHTS["risk_adjustment"]
        )

        results.append(FactorScores(
            ticker             = ticker,
            quality            = q, momentum = m, valuation = v,
            earnings_revisions = r, earnings_quality = eq,
            moat               = mo, capital_allocation = ca,
            risk_adjustment    = ra, ai_research = ai,
            composite          = composite,
        ))

    return results


def rank_scores(scores: list[FactorScores]) -> list[FactorScores]:
    return sorted(scores, key=lambda s: s.composite, reverse=True)
