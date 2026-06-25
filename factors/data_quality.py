"""
V5 Phase 2 — Data Quality Engine

Scores each ticker's data on three dimensions:
  1. Freshness      — how recently the data was retrieved
  2. Coverage       — fraction of expected fields populated
  3. Source Quality — provider tier (yfinance < FMP < Bloomberg)

Output: DataQualityScore (0-100) with component breakdown.
Higher score = better quality data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Source quality tiers (0-100)
SOURCE_QUALITY = {
    "bloomberg": 100,
    "fmp":       75,
    "edgar":     85,
    "yfinance":  60,
    "manual":    70,
    "unknown":   40,
}

# Expected fields for a fully-populated ticker record
_EXPECTED_FIELDS = [
    "price", "mkt_cap", "avg_volume",
    "quality", "momentum", "valuation", "earnings_revisions",
    "earnings_quality", "moat", "capital_allocation", "risk_adjustment",
    "quality_detail", "revisions_detail", "moat_detail",
    "earnings_quality_detail", "capital_allocation_detail", "risk_detail",
]


@dataclass
class DataQualityScore:
    ticker: str
    overall: float                  # 0-100
    freshness_score: float          # 0-100
    coverage_score: float           # 0-100
    source_score: float             # 0-100
    fields_populated: int = 0
    fields_total: int = len(_EXPECTED_FIELDS)
    data_age_hours: float | None = None
    sources_used: list[str] = field(default_factory=list)
    label: str = ""                 # "Excellent" / "Good" / "Fair" / "Poor"

    def to_dict(self) -> dict:
        return {
            "overall":           round(self.overall, 1),
            "freshness_score":   round(self.freshness_score, 1),
            "coverage_score":    round(self.coverage_score, 1),
            "source_score":      round(self.source_score, 1),
            "fields_populated":  self.fields_populated,
            "fields_total":      self.fields_total,
            "data_age_hours":    round(self.data_age_hours, 1) if self.data_age_hours is not None else None,
            "sources_used":      self.sources_used,
            "label":             self.label,
        }


def _freshness_score(generated_at_iso: str | None) -> tuple[float, float | None]:
    """Return (freshness_score 0-100, age_hours)."""
    if not generated_at_iso:
        return 40.0, None
    try:
        generated = datetime.fromisoformat(generated_at_iso)
        if generated.tzinfo is None:
            generated = generated.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        hours = (now - generated).total_seconds() / 3600.0
        if hours < 1:
            return 100.0, hours
        elif hours < 24:
            return 90.0, hours
        elif hours < 168:    # 7 days
            return 70.0, hours
        elif hours < 720:    # 30 days
            return 40.0, hours
        else:
            return 20.0, hours
    except Exception:
        return 40.0, None


def _coverage_score(record: dict) -> tuple[float, int]:
    """Return (coverage_score 0-100, n_populated)."""
    populated = sum(
        1 for f in _EXPECTED_FIELDS
        if record.get(f) is not None
    )
    ratio = populated / len(_EXPECTED_FIELDS)
    score = ratio * 100.0
    return score, populated


def _source_score(record: dict) -> tuple[float, list[str]]:
    """Return (source_quality_score 0-100, list of sources detected)."""
    sources = []
    # Infer sources from what's populated
    if record.get("quality_detail") or record.get("moat_detail"):
        sources.append("fmp")
    if record.get("revisions_detail"):
        rd = record["revisions_detail"]
        if isinstance(rd, dict) and rd.get("eps_7d_change") is not None:
            sources.append("yfinance")
        else:
            sources.append("fmp")
    if record.get("ai_detail") and isinstance(record["ai_detail"], dict):
        ad = record["ai_detail"]
        if ad.get("sec_filing"):
            sources.append("edgar")
    if not sources:
        sources.append("unknown")

    avg = sum(SOURCE_QUALITY.get(s, 40) for s in sources) / len(sources)
    return avg, sources


def compute_data_quality(
    ticker: str,
    record: dict,
    generated_at_iso: str | None = None,
) -> DataQualityScore:
    """
    Compute DataQualityScore for one ticker.

    Args:
        ticker:           the ticker symbol
        record:           one entry from scored_universe["universe"]
        generated_at_iso: ISO timestamp of when scored_universe was generated
    """
    fresh_score, age_hours = _freshness_score(generated_at_iso)
    cov_score, n_pop       = _coverage_score(record)
    src_score, sources     = _source_score(record)

    # Weighted overall: freshness 30% + coverage 45% + source 25%
    overall = 0.30 * fresh_score + 0.45 * cov_score + 0.25 * src_score
    overall = max(0.0, min(100.0, overall))

    if overall >= 85:
        label = "Excellent"
    elif overall >= 70:
        label = "Good"
    elif overall >= 50:
        label = "Fair"
    else:
        label = "Poor"

    return DataQualityScore(
        ticker           = ticker,
        overall          = overall,
        freshness_score  = fresh_score,
        coverage_score   = cov_score,
        source_score     = src_score,
        fields_populated = n_pop,
        fields_total     = len(_EXPECTED_FIELDS),
        data_age_hours   = age_hours,
        sources_used     = sources,
        label            = label,
    )
