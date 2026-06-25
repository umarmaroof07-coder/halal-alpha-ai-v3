"""
Shared dataclass, cache helpers, and Claude call wrapper for all AI Research
analyzers. No analyzer should import anthropic directly — use _call_claude().

V6 Role: AI is a qualitative risk-review layer, not a ranking driver.
  - Max effective weight in composite: ~3% (30% of Moat × 10% Moat weight).
  - Confidence < 0.30 → score clamped to neutral 50 (no signal).
  - One-way dampening in moat blend: AI can only reduce moat, never boost it.
  - Primary outputs: thesis_risks, accounting_concerns, management_concerns,
    moat_concerns, thesis_breakers (all carried as qualitative lists).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

NEUTRAL_SCORE = 50.0
_DB_PATH = Path("data/cache/ai_research_cache.db")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

_CONF_MIN = 0.30   # confidence below this → score forced to neutral 50


@dataclass
class AIAnalysisResult:
    ticker: str
    score: float = NEUTRAL_SCORE        # 0–100; clamped to 50 when confidence < _CONF_MIN
    confidence: float = 0.0             # 0–1
    rationale: str = "No source text available."
    # ── Quantitative gate (legacy, kept for compatibility) ─────────────────
    red_flags: list[str] = field(default_factory=list)
    positives: list[str] = field(default_factory=list)
    # ── Qualitative risk categories (V6) ──────────────────────────────────
    thesis_risks:         list[str] = field(default_factory=list)
    accounting_concerns:  list[str] = field(default_factory=list)
    management_concerns:  list[str] = field(default_factory=list)
    moat_concerns:        list[str] = field(default_factory=list)
    thesis_breakers:      list[str] = field(default_factory=list)
    source: str = ""                    # sec_filing | transcript | moat | management
    as_of_date: str = ""               # ISO date of source material
    cache_hit: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def neutral(ticker: str, source: str, as_of_date: str = "", reason: str = "No source text available.") -> AIAnalysisResult:
    return AIAnalysisResult(
        ticker=ticker,
        score=NEUTRAL_SCORE,
        confidence=0.0,
        rationale=reason,
        source=source,
        as_of_date=as_of_date,
    )


def effective_score(raw_score: float, confidence: float) -> float:
    """
    Return the confidence-adjusted AI score used for ranking/moat blending.

    Rules (in order):
      1. If confidence < 0.30 → return exactly NEUTRAL_SCORE (no signal).
      2. Otherwise → (raw_score − 50) × confidence + 50  (continuous fade).
    """
    if confidence < _CONF_MIN:
        return NEUTRAL_SCORE
    return (raw_score - NEUTRAL_SCORE) * confidence + NEUTRAL_SCORE


# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------

def _ensure_db() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ai_research_cache (
            cache_key  TEXT PRIMARY KEY,
            ticker     TEXT NOT NULL,
            analyzer   TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            ttl_days   INTEGER NOT NULL,
            payload    TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def _cache_key(ticker: str, analyzer: str, as_of_date: str) -> str:
    raw = json.dumps({"ticker": ticker, "analyzer": analyzer, "as_of_date": as_of_date}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def cache_get(ticker: str, analyzer: str, as_of_date: str) -> AIAnalysisResult | None:
    if not as_of_date:
        return None
    key = _cache_key(ticker, analyzer, as_of_date)
    try:
        conn = _ensure_db()
        row = conn.execute(
            "SELECT payload, fetched_at, ttl_days FROM ai_research_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()
        conn.close()
        if row is None:
            return None
        payload_str, fetched_at_str, ttl_days = row
        fetched_at = datetime.fromisoformat(fetched_at_str)
        age_days = (datetime.now(timezone.utc) - fetched_at.replace(tzinfo=timezone.utc)).days
        if age_days > ttl_days:
            return None
        data = json.loads(payload_str)
        result = AIAnalysisResult(**data)
        result.cache_hit = True
        return result
    except Exception as exc:
        log.warning("AI cache read error: %s", exc)
        return None


def cache_set(result: AIAnalysisResult, analyzer: str, ttl_days: int) -> None:
    if not result.as_of_date:
        return
    key = _cache_key(result.ticker, analyzer, result.as_of_date)
    payload = json.dumps(result.to_dict())
    fetched_at = datetime.now(timezone.utc).isoformat()
    try:
        conn = _ensure_db()
        conn.execute("""
            INSERT OR REPLACE INTO ai_research_cache
              (cache_key, ticker, analyzer, as_of_date, fetched_at, ttl_days, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (key, result.ticker, analyzer, result.as_of_date, fetched_at, ttl_days, payload))
        conn.commit()
        conn.close()
    except Exception as exc:
        log.warning("AI cache write error: %s", exc)


# ---------------------------------------------------------------------------
# Claude call wrapper
# ---------------------------------------------------------------------------

def _call_claude(prompt: str, model: str, api_key: str) -> str | None:
    """
    Call Anthropic Claude with *prompt* and return the text response.
    Returns None on any error. Never logs the api_key.
    """
    try:
        import anthropic  # type: ignore
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as exc:
        log.warning("Claude API call failed: %s", type(exc).__name__)
        return None


def parse_claude_json(raw: str | None, ticker: str, source: str, as_of_date: str) -> AIAnalysisResult | None:
    """
    Parse Claude's JSON response into an AIAnalysisResult.
    Returns None if raw is None or JSON is malformed — caller falls back to neutral.
    """
    if not raw:
        return None
    try:
        # Claude sometimes wraps JSON in markdown code fences
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        data: dict[str, Any] = json.loads(text)
        score = float(data.get("score", NEUTRAL_SCORE))
        score = max(0.0, min(100.0, score))
        confidence = float(data.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
        return AIAnalysisResult(
            ticker=ticker,
            score=score,
            confidence=confidence,
            rationale=str(data.get("rationale", "")),
            red_flags=list(data.get("red_flags", [])),
            positives=list(data.get("positives", [])),
            thesis_risks=list(data.get("thesis_risks", [])),
            accounting_concerns=list(data.get("accounting_concerns", [])),
            management_concerns=list(data.get("management_concerns", [])),
            moat_concerns=list(data.get("moat_concerns", [])),
            thesis_breakers=list(data.get("thesis_breakers", [])),
            source=source,
            as_of_date=as_of_date,
        )
    except Exception as exc:
        log.warning("Failed to parse Claude JSON for %s/%s: %s", ticker, source, exc)
        return None
