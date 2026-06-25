"""
Management Scorer — scores management quality using Claude.

Draws from a combination of:
  - Earnings call transcript (tone, candor, specific commitments)
  - SEC filing MD&A (capital allocation decisions, stated strategy, execution track record)

Assesses:
  - Candor: do they address problems directly or use corporate-speak?
  - Capital allocation: buybacks at fair value, disciplined M&A, appropriate capex?
  - Shareholder alignment: insider ownership signals, compensation structure mentions
  - Guidance accuracy signals: did past stated targets get met (if mentioned)?
  - Execution: do they deliver on what they say?

If both texts are None/empty → neutral 50, confidence 0.0.
If only one is available, scores from that source with lower confidence.
"""

from __future__ import annotations

import logging

from config.settings import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, AI_CACHE_TTL_FILINGS_DAYS
from ai_research._base import (
    AIAnalysisResult,
    neutral,
    cache_get,
    cache_set,
    _call_claude,
    parse_claude_json,
)

log = logging.getLogger(__name__)

_ANALYZER = "management"
_TTL = AI_CACHE_TTL_FILINGS_DAYS

_PROMPT_TEMPLATE = """\
You are a senior forensic analyst assessing management quality and credibility for {ticker}.
Your job is to surface CONCERNS about management first. Incompetent or misaligned management is the #1 destroyer of long-term equity value.
Analyze ONLY the text provided below. Do not use any prior knowledge about this company.

Evaluate management across these dimensions:
1. Candor — do they acknowledge problems directly, or pivot with corporate-speak and blame externals?
2. Capital allocation — is cash being used wisely (buybacks at fair value, no empire-building M&A)?
3. Incentive alignment — any signals about insider ownership, compensation structure, or shareholder-friendly actions?
4. Past guidance vs. delivery — did they hit what they said they would? Any history of sandbagging or miss-and-reset?
5. Execution signals — do past commitments appear to have been met, based on what the text says?

Score 0–100 where:
  0–30 = red flags (misleading language, value-destroying capital allocation, entrenchment)
  31–50 = mediocre or insufficient evidence to assess
  51–70 = competent, reasonable alignment
  71–100 = exceptional track record, honest, proven

Return ONLY valid JSON in this exact format — no prose before or after:
{{
  "score": <float 0-100>,
  "confidence": <float 0-1, where 0=insufficient text, 1=very confident>,
  "rationale": "<2-3 sentence summary of the most important management signals>",
  "red_flags": ["<factual management concern from the text>"],
  "positives": ["<factual management strength from the text>"],
  "management_concerns": ["<specific candor, capital allocation, or alignment concern>"],
  "thesis_risks": ["<forward-looking risk management is signalling, downplaying, or avoiding>"],
  "thesis_breakers": ["<single statement revealing fundamental management failure, if any>"]
}}

Rules:
- management_concerns are specific, factual — not vague ("management seems evasive").
- thesis_risks are what could go wrong based on what management is saying or not saying.
- thesis_breakers are disqualifying facts (e.g. "CFO resigned effective immediately", "guidance withdrawn with no timeline").
- If no concerns exist in a category, return an empty list [].
- If text is insufficient, set confidence below 0.30.
- Do NOT hallucinate. Every item must come from the text.

{source_section}
"""

_MAX_TEXT_CHARS = 6_000  # per source


def score_management(
    ticker: str,
    transcript_text: str | None,
    sec_filing_text: str | None,
    as_of_date: str,
) -> AIAnalysisResult:
    """
    Score management quality for *ticker*.

    Parameters
    ----------
    ticker : str
    transcript_text : str | None
        Earnings call transcript text. Pass None if unavailable.
    sec_filing_text : str | None
        SEC filing (10-K/10-Q) text. Pass None if unavailable.
    as_of_date : str
        ISO date used as cache key.

    Returns
    -------
    AIAnalysisResult with source="management".
    """
    has_transcript = bool(transcript_text and transcript_text.strip())
    has_filing = bool(sec_filing_text and sec_filing_text.strip())

    if not has_transcript and not has_filing:
        return neutral(ticker, _ANALYZER, as_of_date)

    if not ANTHROPIC_API_KEY:
        log.debug("ANTHROPIC_API_KEY not set — returning neutral for %s management", ticker)
        return neutral(ticker, _ANALYZER, as_of_date, reason="API key not configured.")

    cached = cache_get(ticker, _ANALYZER, as_of_date)
    if cached:
        return cached

    sections: list[str] = []
    if has_transcript:
        sections.append(
            "--- EARNINGS CALL TRANSCRIPT START ---\n"
            + transcript_text[:_MAX_TEXT_CHARS]  # type: ignore[index]
            + "\n--- EARNINGS CALL TRANSCRIPT END ---"
        )
    if has_filing:
        sections.append(
            "--- SEC FILING (MD&A SECTION) START ---\n"
            + sec_filing_text[:_MAX_TEXT_CHARS]  # type: ignore[index]
            + "\n--- SEC FILING (MD&A SECTION) END ---"
        )

    source_section = "\n\n".join(sections)
    prompt = _PROMPT_TEMPLATE.format(ticker=ticker, source_section=source_section)

    raw = _call_claude(prompt, ANTHROPIC_MODEL, ANTHROPIC_API_KEY)
    result = parse_claude_json(raw, ticker, _ANALYZER, as_of_date)

    if result is None:
        result = neutral(ticker, _ANALYZER, as_of_date, reason="Claude response could not be parsed.")

    cache_set(result, _ANALYZER, _TTL)
    return result
