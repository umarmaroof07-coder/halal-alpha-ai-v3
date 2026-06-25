"""
Earnings Call Transcript Analyzer — scores a transcript using Claude.

Looks for:
  - Guidance language (raised / maintained / lowered / withdrawn)
  - Management tone (confident vs. defensive / hedging)
  - Capital allocation commentary (buybacks, dividends, capex plans)
  - New product/market signals
  - Analyst question patterns (probing on specific risks)

If transcript_text is None or empty → neutral 50, confidence 0.0.
If ANTHROPIC_API_KEY is missing → neutral 50, confidence 0.0.
Results cached by ticker + as_of_date for AI_CACHE_TTL_TRANSCRIPTS_DAYS days.
"""

from __future__ import annotations

import logging

from config.settings import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, AI_CACHE_TTL_TRANSCRIPTS_DAYS
from ai_research._base import (
    AIAnalysisResult,
    neutral,
    cache_get,
    cache_set,
    _call_claude,
    parse_claude_json,
)

log = logging.getLogger(__name__)

_ANALYZER = "transcript"
_TTL = AI_CACHE_TTL_TRANSCRIPTS_DAYS

_PROMPT_TEMPLATE = """\
You are a senior buy-side analyst reviewing an earnings call transcript for {ticker}.
Your job is RISK IDENTIFICATION. Surface what analysts are probing for and what management is avoiding.
Analyze ONLY the text provided below. Do not use any prior knowledge about this company.

Assess these dimensions:
1. Guidance: raised, maintained, lowered, or withdrawn? Note any softening language.
2. Tone: is management specific and confident, or vague, defensive, or over-using "challenges"?
3. Analyst probing: what risks are analysts pressing on? What did management deflect or not answer directly?
4. Management credibility: are they addressing problems honestly, or pivoting with corporate-speak?
5. Capital allocation signals: any commentary on buybacks, M&A, debt, or capex cuts?

Return ONLY valid JSON in this exact format — no prose before or after:
{{
  "score": <float 0-100, where 50=neutral, >50=positive signals, <50=warning signals>,
  "confidence": <float 0-1, where 0=insufficient text, 1=very confident>,
  "rationale": "<2-3 sentence summary of the most important signals>",
  "red_flags": ["<factual concern from the transcript>"],
  "positives": ["<factual positive from the transcript>"],
  "management_concerns": ["<candor, credibility, or capital allocation concern from the text>"],
  "thesis_risks": ["<risk analysts pressed on, or management flagged, that could derail the thesis>"],
  "thesis_breakers": ["<single statement that would make you sell immediately, if any>"]
}}

Rules:
- management_concerns focus on how management communicated, not just what they said.
- thesis_risks are forward-looking risks surfaced in this call.
- thesis_breakers are severe disqualifiers (e.g. "CEO said demand has materially slowed in core segment").
- If no concerns exist in a category, return an empty list [].
- If text is insufficient, set confidence below 0.30.
- Do NOT hallucinate. Every item must come from the text.

--- TRANSCRIPT TEXT START ---
{transcript_text}
--- TRANSCRIPT TEXT END ---
"""

_MAX_TEXT_CHARS = 12_000


def analyze_transcript(
    ticker: str,
    transcript_text: str | None,
    as_of_date: str,
) -> AIAnalysisResult:
    """
    Score an earnings call transcript for *ticker*.

    Parameters
    ----------
    ticker : str
    transcript_text : str | None
        Raw text of the earnings call. Pass None if unavailable.
    as_of_date : str
        ISO date of the earnings call (e.g. "2024-02-15"). Used as cache key.

    Returns
    -------
    AIAnalysisResult with source="transcript".
    """
    if not transcript_text or not transcript_text.strip():
        return neutral(ticker, _ANALYZER, as_of_date)

    if not ANTHROPIC_API_KEY:
        log.debug("ANTHROPIC_API_KEY not set — returning neutral for %s transcript", ticker)
        return neutral(ticker, _ANALYZER, as_of_date, reason="API key not configured.")

    cached = cache_get(ticker, _ANALYZER, as_of_date)
    if cached:
        return cached

    truncated = transcript_text[:_MAX_TEXT_CHARS]
    prompt = _PROMPT_TEMPLATE.format(ticker=ticker, transcript_text=truncated)

    raw = _call_claude(prompt, ANTHROPIC_MODEL, ANTHROPIC_API_KEY)
    result = parse_claude_json(raw, ticker, _ANALYZER, as_of_date)

    if result is None:
        result = neutral(ticker, _ANALYZER, as_of_date, reason="Claude response could not be parsed.")

    cache_set(result, _ANALYZER, _TTL)
    return result
