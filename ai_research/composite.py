"""
AI Research Composite — combines the four sub-analyzers into a final 0–100 score.

Sub-weights (from config/weights.py):
  transcript   30%
  sec_filing   30%
  moat         25%
  management   15%

Backtest guard:
  If is_backtest=True → return BACKTEST_AI_NEUTRAL (50.0) immediately.
  No Claude calls are made. This prevents look-ahead bias.

Missing API key:
  If ANTHROPIC_API_KEY is not set → all sub-scores are neutral 50.

Confidence:
  Overall confidence = weighted average of sub-confidences.
  Sub-scores with confidence=0 contribute to the weighted average (pulling
  overall confidence down, which is correct — partial data = lower trust).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from config.settings import ANTHROPIC_API_KEY, BACKTEST_AI_NEUTRAL
from config.weights import AI_RESEARCH_WEIGHTS
from ai_research._base import AIAnalysisResult, neutral as _neutral
from ai_research.sec_filing_analyzer import analyze_sec_filing
from ai_research.transcript_analyzer import analyze_transcript
from ai_research.moat_scorer import score_moat
from ai_research.management_scorer import score_management

log = logging.getLogger(__name__)


@dataclass
class AIResearchResult:
    ticker: str
    score: float                               # 0–100 final composite
    confidence: float                          # 0–1 weighted average
    is_backtest: bool = False
    sec_filing: AIAnalysisResult | None = None
    transcript: AIAnalysisResult | None = None
    moat: AIAnalysisResult | None = None
    management: AIAnalysisResult | None = None
    # Aggregated risk lists from all sub-analyzers
    all_red_flags:          list[str] = field(default_factory=list)
    all_positives:          list[str] = field(default_factory=list)
    all_thesis_risks:       list[str] = field(default_factory=list)
    all_accounting_concerns:list[str] = field(default_factory=list)
    all_management_concerns:list[str] = field(default_factory=list)
    all_moat_concerns:      list[str] = field(default_factory=list)
    all_thesis_breakers:    list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ticker":            self.ticker,
            "score":             round(self.score, 2),
            "confidence":        round(self.confidence, 3),
            "is_backtest":       self.is_backtest,
            "sec_filing_score":  round(self.sec_filing.score, 2)  if self.sec_filing  else None,
            "transcript_score":  round(self.transcript.score, 2)  if self.transcript  else None,
            "moat_score":        round(self.moat.score, 2)         if self.moat        else None,
            "management_score":  round(self.management.score, 2)  if self.management  else None,
            "all_red_flags":           self.all_red_flags,
            "all_positives":           self.all_positives,
            "all_thesis_risks":        self.all_thesis_risks,
            "all_accounting_concerns": self.all_accounting_concerns,
            "all_management_concerns": self.all_management_concerns,
            "all_moat_concerns":       self.all_moat_concerns,
            "all_thesis_breakers":     self.all_thesis_breakers,
        }


def _backtest_result(ticker: str) -> AIResearchResult:
    return AIResearchResult(
        ticker=ticker,
        score=BACKTEST_AI_NEUTRAL,
        confidence=0.0,
        is_backtest=True,
    )


def run_ai_research(
    ticker: str,
    filing_text: str | None = None,
    transcript_text: str | None = None,
    company_profile: str | None = None,
    as_of_date: str = "",
    is_backtest: bool = False,
) -> AIResearchResult:
    """
    Run all four AI Research sub-analyzers and return a composite result.

    Parameters
    ----------
    ticker : str
    filing_text : str | None
        SEC 10-K/10-Q text.
    transcript_text : str | None
        Earnings call transcript text.
    company_profile : str | None
        Business description for moat scoring.
    as_of_date : str
        ISO date — used as cache key across all sub-analyzers.
    is_backtest : bool
        If True, skip all Claude calls and return BACKTEST_AI_NEUTRAL (50.0).

    Returns
    -------
    AIResearchResult
    """
    if is_backtest:
        log.debug("Backtest mode — AI Research locked to %.1f for %s", BACKTEST_AI_NEUTRAL, ticker)
        return _backtest_result(ticker)

    if not ANTHROPIC_API_KEY:
        log.debug("No ANTHROPIC_API_KEY — returning neutral AI Research for %s", ticker)
        return AIResearchResult(ticker=ticker, score=BACKTEST_AI_NEUTRAL, confidence=0.0)

    # Run all four sub-analyzers
    filing_result     = analyze_sec_filing(ticker, filing_text, as_of_date)
    transcript_result = analyze_transcript(ticker, transcript_text, as_of_date)
    moat_result       = score_moat(ticker, company_profile, as_of_date)
    management_result = score_management(ticker, transcript_text, filing_text, as_of_date)

    weights = AI_RESEARCH_WEIGHTS
    composite_score = (
        filing_result.score     * weights["sec_filing"] +
        transcript_result.score * weights["transcript"] +
        moat_result.score       * weights["moat"]       +
        management_result.score * weights["management"]
    )

    composite_confidence = (
        filing_result.confidence     * weights["sec_filing"] +
        transcript_result.confidence * weights["transcript"] +
        moat_result.confidence       * weights["moat"]       +
        management_result.confidence * weights["management"]
    )

    sub_results = [filing_result, transcript_result, moat_result, management_result]

    def _agg(attr: str) -> list[str]:
        out: list[str] = []
        for r in sub_results:
            out.extend(getattr(r, attr, []))
        return out

    return AIResearchResult(
        ticker=ticker,
        score=round(composite_score, 2),
        confidence=round(composite_confidence, 3),
        is_backtest=False,
        sec_filing=filing_result,
        transcript=transcript_result,
        moat=moat_result,
        management=management_result,
        all_red_flags           = _agg("red_flags"),
        all_positives           = _agg("positives"),
        all_thesis_risks        = _agg("thesis_risks"),
        all_accounting_concerns = _agg("accounting_concerns"),
        all_management_concerns = _agg("management_concerns"),
        all_moat_concerns       = _agg("moat_concerns"),
        all_thesis_breakers     = _agg("thesis_breakers"),
    )


def run_ai_research_batch(
    tickers: list[str],
    filing_texts: dict[str, str] | None = None,
    transcript_texts: dict[str, str] | None = None,
    company_profiles: dict[str, str] | None = None,
    as_of_dates: dict[str, str] | None = None,
    is_backtest: bool = False,
) -> dict[str, AIResearchResult]:
    """
    Run AI Research for multiple tickers. Returns {ticker: AIResearchResult}.
    """
    filing_texts     = filing_texts or {}
    transcript_texts = transcript_texts or {}
    company_profiles = company_profiles or {}
    as_of_dates      = as_of_dates or {}

    results: dict[str, AIResearchResult] = {}
    for ticker in tickers:
        results[ticker] = run_ai_research(
            ticker=ticker,
            filing_text=filing_texts.get(ticker),
            transcript_text=transcript_texts.get(ticker),
            company_profile=company_profiles.get(ticker),
            as_of_date=as_of_dates.get(ticker, ""),
            is_backtest=is_backtest,
        )
    return results
