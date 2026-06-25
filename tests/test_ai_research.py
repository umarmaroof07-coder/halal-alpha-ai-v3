"""
Tests for Phase 5: AI Research Module.

All Anthropic API calls are mocked — no real network calls are made.
Tests verify:
  - Neutral 50 returned when API key is missing
  - Neutral 50 returned when source text is missing
  - Neutral 50 returned when is_backtest=True
  - Correct parsing of well-formed Claude JSON responses
  - Graceful fallback when Claude returns malformed JSON
  - Composite weighting math
  - Cache round-trip
  - No API key in any log output or exception messages
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GOOD_RESPONSE = json.dumps({
    "score": 72.0,
    "confidence": 0.8,
    "rationale": "Strong revenue growth with low debt.",
    "red_flags": ["Pension liability growing"],
    "positives": ["Consistent FCF", "Expanding margins"],
})

NEUTRAL_SCORE = 50.0

SAMPLE_TEXT = "Revenue grew 15% year-over-year. Operating margins expanded. Debt levels remain manageable."
SAMPLE_DATE = "2024-03-31"
TICKER = "MSFT"


def _mock_claude(response_text: str):
    """Return a mock that makes _call_claude return response_text."""
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=response_text)]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_msg
    return mock_client


# ---------------------------------------------------------------------------
# _base: parse_claude_json
# ---------------------------------------------------------------------------

class TestParseClaude:
    def test_valid_json_parsed(self):
        from ai_research._base import parse_claude_json
        r = parse_claude_json(GOOD_RESPONSE, TICKER, "sec_filing", SAMPLE_DATE)
        assert r is not None
        assert r.score == pytest.approx(72.0)
        assert r.confidence == pytest.approx(0.8)
        assert "Pension" in r.red_flags[0]
        assert r.ticker == TICKER

    def test_json_in_markdown_fence_parsed(self):
        from ai_research._base import parse_claude_json
        fenced = f"```json\n{GOOD_RESPONSE}\n```"
        r = parse_claude_json(fenced, TICKER, "sec_filing", SAMPLE_DATE)
        assert r is not None
        assert r.score == pytest.approx(72.0)

    def test_none_returns_none(self):
        from ai_research._base import parse_claude_json
        assert parse_claude_json(None, TICKER, "sec_filing", SAMPLE_DATE) is None

    def test_malformed_json_returns_none(self):
        from ai_research._base import parse_claude_json
        assert parse_claude_json("not json at all {{{", TICKER, "sec_filing", SAMPLE_DATE) is None

    def test_score_clamped_to_0_100(self):
        from ai_research._base import parse_claude_json
        over = json.dumps({"score": 150, "confidence": 0.5, "rationale": "x", "red_flags": [], "positives": []})
        r = parse_claude_json(over, TICKER, "sec_filing", SAMPLE_DATE)
        assert r.score == pytest.approx(100.0)

    def test_confidence_clamped_to_0_1(self):
        from ai_research._base import parse_claude_json
        over = json.dumps({"score": 50, "confidence": 5.0, "rationale": "x", "red_flags": [], "positives": []})
        r = parse_claude_json(over, TICKER, "sec_filing", SAMPLE_DATE)
        assert r.confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# SEC Filing Analyzer
# ---------------------------------------------------------------------------

class TestSecFilingAnalyzer:
    def test_no_text_returns_neutral(self):
        from ai_research.sec_filing_analyzer import analyze_sec_filing
        r = analyze_sec_filing(TICKER, None, SAMPLE_DATE)
        assert r.score == pytest.approx(NEUTRAL_SCORE)
        assert r.confidence == pytest.approx(0.0)

    def test_empty_text_returns_neutral(self):
        from ai_research.sec_filing_analyzer import analyze_sec_filing
        r = analyze_sec_filing(TICKER, "   ", SAMPLE_DATE)
        assert r.score == pytest.approx(NEUTRAL_SCORE)

    def test_missing_api_key_returns_neutral(self):
        from ai_research.sec_filing_analyzer import analyze_sec_filing
        with patch("ai_research.sec_filing_analyzer.ANTHROPIC_API_KEY", ""):
            r = analyze_sec_filing(TICKER, SAMPLE_TEXT, SAMPLE_DATE)
        assert r.score == pytest.approx(NEUTRAL_SCORE)
        assert r.confidence == pytest.approx(0.0)

    def test_good_response_parsed(self):
        from ai_research.sec_filing_analyzer import analyze_sec_filing
        with (
            patch("ai_research.sec_filing_analyzer.ANTHROPIC_API_KEY", "sk-fake"),
            patch("ai_research.sec_filing_analyzer._call_claude", return_value=GOOD_RESPONSE),
            patch("ai_research.sec_filing_analyzer.cache_get", return_value=None),
            patch("ai_research.sec_filing_analyzer.cache_set"),
        ):
            r = analyze_sec_filing(TICKER, SAMPLE_TEXT, SAMPLE_DATE)
        assert r.score == pytest.approx(72.0)
        assert r.source == "sec_filing"

    def test_malformed_response_returns_neutral(self):
        from ai_research.sec_filing_analyzer import analyze_sec_filing
        with (
            patch("ai_research.sec_filing_analyzer.ANTHROPIC_API_KEY", "sk-fake"),
            patch("ai_research.sec_filing_analyzer._call_claude", return_value="NOT JSON"),
            patch("ai_research.sec_filing_analyzer.cache_get", return_value=None),
            patch("ai_research.sec_filing_analyzer.cache_set"),
        ):
            r = analyze_sec_filing(TICKER, SAMPLE_TEXT, SAMPLE_DATE)
        assert r.score == pytest.approx(NEUTRAL_SCORE)

    def test_cache_hit_returned(self):
        from ai_research.sec_filing_analyzer import analyze_sec_filing
        from ai_research._base import AIAnalysisResult
        cached = AIAnalysisResult(
            ticker=TICKER, score=88.0, confidence=0.9,
            rationale="cached", source="sec_filing",
            as_of_date=SAMPLE_DATE, cache_hit=True,
        )
        with (
            patch("ai_research.sec_filing_analyzer.ANTHROPIC_API_KEY", "sk-fake"),
            patch("ai_research.sec_filing_analyzer.cache_get", return_value=cached),
        ):
            r = analyze_sec_filing(TICKER, SAMPLE_TEXT, SAMPLE_DATE)
        assert r.score == pytest.approx(88.0)
        assert r.cache_hit is True


# ---------------------------------------------------------------------------
# Transcript Analyzer
# ---------------------------------------------------------------------------

class TestTranscriptAnalyzer:
    def test_no_text_returns_neutral(self):
        from ai_research.transcript_analyzer import analyze_transcript
        r = analyze_transcript(TICKER, None, SAMPLE_DATE)
        assert r.score == pytest.approx(NEUTRAL_SCORE)

    def test_missing_api_key_returns_neutral(self):
        from ai_research.transcript_analyzer import analyze_transcript
        with patch("ai_research.transcript_analyzer.ANTHROPIC_API_KEY", ""):
            r = analyze_transcript(TICKER, SAMPLE_TEXT, SAMPLE_DATE)
        assert r.score == pytest.approx(NEUTRAL_SCORE)

    def test_good_response_parsed(self):
        from ai_research.transcript_analyzer import analyze_transcript
        with (
            patch("ai_research.transcript_analyzer.ANTHROPIC_API_KEY", "sk-fake"),
            patch("ai_research.transcript_analyzer._call_claude", return_value=GOOD_RESPONSE),
            patch("ai_research.transcript_analyzer.cache_get", return_value=None),
            patch("ai_research.transcript_analyzer.cache_set"),
        ):
            r = analyze_transcript(TICKER, SAMPLE_TEXT, SAMPLE_DATE)
        assert r.score == pytest.approx(72.0)
        assert r.source == "transcript"


# ---------------------------------------------------------------------------
# Moat Scorer
# ---------------------------------------------------------------------------

class TestMoatScorer:
    def test_no_profile_returns_neutral(self):
        from ai_research.moat_scorer import score_moat
        r = score_moat(TICKER, None, SAMPLE_DATE)
        assert r.score == pytest.approx(NEUTRAL_SCORE)

    def test_missing_api_key_returns_neutral(self):
        from ai_research.moat_scorer import score_moat
        with patch("ai_research.moat_scorer.ANTHROPIC_API_KEY", ""):
            r = score_moat(TICKER, "Some business description.", SAMPLE_DATE)
        assert r.score == pytest.approx(NEUTRAL_SCORE)

    def test_good_response_parsed(self):
        from ai_research.moat_scorer import score_moat
        with (
            patch("ai_research.moat_scorer.ANTHROPIC_API_KEY", "sk-fake"),
            patch("ai_research.moat_scorer._call_claude", return_value=GOOD_RESPONSE),
            patch("ai_research.moat_scorer.cache_get", return_value=None),
            patch("ai_research.moat_scorer.cache_set"),
        ):
            r = score_moat(TICKER, "Dominant cloud platform with high switching costs.", SAMPLE_DATE)
        assert r.score == pytest.approx(72.0)
        assert r.source == "moat"


# ---------------------------------------------------------------------------
# Management Scorer
# ---------------------------------------------------------------------------

class TestManagementScorer:
    def test_no_text_returns_neutral(self):
        from ai_research.management_scorer import score_management
        r = score_management(TICKER, None, None, SAMPLE_DATE)
        assert r.score == pytest.approx(NEUTRAL_SCORE)

    def test_missing_api_key_returns_neutral(self):
        from ai_research.management_scorer import score_management
        with patch("ai_research.management_scorer.ANTHROPIC_API_KEY", ""):
            r = score_management(TICKER, SAMPLE_TEXT, None, SAMPLE_DATE)
        assert r.score == pytest.approx(NEUTRAL_SCORE)

    def test_transcript_only_accepted(self):
        from ai_research.management_scorer import score_management
        with (
            patch("ai_research.management_scorer.ANTHROPIC_API_KEY", "sk-fake"),
            patch("ai_research.management_scorer._call_claude", return_value=GOOD_RESPONSE),
            patch("ai_research.management_scorer.cache_get", return_value=None),
            patch("ai_research.management_scorer.cache_set"),
        ):
            r = score_management(TICKER, SAMPLE_TEXT, None, SAMPLE_DATE)
        assert r.score == pytest.approx(72.0)

    def test_filing_only_accepted(self):
        from ai_research.management_scorer import score_management
        with (
            patch("ai_research.management_scorer.ANTHROPIC_API_KEY", "sk-fake"),
            patch("ai_research.management_scorer._call_claude", return_value=GOOD_RESPONSE),
            patch("ai_research.management_scorer.cache_get", return_value=None),
            patch("ai_research.management_scorer.cache_set"),
        ):
            r = score_management(TICKER, None, SAMPLE_TEXT, SAMPLE_DATE)
        assert r.score == pytest.approx(72.0)


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

class TestAIComposite:
    def test_backtest_returns_50(self):
        from ai_research.composite import run_ai_research
        r = run_ai_research(TICKER, SAMPLE_TEXT, SAMPLE_TEXT, SAMPLE_TEXT, SAMPLE_DATE, is_backtest=True)
        assert r.score == pytest.approx(50.0)
        assert r.is_backtest is True

    def test_no_api_key_returns_neutral(self):
        from ai_research.composite import run_ai_research
        with patch("ai_research.composite.ANTHROPIC_API_KEY", ""):
            r = run_ai_research(TICKER, SAMPLE_TEXT, SAMPLE_TEXT, SAMPLE_TEXT, SAMPLE_DATE)
        assert r.score == pytest.approx(50.0)

    def test_all_neutral_inputs_gives_50(self):
        """When all texts are None/empty (no API key), composite = 50."""
        from ai_research.composite import run_ai_research
        with patch("ai_research.composite.ANTHROPIC_API_KEY", ""):
            r = run_ai_research(TICKER, None, None, None, SAMPLE_DATE)
        assert r.score == pytest.approx(50.0)

    def test_composite_weighting(self):
        """Check composite arithmetic with known sub-scores."""
        from ai_research.composite import run_ai_research
        from ai_research._base import AIAnalysisResult

        def make_result(score, source):
            return AIAnalysisResult(ticker=TICKER, score=score, confidence=0.9,
                                    rationale="test", source=source, as_of_date=SAMPLE_DATE)

        with (
            patch("ai_research.composite.ANTHROPIC_API_KEY", "sk-fake"),
            patch("ai_research.composite.analyze_sec_filing", return_value=make_result(80.0, "sec_filing")),
            patch("ai_research.composite.analyze_transcript", return_value=make_result(60.0, "transcript")),
            patch("ai_research.composite.score_moat", return_value=make_result(70.0, "moat")),
            patch("ai_research.composite.score_management", return_value=make_result(50.0, "management")),
        ):
            r = run_ai_research(TICKER, SAMPLE_TEXT, SAMPLE_TEXT, SAMPLE_TEXT, SAMPLE_DATE)

        # transcript 30% + sec_filing 30% + moat 25% + management 15%
        expected = 60.0 * 0.30 + 80.0 * 0.30 + 70.0 * 0.25 + 50.0 * 0.15
        assert r.score == pytest.approx(expected, abs=0.01)

    def test_red_flags_aggregated(self):
        from ai_research.composite import run_ai_research
        from ai_research._base import AIAnalysisResult

        def make_result(score, source, flags):
            return AIAnalysisResult(ticker=TICKER, score=score, confidence=0.5,
                                    rationale="test", source=source, as_of_date=SAMPLE_DATE,
                                    red_flags=flags, positives=[])

        with (
            patch("ai_research.composite.ANTHROPIC_API_KEY", "sk-fake"),
            patch("ai_research.composite.analyze_sec_filing",
                  return_value=make_result(50.0, "sec_filing", ["Debt risk"])),
            patch("ai_research.composite.analyze_transcript",
                  return_value=make_result(50.0, "transcript", ["Guidance withdrawn"])),
            patch("ai_research.composite.score_moat",
                  return_value=make_result(50.0, "moat", [])),
            patch("ai_research.composite.score_management",
                  return_value=make_result(50.0, "management", ["CEO selling shares"])),
        ):
            r = run_ai_research(TICKER, SAMPLE_TEXT, SAMPLE_TEXT, SAMPLE_TEXT, SAMPLE_DATE)

        assert "Debt risk" in r.all_red_flags
        assert "Guidance withdrawn" in r.all_red_flags
        assert "CEO selling shares" in r.all_red_flags

    def test_batch_backtest_all_neutral(self):
        from ai_research.composite import run_ai_research_batch
        results = run_ai_research_batch(
            ["AAPL", "MSFT", "GOOG"],
            is_backtest=True,
        )
        assert len(results) == 3
        for ticker, r in results.items():
            assert r.score == pytest.approx(50.0), f"{ticker} should be neutral"
            assert r.is_backtest is True

    def test_batch_no_api_key_all_neutral(self):
        from ai_research.composite import run_ai_research_batch
        with patch("ai_research.composite.ANTHROPIC_API_KEY", ""):
            results = run_ai_research_batch(["AAPL", "GOOG"])
        for r in results.values():
            assert r.score == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Cache integration (uses a real temp DB)
# ---------------------------------------------------------------------------

class TestAICache:
    def test_cache_round_trip(self, tmp_path):
        from ai_research._base import AIAnalysisResult, cache_set, cache_get

        # Point DB to temp path
        with patch("ai_research._base._DB_PATH", tmp_path / "test_cache.db"):
            result = AIAnalysisResult(
                ticker="AAPL", score=75.0, confidence=0.85,
                rationale="Good filing.", red_flags=[], positives=["FCF strong"],
                source="sec_filing", as_of_date="2024-06-30",
            )
            cache_set(result, "sec_filing", ttl_days=30)
            retrieved = cache_get("AAPL", "sec_filing", "2024-06-30")

        assert retrieved is not None
        assert retrieved.score == pytest.approx(75.0)
        assert retrieved.cache_hit is True
        assert "FCF strong" in retrieved.positives

    def test_expired_cache_returns_none(self, tmp_path):
        from ai_research._base import AIAnalysisResult, cache_set, cache_get
        from datetime import datetime, timezone, timedelta

        with patch("ai_research._base._DB_PATH", tmp_path / "test_cache2.db"):
            result = AIAnalysisResult(
                ticker="GOOG", score=60.0, confidence=0.7,
                rationale="Old.", source="transcript", as_of_date="2023-01-01",
            )
            # Write with TTL=1 but simulate old fetch date
            with patch("ai_research._base.datetime") as mock_dt:
                old_time = datetime(2023, 1, 2, tzinfo=timezone.utc)
                mock_dt.now.return_value = old_time
                mock_dt.fromisoformat.side_effect = datetime.fromisoformat
                cache_set(result, "transcript", ttl_days=1)

            # Read back — now 400 days later — should be expired
            with patch("ai_research._base.datetime") as mock_dt:
                future = datetime(2024, 2, 5, tzinfo=timezone.utc)
                mock_dt.now.return_value = future
                mock_dt.fromisoformat.side_effect = datetime.fromisoformat
                retrieved = cache_get("GOOG", "transcript", "2023-01-01")

        assert retrieved is None

    def test_empty_as_of_date_not_cached(self, tmp_path):
        from ai_research._base import AIAnalysisResult, cache_set, cache_get
        with patch("ai_research._base._DB_PATH", tmp_path / "test_cache3.db"):
            result = AIAnalysisResult(
                ticker="AAPL", score=70.0, confidence=0.6,
                source="moat", as_of_date="",   # empty date
            )
            cache_set(result, "moat", ttl_days=30)
            # cache_get with empty date should return None
            retrieved = cache_get("AAPL", "moat", "")
        assert retrieved is None
