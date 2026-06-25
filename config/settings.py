import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# API Keys — loaded from .env, never hardcoded
# ---------------------------------------------------------------------------

FMP_API_KEY: str = os.getenv("FMP_API_KEY", "")
FINNHUB_API_KEY: str = os.getenv("FINNHUB_API_KEY") or os.getenv("finnhub_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

# ---------------------------------------------------------------------------
# Provider URLs — modular so new providers slot in without touching callers
# ---------------------------------------------------------------------------

FMP_BASE_URL = "https://financialmodelingprep.com/stable"
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

ANTHROPIC_MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Portfolio constants
# ---------------------------------------------------------------------------

ACCOUNT_SIZE: float = 2_000.0
MAX_POSITIONS: int = 5
MAX_STOCK_PRICE: float = 1_000.0

# Universe + portfolio filters — single source of truth used by both
# screener/live_universe.py and portfolio/constraints.py
PORTFOLIO_MIN_MARKET_CAP: float   = 2_000_000_000   # $2B
PORTFOLIO_MIN_AVG_VOLUME: float   = 500_000          # shares/day
UNIVERSE_MIN_MARKET_CAP:  float   = 2_000_000_000   # $2B  (same as portfolio gate)
UNIVERSE_MIN_AVG_VOLUME:  float   = 500_000          # shares/day

CONVICTION_WEIGHTS: list[float] = [0.30, 0.25, 0.20, 0.15, 0.10]
CONVICTION_DOLLARS: list[float] = [600.0, 500.0, 400.0, 300.0, 200.0]

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

AI_CACHE_TTL_FILINGS_DAYS: int = 30
AI_CACHE_TTL_TRANSCRIPTS_DAYS: int = 7

# ---------------------------------------------------------------------------
# Backtest constants
# ---------------------------------------------------------------------------

BACKTEST_AI_NEUTRAL: float = 50.0      # AI Research score used for all historical periods
BACKTEST_IN_SAMPLE_MONTHS: int = 12
BACKTEST_OUT_SAMPLE_MONTHS: int = 3
