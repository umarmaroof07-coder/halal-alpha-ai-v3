# Halal Alpha AI V3

Shariah-compliant U.S. stock research and recommendation system.

## Data Providers

Providers are tried in order for every data request. The first successful response wins and is cached.

| Priority | Provider | Role |
|----------|----------|------|
| 1 | **FMP** (Financial Modeling Prep) | Primary — financial statements, estimates, ratios, quotes, price targets, earnings calendar |
| 2 | **Finnhub** | Secondary — fills gaps when FMP returns empty or unavailable |
| 3 | **yfinance** | Final fallback — emergency safety net only |

AI Research (earnings transcript analysis, 10-K/10-Q analysis, moat scoring, management scoring) is handled exclusively by the **Anthropic Claude API** and is never part of the data fallback chain.

## Setup

```bash
cp .env.example .env
# Fill in all three keys in .env

pip install -r requirements.txt
```

## Run

```bash
# Dashboard
streamlit run app.py

# CLI
python main.py
```

## Test

```bash
pytest tests/
```

## Portfolio rules

- $2,000 account · Top 5 Shariah-compliant U.S. stocks only
- Conviction weights: 30% / 25% / 20% / 15% / 10%
- Dollar allocations: $600 / $500 / $400 / $300 / $200
- Stock price must be under $1,000
- No options, leverage, shorting, penny stocks, meme stocks, or SPACs

## Factor weights

| Factor | Weight |
|--------|--------|
| Quality | 35% |
| Momentum | 25% |
| Earnings Revisions | 15% |
| Valuation | 10% |
| AI Research | 15% |

## Safety rule

Recommendations may **only** come from `safe_recommendations(portfolio_result, compliant_universe)`.
Never from raw rankings, factor outputs, or unconstrained dataframes.

## Backtest integrity

During historical backtests, AI Research is locked to **50 (neutral)**.
No current filings or transcripts are used for past periods.
