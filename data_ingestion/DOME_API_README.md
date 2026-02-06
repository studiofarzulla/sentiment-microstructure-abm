# Dome API Integration - Prediction Market Sentiment

**Informed Sentiment from Prediction Markets**

Dome API provides access to prediction market data from Polymarket and Kalshi. Prediction markets aggregate information from participants with real money at stake, providing a unique "informed sentiment" signal distinct from social media (retail) and institutional indicators (ASRI).

## Why Prediction Markets?

Prediction markets offer several advantages for sentiment analysis:

1. **Informed Participants**: People betting real money have incentives to be accurate
2. **Aggregated Wisdom**: Prices reflect collective beliefs, not individual opinions
3. **Real-Time Updates**: Markets update continuously as new information arrives
4. **Microstructure Data**: Orderbooks and trade history similar to traditional exchanges

## Quick Start

### 1. Get API Key

Sign up at [docs.domeapi.io](https://docs.domeapi.io) and get your free API key.

### 2. Configure Environment

```bash
# Add to .env file
DOME_API_KEY=your_api_key_here
```

### 3. Basic Usage

```python
from data_ingestion.dome_client import DomeAPIClient

# Initialize client
client = DomeAPIClient(platform='polymarket')

# Get a single market
market = client.get_market('will-bitcoin-price-be-above-50000-usd-on-december-31-2025')
print(f"Price: {market.current_price:.2%}")
print(f"Sentiment: {market.sentiment:.3f}")  # [-1, 1]

# Get aggregated crypto sentiment
sentiment = client.get_crypto_sentiment()
print(f"Weighted Sentiment: {sentiment.weighted_sentiment:.3f}")
print(f"Uncertainty: {sentiment.sentiment_std:.3f}")
```

### 4. CLI Usage

```bash
# Get a specific market
python -m data_ingestion.dome_client --market will-bitcoin-price-be-above-50000-usd-on-december-31-2025

# Get aggregated crypto sentiment
python -m data_ingestion.dome_client --crypto-sentiment

# Search for markets
python -m data_ingestion.dome_client --search bitcoin
```

## Integration with SignalComposer

Add prediction market sentiment as a third dimension to your existing sentiment blend:

```python
from data_ingestion.dome_integration_example import EnhancedSignalComposer

# Initialize with prediction market weight
composer = EnhancedSignalComposer(
    prediction_weight=0.20,  # 20% weight for prediction markets
    macro_weight=0.30,       # 30% for ASRI
    micro_weight=0.50,       # 50% for Reddit/CryptoBERT
)

# Compose enhanced tick
tick = composer.compose_enhanced(
    macro_signals=asri_signals,
    micro_sentiment=(reddit_sent, epi, aleat),
    prediction_market_slugs=['will-bitcoin-price-be-above-50000-usd-on-december-31-2025'],
    price=45000.0,
)
```

## Sentiment Conversion

Prediction market prices (probabilities) are converted to sentiment scores:

- **Price 0.8** (80% chance) → **+0.6 sentiment** (bullish)
- **Price 0.5** (50% chance) → **0.0 sentiment** (neutral)
- **Price 0.2** (20% chance) → **-0.6 sentiment** (bearish)

Formula: `sentiment = 2 * (price - 0.5)`

## Crypto Markets

The client includes common crypto-related market slugs:

- `will-bitcoin-price-be-above-50000-usd-on-december-31-2025`
- `will-bitcoin-price-be-above-60000-usd-on-december-31-2025`
- `will-bitcoin-price-be-above-70000-usd-on-december-31-2025`
- `will-ethereum-price-be-above-3000-usd-on-december-31-2025`
- `will-bitcoin-reach-100000-usd-before-2026`

You can also search for markets:

```python
markets = client.search_markets('bitcoin')
```

## API Endpoints

The client supports:

- **Markets**: Get market information and prices
- **Activity**: Recent trades/activity (for price extraction)
- **Orderbook**: Current orderbook (for microstructure analysis)
- **Trade History**: Historical trades
- **Orderbook History**: Historical orderbook snapshots

## Rate Limits

Free tier: **1 QPS, 10 queries per 10 seconds**

The client automatically handles rate limiting with appropriate delays.

## Research Applications

### 1. Sentiment Comparison

Compare prediction market sentiment vs Reddit sentiment:

```python
pred_sentiment = client.get_crypto_sentiment()
reddit_sentiment = 0.3  # From CryptoBERT

divergence = pred_sentiment.weighted_sentiment - reddit_sentiment
print(f"Prediction markets more bullish: {divergence > 0}")
```

### 2. Microstructure Analysis

Analyze prediction market orderbooks similar to Binance:

```python
orderbook = client.get_orderbook('will-bitcoin-price-be-above-50000-usd-on-december-31-2025')
# Compare spread, depth, imbalance with crypto exchange orderbooks
```

### 3. Cross-Market Arbitrage

Model arbitrage opportunities between prediction markets and spot markets:

```python
# If prediction market says 80% chance BTC > $50k
# But spot price is $45k, is there an arbitrage opportunity?
```

## Example Output

```
============================================================
CRYPTO PREDICTION MARKET SENTIMENT
============================================================
Platform: polymarket
Markets: 5
Mean Sentiment: 0.342
Weighted Sentiment: 0.387
Uncertainty (std): 0.156
Total Volume 24h: $1,234,567
Price Range: 0.45 - 0.85

For SignalComposer: sentiment=0.387, uncertainty=0.312
```

## Files

- `dome_client.py` - Main API client
- `dome_integration_example.py` - Integration with SignalComposer
- `DOME_API_README.md` - This file

## Requirements

- `requests` (already in requirements.txt)
- `python-dotenv` (already in requirements.txt)

## References

- [Dome API Documentation](https://docs.domeapi.io)
- [Polymarket](https://polymarket.com)
- [Kalshi](https://kalshi.com)

## Notes

- Prediction markets may have limited liquidity for some crypto markets
- Prices reflect probabilities, not direct sentiment (need conversion)
- Free tier rate limits may require batching for multiple markets
- Some markets may not have recent activity (price may be stale)
