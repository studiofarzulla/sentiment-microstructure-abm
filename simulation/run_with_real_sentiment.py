"""
Run simulation with real CryptoBERT sentiment analysis.

This script demonstrates the full pipeline:
1. Fetch recent crypto news/posts
2. Analyze with CryptoBERT + MC Dropout
3. Feed into Mesa multi-agent simulation
4. Generate results

Author: Murad Farzulla
Date: December 2025
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Simulation imports
from simulation.market_env import create_default_market, CryptoMarketModel
from simulation.order_book import OrderBook

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# CryptoBERT Sentiment Analyzer
# ============================================================================

class CryptoBERTSentiment:
    """CryptoBERT sentiment analyzer with MC Dropout uncertainty."""

    def __init__(self, n_mc_samples: int = 20, cache_dir: str = "/tmp/hf_cache"):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch.nn.functional as F

        self.torch = torch
        self.F = F

        os.environ['HF_HOME'] = cache_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading CryptoBERT on {self.device}...")

        model_name = "ElKulako/cryptobert"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.n_mc_samples = n_mc_samples

        # Enable dropout for MC sampling
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

        logger.info(f"CryptoBERT loaded with {n_mc_samples} MC samples")

    def analyze(self, text: str) -> Tuple[float, float, float]:
        """
        Analyze sentiment with uncertainty.

        Returns: (sentiment, epistemic_uncertainty, aleatoric_uncertainty)
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        all_probs = []
        with self.torch.no_grad():
            for _ in range(self.n_mc_samples):
                outputs = self.model(**inputs)
                probs = self.F.softmax(outputs.logits, dim=-1)
                all_probs.append(probs.cpu().numpy())

        all_probs = np.array(all_probs).squeeze()
        mean_probs = all_probs.mean(axis=0)

        # Sentiment: -1*P(bearish) + 0*P(neutral) + 1*P(bullish)
        sentiment = -1 * mean_probs[0] + 0 * mean_probs[1] + 1 * mean_probs[2]

        # Epistemic: variance across MC samples
        epistemic = all_probs.var(axis=0).mean()

        # Aleatoric: entropy of mean prediction
        aleatoric = -np.sum(mean_probs * np.log(mean_probs + 1e-10))

        return float(sentiment), float(epistemic), float(aleatoric)

    def analyze_batch(self, texts: List[str]) -> List[Tuple[float, float, float]]:
        """Analyze multiple texts."""
        return [self.analyze(t) for t in texts]


# ============================================================================
# Sample News/Posts for Demo
# ============================================================================

SAMPLE_CRYPTO_NEWS = [
    # Bullish
    "Breaking: Bitcoin ETF officially approved by SEC! Institutional money incoming. BTC to the moon!",
    "Ethereum hits new all-time high as DeFi TVL surpasses $100B. Incredible growth.",
    "MicroStrategy buys another 10,000 BTC. Saylor remains incredibly bullish on Bitcoin.",
    "Grayscale converts GBTC to spot ETF. Massive inflows expected this week.",
    "Bitcoin hashrate reaches new ATH. Network security stronger than ever.",

    # Neutral
    "Bitcoin trading sideways around $42,000. Market waiting for next catalyst.",
    "Crypto market cap steady at $1.7 trillion. Low volatility week ahead.",
    "SEC schedules meeting to discuss crypto regulations next Tuesday.",
    "Binance reports normal trading volumes. No significant changes.",
    "Bitcoin dominance hovers around 52%. Altcoins relatively stable.",

    # Bearish
    "FTX collapse aftermath: More exchanges face liquidity concerns.",
    "SEC sues major crypto exchange for securities violations.",
    "Bitcoin drops 10% as whale wallets move to exchanges.",
    "Crypto winter continues: Another DeFi protocol exploited for $50M.",
    "Mt. Gox creditors finally receiving payouts, massive selling pressure expected.",

    # Mixed/Uncertain
    "Is this a dead cat bounce or real recovery? Analysts divided.",
    "Whale alert: 10,000 BTC moved from cold storage. Intentions unclear.",
    "Conflicting signals in crypto market. Volume weak but price holding.",
    "China rumors resurface. Market uncertain about regulatory impact.",
    "Bitcoin forms bearish pattern on 4H but bullish on daily. Mixed signals.",
]


def create_sentiment_sequence(
    analyzer: CryptoBERTSentiment,
    news_items: List[str],
    n_steps: int = 500,
    news_interval: int = 25,
    decay_rate: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a time series of sentiment by cycling through news items.

    Sentiment decays toward neutral between news events.

    Returns: (sentiment, epistemic, aleatoric, regime_labels)
    """
    sentiment = np.zeros(n_steps)
    epistemic = np.zeros(n_steps)
    aleatoric = np.zeros(n_steps)
    regime_labels = np.empty(n_steps, dtype=object)

    current_sent = 0.0
    current_epi = 0.05
    current_aleat = 0.2

    news_idx = 0

    for t in range(n_steps):
        # New news item arrives
        if t % news_interval == 0 and news_idx < len(news_items):
            text = news_items[news_idx]
            logger.info(f"Step {t}: Analyzing news #{news_idx+1}")

            sent, epi, aleat = analyzer.analyze(text)
            current_sent = sent
            current_epi = epi
            current_aleat = aleat
            news_idx += 1

        else:
            # Decay toward neutral
            current_sent *= decay_rate
            current_epi = 0.05 + abs(current_sent) * 0.02
            current_aleat = 0.15 + (1 - abs(current_sent)) * 0.1

        # Add small noise
        noisy_sent = np.clip(current_sent + np.random.normal(0, 0.02), -1, 1)

        sentiment[t] = noisy_sent
        epistemic[t] = current_epi + np.random.uniform(0, 0.01)
        aleatoric[t] = current_aleat + np.random.uniform(0, 0.02)

        # Classify regime
        if noisy_sent > 0.2:
            regime_labels[t] = 'bullish'
        elif noisy_sent < -0.2:
            regime_labels[t] = 'bearish'
        else:
            regime_labels[t] = 'neutral'

    return sentiment, epistemic, aleatoric, regime_labels


def run_simulation_with_real_sentiment(
    n_steps: int = 500,
    n_market_makers: int = 2,
    n_informed: int = 5,
    n_noise: int = 10,
    initial_price: float = 42000.0,  # Realistic BTC price
    seed: int = 42,
) -> pd.DataFrame:
    """Run full simulation with CryptoBERT sentiment."""

    logger.info("=" * 60)
    logger.info("SENTIMENT-MICROSTRUCTURE ABM: REAL SENTIMENT RUN")
    logger.info("=" * 60)

    # Initialize sentiment analyzer
    logger.info("\n1. Loading CryptoBERT sentiment analyzer...")
    analyzer = CryptoBERTSentiment(n_mc_samples=15)  # Fewer samples for speed

    # Generate sentiment sequence from real news
    logger.info("\n2. Generating sentiment sequence from news...")
    sentiment, epistemic, aleatoric, regimes = create_sentiment_sequence(
        analyzer, SAMPLE_CRYPTO_NEWS, n_steps=n_steps
    )

    logger.info(f"   Sentiment range: [{sentiment.min():.2f}, {sentiment.max():.2f}]")
    logger.info(f"   Mean epistemic: {epistemic.mean():.3f}")
    logger.info(f"   Mean aleatoric: {aleatoric.mean():.3f}")

    # Create market
    logger.info("\n3. Creating multi-agent market...")
    model = create_default_market(
        n_market_makers=n_market_makers,
        n_informed=n_informed,
        n_noise=n_noise,
        initial_price=initial_price,
        seed=seed,
    )

    logger.info(f"   Market makers: {n_market_makers}")
    logger.info(f"   Informed traders: {n_informed}")
    logger.info(f"   Noise traders: {n_noise}")

    # Create sentiment generator from pre-computed values
    def sentiment_gen(step):
        return (
            sentiment[step],
            epistemic[step],
            aleatoric[step],
            regimes[step]
        )

    # Run simulation
    logger.info(f"\n4. Running {n_steps}-step simulation...")
    history = model.run_simulation(n_steps, sentiment_generator=sentiment_gen)

    # Convert to DataFrame
    df = pd.DataFrame(history)

    # Add sentiment columns
    df['sentiment'] = sentiment[:len(df)]
    df['epistemic_uncertainty'] = epistemic[:len(df)]
    df['aleatoric_uncertainty'] = aleatoric[:len(df)]
    df['regime'] = regimes[:len(df)]

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SIMULATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total steps: {len(df)}")
    logger.info(f"Total trades: {df['trade_count'].iloc[-1]}")
    logger.info(f"Total volume: {df['total_volume'].iloc[-1]:.2f}")

    # Handle None values in price
    valid_prices = df['mid_price'].dropna()
    if len(valid_prices) > 0:
        logger.info(f"Price: ${valid_prices.iloc[0]:.2f} -> ${valid_prices.iloc[-1]:.2f}")
    else:
        logger.warning("No valid price data!")

    # Handle None values in spread
    valid_spread = df['spread_bps'].dropna()
    if len(valid_spread) > 0:
        logger.info(f"Mean spread: {valid_spread.mean():.2f} bps")

    # Correlations (only on valid data)
    logger.info("\nKey Correlations:")
    valid_df = df.dropna(subset=['spread_bps'])
    if len(valid_df) > 10:
        logger.info(f"  Sentiment-Spread: {valid_df['sentiment'].corr(valid_df['spread_bps']):.3f}")
        logger.info(f"  Epistemic-Spread: {valid_df['epistemic_uncertainty'].corr(valid_df['spread_bps']):.3f}")
        logger.info(f"  Aleatoric-Spread: {valid_df['aleatoric_uncertainty'].corr(valid_df['spread_bps']):.3f}")
    else:
        logger.warning("Insufficient data for correlation analysis")

    return df


def main():
    # Run simulation
    df = run_simulation_with_real_sentiment(
        n_steps=500,
        n_market_makers=2,
        n_informed=5,
        n_noise=10,
        initial_price=42000.0,
        seed=42,
    )

    # Save results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "real_sentiment_results.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"\nSaved results to: {output_path}")

    return df


if __name__ == "__main__":
    main()
