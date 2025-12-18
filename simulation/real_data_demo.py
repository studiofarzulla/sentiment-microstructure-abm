"""
Real Data Demo - Fetch Reddit Posts and Run Simulation

This script demonstrates the full pipeline with REAL data:
1. Fetch recent posts from crypto subreddits (no Kafka needed)
2. Analyze sentiment with CryptoBERT + MC Dropout
3. Run multi-agent simulation

Usage:
    # Requires Reddit API credentials in .env
    HF_HOME=/tmp/hf_cache python simulation/real_data_demo.py

Author: Murad Farzulla
Date: December 2025
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import Reddit client
try:
    import praw
    from dotenv import load_dotenv
    load_dotenv()
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False

from simulation.market_env import create_default_market
from simulation.run_with_real_sentiment import CryptoBERTSentiment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Reddit Data Fetcher (Batch Mode - No Kafka)
# ============================================================================

class RedditFetcher:
    """Fetch recent posts from crypto subreddits without streaming."""

    def __init__(self):
        if not REDDIT_AVAILABLE:
            raise ImportError("praw not installed. Run: pip install praw python-dotenv")

        # Check for credentials
        client_id = os.getenv('REDDIT_CLIENT_ID')
        client_secret = os.getenv('REDDIT_CLIENT_SECRET')

        if not client_id or not client_secret:
            raise ValueError(
                "Reddit API credentials not found.\n"
                "Create a .env file with:\n"
                "  REDDIT_CLIENT_ID=your_client_id\n"
                "  REDDIT_CLIENT_SECRET=your_client_secret\n"
                "  REDDIT_USER_AGENT='sentiment-abm/1.0'"
            )

        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=os.getenv('REDDIT_USER_AGENT', 'sentiment-abm/1.0 by /u/anonymous')
        )

        self.subreddits = [
            'CryptoCurrency',
            'Bitcoin',
            'ethereum',
            'CryptoMarkets',
        ]

    def fetch_recent_posts(
        self,
        n_posts: int = 50,
        hours_back: int = 24,
        sort: str = 'hot'
    ) -> List[dict]:
        """
        Fetch recent posts from crypto subreddits.

        Args:
            n_posts: Max posts per subreddit
            hours_back: Only posts within this time window
            sort: 'hot', 'new', or 'top'
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours_back)
        posts = []

        for sub_name in self.subreddits:
            try:
                subreddit = self.reddit.subreddit(sub_name)

                if sort == 'hot':
                    submissions = subreddit.hot(limit=n_posts)
                elif sort == 'new':
                    submissions = subreddit.new(limit=n_posts)
                else:
                    submissions = subreddit.top(time_filter='day', limit=n_posts)

                for submission in submissions:
                    created = datetime.utcfromtimestamp(submission.created_utc)
                    if created >= cutoff:
                        posts.append({
                            'id': submission.id,
                            'title': submission.title,
                            'selftext': submission.selftext[:500] if submission.selftext else '',
                            'subreddit': sub_name,
                            'score': submission.score,
                            'num_comments': submission.num_comments,
                            'created_utc': created.isoformat(),
                            'url': submission.url,
                        })

                logger.info(f"Fetched from r/{sub_name}: {len([p for p in posts if p['subreddit'] == sub_name])} posts")

            except Exception as e:
                logger.warning(f"Error fetching r/{sub_name}: {e}")
                continue

        return posts

    def get_text_for_analysis(self, posts: List[dict]) -> List[str]:
        """Extract text from posts for sentiment analysis."""
        texts = []
        for post in posts:
            # Combine title and body
            text = post['title']
            if post['selftext']:
                text += ' ' + post['selftext']
            texts.append(text)
        return texts


# ============================================================================
# Fallback: Sample News (if Reddit unavailable)
# ============================================================================

FALLBACK_NEWS = [
    "Bitcoin ETF approved by SEC! Major institutional inflows expected.",
    "Ethereum gas fees drop to lowest level in months, bullish for DeFi.",
    "Crypto market cap hits new ATH as BTC breaks $50k resistance.",
    "SEC announces new crypto regulations, market reacts cautiously.",
    "Major exchange hacked, $100M stolen. Security concerns rise.",
    "Bitcoin mining difficulty reaches all-time high, network stronger.",
    "FTX creditors to receive partial payouts, selling pressure looms.",
    "Layer 2 solutions gain traction as ETH scalability improves.",
    "Whale moves 10,000 BTC to exchange, dumping concerns.",
    "Crypto winter over? Analysts predict bull run continuation.",
]


# ============================================================================
# Main Demo
# ============================================================================

def run_real_data_simulation(
    n_steps: int = 300,
    use_reddit: bool = True,
    n_posts: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Run simulation with real data if available, fallback to samples."""

    logger.info("=" * 60)
    logger.info("REAL DATA DEMONSTRATION")
    logger.info("=" * 60)

    # Try to fetch real Reddit data
    texts = []
    source = "fallback"

    if use_reddit and REDDIT_AVAILABLE:
        try:
            logger.info("\n1. Fetching Reddit posts...")
            fetcher = RedditFetcher()
            posts = fetcher.fetch_recent_posts(n_posts=n_posts, hours_back=24)

            if posts:
                texts = fetcher.get_text_for_analysis(posts)
                source = "reddit"
                logger.info(f"   Fetched {len(texts)} posts from Reddit")
            else:
                logger.warning("   No posts retrieved, using fallback")

        except Exception as e:
            logger.warning(f"   Reddit fetch failed: {e}")
            logger.warning("   Using fallback sample news")

    if not texts:
        texts = FALLBACK_NEWS
        source = "fallback"
        logger.info(f"\n1. Using {len(texts)} fallback news items")

    # Initialize sentiment analyzer
    logger.info("\n2. Loading CryptoBERT...")
    os.environ['HF_HOME'] = '/tmp/hf_cache'
    analyzer = CryptoBERTSentiment(n_mc_samples=15)

    # Analyze all texts
    logger.info("\n3. Analyzing sentiment...")
    sentiments = []
    for i, text in enumerate(texts):
        sent, epi, aleat = analyzer.analyze(text[:512])  # Truncate for BERT
        sentiments.append({
            'text': text[:100] + '...' if len(text) > 100 else text,
            'sentiment': sent,
            'epistemic': epi,
            'aleatoric': aleat,
        })

        if (i + 1) % 10 == 0:
            logger.info(f"   Analyzed {i+1}/{len(texts)} texts")

    # Summary of sentiment
    sent_df = pd.DataFrame(sentiments)
    logger.info(f"\n   Sentiment Summary:")
    logger.info(f"   Mean: {sent_df['sentiment'].mean():.3f}")
    logger.info(f"   Std: {sent_df['sentiment'].std():.3f}")
    logger.info(f"   Range: [{sent_df['sentiment'].min():.3f}, {sent_df['sentiment'].max():.3f}]")

    # Create sentiment time series for simulation
    # Cycle through analyzed sentiments with decay
    def create_sentiment_gen():
        """Generator that cycles through real sentiments with decay."""
        sent_list = sentiments.copy()
        idx = 0
        current_sent = 0.0
        current_epi = 0.05
        current_aleat = 0.2

        def gen(step):
            nonlocal idx, current_sent, current_epi, current_aleat

            # New sentiment every 15 steps
            if step % 15 == 0 and idx < len(sent_list):
                s = sent_list[idx]
                current_sent = s['sentiment']
                current_epi = s['epistemic']
                current_aleat = s['aleatoric']
                idx += 1
            else:
                # Decay toward neutral
                current_sent *= 0.95
                current_epi = 0.05 + abs(current_sent) * 0.02

            # Classify regime
            if current_sent > 0.2:
                regime = 'bullish'
            elif current_sent < -0.2:
                regime = 'bearish'
            else:
                regime = 'neutral'

            # Add noise
            noisy_sent = np.clip(current_sent + np.random.normal(0, 0.02), -1, 1)

            return noisy_sent, current_epi, current_aleat, regime

        return gen

    # Create market
    logger.info("\n4. Creating market...")
    np.random.seed(seed)
    model = create_default_market(
        n_market_makers=2,
        n_informed=5,
        n_noise=10,
        initial_price=42000.0,
        seed=seed,
    )

    # Run simulation
    logger.info(f"\n5. Running {n_steps}-step simulation...")
    sentiment_gen = create_sentiment_gen()
    history = model.run_simulation(n_steps, sentiment_generator=sentiment_gen)

    # Convert to DataFrame
    df = pd.DataFrame(history)

    # Results
    logger.info("\n" + "=" * 60)
    logger.info(f"SIMULATION COMPLETE (Source: {source.upper()})")
    logger.info("=" * 60)
    logger.info(f"Steps: {len(df)}")
    logger.info(f"Trades: {df['trade_count'].iloc[-1]}")

    valid_prices = df['mid_price'].dropna()
    if len(valid_prices) > 0:
        logger.info(f"Price: ${valid_prices.iloc[0]:.2f} -> ${valid_prices.iloc[-1]:.2f}")

    valid_spread = df['spread_bps'].dropna()
    if len(valid_spread) > 0:
        logger.info(f"Mean spread: {valid_spread.mean():.2f} bps")

    # Save
    output_dir = os.path.dirname(os.path.abspath(__file__))
    df.to_csv(os.path.join(output_dir, 'real_data_results.csv'), index=False)
    sent_df.to_csv(os.path.join(output_dir, 'sentiment_analysis.csv'), index=False)
    logger.info(f"\nResults saved to simulation/")

    return df, sent_df


def main():
    # Check environment
    if REDDIT_AVAILABLE:
        logger.info("Reddit API available - will attempt real data fetch")
    else:
        logger.info("Reddit API not available - using fallback news")

    df, sent_df = run_real_data_simulation(
        n_steps=300,
        use_reddit=True,
        n_posts=30,
        seed=42,
    )

    # Show top sentiment items
    print("\n=== TOP SENTIMENT ITEMS ===")
    top_bullish = sent_df.nlargest(3, 'sentiment')
    top_bearish = sent_df.nsmallest(3, 'sentiment')

    print("\nMost Bullish:")
    for _, row in top_bullish.iterrows():
        print(f"  [{row['sentiment']:+.3f}] {row['text']}")

    print("\nMost Bearish:")
    for _, row in top_bearish.iterrows():
        print(f"  [{row['sentiment']:+.3f}] {row['text']}")

    return df


if __name__ == "__main__":
    main()
