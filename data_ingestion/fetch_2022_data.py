"""
Fetch 2022 Bear Market Data for Out-of-Sample Validation

The 2022 crypto bear market provides an excellent out-of-sample test:
- Luna/UST collapse (May 2022)
- 3AC bankruptcy (June 2022)
- FTX collapse (November 2022)
- Sustained fear regime vs 2024's bull market

This validates whether the extremity premium holds in bear markets.

Author: Murad Farzulla
Date: January 2026
"""

import os
import sys
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_ingestion.public_data_fetcher import (
    BinanceHistoricalFetcher,
    FearGreedFetcher,
    PublicDatasetBuilder
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_2022_bear_market_data(output_dir: str = None):
    """
    Fetch 2022 bear market data (Jan 1 - Dec 31, 2022).

    Returns:
        DataFrame with same structure as btc_sentiment_daily.csv
    """
    output_dir = output_dir or os.path.join(
        os.path.dirname(__file__), '..', 'data', 'datasets'
    )
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("FETCHING 2022 BEAR MARKET DATA")
    logger.info("=" * 60)
    logger.info("Period: 2022-01-01 to 2022-12-31")
    logger.info("Key events:")
    logger.info("  - May 2022: Luna/UST collapse")
    logger.info("  - June 2022: 3AC bankruptcy")
    logger.info("  - Nov 2022: FTX collapse")
    logger.info("=" * 60)

    # Use the existing PublicDatasetBuilder
    builder = PublicDatasetBuilder(output_dir=output_dir)

    # Fetch 2022 data
    df = builder.build_daily_dataset(
        start_date="2022-01-01",
        end_date="2022-12-31"
    )

    # Save to separate file
    output_path = os.path.join(output_dir, 'btc_sentiment_2022_bear.csv')
    df.to_csv(output_path, index=False)
    logger.info(f"Saved to: {output_path}")

    # Print summary
    summary = builder.get_summary(df)

    print("\n" + "=" * 60)
    print("2022 BEAR MARKET DATASET SUMMARY")
    print("=" * 60)
    print(f"Total days: {summary['n_days']}")
    print(f"Date range: {summary['date_range'][0]} to {summary['date_range'][1]}")
    print(f"Price range: ${summary['price_range'][0]:,.0f} to ${summary['price_range'][1]:,.0f}")
    print(f"Mean sentiment: {summary['mean_sentiment']:.3f} (vs ~0.1 expected for bear)")

    print(f"\nRegime distribution (2022):")
    for regime, count in sorted(summary['regime_distribution'].items()):
        pct = 100 * count / summary['n_days']
        print(f"  {regime:15s}: {count:3d} days ({pct:5.1f}%)")

    # Compare to expected 2024 distribution
    print("\n★ Expected pattern:")
    print("  2022 (bear): Heavy fear/extreme_fear, minimal greed")
    print("  2024 (bull): Heavy greed/extreme_greed, minimal fear")
    print("  If extremity premium holds, both extremes should show elevated uncertainty")

    return df


def fetch_eth_2022_data(output_dir: str = None):
    """
    Fetch 2022 ETH data for cross-asset validation.
    """
    output_dir = output_dir or os.path.join(
        os.path.dirname(__file__), '..', 'data', 'datasets'
    )

    logger.info("\nFetching ETH 2022 data for cross-asset validation...")

    binance = BinanceHistoricalFetcher()

    eth_df = binance.fetch_range(
        symbol="ETHUSDT",
        interval="1d",
        start_date="2022-01-01",
        end_date="2022-12-31"
    )

    # Compute returns and volatility
    eth_df['returns'] = eth_df['close'].pct_change()
    eth_df['volatility'] = eth_df['returns'].rolling(7).std()
    eth_df['date'] = eth_df['timestamp'].dt.date

    output_path = os.path.join(output_dir, 'eth_price_2022.csv')
    eth_df.to_csv(output_path, index=False)
    logger.info(f"Saved ETH 2022 data to: {output_path}")

    return eth_df


def compare_regime_distributions():
    """
    Load both 2022 and 2024 datasets and compare regime distributions.
    """
    import pandas as pd

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'datasets')

    # Load both datasets
    df_2024 = pd.read_csv(os.path.join(data_dir, 'btc_sentiment_daily.csv'))
    df_2022 = pd.read_csv(os.path.join(data_dir, 'btc_sentiment_2022_bear.csv'))

    print("\n" + "=" * 60)
    print("REGIME DISTRIBUTION COMPARISON")
    print("=" * 60)

    # Regime counts
    regimes = ['extreme_greed', 'greed', 'neutral', 'fear', 'extreme_fear']

    print("\n{:15s}  {:>10s}  {:>10s}".format("Regime", "2022 Bear", "2024 Bull"))
    print("-" * 40)

    for regime in regimes:
        n_2022 = len(df_2022[df_2022['regime'] == regime])
        n_2024 = len(df_2024[df_2024['regime'] == regime])
        pct_2022 = 100 * n_2022 / len(df_2022)
        pct_2024 = 100 * n_2024 / len(df_2024)

        print(f"{regime:15s}  {pct_2022:8.1f}%   {pct_2024:8.1f}%")

    # Summary stats
    print("\n" + "-" * 40)
    print(f"Mean sentiment: 2022={df_2022['macro_sentiment'].mean():+.3f}  2024={df_2024['macro_sentiment'].mean():+.3f}")
    print(f"Mean price:     2022=${df_2022['close'].mean():,.0f}  2024=${df_2024['close'].mean():,.0f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Fetch 2022 bear market data')
    parser.add_argument('--compare', action='store_true',
                        help='Compare 2022 vs 2024 regime distributions')
    parser.add_argument('--eth', action='store_true',
                        help='Also fetch ETH 2022 data')

    args = parser.parse_args()

    if args.compare:
        compare_regime_distributions()
    else:
        # Fetch BTC 2022 data
        df = fetch_2022_bear_market_data()

        # Optionally fetch ETH
        if args.eth:
            eth_df = fetch_eth_2022_data()

        print("\n✓ 2022 bear market data ready for out-of-sample validation")


if __name__ == '__main__':
    main()
