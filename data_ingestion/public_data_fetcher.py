"""
Public Data Fetcher for Sentiment-Microstructure ABM

Fetches freely available sentiment and market data that requires NO API keys.
Perfect for reproducible academic research.

Data Sources:
1. Fear & Greed Index (alternative.me) - 2018-present, daily
2. CoinGecko - Social metrics, prices (free tier)
3. Binance - Order book via WebSocket (public)

This eliminates the Reddit API dependency while providing:
- MACRO sentiment (Fear & Greed = institutional/aggregate)
- MICRO sentiment (to be added via Kaggle tweet dataset)
- Market microstructure (Binance order books)

Author: Murad Farzulla
Date: January 2026
"""

import os
import sys
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from pathlib import Path

import requests
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Fear & Greed Index (Macro Sentiment)
# ============================================================================

class FearGreedFetcher:
    """
    Fetch Crypto Fear & Greed Index from alternative.me
    
    This is a well-known aggregate sentiment indicator that combines:
    - Volatility (25%)
    - Market momentum/volume (25%)
    - Social media (15%)
    - Surveys (15%)
    - Bitcoin dominance (10%)
    - Google Trends (10%)
    
    Perfect as a proxy for "institutional" or "aggregate" sentiment.
    
    API: https://alternative.me/crypto/fear-and-greed-index/
    No API key required!
    """
    
    BASE_URL = "https://api.alternative.me/fng/"
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), '..', 'data', 'cache'
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def fetch_historical(self, limit: int = 0) -> pd.DataFrame:
        """
        Fetch historical Fear & Greed data.
        
        Args:
            limit: Number of days to fetch (0 = all available)
            
        Returns:
            DataFrame with columns: timestamp, value, classification
        """
        logger.info(f"Fetching Fear & Greed Index (limit={limit})...")
        
        params = {'limit': limit, 'format': 'json'}
        
        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            records = []
            for item in data.get('data', []):
                records.append({
                    'timestamp': datetime.fromtimestamp(int(item['timestamp'])),
                    'fear_greed_value': int(item['value']),
                    'fear_greed_class': item['value_classification'],
                })
                
            df = pd.DataFrame(records)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Fetched {len(df)} days of Fear & Greed data")
            logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch Fear & Greed data: {e}")
            raise
            
    def fetch_current(self) -> Dict:
        """Fetch current Fear & Greed value."""
        resp = requests.get(self.BASE_URL, params={'limit': 1}, timeout=10)
        resp.raise_for_status()
        data = resp.json()['data'][0]
        
        return {
            'timestamp': datetime.fromtimestamp(int(data['timestamp'])),
            'value': int(data['value']),
            'classification': data['value_classification'],
        }
    
    def to_sentiment_score(self, fear_greed_value: int) -> float:
        """
        Convert Fear & Greed (0-100) to sentiment score (-1 to 1).
        
        0 = Extreme Fear → -1
        50 = Neutral → 0
        100 = Extreme Greed → +1
        """
        return (fear_greed_value - 50) / 50
    
    def save_to_csv(self, df: pd.DataFrame, filename: str = 'fear_greed_historical.csv'):
        """Save to CSV in cache directory."""
        path = os.path.join(self.cache_dir, filename)
        df.to_csv(path, index=False)
        logger.info(f"Saved to {path}")
        return path


# ============================================================================
# CoinGecko Data (Prices + Social)
# ============================================================================

class CoinGeckoFetcher:
    """
    Fetch price and social data from CoinGecko.
    
    Free tier includes:
    - Current prices
    - Historical prices (OHLC)
    - Social metrics (sentiment votes, community size)
    
    Rate limit: 10-50 calls/minute (free tier)
    """
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers['x-cg-demo-api-key'] = api_key
            
    def fetch_bitcoin_history(self, days: int = 365) -> pd.DataFrame:
        """
        Fetch Bitcoin price history.
        
        Args:
            days: Number of days of history (max 365 for free tier)
            
        Returns:
            DataFrame with OHLC data
        """
        logger.info(f"Fetching {days} days of BTC price history...")
        
        url = f"{self.BASE_URL}/coins/bitcoin/ohlc"
        params = {'vs_currency': 'usd', 'days': days}
        
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        logger.info(f"Fetched {len(df)} price points")
        return df
    
    def fetch_bitcoin_current(self) -> Dict:
        """Fetch current Bitcoin data with social metrics."""
        url = f"{self.BASE_URL}/coins/bitcoin"
        params = {
            'localization': 'false',
            'tickers': 'false',
            'market_data': 'true',
            'community_data': 'true',
            'developer_data': 'false',
        }
        
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        market = data.get('market_data', {})
        community = data.get('community_data', {})
        
        return {
            'timestamp': datetime.utcnow(),
            'price_usd': market.get('current_price', {}).get('usd'),
            'price_change_24h': market.get('price_change_percentage_24h'),
            'volume_24h': market.get('total_volume', {}).get('usd'),
            'sentiment_up_pct': data.get('sentiment_votes_up_percentage'),
            'sentiment_down_pct': data.get('sentiment_votes_down_percentage'),
            'reddit_subscribers': community.get('reddit_subscribers'),
            'twitter_followers': community.get('twitter_followers'),
        }


# ============================================================================
# Binance Historical Data
# ============================================================================

class BinanceHistoricalFetcher:
    """
    Fetch historical klines (candlesticks) from Binance.
    
    No API key needed for public endpoints.
    """
    
    BASE_URL = "https://api.binance.com/api/v3"
    
    def fetch_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch historical klines.
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            interval: Candlestick interval (1m, 5m, 15m, 1h, 4h, 1d)
            start_time: Start datetime
            end_time: End datetime
            limit: Max candles per request (1000)
        """
        url = f"{self.BASE_URL}/klines"
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit,
        }
        
        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)
            
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[col] = df[col].astype(float)
            
        df['trades'] = df['trades'].astype(int)
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades', 'quote_volume']]
    
    def fetch_range(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        Fetch data for a date range (handles pagination).
        
        Args:
            symbol: Trading pair
            interval: Candlestick interval
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
        """
        start = datetime.strptime(start_date, "%Y-%m-%d") if start_date else datetime.utcnow() - timedelta(days=30)
        end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.utcnow()
        
        logger.info(f"Fetching {symbol} {interval} data from {start} to {end}")
        
        all_data = []
        current_start = start
        
        while current_start < end:
            df = self.fetch_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=end,
                limit=1000,
            )
            
            if len(df) == 0:
                break
                
            all_data.append(df)
            current_start = df['timestamp'].max() + timedelta(hours=1)
            
            logger.info(f"  Fetched up to {df['timestamp'].max()}")
            time.sleep(0.1)  # Rate limit
            
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result = result.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            logger.info(f"Total: {len(result)} candles")
            return result
        else:
            return pd.DataFrame()


# ============================================================================
# Combined Dataset Builder
# ============================================================================

class PublicDatasetBuilder:
    """
    Build a combined dataset from all public sources.
    
    Creates a time-aligned dataset with:
    - Price data (Binance)
    - Fear & Greed Index (alternative.me)
    - Social metrics (CoinGecko)
    """
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(__file__), '..', 'data', 'datasets'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.fear_greed = FearGreedFetcher()
        self.coingecko = CoinGeckoFetcher()
        self.binance = BinanceHistoricalFetcher()
        
    def build_daily_dataset(
        self,
        start_date: str = "2023-01-01",
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        Build daily dataset combining all sources.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
        """
        end_date = end_date or datetime.utcnow().strftime("%Y-%m-%d")
        
        logger.info("=" * 60)
        logger.info("BUILDING PUBLIC DATASET")
        logger.info("=" * 60)
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # 1. Fetch Fear & Greed (daily)
        logger.info("\n1. Fetching Fear & Greed Index...")
        fg_df = self.fear_greed.fetch_historical(limit=0)
        fg_df['date'] = fg_df['timestamp'].dt.date
        
        # Convert to sentiment score
        fg_df['macro_sentiment'] = fg_df['fear_greed_value'].apply(
            self.fear_greed.to_sentiment_score
        )
        
        # 2. Fetch Binance daily data
        logger.info("\n2. Fetching Binance price data...")
        btc_df = self.binance.fetch_range(
            symbol="BTCUSDT",
            interval="1d",
            start_date=start_date,
            end_date=end_date,
        )
        btc_df['date'] = btc_df['timestamp'].dt.date
        
        # Compute returns and volatility
        btc_df['returns'] = btc_df['close'].pct_change()
        btc_df['volatility'] = btc_df['returns'].rolling(7).std()
        btc_df['volume_ma'] = btc_df['volume'].rolling(7).mean()
        
        # 3. Merge datasets
        logger.info("\n3. Merging datasets...")
        
        # Start with price data
        df = btc_df[['date', 'open', 'high', 'low', 'close', 'volume', 
                     'trades', 'returns', 'volatility']].copy()
        
        # Merge Fear & Greed
        fg_merge = fg_df[['date', 'fear_greed_value', 'fear_greed_class', 'macro_sentiment']]
        df = df.merge(fg_merge, on='date', how='left')
        
        # Fill missing sentiment with forward fill
        df['macro_sentiment'] = df['macro_sentiment'].fillna(method='ffill')
        df['fear_greed_value'] = df['fear_greed_value'].fillna(method='ffill')
        
        # Add derived features
        df['sentiment_momentum'] = df['macro_sentiment'].diff()
        df['price_sentiment_corr'] = df['returns'].rolling(7).corr(df['macro_sentiment'])
        
        # Classify regime based on Fear & Greed
        def classify_regime(fg_value):
            if pd.isna(fg_value):
                return 'unknown'
            elif fg_value <= 25:
                return 'extreme_fear'
            elif fg_value <= 45:
                return 'fear'
            elif fg_value <= 55:
                return 'neutral'
            elif fg_value <= 75:
                return 'greed'
            else:
                return 'extreme_greed'
                
        df['regime'] = df['fear_greed_value'].apply(classify_regime)
        
        # Filter to date range
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
        
        logger.info(f"\nFinal dataset: {len(df)} days")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, name: str = "btc_sentiment_daily"):
        """Save dataset to CSV."""
        path = os.path.join(self.output_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        logger.info(f"Saved to {path}")
        return path
    
    def get_summary(self, df: pd.DataFrame) -> Dict:
        """Get dataset summary statistics."""
        return {
            'n_days': len(df),
            'date_range': [str(df['date'].min()), str(df['date'].max())],
            'price_range': [float(df['close'].min()), float(df['close'].max())],
            'sentiment_range': [float(df['macro_sentiment'].min()), float(df['macro_sentiment'].max())],
            'regime_distribution': df['regime'].value_counts().to_dict(),
            'mean_sentiment': float(df['macro_sentiment'].mean()),
            'sentiment_volatility': float(df['macro_sentiment'].std()),
        }


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch public crypto sentiment data')
    parser.add_argument('--start', default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', default=None, help='Output directory')
    parser.add_argument('--fear-greed-only', action='store_true', help='Only fetch Fear & Greed')
    
    args = parser.parse_args()
    
    if args.fear_greed_only:
        fetcher = FearGreedFetcher()
        df = fetcher.fetch_historical(limit=0)
        df['sentiment'] = df['fear_greed_value'].apply(fetcher.to_sentiment_score)
        
        output_dir = args.output or 'data/datasets'
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'fear_greed_full.csv')
        df.to_csv(path, index=False)
        
        print(f"\n✅ Saved {len(df)} days to {path}")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Sentiment range: [{df['sentiment'].min():.2f}, {df['sentiment'].max():.2f}]")
        
    else:
        builder = PublicDatasetBuilder(output_dir=args.output)
        df = builder.build_daily_dataset(start_date=args.start, end_date=args.end)
        path = builder.save_dataset(df)
        
        summary = builder.get_summary(df)
        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print(f"\nSaved to: {path}")


if __name__ == '__main__':
    main()
