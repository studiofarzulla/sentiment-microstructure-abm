"""
Dome API Client for Prediction Market Data

Fetches prediction market data from Polymarket and Kalshi to extract
"informed sentiment" signals. Prediction markets aggregate information
from participants with real money at stake, providing a unique
sentiment signal distinct from social media (retail) and institutional
indicators (ASRI).

Key Features:
- Market prices → sentiment conversion (high price = bullish)
- Orderbook history for microstructure analysis
- Trade history for volume/activity metrics
- Wallet analytics for trader behavior
- Crypto-focused market discovery

Author: Murad Farzulla
Date: January 2026
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PredictionMarket:
    """Represents a prediction market with key metrics."""
    market_slug: str
    question: str
    platform: str  # 'polymarket' or 'kalshi'
    current_price: float  # [0, 1] probability
    volume_24h: Optional[float] = None
    liquidity: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Sentiment conversion
    sentiment: float = 0.0  # [-1, 1] derived from price
    
    def to_sentiment(self) -> float:
        """
        Convert prediction market price to sentiment.
        
        For crypto markets like "BTC > $50k by EOY":
        - Price 0.8 (80% chance) → +0.6 sentiment (bullish)
        - Price 0.5 (50% chance) → 0.0 sentiment (neutral)
        - Price 0.2 (20% chance) → -0.6 sentiment (bearish)
        
        Uses sigmoid-like mapping: sentiment = 2 * (price - 0.5)
        """
        # Linear mapping: 0 → -1, 0.5 → 0, 1 → +1
        return 2.0 * (self.current_price - 0.5)


@dataclass
class PredictionMarketSentiment:
    """Aggregated sentiment from multiple prediction markets."""
    timestamp: datetime
    platform: str
    markets: List[PredictionMarket]
    
    # Aggregated metrics
    mean_sentiment: float
    weighted_sentiment: float  # Volume-weighted
    market_count: int
    total_volume_24h: float
    
    # Uncertainty metrics
    sentiment_std: float  # Higher = disagreement
    price_range: Tuple[float, float]  # Min/max prices
    
    def to_sentiment_tick_component(self) -> Tuple[float, float]:
        """
        Convert to (sentiment, uncertainty) tuple for SignalComposer.
        
        Returns:
            (sentiment, epistemic_uncertainty)
            - sentiment: [-1, 1] weighted average
            - epistemic_uncertainty: [0, 1] based on disagreement
        """
        sentiment = self.weighted_sentiment
        
        # Uncertainty: high std = disagreement = high uncertainty
        # Normalize std (max ~0.5 for binary markets) to [0, 1]
        epistemic = min(1.0, self.sentiment_std / 0.5) if self.sentiment_std > 0 else 0.1
        
        return sentiment, epistemic


class DomeAPIClient:
    """
    Client for Dome API prediction market data.
    
    Supports both Polymarket and Kalshi platforms.
    Focuses on crypto-related markets for sentiment extraction.
    """
    
    BASE_URL = "https://api.domeapi.io/v1"
    
    # Common crypto-related market slugs (Polymarket)
    CRYPTO_MARKETS = [
        "will-bitcoin-price-be-above-50000-usd-on-december-31-2025",
        "will-bitcoin-price-be-above-60000-usd-on-december-31-2025",
        "will-bitcoin-price-be-above-70000-usd-on-december-31-2025",
        "will-ethereum-price-be-above-3000-usd-on-december-31-2025",
        "will-bitcoin-reach-100000-usd-before-2026",
        "will-bitcoin-etf-approval-happen-in-2025",
    ]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        platform: str = "polymarket",
        rate_limit_delay: float = 1.1,  # Free tier: 1 QPS
    ):
        """
        Initialize Dome API client.
        
        Args:
            api_key: Dome API key (defaults to DOME_API_KEY env var)
            platform: 'polymarket' or 'kalshi'
            rate_limit_delay: Delay between requests (seconds)
        """
        self.api_key = api_key or os.getenv('DOME_API_KEY')
        if not self.api_key:
            raise ValueError(
                "DOME_API_KEY not found. Set it in .env or pass as argument.\n"
                "Get your free API key at: https://docs.domeapi.io"
            )
        
        self.platform = platform.lower()
        if self.platform not in ['polymarket', 'kalshi']:
            raise ValueError(f"Platform must be 'polymarket' or 'kalshi', got '{self.platform}'")
        
        self.rate_limit_delay = rate_limit_delay
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        
        # Headers
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        
        logger.info(f"Initialized Dome API client for {self.platform}")
    
    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
    ) -> Dict:
        """
        Make authenticated API request with rate limiting.
        
        Args:
            endpoint: API endpoint (e.g., '/polymarket/markets')
            params: Query parameters
            
        Returns:
            JSON response as dict
            
        Raises:
            requests.HTTPError: If request fails
        """
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            time.sleep(self.rate_limit_delay)  # Rate limiting
            
            response = self.session.get(
                url,
                headers=self.headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning(f"Rate limited. Waiting {self.rate_limit_delay * 2}s...")
                time.sleep(self.rate_limit_delay * 2)
                return self._make_request(endpoint, params)  # Retry
            else:
                logger.error(f"API request failed: {e}")
                raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    def get_market(
        self,
        market_slug: str,
    ) -> Optional[PredictionMarket]:
        """
        Fetch a single market by slug.
        
        Args:
            market_slug: Market identifier (e.g., 'will-bitcoin-price-be-above-50000-usd-on-december-31-2025')
            
        Returns:
            PredictionMarket object or None if not found
        """
        endpoint = f"/{self.platform}/markets"
        params = {'market_slug': market_slug}
        
        try:
            data = self._make_request(endpoint, params)
            
            # Parse response - Dome API returns {"markets": [...]}
            if isinstance(data, dict) and 'markets' in data:
                markets_list = data['markets']
                if markets_list and len(markets_list) > 0:
                    market_data = markets_list[0]
                else:
                    logger.warning(f"No markets found for {market_slug}")
                    return None
            elif isinstance(data, list) and len(data) > 0:
                market_data = data[0]
            elif isinstance(data, dict):
                market_data = data
            else:
                logger.warning(f"Unexpected response format for {market_slug}")
                return None
            
            # Try to get price from market-price endpoint or orderbook
            price = self._get_market_price(market_slug, market_data)
            
            # Extract volume (may be in different fields)
            volume_24h = (
                market_data.get('volume_24h') or
                market_data.get('volume_1_day') or
                market_data.get('volume_1_week', 0) / 7  # Approximate daily
            )
            
            market = PredictionMarket(
                market_slug=market_data.get('market_slug', market_slug),
                question=market_data.get('title') or market_data.get('question', market_slug),
                platform=self.platform,
                current_price=price if price is not None else 0.5,  # Default to neutral if unknown
                volume_24h=volume_24h,
                liquidity=market_data.get('liquidity'),
            )
            
            market.sentiment = market.to_sentiment()
            
            return market
            
        except Exception as e:
            logger.error(f"Failed to fetch market {market_slug}: {e}")
            return None
    
    def _get_market_price(
        self,
        market_slug: str,
        market_data: Dict,
    ) -> Optional[float]:
        """
        Get current market price from various endpoints.
        
        Tries multiple methods:
        1. Direct price field in market data
        2. Market price endpoint (if available)
        3. Orderbook best bid/ask
        4. Activity/trade history latest price
        """
        # Method 1: Check if price is in market data
        price = self._extract_price(market_data)
        if price is not None:
            return price
        
        # Method 2: Try activity endpoint for latest trade price
        try:
            activity = self.get_activity(market_slug, limit=1)
            if activity and len(activity) > 0:
                latest = activity[0]
                # Extract price from trade/activity data
                price = self._extract_price(latest)
                if price is not None:
                    return price
        except Exception as e:
            logger.debug(f"Could not get price from activity: {e}")
        
        # Method 3: Try orderbook for mid-price
        try:
            orderbook = self.get_orderbook(market_slug)
            if orderbook:
                price = self._extract_price_from_orderbook(orderbook)
                if price is not None:
                    return price
        except Exception as e:
            logger.debug(f"Could not get price from orderbook: {e}")
        
        return None
    
    def _extract_price_from_orderbook(self, orderbook: Dict) -> Optional[float]:
        """Extract mid-price from orderbook data."""
        # Try common orderbook structures
        if 'bids' in orderbook and 'asks' in orderbook:
            bids = orderbook['bids']
            asks = orderbook['asks']
            if bids and asks:
                best_bid = float(bids[0][0]) if isinstance(bids[0], list) else float(bids[0].get('price', 0))
                best_ask = float(asks[0][0]) if isinstance(asks[0], list) else float(asks[0].get('price', 0))
                if best_bid > 0 and best_ask > 0:
                    return (best_bid + best_ask) / 2.0
        return None
    
    def get_orderbook(
        self,
        market_slug: str,
    ) -> Optional[Dict]:
        """Get current orderbook for a market."""
        endpoint = f"/{self.platform}/orderbook"
        params = {'market_slug': market_slug}
        
        try:
            return self._make_request(endpoint, params)
        except Exception as e:
            logger.debug(f"Orderbook endpoint not available or failed: {e}")
            return None
    
    def get_activity(
        self,
        market_slug: str,
        limit: int = 10,
    ) -> Optional[List[Dict]]:
        """Get recent activity/trades for a market."""
        endpoint = f"/{self.platform}/activity"
        params = {
            'market_slug': market_slug,
            'limit': limit,
        }
        
        try:
            data = self._make_request(endpoint, params)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'activities' in data:
                return data['activities']
            elif isinstance(data, dict) and 'activity' in data:
                return data['activity']
            elif isinstance(data, dict) and 'trades' in data:
                return data['trades']
            return []
        except Exception as e:
            logger.debug(f"Activity endpoint not available or failed: {e}")
            return None
    
    def _extract_price(self, data: Dict) -> Optional[float]:
        """
        Extract current price from market/trade/activity data.
        
        Platform-specific parsing. Returns [0, 1] probability.
        """
        # Try common field names
        for field in ['price', 'current_price', 'yes_price', 'probability', 'prob', 
                      'last_price', 'trade_price', 'execution_price']:
            if field in data:
                price = data[field]
                if isinstance(price, (int, float)):
                    # Ensure in [0, 1] range
                    return max(0.0, min(1.0, float(price)))
                elif isinstance(price, str):
                    try:
                        return max(0.0, min(1.0, float(price)))
                    except ValueError:
                        continue
        
        # Try nested structures (side_a/side_b for Polymarket)
        if 'side_a' in data and 'side_b' in data:
            # Check if there's a price in side_a (Yes side)
            side_a = data['side_a']
            if isinstance(side_a, dict):
                price = self._extract_price(side_a)
                if price is not None:
                    return price
        
        # Try outcomes array
        if 'outcomes' in data:
            outcomes = data['outcomes']
            if isinstance(outcomes, list) and len(outcomes) > 0:
                # Usually first outcome is "Yes"
                first_outcome = outcomes[0]
                return self._extract_price(first_outcome)
        
        return None
    
    def get_multiple_markets(
        self,
        market_slugs: List[str],
    ) -> List[PredictionMarket]:
        """
        Fetch multiple markets.
        
        Args:
            market_slugs: List of market identifiers
            
        Returns:
            List of PredictionMarket objects (may be shorter if some fail)
        """
        markets = []
        
        for slug in market_slugs:
            market = self.get_market(slug)
            if market:
                markets.append(market)
            time.sleep(self.rate_limit_delay)  # Rate limit between markets
        
        logger.info(f"Fetched {len(markets)}/{len(market_slugs)} markets")
        return markets
    
    def get_crypto_sentiment(
        self,
        market_slugs: Optional[List[str]] = None,
    ) -> Optional[PredictionMarketSentiment]:
        """
        Get aggregated sentiment from crypto prediction markets.
        
        Args:
            market_slugs: List of market slugs (defaults to CRYPTO_MARKETS)
            
        Returns:
            PredictionMarketSentiment with aggregated metrics
        """
        slugs = market_slugs or self.CRYPTO_MARKETS
        markets = self.get_multiple_markets(slugs)
        
        if not markets:
            logger.warning("No markets fetched, cannot compute sentiment")
            return None
        
        # Aggregate metrics
        sentiments = [m.sentiment for m in markets]
        volumes = [m.volume_24h or 0.0 for m in markets]
        total_volume = sum(volumes)
        
        # Mean sentiment
        mean_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
        
        # Volume-weighted sentiment
        if total_volume > 0:
            weighted_sentiment = sum(
                s * v for s, v in zip(sentiments, volumes)
            ) / total_volume
        else:
            weighted_sentiment = mean_sentiment
        
        # Uncertainty (std of sentiments)
        if len(sentiments) > 1:
            import numpy as np
            sentiment_std = float(np.std(sentiments))
        else:
            sentiment_std = 0.0
        
        # Price range
        prices = [m.current_price for m in markets]
        price_range = (min(prices), max(prices)) if prices else (0.0, 1.0)
        
        return PredictionMarketSentiment(
            timestamp=datetime.utcnow(),
            platform=self.platform,
            markets=markets,
            mean_sentiment=mean_sentiment,
            weighted_sentiment=weighted_sentiment,
            market_count=len(markets),
            total_volume_24h=total_volume,
            sentiment_std=sentiment_std,
            price_range=price_range,
        )
    
    def get_orderbook_history(
        self,
        market_slug: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Optional[Dict]:
        """
        Fetch orderbook history for a market.
        
        Useful for microstructure analysis similar to Binance orderbooks.
        """
        endpoint = f"/{self.platform}/orderbook-history"
        params = {'market_slug': market_slug}
        
        if start_time:
            params['start_time'] = start_time.isoformat()
        if end_time:
            params['end_time'] = end_time.isoformat()
        
        try:
            return self._make_request(endpoint, params)
        except Exception as e:
            logger.error(f"Failed to fetch orderbook history: {e}")
            return None
    
    def get_trade_history(
        self,
        market_slug: str,
        limit: int = 100,
    ) -> Optional[List[Dict]]:
        """
        Fetch recent trade history for a market.
        
        Useful for volume/activity metrics.
        """
        endpoint = f"/{self.platform}/trade-history"
        params = {
            'market_slug': market_slug,
            'limit': limit,
        }
        
        try:
            data = self._make_request(endpoint, params)
            return data if isinstance(data, list) else [data]
        except Exception as e:
            logger.error(f"Failed to fetch trade history: {e}")
            return None
    
    def search_markets(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Dict]:
        """
        Search for markets by keyword.
        
        Useful for discovering crypto-related markets.
        """
        endpoint = f"/{self.platform}/markets"
        params = {
            'search': query,
            'limit': limit,
        }
        
        try:
            data = self._make_request(endpoint, params)
            return data if isinstance(data, list) else [data]
        except Exception as e:
            logger.error(f"Failed to search markets: {e}")
            return []


# ============================================================================
# Convenience Functions
# ============================================================================

def get_prediction_market_sentiment(
    api_key: Optional[str] = None,
    market_slugs: Optional[List[str]] = None,
) -> Optional[Tuple[float, float]]:
    """
    Quick function to get prediction market sentiment.
    
    Returns:
        (sentiment, epistemic_uncertainty) tuple for SignalComposer
        or None if unavailable
    """
    client = DomeAPIClient(api_key=api_key)
    sentiment_obj = client.get_crypto_sentiment(market_slugs=market_slugs)
    
    if sentiment_obj:
        return sentiment_obj.to_sentiment_tick_component()
    return None


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI for testing Dome API client."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dome API prediction market client')
    parser.add_argument('--market', help='Market slug to fetch')
    parser.add_argument('--crypto-sentiment', action='store_true', 
                       help='Get aggregated crypto sentiment')
    parser.add_argument('--search', help='Search for markets')
    parser.add_argument('--platform', choices=['polymarket', 'kalshi'], 
                       default='polymarket', help='Platform to query')
    
    args = parser.parse_args()
    
    try:
        client = DomeAPIClient(platform=args.platform)
        
        if args.market:
            market = client.get_market(args.market)
            if market:
                print(f"\nMarket: {market.question}")
                print(f"Price: {market.current_price:.2%}")
                print(f"Sentiment: {market.sentiment:.3f}")
                print(f"Volume 24h: {market.volume_24h}")
            else:
                print(f"Market '{args.market}' not found")
        
        elif args.crypto_sentiment:
            sentiment = client.get_crypto_sentiment()
            if sentiment:
                print(f"\n{'='*60}")
                print("CRYPTO PREDICTION MARKET SENTIMENT")
                print(f"{'='*60}")
                print(f"Platform: {sentiment.platform}")
                print(f"Markets: {sentiment.market_count}")
                print(f"Mean Sentiment: {sentiment.mean_sentiment:.3f}")
                print(f"Weighted Sentiment: {sentiment.weighted_sentiment:.3f}")
                print(f"Uncertainty (std): {sentiment.sentiment_std:.3f}")
                print(f"Total Volume 24h: ${sentiment.total_volume_24h:,.0f}")
                print(f"Price Range: {sentiment.price_range[0]:.2%} - {sentiment.price_range[1]:.2%}")
                
                sent, unc = sentiment.to_sentiment_tick_component()
                print(f"\nFor SignalComposer: sentiment={sent:.3f}, uncertainty={unc:.3f}")
            else:
                print("Failed to fetch sentiment")
        
        elif args.search:
            markets = client.search_markets(args.search)
            print(f"\nFound {len(markets)} markets:")
            for m in markets[:10]:
                slug = m.get('slug', m.get('market_slug', 'unknown'))
                question = m.get('question', slug)
                print(f"  - {slug}")
                print(f"    {question}")
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()
