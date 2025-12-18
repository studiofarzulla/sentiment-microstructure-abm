"""
ASRI Data Adapter

Wraps ASRI project's data connectors for use in sentiment-microstructure-abm.
Handles async fetching, error recovery, and data normalization.
"""

import sys
import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict
from pathlib import Path

import numpy as np

# Add ASRI to path
ASRI_PATH = Path("/home/purrpower/Resurrexi/projects/resurrexi-projects/asri/src")
if str(ASRI_PATH) not in sys.path:
    sys.path.insert(0, str(ASRI_PATH))

from .models import MacroSignals

logger = logging.getLogger(__name__)


class ASRIAdapter:
    """
    Adapter for ASRI data sources.

    Provides a unified interface for fetching macro sentiment signals
    from DeFiLlama, Google News, CoinGecko, and FRED.

    All methods handle errors gracefully and return partial data when possible.
    """

    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        coingecko_api_key: Optional[str] = None,
        cache_ttl_seconds: int = 300,  # 5 minute cache
    ):
        self.fred_api_key = fred_api_key or os.getenv('FRED_API_KEY')
        self.coingecko_api_key = coingecko_api_key or os.getenv('COINGECKO_API_KEY')
        self.cache_ttl = cache_ttl_seconds

        # Cache for expensive API calls
        self._cache: Dict[str, Tuple[datetime, any]] = {}

        # Lazy-loaded clients
        self._news_aggregator = None
        self._defillama_client = None
        self._coingecko_client = None
        self._fred_connector = None

    async def _get_cached_or_fetch(self, key: str, fetch_fn) -> any:
        """Check cache or fetch fresh data."""
        now = datetime.utcnow()
        if key in self._cache:
            cached_time, cached_data = self._cache[key]
            if (now - cached_time).total_seconds() < self.cache_ttl:
                return cached_data

        data = await fetch_fn()
        self._cache[key] = (now, data)
        return data

    async def fetch_news_sentiment(self) -> dict:
        """
        Fetch regulatory sentiment from Google News RSS.

        Returns dict with:
            - score: 0-100 (higher = more regulatory risk)
            - sentiment: -1 to 1 (converted)
            - article_count, regulatory_count, top_headlines
        """
        try:
            from asri.ingestion.news import NewsAggregator

            if self._news_aggregator is None:
                self._news_aggregator = NewsAggregator()

            result = await self._get_cached_or_fetch(
                'news_sentiment',
                self._news_aggregator.calculate_regulatory_sentiment
            )

            # Convert 0-100 risk to -1 to 1 sentiment
            # High risk (100) = bearish (-1), Low risk (0) = bullish (1)
            sentiment = 1.0 - 2.0 * (result['score'] / 100.0)

            return {
                'score': result['score'],
                'sentiment': sentiment,
                'article_count': result['article_count'],
                'regulatory_count': result['regulatory_count'],
                'avg_sentiment': result.get('avg_sentiment', 0),
                'top_headlines': result.get('top_headlines', []),
            }

        except Exception as e:
            logger.warning(f"News sentiment fetch failed: {e}")
            return {
                'score': 50.0,  # Neutral fallback
                'sentiment': 0.0,
                'article_count': 0,
                'regulatory_count': 0,
                'avg_sentiment': 0,
                'top_headlines': [],
            }

    async def fetch_defi_data(self) -> dict:
        """
        Fetch DeFi metrics from DeFiLlama.

        Returns dict with:
            - total_tvl: Current TVL in USD
            - tvl_change_24h: Percentage change
            - stablecoins: List of stablecoin data
            - peg_deviations: Max deviation from $1
        """
        try:
            from asri.ingestion.defillama import DeFiLlamaClient

            if self._defillama_client is None:
                self._defillama_client = DeFiLlamaClient()

            async def fetch_all_defi():
                tvl_task = self._defillama_client.get_total_tvl()
                stables_task = self._defillama_client.get_stablecoins()

                results = await asyncio.gather(
                    tvl_task, stables_task,
                    return_exceptions=True
                )
                return results

            tvl_data, stablecoins = await self._get_cached_or_fetch(
                'defi_data', fetch_all_defi
            )

            # Handle exceptions in results
            if isinstance(tvl_data, Exception):
                logger.warning(f"TVL fetch failed: {tvl_data}")
                tvl_data = None

            if isinstance(stablecoins, Exception):
                logger.warning(f"Stablecoins fetch failed: {stablecoins}")
                stablecoins = []

            # Calculate peg deviation
            max_peg_deviation = 0.0
            if stablecoins:
                for sc in stablecoins[:10]:  # Top 10 stablecoins
                    if hasattr(sc, 'price') and sc.price:
                        deviation = abs(sc.price - 1.0)
                        max_peg_deviation = max(max_peg_deviation, deviation)

            return {
                'total_tvl': tvl_data,
                'tvl_change_24h': None,  # Would need historical data
                'stablecoins': stablecoins,
                'peg_deviation': max_peg_deviation,
            }

        except Exception as e:
            logger.warning(f"DeFi data fetch failed: {e}")
            return {
                'total_tvl': None,
                'tvl_change_24h': None,
                'stablecoins': [],
                'peg_deviation': None,
            }

    async def fetch_tradfi_data(self) -> dict:
        """
        Fetch TradFi indicators from FRED.

        Returns dict with:
            - vix: Current VIX level
            - treasury_10y: 10-year Treasury yield
            - yield_spread: 10Y-2Y spread
        """
        if not self.fred_api_key:
            logger.info("FRED API key not set, skipping TradFi data")
            return {'vix': None, 'treasury_10y': None, 'yield_spread': None}

        try:
            from asri.ingestion.fred import FREDConnector

            if self._fred_connector is None:
                self._fred_connector = FREDConnector(api_key=self.fred_api_key)

            async def fetch_fred():
                # Fetch last 5 days to ensure we have recent data
                start = (datetime.utcnow() - timedelta(days=5)).strftime('%Y-%m-%d')

                vix_task = self._fred_connector.fetch_series('VIXCLS', start_date=start)
                t10y_task = self._fred_connector.fetch_series('DGS10', start_date=start)
                spread_task = self._fred_connector.fetch_series('T10Y2Y', start_date=start)

                return await asyncio.gather(
                    vix_task, t10y_task, spread_task,
                    return_exceptions=True
                )

            vix_data, t10y_data, spread_data = await self._get_cached_or_fetch(
                'fred_data', fetch_fred
            )

            # Extract most recent values
            def get_latest(data):
                if isinstance(data, Exception) or not data:
                    return None
                if isinstance(data, list) and len(data) > 0:
                    return data[-1].get('value') if isinstance(data[-1], dict) else data[-1]
                return None

            return {
                'vix': get_latest(vix_data),
                'treasury_10y': get_latest(t10y_data),
                'yield_spread': get_latest(spread_data),
            }

        except Exception as e:
            logger.warning(f"FRED data fetch failed: {e}")
            return {'vix': None, 'treasury_10y': None, 'yield_spread': None}

    async def fetch_all(self) -> MacroSignals:
        """
        Fetch all available macro signals concurrently.

        Returns MacroSignals dataclass with all available data.
        Gracefully handles partial failures.
        """
        logger.info("Fetching all ASRI data sources...")

        # Fetch all sources concurrently
        news_task = self.fetch_news_sentiment()
        defi_task = self.fetch_defi_data()
        tradfi_task = self.fetch_tradfi_data()

        news, defi, tradfi = await asyncio.gather(
            news_task, defi_task, tradfi_task,
            return_exceptions=True
        )

        # Handle exceptions
        if isinstance(news, Exception):
            logger.warning(f"News fetch exception: {news}")
            news = {'score': 50, 'sentiment': 0, 'article_count': 0,
                   'regulatory_count': 0, 'top_headlines': []}

        if isinstance(defi, Exception):
            logger.warning(f"DeFi fetch exception: {defi}")
            defi = {'total_tvl': None, 'peg_deviation': None}

        if isinstance(tradfi, Exception):
            logger.warning(f"TradFi fetch exception: {tradfi}")
            tradfi = {'vix': None, 'treasury_10y': None, 'yield_spread': None}

        # Normalize VIX to [0, 1] range (10-50 typical range)
        vix_normalized = None
        if tradfi.get('vix') is not None:
            vix_normalized = np.clip((tradfi['vix'] - 10) / 40, 0, 1)

        # Compute ASRI alert level based on available data
        alert_level = self._compute_alert_level(news, defi, tradfi)

        return MacroSignals(
            timestamp=datetime.utcnow(),
            regulatory_sentiment=news['sentiment'],
            regulatory_score_raw=news['score'],
            article_count=news['article_count'],
            regulatory_article_count=news['regulatory_count'],
            top_headlines=news['top_headlines'],
            total_tvl=defi.get('total_tvl'),
            peg_stability=defi.get('peg_deviation'),
            vix_level=tradfi.get('vix'),
            vix_normalized=vix_normalized,
            treasury_10y=tradfi.get('treasury_10y'),
            yield_spread=tradfi.get('yield_spread'),
            asri_alert_level=alert_level,
        )

    def _compute_alert_level(self, news: dict, defi: dict, tradfi: dict) -> str:
        """Compute simple alert level from available data."""
        risk_score = 0
        factors = 0

        # News contributes
        if news.get('score') is not None:
            risk_score += news['score']
            factors += 1

        # High VIX contributes
        if tradfi.get('vix') is not None:
            vix_contribution = np.clip((tradfi['vix'] - 15) / 35 * 100, 0, 100)
            risk_score += vix_contribution
            factors += 1

        # Peg deviation contributes
        if defi.get('peg_deviation') is not None:
            peg_contribution = min(defi['peg_deviation'] * 1000, 100)  # 0.1 deviation = 100
            risk_score += peg_contribution
            factors += 1

        if factors == 0:
            return 'unknown'

        avg_risk = risk_score / factors

        if avg_risk >= 80:
            return 'critical'
        elif avg_risk >= 65:
            return 'high'
        elif avg_risk >= 50:
            return 'elevated'
        elif avg_risk >= 35:
            return 'moderate'
        else:
            return 'low'

    async def close(self):
        """Clean up async clients."""
        if self._news_aggregator:
            await self._news_aggregator.close()
        if self._defillama_client:
            await self._defillama_client.close()
        # FRED connector doesn't need closing


# Convenience function for sync usage
def fetch_macro_signals_sync() -> MacroSignals:
    """Synchronous wrapper for fetching macro signals."""
    adapter = ASRIAdapter()
    return asyncio.run(adapter.fetch_all())
