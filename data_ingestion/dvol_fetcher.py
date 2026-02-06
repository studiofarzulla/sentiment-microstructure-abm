"""
Deribit DVOL (BTC Implied Volatility) Fetcher

Fetches the DVOL index from Deribit's public API - the crypto-native
equivalent of VIX for Bitcoin options implied volatility.

DVOL advantages over VIX for crypto research:
- Native to crypto derivatives market (Deribit is largest BTC options venue)
- Reflects actual BTC option pricing, not equity market fear
- More responsive to crypto-specific events
- Eliminates TradFi contagion assumptions

API: https://docs.deribit.com/ (public, no API key required for market data)

Author: Murad Farzulla
Date: January 2026
"""

import os
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import requests
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DVOLReading:
    """Single DVOL observation."""
    timestamp: datetime
    dvol: float  # Annualized implied volatility percentage
    dvol_normalized: float  # Normalized to [0, 1] range
    btc_price: Optional[float] = None
    realized_vol_30d: Optional[float] = None  # Fallback measure


class DeribitDVOLFetcher:
    """
    Fetch Bitcoin Implied Volatility (DVOL) from Deribit.

    DVOL is Deribit's volatility index, similar to VIX but for BTC options.
    It represents 30-day annualized implied volatility derived from
    BTC options across multiple strikes.

    Typical DVOL ranges:
    - 30-50: Low volatility (crypto calm)
    - 50-80: Normal volatility
    - 80-120: Elevated volatility
    - 120+: High volatility / crisis

    Public API - no authentication required for market data.
    """

    BASE_URL = "https://www.deribit.com/api/v2"

    # Normalization parameters based on historical DVOL behavior
    DVOL_LOW = 30.0   # Historically low DVOL
    DVOL_HIGH = 150.0  # Crisis-level DVOL

    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), '..', 'data', 'cache'
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        self.session = requests.Session()

    def fetch_current_dvol(self) -> DVOLReading:
        """
        Fetch current DVOL value.

        Uses get_volatility_index_data endpoint with short lookback to get latest.

        Returns:
            DVOLReading with current implied volatility
        """
        logger.info("Fetching current DVOL from Deribit...")

        try:
            # Get recent volatility data (last 2 days to ensure we have latest)
            end_ts = int(datetime.utcnow().timestamp() * 1000)
            start_ts = end_ts - (2 * 24 * 60 * 60 * 1000)  # 2 days ago

            dvol_resp = self._api_call("public/get_volatility_index_data", {
                "currency": "BTC",
                "resolution": "1D",
                "start_timestamp": start_ts,
                "end_timestamp": end_ts,
            })

            data = dvol_resp.get("data", [])
            if not data:
                logger.warning("No DVOL data returned, falling back to realized vol")
                return self._fallback_volatility()

            # Latest candle's close price is current DVOL
            # Format: [timestamp, open, high, low, close]
            latest = data[-1]
            dvol_value = latest[4]  # Close price

            # Also get BTC price for context
            try:
                btc_resp = self._api_call("public/get_index_price", {
                    "index_name": "btc_usd"
                })
                btc_price = btc_resp.get("index_price")
            except Exception:
                btc_price = None

            return DVOLReading(
                timestamp=datetime.utcfromtimestamp(latest[0] / 1000),
                dvol=dvol_value,
                dvol_normalized=self._normalize_dvol(dvol_value),
                btc_price=btc_price,
            )

        except Exception as e:
            logger.warning(f"DVOL fetch failed: {e}, falling back to realized vol")
            return self._fallback_volatility()

    def fetch_historical_dvol(
        self,
        start_time: datetime = None,
        end_time: datetime = None,
        resolution: str = "1D",  # 1D (daily) or smaller intervals
    ) -> pd.DataFrame:
        """
        Fetch historical DVOL data via get_volatility_index_data endpoint.

        Args:
            start_time: Start of period (default: 90 days ago)
            end_time: End of period (default: now)
            resolution: Candle resolution (1D = daily)

        Returns:
            DataFrame with timestamp, dvol, dvol_normalized, open, high, low columns
        """
        end_time = end_time or datetime.utcnow()
        start_time = start_time or (end_time - timedelta(days=90))

        logger.info(f"Fetching DVOL history from {start_time} to {end_time}")

        # Convert to Unix timestamps (milliseconds)
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        try:
            resp = self._api_call("public/get_volatility_index_data", {
                "currency": "BTC",
                "resolution": resolution,
                "start_timestamp": start_ms,
                "end_timestamp": end_ms,
            })

            data = resp.get("data", [])
            if not data:
                logger.warning("No DVOL history available, returning empty DataFrame")
                return pd.DataFrame(columns=['timestamp', 'dvol', 'dvol_normalized', 'open', 'high', 'low'])

            # Format: [timestamp, open, high, low, close]
            records = []
            for candle in data:
                ts, open_, high, low, close = candle
                records.append({
                    'timestamp': datetime.utcfromtimestamp(ts / 1000),
                    'dvol': close,
                    'dvol_normalized': self._normalize_dvol(close),
                    'open': open_,
                    'high': high,
                    'low': low,
                })

            df = pd.DataFrame(records)
            df = df.sort_values('timestamp').reset_index(drop=True)

            logger.info(f"Fetched {len(df)} DVOL observations")
            logger.info(f"DVOL range: {df['dvol'].min():.1f}% - {df['dvol'].max():.1f}%")

            return df

        except Exception as e:
            logger.error(f"Failed to fetch DVOL history: {e}")
            return pd.DataFrame(columns=['timestamp', 'dvol', 'dvol_normalized', 'open', 'high', 'low'])

    def fetch_realized_volatility(
        self,
        days: int = 30,
        symbol: str = "BTCUSDT",
    ) -> float:
        """
        Calculate realized BTC volatility as fallback when DVOL unavailable.

        Uses Binance price data to compute annualized historical volatility.

        Args:
            days: Lookback period for volatility calculation
            symbol: Trading pair

        Returns:
            Annualized realized volatility (e.g., 0.65 = 65%)
        """
        logger.info(f"Computing {days}-day realized BTC volatility...")

        # Fetch from Binance (public, no auth needed)
        binance_url = "https://api.binance.com/api/v3/klines"

        resp = requests.get(binance_url, params={
            "symbol": symbol,
            "interval": "1d",
            "limit": days + 1,
        }, timeout=30)
        resp.raise_for_status()

        data = resp.json()
        closes = [float(candle[4]) for candle in data]  # Close prices

        # Compute log returns
        returns = np.diff(np.log(closes))

        # Annualized volatility
        realized_vol = np.std(returns) * np.sqrt(365) * 100  # As percentage

        logger.info(f"{days}-day realized vol: {realized_vol:.1f}%")

        return realized_vol

    def get_volatility_reading(
        self,
        prefer_implied: bool = True,
    ) -> DVOLReading:
        """
        Get volatility reading, preferring DVOL but falling back to realized vol.

        This is the main interface for the uncertainty decomposer.

        Args:
            prefer_implied: If True, try DVOL first; if False, use realized vol

        Returns:
            DVOLReading with best available volatility measure
        """
        if prefer_implied:
            try:
                reading = self.fetch_current_dvol()
                if reading.dvol is not None and reading.dvol > 0:
                    return reading
            except Exception as e:
                logger.warning(f"DVOL fetch failed: {e}")

        # Fallback to realized volatility
        return self._fallback_volatility()

    def _fallback_volatility(self) -> DVOLReading:
        """Compute realized volatility as fallback."""
        try:
            realized_vol = self.fetch_realized_volatility(days=30)

            return DVOLReading(
                timestamp=datetime.utcnow(),
                dvol=realized_vol,  # Use realized as proxy
                dvol_normalized=self._normalize_dvol(realized_vol),
                realized_vol_30d=realized_vol,
            )
        except Exception as e:
            logger.error(f"Realized vol fallback also failed: {e}")
            # Return moderate default
            return DVOLReading(
                timestamp=datetime.utcnow(),
                dvol=60.0,  # Moderate default
                dvol_normalized=0.25,
                realized_vol_30d=None,
            )

    def _normalize_dvol(self, dvol: float) -> float:
        """
        Normalize DVOL to [0, 1] range for uncertainty calculations.

        Mapping:
        - DVOL 30 (very low) -> 0.0
        - DVOL 150 (crisis) -> 1.0
        """
        if dvol is None:
            return 0.25  # Default moderate

        normalized = (dvol - self.DVOL_LOW) / (self.DVOL_HIGH - self.DVOL_LOW)
        return np.clip(normalized, 0.0, 1.0)

    def _api_call(self, endpoint: str, params: dict = None) -> dict:
        """Make Deribit API call."""
        url = f"{self.BASE_URL}/{endpoint}"

        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()

        data = resp.json()

        if "error" in data:
            raise RuntimeError(f"Deribit API error: {data['error']}")

        return data.get("result", {})

    def save_to_csv(self, df: pd.DataFrame, filename: str = "dvol_historical.csv"):
        """Save DVOL data to cache directory."""
        path = os.path.join(self.cache_dir, filename)
        df.to_csv(path, index=False)
        logger.info(f"Saved DVOL data to {path}")
        return path


class CryptoVolatilityAggregator:
    """
    Aggregates multiple crypto volatility sources for robust aleatoric uncertainty.

    Primary: DVOL (crypto implied vol)
    Secondary: VIX (TradFi contagion signal)
    Fallback: Realized BTC volatility

    This replaces the VIX-only approach in the original uncertainty decomposer.
    """

    def __init__(
        self,
        dvol_weight: float = 0.70,  # Primary crypto signal
        vix_weight: float = 0.30,   # TradFi contagion
    ):
        self.dvol_weight = dvol_weight
        self.vix_weight = vix_weight
        self.dvol_fetcher = DeribitDVOLFetcher()

    def get_composite_volatility(
        self,
        vix_level: Optional[float] = None,
        vix_normalized: Optional[float] = None,
    ) -> Tuple[float, Dict]:
        """
        Compute composite crypto volatility metric.

        Args:
            vix_level: Raw VIX value (if available from FRED)
            vix_normalized: Pre-normalized VIX [0, 1]

        Returns:
            (composite_volatility, breakdown_dict)
            - composite_volatility: [0, 1] blended measure
            - breakdown_dict: Component values for diagnostics
        """
        # Get DVOL
        dvol_reading = self.dvol_fetcher.get_volatility_reading(prefer_implied=True)
        dvol_norm = dvol_reading.dvol_normalized

        # Normalize VIX if raw value provided
        if vix_normalized is not None:
            vix_norm = vix_normalized
        elif vix_level is not None:
            # VIX normalization: 10 = low, 50 = crisis
            vix_norm = np.clip((vix_level - 10) / 40, 0, 1)
        else:
            vix_norm = 0.25  # Default moderate when no TradFi data

        # Weighted composite
        composite = (
            self.dvol_weight * dvol_norm +
            self.vix_weight * vix_norm
        )

        breakdown = {
            'dvol_raw': dvol_reading.dvol,
            'dvol_normalized': dvol_norm,
            'dvol_source': 'implied' if dvol_reading.realized_vol_30d is None else 'realized',
            'vix_normalized': vix_norm,
            'composite': composite,
            'weights': {'dvol': self.dvol_weight, 'vix': self.vix_weight},
        }

        return composite, breakdown


# ============================================================================
# CLI for testing
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Fetch Deribit DVOL data')
    parser.add_argument('--current', action='store_true', help='Fetch current DVOL')
    parser.add_argument('--history', action='store_true', help='Fetch historical DVOL')
    parser.add_argument('--days', type=int, default=90, help='Days of history')
    parser.add_argument('--realized', action='store_true', help='Compute realized vol')
    parser.add_argument('--composite', action='store_true', help='Test composite aggregator')

    args = parser.parse_args()

    fetcher = DeribitDVOLFetcher()

    if args.current:
        reading = fetcher.fetch_current_dvol()
        print(f"\nCurrent DVOL:")
        print(f"  Value: {reading.dvol:.1f}%")
        print(f"  Normalized: {reading.dvol_normalized:.3f}")
        print(f"  BTC Price: ${reading.btc_price:,.0f}" if reading.btc_price else "")
        print(f"  Timestamp: {reading.timestamp}")

    if args.history:
        end = datetime.utcnow()
        start = end - timedelta(days=args.days)
        df = fetcher.fetch_historical_dvol(start_time=start, end_time=end)

        if len(df) > 0:
            path = fetcher.save_to_csv(df)
            print(f"\nHistorical DVOL saved to {path}")
            print(f"  Records: {len(df)}")
            print(f"  Range: {df['dvol'].min():.1f}% - {df['dvol'].max():.1f}%")
            print(f"  Mean: {df['dvol'].mean():.1f}%")
        else:
            print("\nNo historical data available")

    if args.realized:
        vol = fetcher.fetch_realized_volatility(days=30)
        print(f"\n30-day Realized BTC Volatility: {vol:.1f}%")
        print(f"  Normalized: {fetcher._normalize_dvol(vol):.3f}")

    if args.composite:
        aggregator = CryptoVolatilityAggregator()
        composite, breakdown = aggregator.get_composite_volatility(vix_level=22.5)

        print(f"\nComposite Volatility:")
        print(f"  DVOL (raw): {breakdown['dvol_raw']:.1f}%")
        print(f"  DVOL (norm): {breakdown['dvol_normalized']:.3f}")
        print(f"  DVOL source: {breakdown['dvol_source']}")
        print(f"  VIX (norm): {breakdown['vix_normalized']:.3f}")
        print(f"  Composite: {breakdown['composite']:.3f}")

    # Default: show all
    if not any([args.current, args.history, args.realized, args.composite]):
        reading = fetcher.get_volatility_reading(prefer_implied=True)
        print(f"\nBest Available Volatility:")
        print(f"  DVOL: {reading.dvol:.1f}%")
        print(f"  Normalized: {reading.dvol_normalized:.3f}")
        print(f"  Source: {'implied' if reading.realized_vol_30d is None else 'realized (fallback)'}")


if __name__ == '__main__':
    main()
