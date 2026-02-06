"""
Historical Data Replay System for Real-Data Simulation

This module provides infrastructure to replay historical market data
(order books, sentiment, prices) through the ABM simulation.

Key Features:
- Load timestamped data from CSV/Parquet files
- Align multiple data streams by timestamp
- Feed synchronized data to Mesa simulation
- Support for backtesting and validation

Usage:
    # Load historical data
    loader = DataReplayLoader()
    loader.load_orderbook_data("binance_btcusdt_2024.csv")
    loader.load_sentiment_data("reddit_sentiment_2024.csv")
    
    # Create replay generator
    replay_gen = loader.create_replay_generator()
    
    # Run simulation with real data
    model = create_default_market()
    for tick in replay_gen:
        model.set_market_state_from_tick(tick)
        model.step()

Author: Murad Farzulla
Date: January 2026
"""

import os
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Generator, Union, Any
from pathlib import Path
import numpy as np

# Try pandas import (should be available)
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class OrderBookTick:
    """Single order book snapshot from historical data."""
    timestamp: datetime
    symbol: str
    best_bid: float
    best_ask: float
    mid_price: float
    spread: float
    spread_bps: float
    bid_volume: float
    ask_volume: float
    imbalance: float
    # Optional: full depth
    bids: List[Tuple[float, float]] = field(default_factory=list)
    asks: List[Tuple[float, float]] = field(default_factory=list)
    source: str = "binance"
    
    @property
    def is_valid(self) -> bool:
        return self.mid_price > 0 and self.spread >= 0


@dataclass  
class SentimentTick:
    """Single sentiment observation from historical data."""
    timestamp: datetime
    sentiment: float  # [-1, 1]
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    regime: str = "neutral"
    # Optional extended fields
    retail_sentiment: float = 0.0
    institutional_sentiment: float = 0.0
    divergence: float = 0.0
    source: str = "reddit"
    text_snippet: str = ""
    
    @property
    def total_uncertainty(self) -> float:
        return self.epistemic_uncertainty + self.aleatoric_uncertainty


@dataclass
class ReplayTick:
    """
    Combined tick for simulation consumption.
    
    Merges order book and sentiment data at a given timestamp.
    This is what the simulation actually receives each step.
    """
    timestamp: datetime
    step: int
    
    # Order book state
    mid_price: float
    best_bid: float
    best_ask: float
    spread_bps: float
    imbalance: float
    
    # Sentiment state
    sentiment: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    regime: str
    
    # Metadata
    orderbook_age_ms: float = 0.0  # How stale is the orderbook data
    sentiment_age_ms: float = 0.0  # How stale is the sentiment data
    is_interpolated: bool = False  # Was data interpolated vs actual observation
    
    def to_sentiment_tuple(self) -> Tuple[float, float, float, str]:
        """Return (sentiment, epistemic, aleatoric, regime) for simulation."""
        return (
            self.sentiment,
            self.epistemic_uncertainty,
            self.aleatoric_uncertainty,
            self.regime
        )
    
    def to_dict(self) -> dict:
        """Serialize for logging/CSV."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'step': self.step,
            'mid_price': self.mid_price,
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'spread_bps': self.spread_bps,
            'imbalance': self.imbalance,
            'sentiment': self.sentiment,
            'epistemic_uncertainty': self.epistemic_uncertainty,
            'aleatoric_uncertainty': self.aleatoric_uncertainty,
            'regime': self.regime,
            'orderbook_age_ms': self.orderbook_age_ms,
            'sentiment_age_ms': self.sentiment_age_ms,
            'is_interpolated': self.is_interpolated,
        }


# ============================================================================
# Data Loaders
# ============================================================================

class OrderBookLoader:
    """Load order book data from files."""
    
    REQUIRED_COLUMNS = ['timestamp', 'mid_price', 'best_bid', 'best_ask']
    OPTIONAL_COLUMNS = ['spread', 'spread_bps', 'bid_volume', 'ask_volume', 'imbalance', 'symbol']
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.symbol: str = "BTCUSDT"
        
    def load_csv(self, path: str, symbol: str = "BTCUSDT") -> 'OrderBookLoader':
        """Load order book data from CSV."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required for data loading")
            
        logger.info(f"Loading order book data from {path}")
        df = pd.read_csv(path)
        
        # Validate required columns
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Parse timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Compute derived columns if missing
        if 'spread' not in df.columns:
            df['spread'] = df['best_ask'] - df['best_bid']
        if 'spread_bps' not in df.columns:
            df['spread_bps'] = (df['spread'] / df['mid_price']) * 10000
        if 'imbalance' not in df.columns:
            df['imbalance'] = 0.0
        if 'bid_volume' not in df.columns:
            df['bid_volume'] = 1.0
        if 'ask_volume' not in df.columns:
            df['ask_volume'] = 1.0
            
        self.data = df
        self.symbol = symbol
        
        logger.info(f"Loaded {len(df)} order book snapshots")
        logger.info(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return self
    
    def load_from_binance_format(self, path: str) -> 'OrderBookLoader':
        """Load data in Binance client output format."""
        return self.load_csv(path)  # Same format currently
    
    def get_tick_at(self, timestamp: datetime) -> Optional[OrderBookTick]:
        """Get order book tick at or before timestamp."""
        if self.data is None or len(self.data) == 0:
            return None
            
        # Find most recent tick before timestamp
        mask = self.data['timestamp'] <= timestamp
        if not mask.any():
            return None
            
        row = self.data[mask].iloc[-1]
        
        return OrderBookTick(
            timestamp=row['timestamp'].to_pydatetime(),
            symbol=self.symbol,
            best_bid=float(row['best_bid']),
            best_ask=float(row['best_ask']),
            mid_price=float(row['mid_price']),
            spread=float(row.get('spread', row['best_ask'] - row['best_bid'])),
            spread_bps=float(row.get('spread_bps', 0)),
            bid_volume=float(row.get('bid_volume', 1.0)),
            ask_volume=float(row.get('ask_volume', 1.0)),
            imbalance=float(row.get('imbalance', 0.0)),
        )
    
    def iter_ticks(self) -> Generator[OrderBookTick, None, None]:
        """Iterate through all order book ticks."""
        if self.data is None:
            return
            
        for _, row in self.data.iterrows():
            yield OrderBookTick(
                timestamp=row['timestamp'].to_pydatetime(),
                symbol=self.symbol,
                best_bid=float(row['best_bid']),
                best_ask=float(row['best_ask']),
                mid_price=float(row['mid_price']),
                spread=float(row.get('spread', row['best_ask'] - row['best_bid'])),
                spread_bps=float(row.get('spread_bps', 0)),
                bid_volume=float(row.get('bid_volume', 1.0)),
                ask_volume=float(row.get('ask_volume', 1.0)),
                imbalance=float(row.get('imbalance', 0.0)),
            )


class SentimentLoader:
    """Load sentiment data from files."""
    
    REQUIRED_COLUMNS = ['timestamp', 'sentiment']
    OPTIONAL_COLUMNS = ['epistemic_uncertainty', 'aleatoric_uncertainty', 'regime', 
                        'retail_sentiment', 'institutional_sentiment', 'text']
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        
    def load_csv(self, path: str) -> 'SentimentLoader':
        """Load sentiment data from CSV."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required for data loading")
            
        logger.info(f"Loading sentiment data from {path}")
        df = pd.read_csv(path)
        
        # Validate required columns
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Parse timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Fill missing columns with defaults
        if 'epistemic_uncertainty' not in df.columns:
            df['epistemic_uncertainty'] = 0.05
        if 'aleatoric_uncertainty' not in df.columns:
            df['aleatoric_uncertainty'] = 0.2
        if 'regime' not in df.columns:
            df['regime'] = df['sentiment'].apply(
                lambda s: 'bullish' if s > 0.2 else ('bearish' if s < -0.2 else 'neutral')
            )
            
        self.data = df
        
        logger.info(f"Loaded {len(df)} sentiment observations")
        logger.info(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"Sentiment range: [{df['sentiment'].min():.3f}, {df['sentiment'].max():.3f}]")
        
        return self
    
    def get_tick_at(self, timestamp: datetime) -> Optional[SentimentTick]:
        """Get sentiment tick at or before timestamp."""
        if self.data is None or len(self.data) == 0:
            return None
            
        # Find most recent tick before timestamp
        mask = self.data['timestamp'] <= timestamp
        if not mask.any():
            return None
            
        row = self.data[mask].iloc[-1]
        
        return SentimentTick(
            timestamp=row['timestamp'].to_pydatetime(),
            sentiment=float(row['sentiment']),
            epistemic_uncertainty=float(row.get('epistemic_uncertainty', 0.05)),
            aleatoric_uncertainty=float(row.get('aleatoric_uncertainty', 0.2)),
            regime=str(row.get('regime', 'neutral')),
            retail_sentiment=float(row.get('retail_sentiment', row['sentiment'])),
            institutional_sentiment=float(row.get('institutional_sentiment', 0.0)),
            text_snippet=str(row.get('text', ''))[:100],
        )
    
    def iter_ticks(self) -> Generator[SentimentTick, None, None]:
        """Iterate through all sentiment ticks."""
        if self.data is None:
            return
            
        for _, row in self.data.iterrows():
            yield SentimentTick(
                timestamp=row['timestamp'].to_pydatetime(),
                sentiment=float(row['sentiment']),
                epistemic_uncertainty=float(row.get('epistemic_uncertainty', 0.05)),
                aleatoric_uncertainty=float(row.get('aleatoric_uncertainty', 0.2)),
                regime=str(row.get('regime', 'neutral')),
            )


# ============================================================================
# Data Replay Loader (Main Class)
# ============================================================================

class DataReplayLoader:
    """
    Main class for loading and replaying historical data.
    
    Handles:
    - Loading multiple data sources (order book, sentiment)
    - Timestamp alignment between sources
    - Interpolation for missing data
    - Generation of ReplayTicks for simulation
    
    Example:
        loader = DataReplayLoader()
        loader.load_orderbook_data("orderbooks.csv")
        loader.load_sentiment_data("sentiment.csv")
        
        # Run simulation
        model = create_default_market()
        for tick in loader.replay(step_interval_ms=500):
            model.set_sentiment(*tick.to_sentiment_tuple())
            model.step()
    """
    
    def __init__(
        self,
        interpolate_missing: bool = True,
        max_stale_ms: float = 60000,  # 1 minute max staleness
    ):
        """
        Initialize replay loader.
        
        Args:
            interpolate_missing: Whether to interpolate between observations
            max_stale_ms: Maximum allowed staleness before marking data as stale
        """
        self.interpolate_missing = interpolate_missing
        self.max_stale_ms = max_stale_ms
        
        self.orderbook_loader = OrderBookLoader()
        self.sentiment_loader = SentimentLoader()
        
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None
        
    def load_orderbook_data(self, path: str, symbol: str = "BTCUSDT") -> 'DataReplayLoader':
        """Load order book data from CSV file."""
        self.orderbook_loader.load_csv(path, symbol)
        self._update_time_range()
        return self
        
    def load_sentiment_data(self, path: str) -> 'DataReplayLoader':
        """Load sentiment data from CSV file."""
        self.sentiment_loader.load_csv(path)
        self._update_time_range()
        return self
    
    def _update_time_range(self):
        """Update the common time range across all loaded data."""
        starts = []
        ends = []
        
        if self.orderbook_loader.data is not None:
            starts.append(self.orderbook_loader.data['timestamp'].min())
            ends.append(self.orderbook_loader.data['timestamp'].max())
            
        if self.sentiment_loader.data is not None:
            starts.append(self.sentiment_loader.data['timestamp'].min())
            ends.append(self.sentiment_loader.data['timestamp'].max())
            
        if starts:
            self._start_time = max(starts).to_pydatetime()
            self._end_time = min(ends).to_pydatetime()
            
            logger.info(f"Common time range: {self._start_time} to {self._end_time}")
    
    def get_time_range(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get the common time range of loaded data."""
        return self._start_time, self._end_time
    
    def replay(
        self,
        step_interval_ms: float = 500,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        max_steps: Optional[int] = None,
    ) -> Generator[ReplayTick, None, None]:
        """
        Generate replay ticks at fixed intervals.
        
        Args:
            step_interval_ms: Time between simulation steps in milliseconds
            start_time: Override start time (default: data start)
            end_time: Override end time (default: data end)
            max_steps: Maximum number of steps to generate
            
        Yields:
            ReplayTick for each simulation step
        """
        # Determine time range
        t_start = start_time or self._start_time
        t_end = end_time or self._end_time
        
        if t_start is None or t_end is None:
            raise ValueError("No data loaded. Call load_orderbook_data() and/or load_sentiment_data() first.")
        
        # Initialize state
        current_time = t_start
        step = 0
        interval = timedelta(milliseconds=step_interval_ms)
        
        # Track last known values for interpolation
        last_ob: Optional[OrderBookTick] = None
        last_sent: Optional[SentimentTick] = None
        
        logger.info(f"Starting replay from {t_start} to {t_end}")
        logger.info(f"Step interval: {step_interval_ms}ms")
        
        while current_time <= t_end:
            if max_steps is not None and step >= max_steps:
                break
                
            # Get current order book state
            ob_tick = self.orderbook_loader.get_tick_at(current_time)
            if ob_tick is not None:
                last_ob = ob_tick
                ob_age_ms = (current_time - ob_tick.timestamp).total_seconds() * 1000
            else:
                ob_age_ms = self.max_stale_ms + 1  # Mark as stale
                
            # Get current sentiment state
            sent_tick = self.sentiment_loader.get_tick_at(current_time)
            if sent_tick is not None:
                last_sent = sent_tick
                sent_age_ms = (current_time - sent_tick.timestamp).total_seconds() * 1000
            else:
                sent_age_ms = self.max_stale_ms + 1  # Mark as stale
            
            # Build replay tick
            if last_ob is not None and last_sent is not None:
                tick = ReplayTick(
                    timestamp=current_time,
                    step=step,
                    mid_price=last_ob.mid_price,
                    best_bid=last_ob.best_bid,
                    best_ask=last_ob.best_ask,
                    spread_bps=last_ob.spread_bps,
                    imbalance=last_ob.imbalance,
                    sentiment=last_sent.sentiment,
                    epistemic_uncertainty=last_sent.epistemic_uncertainty,
                    aleatoric_uncertainty=last_sent.aleatoric_uncertainty,
                    regime=last_sent.regime,
                    orderbook_age_ms=ob_age_ms,
                    sentiment_age_ms=sent_age_ms,
                    is_interpolated=(ob_age_ms > step_interval_ms or sent_age_ms > step_interval_ms),
                )
                yield tick
                
            elif last_ob is not None:
                # Only order book available
                tick = ReplayTick(
                    timestamp=current_time,
                    step=step,
                    mid_price=last_ob.mid_price,
                    best_bid=last_ob.best_bid,
                    best_ask=last_ob.best_ask,
                    spread_bps=last_ob.spread_bps,
                    imbalance=last_ob.imbalance,
                    sentiment=0.0,  # Neutral default
                    epistemic_uncertainty=0.1,  # High uncertainty
                    aleatoric_uncertainty=0.3,
                    regime='neutral',
                    orderbook_age_ms=ob_age_ms,
                    sentiment_age_ms=self.max_stale_ms + 1,
                    is_interpolated=True,
                )
                yield tick
                
            elif last_sent is not None:
                # Only sentiment available (unusual, but handle it)
                tick = ReplayTick(
                    timestamp=current_time,
                    step=step,
                    mid_price=100.0,  # Placeholder
                    best_bid=99.9,
                    best_ask=100.1,
                    spread_bps=20.0,
                    imbalance=0.0,
                    sentiment=last_sent.sentiment,
                    epistemic_uncertainty=last_sent.epistemic_uncertainty,
                    aleatoric_uncertainty=last_sent.aleatoric_uncertainty,
                    regime=last_sent.regime,
                    orderbook_age_ms=self.max_stale_ms + 1,
                    sentiment_age_ms=sent_age_ms,
                    is_interpolated=True,
                )
                yield tick
            
            # Advance time
            current_time += interval
            step += 1
            
            # Progress logging
            if step % 1000 == 0:
                progress = (current_time - t_start) / (t_end - t_start) * 100
                logger.info(f"Replay progress: {progress:.1f}% (step {step})")
        
        logger.info(f"Replay complete: {step} steps generated")
    
    def create_sentiment_generator(
        self,
        step_interval_ms: float = 500,
        **kwargs
    ):
        """
        Create a sentiment generator function for use with model.run_simulation().
        
        Returns a callable that takes step number and returns sentiment tuple.
        
        Usage:
            loader = DataReplayLoader()
            loader.load_sentiment_data("sentiment.csv")
            
            gen = loader.create_sentiment_generator()
            model.run_simulation(1000, sentiment_generator=gen)
        """
        # Pre-generate all ticks
        ticks = list(self.replay(step_interval_ms=step_interval_ms, **kwargs))
        
        def generator(step: int) -> Tuple[float, float, float, str]:
            if step < len(ticks):
                return ticks[step].to_sentiment_tuple()
            else:
                # Past end of data - return last known
                return ticks[-1].to_sentiment_tuple() if ticks else (0.0, 0.1, 0.2, 'neutral')
        
        return generator
    
    def get_summary(self) -> dict:
        """Get summary statistics of loaded data."""
        summary = {
            'time_range': {
                'start': self._start_time.isoformat() if self._start_time else None,
                'end': self._end_time.isoformat() if self._end_time else None,
            }
        }
        
        if self.orderbook_loader.data is not None:
            ob_df = self.orderbook_loader.data
            summary['orderbook'] = {
                'count': len(ob_df),
                'price_range': [float(ob_df['mid_price'].min()), float(ob_df['mid_price'].max())],
                'mean_spread_bps': float(ob_df['spread_bps'].mean()) if 'spread_bps' in ob_df else None,
            }
            
        if self.sentiment_loader.data is not None:
            sent_df = self.sentiment_loader.data
            summary['sentiment'] = {
                'count': len(sent_df),
                'mean': float(sent_df['sentiment'].mean()),
                'std': float(sent_df['sentiment'].std()),
                'range': [float(sent_df['sentiment'].min()), float(sent_df['sentiment'].max())],
            }
            
        return summary


# ============================================================================
# Sample Data Generator (for testing)
# ============================================================================

def generate_sample_orderbook_data(
    n_points: int = 1000,
    start_price: float = 42000.0,
    volatility: float = 0.001,
    base_spread_bps: float = 5.0,
    start_time: Optional[datetime] = None,
    interval_ms: float = 100,
) -> pd.DataFrame:
    """
    Generate sample order book data for testing.
    
    Returns DataFrame in standard format.
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas required")
        
    start_time = start_time or datetime.utcnow()
    
    timestamps = [start_time + timedelta(milliseconds=i * interval_ms) for i in range(n_points)]
    
    # Random walk for price
    returns = np.random.normal(0, volatility, n_points)
    prices = start_price * np.exp(np.cumsum(returns))
    
    # Spread varies with volatility
    spreads_bps = base_spread_bps + np.abs(returns) * 1000
    spreads = prices * spreads_bps / 10000
    
    # Imbalance oscillates
    imbalances = np.sin(np.arange(n_points) * 0.05) * 0.3 + np.random.normal(0, 0.1, n_points)
    imbalances = np.clip(imbalances, -1, 1)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'mid_price': prices,
        'best_bid': prices - spreads / 2,
        'best_ask': prices + spreads / 2,
        'spread': spreads,
        'spread_bps': spreads_bps,
        'bid_volume': np.random.uniform(0.5, 5.0, n_points),
        'ask_volume': np.random.uniform(0.5, 5.0, n_points),
        'imbalance': imbalances,
        'symbol': 'BTCUSDT',
    })
    
    return df


def generate_sample_sentiment_data(
    n_points: int = 200,
    start_time: Optional[datetime] = None,
    interval_minutes: float = 5.0,
    regime_switch_prob: float = 0.05,
) -> pd.DataFrame:
    """
    Generate sample sentiment data for testing.
    
    Includes regime switching and uncertainty variation.
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas required")
        
    start_time = start_time or datetime.utcnow()
    
    timestamps = [start_time + timedelta(minutes=i * interval_minutes) for i in range(n_points)]
    
    # Regime-switching sentiment
    regimes = ['neutral']
    sentiment = [0.0]
    
    for i in range(1, n_points):
        # Random regime switch
        if np.random.random() < regime_switch_prob:
            new_regime = np.random.choice(['bullish', 'bearish', 'neutral'])
            regimes.append(new_regime)
        else:
            regimes.append(regimes[-1])
        
        # Sentiment based on regime
        if regimes[-1] == 'bullish':
            base_sent = 0.4
        elif regimes[-1] == 'bearish':
            base_sent = -0.4
        else:
            base_sent = 0.0
        
        # AR(1) process with regime mean
        sent = 0.8 * sentiment[-1] + 0.2 * base_sent + np.random.normal(0, 0.1)
        sentiment.append(np.clip(sent, -1, 1))
    
    sentiment = np.array(sentiment)
    
    # Uncertainty varies inversely with sentiment magnitude
    epistemic = 0.03 + (1 - np.abs(sentiment)) * 0.05 + np.random.uniform(0, 0.02, n_points)
    aleatoric = 0.15 + (1 - np.abs(sentiment)) * 0.1 + np.random.uniform(0, 0.05, n_points)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'sentiment': sentiment,
        'epistemic_uncertainty': epistemic,
        'aleatoric_uncertainty': aleatoric,
        'regime': regimes,
    })
    
    return df


def create_sample_dataset(
    output_dir: str,
    duration_hours: float = 1.0,
    orderbook_interval_ms: float = 100,
    sentiment_interval_min: float = 5.0,
) -> Tuple[str, str]:
    """
    Create sample dataset files for testing.
    
    Returns paths to generated files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = datetime.utcnow()
    
    # Generate order book data
    n_ob_points = int(duration_hours * 3600 * 1000 / orderbook_interval_ms)
    ob_df = generate_sample_orderbook_data(
        n_points=n_ob_points,
        start_time=start_time,
        interval_ms=orderbook_interval_ms,
    )
    ob_path = os.path.join(output_dir, 'sample_orderbook.csv')
    ob_df.to_csv(ob_path, index=False)
    
    # Generate sentiment data
    n_sent_points = int(duration_hours * 60 / sentiment_interval_min)
    sent_df = generate_sample_sentiment_data(
        n_points=n_sent_points,
        start_time=start_time,
        interval_minutes=sentiment_interval_min,
    )
    sent_path = os.path.join(output_dir, 'sample_sentiment.csv')
    sent_df.to_csv(sent_path, index=False)
    
    logger.info(f"Created sample dataset in {output_dir}")
    logger.info(f"  Order book: {n_ob_points} points ({ob_path})")
    logger.info(f"  Sentiment: {n_sent_points} points ({sent_path})")
    
    return ob_path, sent_path


# ============================================================================
# CLI / Demo
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Data Replay System Demo')
    parser.add_argument('--generate', action='store_true', help='Generate sample data')
    parser.add_argument('--output-dir', default='/tmp/sample_data', help='Output directory')
    parser.add_argument('--duration', type=float, default=1.0, help='Duration in hours')
    
    args = parser.parse_args()
    
    if args.generate:
        ob_path, sent_path = create_sample_dataset(
            args.output_dir,
            duration_hours=args.duration,
        )
        
        # Test loading
        print("\n" + "="*60)
        print("TESTING DATA REPLAY")
        print("="*60)
        
        loader = DataReplayLoader()
        loader.load_orderbook_data(ob_path)
        loader.load_sentiment_data(sent_path)
        
        print(f"\nSummary: {loader.get_summary()}")
        
        # Generate a few ticks
        print("\nFirst 5 replay ticks:")
        for i, tick in enumerate(loader.replay(step_interval_ms=500, max_steps=5)):
            print(f"  Step {tick.step}: price=${tick.mid_price:.2f}, sent={tick.sentiment:+.3f}, regime={tick.regime}")
        
        print("\n✅ Data replay system working!")
    else:
        parser.print_help()
