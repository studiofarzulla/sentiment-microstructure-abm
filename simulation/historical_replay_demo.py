"""
Historical Replay Demo - Run Simulation with Real Data

This script demonstrates the full pipeline for running simulations
with historical market data:

1. Generate sample data (or load real data)
2. Replay through the Mesa ABM
3. Analyze results and compare to stylized facts

Usage:
    # Generate sample data and run demo
    python simulation/historical_replay_demo.py --generate
    
    # Run with existing data files
    python simulation/historical_replay_demo.py \
        --orderbook data/orderbook.csv \
        --sentiment data/sentiment.csv

Author: Murad Farzulla
Date: January 2026
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from simulation.market_env import create_default_market, CryptoMarketModel
from simulation.data_replay import (
    DataReplayLoader,
    create_sample_dataset,
    generate_sample_orderbook_data,
    generate_sample_sentiment_data,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Analysis Functions
# ============================================================================

def compute_stylized_facts(df: pd.DataFrame) -> dict:
    """
    Compute stylized facts from simulation results.
    
    Tests for:
    - Fat tails in returns
    - Volatility clustering
    - Spread mean-reversion
    """
    results = {}
    
    # Compute returns
    df = df.copy()
    df['returns'] = df['mid_price'].pct_change()
    returns = df['returns'].dropna()
    
    if len(returns) < 50:
        logger.warning("Insufficient data for stylized facts analysis")
        return {'error': 'insufficient_data'}
    
    # 1. Fat tails (excess kurtosis)
    kurtosis = returns.kurtosis()
    results['kurtosis'] = float(kurtosis)
    results['fat_tails'] = kurtosis > 0  # Normal = 0, fat tails > 0
    
    # 2. Volatility clustering (ACF of absolute returns)
    abs_returns = returns.abs()
    if len(abs_returns) > 20:
        # Lag-1 autocorrelation of |returns|
        acf_lag1 = abs_returns.autocorr(lag=1)
        acf_lag10 = abs_returns.autocorr(lag=10) if len(abs_returns) > 10 else 0
        results['vol_clustering_lag1'] = float(acf_lag1) if not np.isnan(acf_lag1) else 0.0
        results['vol_clustering_lag10'] = float(acf_lag10) if not np.isnan(acf_lag10) else 0.0
        results['volatility_clusters'] = acf_lag1 > 0.1
    
    # 3. Spread statistics
    spread_series = df['spread_bps'].dropna()
    if len(spread_series) > 10:
        results['spread_mean'] = float(spread_series.mean())
        results['spread_std'] = float(spread_series.std())
        results['spread_autocorr'] = float(spread_series.autocorr(lag=1)) if len(spread_series) > 1 else 0.0
    
    # 4. Return statistics
    results['return_mean'] = float(returns.mean())
    results['return_std'] = float(returns.std())
    results['return_skew'] = float(returns.skew())
    
    # 5. Sentiment-price correlation
    if 'sentiment' in df.columns:
        valid = df[['sentiment', 'returns']].dropna()
        if len(valid) > 10:
            corr = valid['sentiment'].corr(valid['returns'])
            results['sentiment_return_corr'] = float(corr) if not np.isnan(corr) else 0.0
    
    return results


def plot_results(df: pd.DataFrame, output_path: str = None):
    """Generate summary plots of simulation results."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Historical Replay Simulation Results', fontsize=14, fontweight='bold')
    
    # 1. Price trajectory
    ax = axes[0, 0]
    ax.plot(df['step'], df['mid_price'], 'b-', linewidth=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mid Price ($)')
    ax.set_title('Price Trajectory')
    ax.grid(True, alpha=0.3)
    
    # 2. Sentiment over time
    ax = axes[0, 1]
    if 'sentiment' in df.columns:
        ax.plot(df['step'], df['sentiment'], 'g-', linewidth=0.8)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=0.2, color='green', linestyle=':', alpha=0.5, label='Bullish threshold')
        ax.axhline(y=-0.2, color='red', linestyle=':', alpha=0.5, label='Bearish threshold')
        ax.fill_between(df['step'], df['sentiment'], 0, 
                       where=df['sentiment'] > 0, alpha=0.3, color='green')
        ax.fill_between(df['step'], df['sentiment'], 0,
                       where=df['sentiment'] < 0, alpha=0.3, color='red')
    ax.set_xlabel('Step')
    ax.set_ylabel('Sentiment')
    ax.set_title('Sentiment Signal')
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)
    
    # 3. Spread over time
    ax = axes[1, 0]
    valid_spread = df['spread_bps'].dropna()
    if len(valid_spread) > 0:
        ax.plot(df.loc[valid_spread.index, 'step'], valid_spread, 'orange', linewidth=0.8)
        ax.axhline(y=valid_spread.mean(), color='red', linestyle='--', 
                  label=f'Mean: {valid_spread.mean():.1f} bps')
    ax.set_xlabel('Step')
    ax.set_ylabel('Spread (bps)')
    ax.set_title('Bid-Ask Spread')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Return distribution
    ax = axes[1, 1]
    returns = df['mid_price'].pct_change().dropna()
    if len(returns) > 10:
        ax.hist(returns * 100, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--')
        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Density')
        ax.set_title(f'Return Distribution (kurtosis={returns.kurtosis():.2f})')
    ax.grid(True, alpha=0.3)
    
    # 5. Cumulative volume
    ax = axes[2, 0]
    ax.plot(df['step'], df['total_volume'], 'purple', linewidth=1.0)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative Volume')
    ax.set_title('Trading Volume')
    ax.grid(True, alpha=0.3)
    
    # 6. Trade count
    ax = axes[2, 1]
    ax.plot(df['step'], df['trade_count'], 'brown', linewidth=1.0)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative Trades')
    ax.set_title('Trade Count')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# Main Demo Functions
# ============================================================================

def run_with_sample_data(
    duration_hours: float = 0.5,
    step_interval_ms: float = 500,
    seed: int = 42,
    output_dir: str = None,
) -> pd.DataFrame:
    """
    Run simulation with generated sample data.
    
    This is useful for testing and development when real data is not available.
    """
    logger.info("=" * 70)
    logger.info("HISTORICAL REPLAY DEMO - SAMPLE DATA")
    logger.info("=" * 70)
    
    # Create output directory
    output_dir = output_dir or os.path.join(os.path.dirname(__file__), 'replay_output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate sample data
    logger.info("\n1. Generating sample historical data...")
    ob_path, sent_path = create_sample_dataset(
        output_dir,
        duration_hours=duration_hours,
        orderbook_interval_ms=100,
        sentiment_interval_min=2.0,
    )
    
    # Load into replay system
    logger.info("\n2. Loading data into replay system...")
    loader = DataReplayLoader()
    loader.load_orderbook_data(ob_path)
    loader.load_sentiment_data(sent_path)
    
    summary = loader.get_summary()
    logger.info(f"   Order book: {summary['orderbook']['count']} snapshots")
    logger.info(f"   Sentiment: {summary['sentiment']['count']} observations")
    logger.info(f"   Price range: ${summary['orderbook']['price_range'][0]:.2f} - ${summary['orderbook']['price_range'][1]:.2f}")
    
    # Create market model
    logger.info("\n3. Creating multi-agent market...")
    np.random.seed(seed)
    
    # Get initial price from data
    first_tick = next(loader.replay(step_interval_ms=step_interval_ms, max_steps=1))
    initial_price = first_tick.mid_price
    
    model = create_default_market(
        n_market_makers=3,
        n_informed=5,
        n_noise=15,
        initial_price=initial_price,
        seed=seed,
    )
    
    logger.info(f"   Initial price: ${initial_price:.2f}")
    logger.info(f"   Agents: {len(model._agents)}")
    
    # Run replay simulation
    logger.info("\n4. Running historical replay...")
    replay_gen = loader.replay(step_interval_ms=step_interval_ms)
    history = model.run_replay(replay_gen, price_tracking=True, log_interval=200)
    
    # Convert to DataFrame
    df = pd.DataFrame(history)
    
    # Add replay metadata
    df['data_source'] = 'sample'
    df['replay_interval_ms'] = step_interval_ms
    
    # Results summary
    logger.info("\n" + "=" * 70)
    logger.info("SIMULATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total steps: {len(df)}")
    logger.info(f"Total trades: {df['trade_count'].iloc[-1]}")
    logger.info(f"Total volume: {df['total_volume'].iloc[-1]:.2f}")
    
    valid_prices = df['mid_price'].dropna()
    if len(valid_prices) > 0:
        logger.info(f"Price: ${valid_prices.iloc[0]:.2f} -> ${valid_prices.iloc[-1]:.2f}")
        returns = valid_prices.pct_change().dropna()
        if len(returns) > 0:
            logger.info(f"Total return: {(valid_prices.iloc[-1]/valid_prices.iloc[0] - 1)*100:.2f}%")
    
    valid_spread = df['spread_bps'].dropna()
    if len(valid_spread) > 0:
        logger.info(f"Mean spread: {valid_spread.mean():.2f} bps")
    
    # Stylized facts
    logger.info("\n--- Stylized Facts Analysis ---")
    facts = compute_stylized_facts(df)
    for key, value in facts.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Save results
    results_path = os.path.join(output_dir, 'replay_results.csv')
    df.to_csv(results_path, index=False)
    logger.info(f"\nResults saved to: {results_path}")
    
    # Generate plots
    plot_path = os.path.join(output_dir, 'replay_analysis.png')
    plot_results(df, plot_path)
    
    return df


def run_with_real_data(
    orderbook_path: str,
    sentiment_path: str,
    step_interval_ms: float = 500,
    seed: int = 42,
    output_dir: str = None,
) -> pd.DataFrame:
    """
    Run simulation with real historical data files.
    
    Args:
        orderbook_path: Path to order book CSV
        sentiment_path: Path to sentiment CSV  
        step_interval_ms: Simulation step interval
        seed: Random seed
        output_dir: Output directory for results
    """
    logger.info("=" * 70)
    logger.info("HISTORICAL REPLAY DEMO - REAL DATA")
    logger.info("=" * 70)
    
    # Validate inputs
    if not os.path.exists(orderbook_path):
        raise FileNotFoundError(f"Order book file not found: {orderbook_path}")
    if not os.path.exists(sentiment_path):
        raise FileNotFoundError(f"Sentiment file not found: {sentiment_path}")
    
    # Create output directory
    output_dir = output_dir or os.path.join(os.path.dirname(__file__), 'replay_output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    logger.info("\n1. Loading historical data...")
    loader = DataReplayLoader()
    loader.load_orderbook_data(orderbook_path)
    loader.load_sentiment_data(sentiment_path)
    
    summary = loader.get_summary()
    logger.info(f"   Order book: {summary['orderbook']['count']} snapshots")
    logger.info(f"   Sentiment: {summary['sentiment']['count']} observations")
    logger.info(f"   Time range: {summary['time_range']['start']} to {summary['time_range']['end']}")
    
    # Create market
    logger.info("\n2. Creating market model...")
    np.random.seed(seed)
    
    first_tick = next(loader.replay(step_interval_ms=step_interval_ms, max_steps=1))
    initial_price = first_tick.mid_price
    
    model = create_default_market(
        n_market_makers=3,
        n_informed=5,
        n_noise=15,
        initial_price=initial_price,
        seed=seed,
    )
    
    # Run replay
    logger.info("\n3. Running replay simulation...")
    replay_gen = loader.replay(step_interval_ms=step_interval_ms)
    history = model.run_replay(replay_gen, price_tracking=True)
    
    # Process results
    df = pd.DataFrame(history)
    df['data_source'] = 'real'
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    logger.info(f"Steps: {len(df)}, Trades: {df['trade_count'].iloc[-1]}")
    
    facts = compute_stylized_facts(df)
    logger.info("\nStylized Facts:")
    for k, v in facts.items():
        logger.info(f"  {k}: {v}")
    
    # Save
    results_path = os.path.join(output_dir, 'replay_results.csv')
    df.to_csv(results_path, index=False)
    plot_results(df, os.path.join(output_dir, 'replay_analysis.png'))
    
    return df


# ============================================================================
# Quick Test Function
# ============================================================================

def quick_test():
    """Quick test to verify the replay system works."""
    logger.info("Running quick test...")
    
    # Generate minimal data
    start_time = datetime.utcnow()
    
    ob_df = generate_sample_orderbook_data(
        n_points=100,
        start_price=42000,
        start_time=start_time,
        interval_ms=100,
    )
    
    sent_df = generate_sample_sentiment_data(
        n_points=10,
        start_time=start_time,
        interval_minutes=1.0,
    )
    
    # Save to temp
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        ob_path = os.path.join(tmpdir, 'ob.csv')
        sent_path = os.path.join(tmpdir, 'sent.csv')
        
        ob_df.to_csv(ob_path, index=False)
        sent_df.to_csv(sent_path, index=False)
        
        # Load and replay
        loader = DataReplayLoader()
        loader.load_orderbook_data(ob_path)
        loader.load_sentiment_data(sent_path)
        
        # Create model
        model = create_default_market(initial_price=42000, seed=42)
        
        # Run short replay
        ticks = list(loader.replay(step_interval_ms=500, max_steps=50))
        
        for tick in ticks[:5]:
            model.set_sentiment(*tick.to_sentiment_tuple())
            model.step()
        
        logger.info(f"✓ Replayed {len(ticks)} ticks successfully")
        logger.info(f"✓ Final trades: {model.order_book.trade_count}")
        logger.info("✓ Quick test passed!")
        
        return True


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run ABM simulation with historical data replay',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate sample data and run demo
  python historical_replay_demo.py --generate
  
  # Run with custom duration
  python historical_replay_demo.py --generate --duration 2.0
  
  # Run with real data files
  python historical_replay_demo.py --orderbook data/ob.csv --sentiment data/sent.csv
  
  # Quick test
  python historical_replay_demo.py --test
        """
    )
    
    parser.add_argument('--generate', action='store_true',
                       help='Generate sample data and run demo')
    parser.add_argument('--orderbook', type=str,
                       help='Path to order book CSV file')
    parser.add_argument('--sentiment', type=str,
                       help='Path to sentiment CSV file')
    parser.add_argument('--duration', type=float, default=0.5,
                       help='Duration in hours for sample data (default: 0.5)')
    parser.add_argument('--interval', type=float, default=500,
                       help='Step interval in milliseconds (default: 500)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output', type=str,
                       help='Output directory for results')
    parser.add_argument('--test', action='store_true',
                       help='Run quick test')
    
    args = parser.parse_args()
    
    if args.test:
        success = quick_test()
        return 0 if success else 1
    
    if args.generate:
        df = run_with_sample_data(
            duration_hours=args.duration,
            step_interval_ms=args.interval,
            seed=args.seed,
            output_dir=args.output,
        )
        return 0
    
    if args.orderbook and args.sentiment:
        df = run_with_real_data(
            orderbook_path=args.orderbook,
            sentiment_path=args.sentiment,
            step_interval_ms=args.interval,
            seed=args.seed,
            output_dir=args.output,
        )
        return 0
    
    # No args - show help
    parser.print_help()
    return 1


if __name__ == '__main__':
    exit(main())
