"""
Run ABM Simulation with Real Fear & Greed Data

This script runs the full simulation pipeline using:
- Real BTC prices from Binance
- Real sentiment from Fear & Greed Index
- Multi-agent market microstructure

Output: Simulation results for calibration and paper figures.

Author: Murad Farzulla
Date: January 2026
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from simulation.market_env import create_default_market, CryptoMarketModel
from simulation.market_env import MarketMakerAgent, InformedTraderAgent, NoiseTraderAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Loading
# ============================================================================

def load_real_data(data_path: str = None) -> pd.DataFrame:
    """Load the Fear & Greed + Binance dataset."""
    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'datasets', 'btc_sentiment_daily.csv'
        )
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    logger.info(f"Loaded {len(df)} days of real data")
    logger.info(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    return df


def create_intraday_sentiment(daily_df: pd.DataFrame, steps_per_day: int = 100) -> pd.DataFrame:
    """
    Expand daily sentiment to intraday resolution with realistic noise.
    
    Daily Fear & Greed → intraday sentiment with:
    - AR(1) persistence within day
    - News shocks (jumps)
    - Mean-reversion to daily value
    """
    records = []
    
    for _, day in daily_df.iterrows():
        daily_sent = day['macro_sentiment']
        daily_price = day['close']
        regime = day['regime']
        
        # Intraday sentiment starts at daily value
        current_sent = daily_sent
        
        for step in range(steps_per_day):
            # AR(1) with mean reversion to daily value
            noise = np.random.normal(0, 0.02)
            mean_revert = 0.05 * (daily_sent - current_sent)
            current_sent = 0.95 * current_sent + mean_revert + noise
            current_sent = np.clip(current_sent, -1, 1)
            
            # Occasional news shock (2% chance)
            if np.random.random() < 0.02:
                shock = np.random.choice([-1, 1]) * np.random.uniform(0.1, 0.3)
                current_sent = np.clip(current_sent + shock, -1, 1)
            
            # Compute uncertainties based on regime
            if regime in ['extreme_fear', 'extreme_greed']:
                epistemic = 0.08 + np.random.uniform(0, 0.03)
                aleatoric = 0.25 + np.random.uniform(0, 0.05)
            else:
                epistemic = 0.04 + np.random.uniform(0, 0.02)
                aleatoric = 0.15 + np.random.uniform(0, 0.03)
            
            records.append({
                'date': day['date'],
                'step': step,
                'daily_sentiment': daily_sent,
                'intraday_sentiment': current_sent,
                'epistemic_uncertainty': epistemic,
                'aleatoric_uncertainty': aleatoric,
                'regime': regime,
                'daily_price': daily_price,
            })
    
    return pd.DataFrame(records)


# ============================================================================
# Simulation Runner
# ============================================================================

def run_simulation_on_real_data(
    data_df: pd.DataFrame,
    n_days: int = None,
    steps_per_day: int = 100,
    n_market_makers: int = 3,
    n_informed: int = 5,
    n_noise: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run ABM simulation driven by real sentiment data.
    
    Args:
        data_df: Daily data with sentiment and prices
        n_days: Number of days to simulate (None = all)
        steps_per_day: Simulation steps per calendar day
        n_market_makers: Number of market maker agents
        n_informed: Number of informed trader agents
        n_noise: Number of noise trader agents
        seed: Random seed
        
    Returns:
        DataFrame with simulation results
    """
    np.random.seed(seed)
    
    # Limit days if specified
    if n_days is not None:
        data_df = data_df.head(n_days)
    
    logger.info("=" * 70)
    logger.info("RUNNING ABM WITH REAL DATA")
    logger.info("=" * 70)
    logger.info(f"Days to simulate: {len(data_df)}")
    logger.info(f"Steps per day: {steps_per_day}")
    logger.info(f"Total steps: {len(data_df) * steps_per_day}")
    
    # Create intraday sentiment series
    logger.info("\nExpanding daily sentiment to intraday...")
    intraday_df = create_intraday_sentiment(data_df, steps_per_day)
    
    # Initialize market at first day's price
    initial_price = data_df['close'].iloc[0]
    logger.info(f"Initial price: ${initial_price:,.2f}")
    
    # Create market model
    model = CryptoMarketModel(
        symbol="BTC/USD",
        initial_price=initial_price,
        seed=seed,
    )
    
    # Add agents
    for i in range(n_market_makers):
        mm = MarketMakerAgent(
            f"mm_{i}",
            model,
            base_spread_bps=8.0 + i * 2,
            inventory_aversion=0.001,
            sentiment_sensitivity=0.5,
            uncertainty_sensitivity=1.5,
            quote_size=0.5,
        )
        model.add_agent(mm)
    
    for i in range(n_informed):
        informed = InformedTraderAgent(
            f"informed_{i}",
            model,
            sentiment_threshold=0.15 + i * 0.05,
            uncertainty_threshold=0.12,
            trade_size=0.3,
            position_limit=5.0,
        )
        model.add_agent(informed)
    
    for i in range(n_noise):
        noise = NoiseTraderAgent(
            f"noise_{i}",
            model,
            trade_probability=0.25,
            sentiment_bias=0.15,
            min_size=0.01,
            max_size=0.2,
        )
        model.add_agent(noise)
    
    logger.info(f"Agents: {n_market_makers} MMs, {n_informed} Informed, {n_noise} Noise")
    
    # Run simulation
    logger.info("\nRunning simulation...")
    
    results = []
    current_day_idx = 0
    last_real_price = initial_price
    
    for idx, row in intraday_df.iterrows():
        # Update sentiment
        model.set_sentiment(
            row['intraday_sentiment'],
            row['epistemic_uncertainty'],
            row['aleatoric_uncertainty'],
            row['regime'],
        )
        
        # Track real price for anchoring (changes daily)
        if row['daily_price'] != last_real_price:
            last_real_price = row['daily_price']
            model._anchor_price_to_real(last_real_price, strength=0.15)
        
        # Step simulation
        model.step()
        
        # Record state
        state = model.get_market_state()
        results.append({
            'global_step': idx,
            'date': row['date'],
            'day_step': row['step'],
            'real_price': row['daily_price'],
            'sim_price': state.mid_price,
            'best_bid': state.best_bid,
            'best_ask': state.best_ask,
            'spread_bps': state.spread_bps,
            'imbalance': state.imbalance,
            'daily_sentiment': row['daily_sentiment'],
            'intraday_sentiment': row['intraday_sentiment'],
            'epistemic_uncertainty': row['epistemic_uncertainty'],
            'aleatoric_uncertainty': row['aleatoric_uncertainty'],
            'regime': row['regime'],
            'total_volume': state.total_volume,
            'trade_count': state.trade_count,
            'fills_this_step': len(state.recent_fills),
        })
        
        # Progress logging
        if idx > 0 and idx % (steps_per_day * 10) == 0:
            day_num = idx // steps_per_day
            logger.info(
                f"  Day {day_num}/{len(data_df)}: "
                f"price=${state.mid_price:.0f}, "
                f"spread={state.spread_bps:.1f}bps, "
                f"trades={state.trade_count}"
            )
    
    result_df = pd.DataFrame(results)
    
    logger.info("\n" + "=" * 70)
    logger.info("SIMULATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total steps: {len(result_df)}")
    logger.info(f"Total trades: {result_df['trade_count'].iloc[-1]}")
    logger.info(f"Total volume: {result_df['total_volume'].iloc[-1]:.2f}")
    
    return result_df


# ============================================================================
# Analysis
# ============================================================================

def analyze_results(sim_df: pd.DataFrame, real_df: pd.DataFrame) -> dict:
    """Analyze simulation results and compare to real data."""
    
    results = {}
    
    # Aggregate to daily for comparison
    daily_sim = sim_df.groupby('date').agg({
        'sim_price': ['first', 'last', 'min', 'max', 'mean'],
        'spread_bps': 'mean',
        'trade_count': 'last',
        'total_volume': 'last',
        'intraday_sentiment': 'mean',
    }).reset_index()
    daily_sim.columns = ['date', 'sim_open', 'sim_close', 'sim_low', 'sim_high', 
                         'sim_mean', 'mean_spread_bps', 'trades', 'volume', 'mean_sentiment']
    
    # Compute returns
    daily_sim['sim_returns'] = daily_sim['sim_close'].pct_change()
    
    # Merge with real data
    merged = daily_sim.merge(real_df[['date', 'close', 'returns', 'volatility', 'regime']], on='date')
    merged['real_returns'] = merged['returns']
    
    # 1. Return statistics
    results['sim_return_mean'] = float(daily_sim['sim_returns'].mean())
    results['sim_return_std'] = float(daily_sim['sim_returns'].std())
    results['real_return_mean'] = float(merged['real_returns'].mean())
    results['real_return_std'] = float(merged['real_returns'].std())
    
    # 2. Fat tails (kurtosis)
    results['sim_kurtosis'] = float(daily_sim['sim_returns'].kurtosis())
    results['real_kurtosis'] = float(merged['real_returns'].kurtosis())
    
    # 3. Spread statistics
    results['mean_spread_bps'] = float(sim_df['spread_bps'].mean())
    results['spread_std'] = float(sim_df['spread_bps'].std())
    
    # 4. Sentiment-return correlation
    corr = merged['mean_sentiment'].corr(merged['sim_returns'])
    results['sentiment_return_corr'] = float(corr) if not np.isnan(corr) else 0.0
    
    # 5. Regime-specific spreads
    regime_spreads = sim_df.groupby('regime')['spread_bps'].mean().to_dict()
    results['regime_spreads'] = regime_spreads
    
    # 6. Volatility clustering (ACF of absolute returns)
    abs_returns = daily_sim['sim_returns'].abs().dropna()
    if len(abs_returns) > 10:
        results['vol_clustering_lag1'] = float(abs_returns.autocorr(lag=1))
        results['vol_clustering_lag5'] = float(abs_returns.autocorr(lag=5))
    
    return results


def save_results(sim_df: pd.DataFrame, analysis: dict, output_dir: str):
    """Save simulation results and analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save simulation data
    sim_path = os.path.join(output_dir, 'simulation_results.csv')
    sim_df.to_csv(sim_path, index=False)
    logger.info(f"Saved simulation data to {sim_path}")
    
    # Save analysis
    analysis_path = os.path.join(output_dir, 'analysis_results.json')
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    logger.info(f"Saved analysis to {analysis_path}")
    
    return sim_path, analysis_path


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ABM with real data')
    parser.add_argument('--days', type=int, default=None, help='Number of days (None=all)')
    parser.add_argument('--steps-per-day', type=int, default=100, help='Steps per day')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='results/real_data_run', help='Output directory')
    
    args = parser.parse_args()
    
    # Load real data
    real_df = load_real_data()
    
    # Run simulation
    sim_df = run_simulation_on_real_data(
        real_df,
        n_days=args.days,
        steps_per_day=args.steps_per_day,
        seed=args.seed,
    )
    
    # Analyze
    logger.info("\nAnalyzing results...")
    analysis = analyze_results(sim_df, real_df)
    
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 70)
    for key, value in analysis.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), '..', args.output)
    save_results(sim_df, analysis, output_dir)
    
    return sim_df, analysis


if __name__ == '__main__':
    sim_df, analysis = main()
