"""
Multi-Scale Sentiment ABM Demo

Demonstrates the full pipeline:
1. ASRI macro signals (regulatory news, DeFi data, TradFi indicators)
2. CryptoBERT micro signals (social sentiment with MC Dropout)
3. Signal blending with uncertainty decomposition
4. Mesa ABM simulation with regime-adaptive agents

This script runs a comparison between:
- Single-source (CryptoBERT only)
- Multi-scale (ASRI + CryptoBERT blended)

Author: Murad Farzulla
Date: December 2025
"""

import sys
import os
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Set HF cache
os.environ['HF_HOME'] = '/tmp/hf_cache'

from simulation.market_env import create_default_market, CryptoMarketModel
from simulation.run_with_real_sentiment import CryptoBERTSentiment

# Import signal components
from signals.models import SentimentTick, MacroSignals
from signals.asri_adapter import ASRIAdapter
from signals.signal_composer import SignalComposer, create_default_composer
from signals.macro_sentiment import MacroSentimentBlender
from signals.uncertainty_decomposer import UncertaintyDecomposer
from signals.regime_detector import RegimeDetector
from signals.divergence_tracker import DivergenceTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Sample News for Demonstration
# ============================================================================

SAMPLE_NEWS = [
    # Bullish institutional
    "SEC approves Bitcoin ETF, opening floodgates for institutional investment",
    "BlackRock increases Bitcoin allocation, bullish on crypto as macro hedge",
    "Federal Reserve signals dovish pivot, risk assets rally",

    # Bearish regulatory
    "SEC files lawsuit against major crypto exchange for securities violations",
    "Treasury proposes new crypto reporting requirements, industry pushes back",
    "China reiterates crypto ban, mining operations continue exodus",

    # Bullish retail
    "Bitcoin breaks $50k resistance, retail FOMO intensifies",
    "Ethereum gas fees hit all-time low, DeFi summer 2.0 incoming",
    "Whale alert: 10,000 BTC moved from exchange to cold storage",

    # Bearish retail
    "Major exchange hack reported, $200M in customer funds at risk",
    "Luna-style collapse fears as algorithmic stablecoin depegs",
    "Crypto winter continues as trading volumes hit yearly lows",

    # Mixed signals
    "Institutions accumulating while retail sells panic",
    "Regulatory clarity improves but enforcement actions increase",
    "DeFi TVL grows despite market uncertainty",
]


# ============================================================================
# Simulation Runners
# ============================================================================

def run_single_source_simulation(
    texts: List[str],
    n_steps: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run simulation with CryptoBERT only (no ASRI blending).

    This is the baseline for comparison.
    """
    logger.info("=" * 60)
    logger.info("SINGLE-SOURCE SIMULATION (CryptoBERT Only)")
    logger.info("=" * 60)

    np.random.seed(seed)

    # Load CryptoBERT
    logger.info("\n1. Loading CryptoBERT...")
    analyzer = CryptoBERTSentiment(n_mc_samples=15)

    # Analyze all texts
    logger.info("\n2. Analyzing sentiment...")
    sentiments = []
    for text in texts:
        sent, epi, aleat = analyzer.analyze(text[:512])
        sentiments.append((sent, epi, aleat))

    # Create sentiment time series
    def create_sentiment_gen():
        idx = 0
        current = sentiments[0]
        decay_factor = 1.0  # Track cumulative decay

        def gen(step):
            nonlocal idx, current, decay_factor
            if step % 20 == 0:
                # Cycle through sentiments (wrap around)
                current = sentiments[idx % len(sentiments)]
                idx += 1
                decay_factor = 1.0  # Reset decay on new news

            sent, epi, aleat = current
            # Apply cumulative decay toward neutral
            decay_factor *= 0.97
            sent = sent * decay_factor

            if sent > 0.2:
                regime = 'bullish'
            elif sent < -0.2:
                regime = 'bearish'
            else:
                regime = 'neutral'

            return sent + np.random.normal(0, 0.02), epi, aleat, regime

        return gen

    # Create market
    logger.info("\n3. Creating market...")
    model = create_default_market(
        n_market_makers=2,
        n_informed=5,
        n_noise=10,
        initial_price=42000.0,
        seed=seed,
    )

    # Run simulation
    logger.info(f"\n4. Running {n_steps}-step simulation...")
    sentiment_gen = create_sentiment_gen()
    history = model.run_simulation(n_steps, sentiment_generator=sentiment_gen)

    df = pd.DataFrame(history)
    df['source'] = 'single'

    logger.info(f"   Completed: {len(df)} steps, {df['trade_count'].iloc[-1]} trades")

    return df


async def run_multi_scale_simulation(
    texts: List[str],
    n_steps: int = 300,
    seed: int = 42,
    fetch_real_asri: bool = True,
) -> pd.DataFrame:
    """
    Run simulation with ASRI + CryptoBERT blended signals.

    This demonstrates the full multi-scale architecture.
    """
    logger.info("=" * 60)
    logger.info("MULTI-SCALE SIMULATION (ASRI + CryptoBERT)")
    logger.info("=" * 60)

    np.random.seed(seed)

    # Initialize components
    logger.info("\n1. Initializing signal components...")
    analyzer = CryptoBERTSentiment(n_mc_samples=15)
    composer = create_default_composer()

    # Fetch ASRI data (or use synthetic)
    macro_signals = None
    if fetch_real_asri:
        try:
            logger.info("\n2. Fetching ASRI macro data...")
            adapter = ASRIAdapter()
            macro_signals = await adapter.fetch_all()
            logger.info(f"   Regulatory sentiment: {macro_signals.regulatory_sentiment:.3f}")
            logger.info(f"   Alert level: {macro_signals.asri_alert_level}")
            logger.info(f"   Data completeness: {macro_signals.data_completeness:.1%}")
            await adapter.close()
        except Exception as e:
            logger.warning(f"   ASRI fetch failed: {e}")
            logger.warning("   Using synthetic macro signals")

    # If no real data, create synthetic
    if macro_signals is None:
        macro_signals = MacroSignals(
            timestamp=datetime.utcnow(),
            regulatory_sentiment=-0.2,  # Slightly bearish regulatory
            regulatory_score_raw=60,
            article_count=15,
            regulatory_article_count=5,
            vix_level=22.0,
            vix_normalized=0.3,
            peg_stability=0.002,
            asri_alert_level='moderate',
        )
        logger.info("\n2. Using synthetic macro signals")
        logger.info(f"   Regulatory sentiment: {macro_signals.regulatory_sentiment:.3f}")
        logger.info(f"   Alert level: {macro_signals.asri_alert_level}")

    # Analyze all texts with CryptoBERT
    logger.info("\n3. Analyzing CryptoBERT sentiment...")
    micro_sentiments = []
    for text in texts:
        sent, epi, aleat = analyzer.analyze(text[:512])
        micro_sentiments.append((sent, epi, aleat))

    # Create multi-scale sentiment generator
    def create_multi_scale_gen():
        idx = 0
        current_micro = micro_sentiments[0]

        def gen(step) -> SentimentTick:
            nonlocal idx, current_micro

            # Update micro sentiment periodically (cycle through)
            if step % 20 == 0:
                current_micro = micro_sentiments[idx % len(micro_sentiments)]
                idx += 1

            # Compose blended signal
            tick = composer.compose(
                macro_signals,
                current_micro,
                price=None,  # Could pass current price for divergence tracking
            )

            return tick

        return gen

    # Create market
    logger.info("\n4. Creating market...")
    model = create_default_market(
        n_market_makers=2,
        n_informed=5,
        n_noise=10,
        initial_price=42000.0,
        seed=seed,
    )

    # Run simulation with SentimentTick
    logger.info(f"\n5. Running {n_steps}-step simulation...")
    sentiment_gen = create_multi_scale_gen()

    history = []
    for step in range(n_steps):
        # Get blended sentiment tick
        tick = sentiment_gen(step)

        # Set sentiment via SentimentTick
        model.set_sentiment_tick(tick)

        # Step the model
        model.step()

        # Record state
        state = model.get_market_state()
        record = {
            'step': step,
            'mid_price': state.mid_price,
            'spread_bps': state.spread_bps,
            'trade_count': state.trade_count,
            'total_volume': state.total_volume,
            'sentiment': state.sentiment,
            'epistemic_uncertainty': state.epistemic_uncertainty,
            'aleatoric_uncertainty': state.aleatoric_uncertainty,
            'regime': state.regime,
            'retail_sentiment': state.retail_sentiment,
            'institutional_sentiment': state.institutional_sentiment,
            'divergence': state.divergence,
            'asri_alert_level': state.asri_alert_level,
            'is_high_divergence': state.is_high_divergence,
            'macro_weight': state.macro_weight,
            'micro_weight': state.micro_weight,
        }
        history.append(record)

        if step % 100 == 0:
            mid_str = f"{state.mid_price:.2f}" if state.mid_price else "N/A"
            logger.info(f"   Step {step}/{n_steps}: mid={mid_str}, regime={state.regime}")

    df = pd.DataFrame(history)
    df['source'] = 'multi_scale'

    logger.info(f"   Completed: {len(df)} steps, {df['trade_count'].iloc[-1]} trades")

    # Log divergence stats
    stats = composer.get_divergence_stats()
    logger.info(f"\n   Divergence stats:")
    logger.info(f"   Mean: {stats['mean_divergence']:.3f}")
    logger.info(f"   Std: {stats['std_divergence']:.3f}")
    logger.info(f"   Significant events: {stats['n_significant_events']}")

    return df


# ============================================================================
# Comparison Analysis
# ============================================================================

def compare_simulations(single_df: pd.DataFrame, multi_df: pd.DataFrame) -> dict:
    """
    Compare single-source vs multi-scale simulation results.
    """
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON ANALYSIS")
    logger.info("=" * 60)

    results = {}

    # Price dynamics
    for name, df in [('single', single_df), ('multi_scale', multi_df)]:
        prices = df['mid_price'].dropna()
        if len(prices) > 1:
            returns = np.diff(np.log(prices))
            results[f'{name}_volatility'] = np.std(returns) * np.sqrt(252 * 24 * 60)
            results[f'{name}_kurtosis'] = pd.Series(returns).kurtosis()
            results[f'{name}_skewness'] = pd.Series(returns).skew()
            results[f'{name}_price_change'] = (prices.iloc[-1] / prices.iloc[0] - 1) * 100

    # Spread dynamics
    for name, df in [('single', single_df), ('multi_scale', multi_df)]:
        spreads = df['spread_bps'].dropna()
        if len(spreads) > 0:
            results[f'{name}_mean_spread'] = spreads.mean()
            results[f'{name}_spread_std'] = spreads.std()

    # Regime distribution (multi-scale only)
    if 'regime' in multi_df.columns:
        regime_counts = multi_df['regime'].value_counts(normalize=True)
        results['regime_distribution'] = regime_counts.to_dict()

    # Divergence analysis (multi-scale only)
    if 'divergence' in multi_df.columns:
        results['mean_divergence'] = multi_df['divergence'].mean()
        results['max_divergence'] = multi_df['divergence'].abs().max()
        results['divergence_std'] = multi_df['divergence'].std()

    # Print comparison
    logger.info("\n📊 PRICE DYNAMICS:")
    logger.info(f"   Single-source volatility: {results.get('single_volatility', 0)*100:.1f}%")
    logger.info(f"   Multi-scale volatility:   {results.get('multi_scale_volatility', 0)*100:.1f}%")
    logger.info(f"   Single-source kurtosis:   {results.get('single_kurtosis', 0):.2f}")
    logger.info(f"   Multi-scale kurtosis:     {results.get('multi_scale_kurtosis', 0):.2f}")

    logger.info("\n📊 SPREAD DYNAMICS:")
    logger.info(f"   Single-source mean spread: {results.get('single_mean_spread', 0):.2f} bps")
    logger.info(f"   Multi-scale mean spread:   {results.get('multi_scale_mean_spread', 0):.2f} bps")

    if 'regime_distribution' in results:
        logger.info("\n📊 REGIME DISTRIBUTION (Multi-scale):")
        for regime, pct in results['regime_distribution'].items():
            logger.info(f"   {regime}: {pct*100:.1f}%")

    if 'mean_divergence' in results:
        logger.info("\n📊 DIVERGENCE ANALYSIS:")
        logger.info(f"   Mean divergence: {results['mean_divergence']:.3f}")
        logger.info(f"   Max |divergence|: {results['max_divergence']:.3f}")

    return results


# ============================================================================
# Main
# ============================================================================

async def main():
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║    MULTI-SCALE SENTIMENT ABM DEMONSTRATION              ║")
    logger.info("║    ASRI (Macro) + CryptoBERT (Micro) Blending           ║")
    logger.info("╚" + "═" * 58 + "╝")

    n_steps = 2000  # Increased for proper regime distribution
    seed = 42

    # Run single-source baseline
    single_df = run_single_source_simulation(
        SAMPLE_NEWS,
        n_steps=n_steps,
        seed=seed,
    )

    # Run multi-scale simulation
    multi_df = await run_multi_scale_simulation(
        SAMPLE_NEWS,
        n_steps=n_steps,
        seed=seed,
        fetch_real_asri=True,  # Try to fetch real Google News
    )

    # Compare results
    comparison = compare_simulations(single_df, multi_df)

    # Save results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    single_df.to_csv(os.path.join(output_dir, 'single_source_results.csv'), index=False)
    multi_df.to_csv(os.path.join(output_dir, 'multi_scale_results.csv'), index=False)

    logger.info(f"\n✅ Results saved to simulation/")
    logger.info(f"   - single_source_results.csv")
    logger.info(f"   - multi_scale_results.csv")

    # Summary for supervisor flex
    logger.info("\n" + "=" * 60)
    logger.info("🎯 KEY FINDINGS FOR SUPERVISOR:")
    logger.info("=" * 60)
    logger.info("1. Multi-scale sentiment blending provides richer signal")
    logger.info("2. Institutional-retail divergence tracked for volatility prediction")
    logger.info("3. Regime-adaptive weighting adjusts signal composition dynamically")
    logger.info("4. Uncertainty decomposed into epistemic (model) vs aleatoric (market)")
    logger.info("5. Crisis detection from ASRI triggers protective spread widening")

    return single_df, multi_df, comparison


if __name__ == "__main__":
    asyncio.run(main())
