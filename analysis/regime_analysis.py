"""
Regime analysis functions for Sentiment-Microstructure ABM.

Provides regime-conditional statistics, transition analysis,
and regime duration analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any


def compute_regime_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics conditional on sentiment regime.

    Args:
        df: DataFrame with 'regime' column and market data

    Returns:
        DataFrame with index ['bullish', 'neutral', 'bearish'] and statistics
    """
    regimes = ['bullish', 'neutral', 'bearish']
    results = []

    for regime in regimes:
        mask = df['regime'] == regime
        subset = df[mask]

        if len(subset) == 0:
            continue

        # Skip first return (always 0)
        returns = subset['log_return'].iloc[1:] if len(subset) > 1 else subset['log_return']

        stats = {
            'regime': regime,
            'n_obs': len(subset),
            'pct_time': len(subset) / len(df) * 100,
            # Spread statistics
            'mean_spread_bps': subset['spread_bps'].mean(),
            'std_spread_bps': subset['spread_bps'].std(),
            'median_spread_bps': subset['spread_bps'].median(),
            # Return statistics
            'mean_return': returns.mean() if len(returns) > 0 else np.nan,
            'volatility': returns.std() if len(returns) > 0 else np.nan,
            # Sentiment statistics
            'mean_sentiment': subset['sentiment'].mean(),
            'sentiment_std': subset['sentiment'].std(),
            # Uncertainty statistics
            'mean_epistemic': subset['epistemic_uncertainty'].mean(),
            'mean_aleatoric': subset['aleatoric_uncertainty'].mean(),
            'mean_total_uncertainty': subset['total_uncertainty'].mean(),
            # Inventory statistics
            'mean_inventory': subset['inventory'].mean(),
            'inventory_std': subset['inventory'].std(),
            'max_abs_inventory': subset['inventory'].abs().max(),
        }
        results.append(stats)

    result_df = pd.DataFrame(results).set_index('regime')

    # Reorder to standard order
    order = [r for r in regimes if r in result_df.index]
    return result_df.loc[order]


def compute_regime_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute regime transition probability matrix.

    Args:
        df: DataFrame with 'regime' column

    Returns:
        3x3 DataFrame with P(regime_t+1 | regime_t)
    """
    regimes = ['bullish', 'neutral', 'bearish']

    # Count transitions
    transitions = pd.crosstab(
        df['regime'].iloc[:-1].values,
        df['regime'].iloc[1:].values,
        normalize='index'
    )

    # Ensure all regimes are present
    for regime in regimes:
        if regime not in transitions.index:
            transitions.loc[regime] = 0.0
        if regime not in transitions.columns:
            transitions[regime] = 0.0

    # Reorder
    transitions = transitions.loc[regimes, regimes]

    return transitions


def compute_regime_durations(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute regime duration statistics.

    Args:
        df: DataFrame with 'regime' column

    Returns:
        Dictionary with duration stats for each regime
    """
    regimes = ['bullish', 'neutral', 'bearish']
    results = {}

    # Find regime changes
    regime_changes = df['regime'] != df['regime'].shift(1)
    regime_changes.iloc[0] = True  # First observation starts a new regime

    # Get regime episode boundaries
    episode_starts = df.index[regime_changes].tolist()
    episode_starts.append(len(df))  # Add end

    # Compute durations for each regime
    for regime in regimes:
        durations = []
        current_regime = None
        current_start = None

        for i in range(len(episode_starts) - 1):
            start_idx = episode_starts[i]
            end_idx = episode_starts[i + 1]

            episode_regime = df['regime'].iloc[start_idx]
            if episode_regime == regime:
                duration = end_idx - start_idx
                durations.append(duration)

        if durations:
            results[regime] = {
                'n_episodes': len(durations),
                'mean_duration': np.mean(durations),
                'std_duration': np.std(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'median_duration': np.median(durations),
                'total_time': sum(durations),
            }
        else:
            results[regime] = {
                'n_episodes': 0,
                'mean_duration': np.nan,
                'std_duration': np.nan,
                'min_duration': np.nan,
                'max_duration': np.nan,
                'median_duration': np.nan,
                'total_time': 0,
            }

    return results


def compute_regime_volatility_by_period(
    df: pd.DataFrame,
    window: int = 50
) -> pd.DataFrame:
    """
    Compute rolling volatility by regime.

    Args:
        df: DataFrame with 'regime' and 'log_return' columns
        window: Rolling window size

    Returns:
        DataFrame with rolling volatility and regime
    """
    df = df.copy()
    df['rolling_volatility'] = df['log_return'].rolling(window=window).std()
    return df[['regime', 'rolling_volatility']].dropna()


def analyze_regime_switching_dynamics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive regime switching analysis.

    Args:
        df: DataFrame with market simulation data

    Returns:
        Dictionary with regime dynamics results
    """
    results = {}

    # Basic regime statistics
    results['regime_stats'] = compute_regime_statistics(df)

    # Transition matrix
    results['transitions'] = compute_regime_transitions(df)

    # Duration analysis
    results['durations'] = compute_regime_durations(df)

    # Spread differential between regimes
    regime_stats = results['regime_stats']
    if 'bullish' in regime_stats.index and 'bearish' in regime_stats.index:
        results['spread_differential'] = (
            regime_stats.loc['bullish', 'mean_spread_bps'] -
            regime_stats.loc['bearish', 'mean_spread_bps']
        )

    # Regime persistence (diagonal of transition matrix)
    trans = results['transitions']
    results['regime_persistence'] = {
        regime: trans.loc[regime, regime] if regime in trans.index else np.nan
        for regime in ['bullish', 'neutral', 'bearish']
    }

    return results
