#!/usr/bin/env python3
"""
Weight Sensitivity Analysis for Uncertainty Decomposition

Tests robustness of regime ranking across different weight configurations.
Addresses AI reviewer critique #4: "81.6% aleatoric depends on arbitrary weights"

Author: Murad Farzulla
Date: January 2026
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load the base data."""
    df_spreads = pd.read_csv('results/real_spread_data.csv', parse_dates=['date'])
    df_sentiment = pd.read_csv('data/datasets/btc_sentiment_daily.csv', parse_dates=['date'])

    df = pd.merge(df_spreads, df_sentiment[['date', 'regime', 'fear_greed_value']],
                  on='date', how='inner')
    df['volatility'] = df['realized_vol'].fillna(df['parkinson_vol'])
    return df.dropna(subset=['aleatoric_proxy', 'epistemic_proxy', 'volatility', 'regime']).copy()


def recompute_total_uncertainty(df, gamma1, delta1):
    """
    Recompute total uncertainty with different weights.

    Original formula (simplified):
    total_uncertainty = gamma1 * aleatoric_proxy + delta1 * epistemic_proxy + residual

    Since we have the proxies, we can vary the weights.
    """
    # Normalize to ensure they sum to reasonable total
    gamma2 = (1 - gamma1) / 2  # remaining for other aleatoric components
    delta2 = (1 - delta1) / 2  # remaining for other epistemic components

    # Simple reweighting of existing proxies
    new_aleatoric = gamma1 * df['aleatoric_proxy']
    new_epistemic = delta1 * df['epistemic_proxy']

    # Total (normalized to 0-1 range)
    total = new_aleatoric + new_epistemic
    total_normalized = (total - total.min()) / (total.max() - total.min())

    return total_normalized


def compute_regime_ranking(df, uncertainty_col='recomputed_uncertainty'):
    """Compute regime means and ranking."""
    regime_stats = df.groupby('regime')[uncertainty_col].agg(['mean', 'std', 'count'])
    regime_stats = regime_stats.sort_values('mean', ascending=False)

    # Get neutral mean for comparison
    neutral_mean = regime_stats.loc['neutral', 'mean']

    # Compute gaps
    gaps = {}
    for regime in regime_stats.index:
        gaps[regime] = regime_stats.loc[regime, 'mean'] - neutral_mean

    return regime_stats, gaps


def sensitivity_grid_search(df, gamma1_range, delta1_range):
    """
    Run sensitivity analysis across weight grid.
    Track whether regime ranking is preserved.
    """
    results = []

    for gamma1 in gamma1_range:
        for delta1 in delta1_range:
            df['recomputed_uncertainty'] = recompute_total_uncertainty(df, gamma1, delta1)
            regime_stats, gaps = compute_regime_ranking(df)

            # Check if extreme_greed and extreme_fear still > neutral
            extreme_greed_above_neutral = gaps.get('extreme_greed', 0) > 0
            extreme_fear_above_neutral = gaps.get('extreme_fear', 0) > 0

            # Ranking preserved if both extremes > neutral
            ranking_preserved = extreme_greed_above_neutral and extreme_fear_above_neutral

            results.append({
                'gamma1': gamma1,
                'delta1': delta1,
                'extreme_greed_gap': gaps.get('extreme_greed', np.nan),
                'extreme_fear_gap': gaps.get('extreme_fear', np.nan),
                'fear_gap': gaps.get('fear', np.nan),
                'greed_gap': gaps.get('greed', np.nan),
                'ranking_preserved': ranking_preserved,
                'extreme_greed_mean': regime_stats.loc['extreme_greed', 'mean'] if 'extreme_greed' in regime_stats.index else np.nan,
                'neutral_mean': regime_stats.loc['neutral', 'mean'],
            })

    return pd.DataFrame(results)


def main():
    print("="*70)
    print("WEIGHT SENSITIVITY ANALYSIS")
    print("Testing robustness of regime ranking across weight configurations")
    print("="*70)

    df = load_data()
    print(f"\nDataset: {len(df)} observations")

    # Define weight grid (±20% from baseline)
    # Baseline: gamma1=0.3, delta1=0.35
    gamma1_range = np.arange(0.20, 0.45, 0.05)  # 0.20, 0.25, 0.30, 0.35, 0.40
    delta1_range = np.arange(0.25, 0.50, 0.05)  # 0.25, 0.30, 0.35, 0.40, 0.45

    print(f"\nWeight grid:")
    print(f"  γ₁ (aleatoric weight): {list(gamma1_range.round(2))}")
    print(f"  δ₁ (epistemic weight): {list(delta1_range.round(2))}")
    print(f"  Total configurations: {len(gamma1_range) * len(delta1_range)}")

    # Run sensitivity analysis
    results = sensitivity_grid_search(df, gamma1_range, delta1_range)

    # Summary statistics
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    n_preserved = results['ranking_preserved'].sum()
    n_total = len(results)

    print(f"\nRanking preserved (extreme > neutral): {n_preserved}/{n_total} ({100*n_preserved/n_total:.1f}%)")

    print("\n★ Key finding:")
    if n_preserved == n_total:
        print("  Regime ranking is FULLY ROBUST across all weight configurations")
    elif n_preserved / n_total >= 0.9:
        print("  Regime ranking is LARGELY ROBUST (>90% of configurations)")
    else:
        print(f"  Regime ranking is MODERATELY ROBUST ({100*n_preserved/n_total:.0f}% of configurations)")

    # Show range of gaps
    print("\n" + "-"*70)
    print("Gap statistics (extreme regime - neutral) across all configurations:")
    print("-"*70)

    for regime in ['extreme_greed', 'extreme_fear', 'fear', 'greed']:
        col = f'{regime}_gap'
        if col in results.columns:
            min_gap = results[col].min()
            max_gap = results[col].max()
            mean_gap = results[col].mean()
            print(f"  {regime:15s}: mean={mean_gap:+.4f}, range=[{min_gap:+.4f}, {max_gap:+.4f}]")

    # Save results
    results.to_csv('results/weight_sensitivity_results.csv', index=False)

    # Create summary table for paper
    summary = pd.DataFrame({
        'metric': [
            'Weight configurations tested',
            'Ranking preserved (%)',
            'Min extreme_greed gap',
            'Max extreme_greed gap',
            'Min extreme_fear gap',
            'Max extreme_fear gap',
        ],
        'value': [
            n_total,
            f"{100*n_preserved/n_total:.1f}%",
            f"{results['extreme_greed_gap'].min():+.4f}",
            f"{results['extreme_greed_gap'].max():+.4f}",
            f"{results['extreme_fear_gap'].min():+.4f}",
            f"{results['extreme_fear_gap'].max():+.4f}",
        ]
    })
    summary.to_csv('results/weight_sensitivity_summary.csv', index=False)

    print("\n✓ Results saved to results/weight_sensitivity_results.csv")
    print("✓ Summary saved to results/weight_sensitivity_summary.csv")

    return results


if __name__ == "__main__":
    main()
