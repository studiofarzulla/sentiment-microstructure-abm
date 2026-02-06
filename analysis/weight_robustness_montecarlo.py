#!/usr/bin/env python3
"""
Monte Carlo Weight Robustness Analysis

Addresses reviewer critique: "Weights are heuristic, not estimated"

Tests whether the extremity premium (extreme regimes > neutral) holds under
random weight configurations drawn from a Dirichlet distribution.

This is NOT a substitute for GMM/MLE estimation, but demonstrates that:
1. The qualitative finding is robust to weight specification
2. The extremity premium is not an artifact of specific weight choices
3. Provides a path to formal estimation if reviewers demand it

Author: Murad Farzulla
Date: January 2026
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load the base data (same as weight_sensitivity.py)."""
    df_spreads = pd.read_csv('../results/real_spread_data.csv', parse_dates=['date'])
    df_sentiment = pd.read_csv('../data/datasets/btc_sentiment_daily.csv', parse_dates=['date'])

    df = pd.merge(df_spreads, df_sentiment[['date', 'regime', 'fear_greed_value']],
                  on='date', how='inner')
    df['volatility'] = df['realized_vol'].fillna(df['parkinson_vol'])
    return df.dropna(subset=['aleatoric_proxy', 'epistemic_proxy', 'volatility', 'regime']).copy()


def recompute_uncertainty_with_random_weights(df, weights):
    """
    Recompute total uncertainty with random weight configuration.

    Args:
        df: DataFrame with aleatoric_proxy and epistemic_proxy columns
        weights: Array of 4 weights [w_aleatoric, w_epistemic, w_volatility, w_residual]
                 drawn from Dirichlet, summing to 1

    Returns:
        Normalized uncertainty series
    """
    # Apply weights to components
    # weights[0]: aleatoric proxy weight
    # weights[1]: epistemic proxy weight
    # weights[2]: volatility weight (if available)
    # weights[3]: residual/constant weight

    total = (weights[0] * df['aleatoric_proxy'] +
             weights[1] * df['epistemic_proxy'] +
             weights[2] * df['volatility'] +
             weights[3])  # constant term

    # Normalize to [0, 1]
    total_normalized = (total - total.min()) / (total.max() - total.min())
    return total_normalized


def check_extremity_premium(df, uncertainty_col='mc_uncertainty'):
    """
    Check if extremity premium holds: extreme_greed > neutral AND extreme_fear > neutral

    Returns:
        dict with regime means, gaps, and whether premium preserved
    """
    regime_means = df.groupby('regime')[uncertainty_col].mean()

    neutral_mean = regime_means.get('neutral', np.nan)
    extreme_greed_mean = regime_means.get('extreme_greed', np.nan)
    extreme_fear_mean = regime_means.get('extreme_fear', np.nan)

    greed_gap = extreme_greed_mean - neutral_mean if not np.isnan(extreme_greed_mean) else np.nan
    fear_gap = extreme_fear_mean - neutral_mean if not np.isnan(extreme_fear_mean) else np.nan

    # Premium preserved if BOTH extremes > neutral
    greed_above = greed_gap > 0 if not np.isnan(greed_gap) else False
    fear_above = fear_gap > 0 if not np.isnan(fear_gap) else False
    premium_preserved = greed_above and fear_above

    return {
        'neutral_mean': neutral_mean,
        'extreme_greed_mean': extreme_greed_mean,
        'extreme_fear_mean': extreme_fear_mean,
        'greed_gap': greed_gap,
        'fear_gap': fear_gap,
        'premium_preserved': premium_preserved,
        'greed_above_neutral': greed_above,
        'fear_above_neutral': fear_above
    }


def monte_carlo_weight_robustness(df, n_simulations=1000, alpha_dirichlet=None, seed=42):
    """
    Draw random weights from Dirichlet distribution and test extremity premium.

    Args:
        df: DataFrame with proxies
        n_simulations: Number of Monte Carlo draws
        alpha_dirichlet: Concentration parameters for Dirichlet.
                        Default [1,1,1,1] = uniform over simplex
        seed: Random seed for reproducibility

    Returns:
        dict with:
            - fraction_preserved: % of simulations where premium holds
            - detailed_results: DataFrame of all simulation results
            - confidence_interval: 95% CI for fraction
            - weight_statistics: Summary of weight distributions
    """
    np.random.seed(seed)

    if alpha_dirichlet is None:
        alpha_dirichlet = [1, 1, 1, 1]  # Uniform Dirichlet

    results = []
    all_weights = []

    for sim in range(n_simulations):
        # Draw random weights from Dirichlet (sum to 1)
        weights = np.random.dirichlet(alpha_dirichlet)
        all_weights.append(weights)

        # Recompute uncertainty with these weights
        df['mc_uncertainty'] = recompute_uncertainty_with_random_weights(df, weights)

        # Check if premium preserved
        check = check_extremity_premium(df, 'mc_uncertainty')

        results.append({
            'simulation': sim,
            'w_aleatoric': weights[0],
            'w_epistemic': weights[1],
            'w_volatility': weights[2],
            'w_residual': weights[3],
            **check
        })

    results_df = pd.DataFrame(results)
    all_weights = np.array(all_weights)

    # Compute statistics
    n_preserved = results_df['premium_preserved'].sum()
    fraction_preserved = n_preserved / n_simulations

    # 95% CI for binomial proportion (Wilson score interval)
    from scipy.stats import norm
    z = norm.ppf(0.975)
    p_hat = fraction_preserved
    n = n_simulations

    denominator = 1 + z**2/n
    center = (p_hat + z**2/(2*n)) / denominator
    spread = z * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2)) / denominator
    ci_lower = max(0, center - spread)
    ci_upper = min(1, center + spread)

    # Weight statistics
    weight_stats = {
        'aleatoric': {'mean': all_weights[:, 0].mean(), 'std': all_weights[:, 0].std()},
        'epistemic': {'mean': all_weights[:, 1].mean(), 'std': all_weights[:, 1].std()},
        'volatility': {'mean': all_weights[:, 2].mean(), 'std': all_weights[:, 2].std()},
        'residual': {'mean': all_weights[:, 3].mean(), 'std': all_weights[:, 3].std()}
    }

    return {
        'fraction_preserved': fraction_preserved,
        'n_preserved': n_preserved,
        'n_simulations': n_simulations,
        'confidence_interval': (ci_lower, ci_upper),
        'detailed_results': results_df,
        'weight_statistics': weight_stats,
        'alpha_dirichlet': alpha_dirichlet
    }


def analyze_failure_cases(results_df):
    """
    Analyze weight configurations where extremity premium fails.

    Helps identify which weight regimes break the finding.
    """
    failures = results_df[~results_df['premium_preserved']]

    if len(failures) == 0:
        return None

    analysis = {
        'n_failures': len(failures),
        'greed_failures': (~failures['greed_above_neutral']).sum(),
        'fear_failures': (~failures['fear_above_neutral']).sum(),
        'both_failures': ((~failures['greed_above_neutral']) & (~failures['fear_above_neutral'])).sum(),
        'mean_weights_failure': {
            'aleatoric': failures['w_aleatoric'].mean(),
            'epistemic': failures['w_epistemic'].mean(),
            'volatility': failures['w_volatility'].mean(),
            'residual': failures['w_residual'].mean()
        }
    }

    return analysis


def main():
    print("="*70)
    print("MONTE CARLO WEIGHT ROBUSTNESS ANALYSIS")
    print("Testing extremity premium under random weight configurations")
    print("="*70)

    df = load_data()
    print(f"\nDataset: {len(df)} observations")
    print(f"Regimes: {df['regime'].value_counts().to_dict()}")

    # Run Monte Carlo with uniform Dirichlet
    print("\n" + "-"*70)
    print("Running Monte Carlo simulations with Dirichlet(1,1,1,1)...")
    print("-"*70)

    results = monte_carlo_weight_robustness(df, n_simulations=1000, seed=42)

    # Report results
    print(f"\n{'='*70}")
    print("RESULTS")
    print("="*70)

    pct = results['fraction_preserved'] * 100
    ci_lo, ci_hi = results['confidence_interval']

    print(f"\nExtremity premium preserved: {results['n_preserved']}/{results['n_simulations']} "
          f"({pct:.1f}%)")
    print(f"95% Confidence Interval: [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]")

    print("\nWeight distribution statistics (Dirichlet draws):")
    for comp, stats in results['weight_statistics'].items():
        print(f"  {comp:12s}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")

    # Analyze failures if any
    failure_analysis = analyze_failure_cases(results['detailed_results'])
    if failure_analysis:
        print(f"\n{'-'*70}")
        print("FAILURE ANALYSIS")
        print("-"*70)
        print(f"Total failures: {failure_analysis['n_failures']}")
        print(f"  - Greed below neutral: {failure_analysis['greed_failures']}")
        print(f"  - Fear below neutral: {failure_analysis['fear_failures']}")
        print(f"  - Both below neutral: {failure_analysis['both_failures']}")
        print("\nMean weights in failed configurations:")
        for comp, val in failure_analysis['mean_weights_failure'].items():
            print(f"  {comp:12s}: {val:.3f}")
    else:
        print("\n★ No failures detected - extremity premium is fully robust!")

    # Additional tests with different Dirichlet concentrations
    print(f"\n{'='*70}")
    print("SENSITIVITY TO DIRICHLET CONCENTRATION")
    print("="*70)

    concentration_tests = [
        ([0.5, 0.5, 0.5, 0.5], "Sparse (concentrated at corners)"),
        ([2, 2, 2, 2], "Dense (concentrated at center)"),
        ([5, 1, 1, 1], "Aleatoric-heavy"),
        ([1, 5, 1, 1], "Epistemic-heavy"),
    ]

    for alpha, description in concentration_tests:
        res = monte_carlo_weight_robustness(df, n_simulations=500, alpha_dirichlet=alpha, seed=42)
        print(f"  Dirichlet{alpha} ({description}): "
              f"{res['fraction_preserved']*100:.1f}% preserved")

    # Save results
    results['detailed_results'].to_csv('../results/mc_weight_robustness_results.csv', index=False)

    # Create summary for paper
    summary = pd.DataFrame({
        'metric': [
            'Monte Carlo simulations',
            'Dirichlet concentration',
            'Fraction preserved',
            '95% CI lower',
            '95% CI upper',
            'Failures (greed < neutral)',
            'Failures (fear < neutral)',
        ],
        'value': [
            results['n_simulations'],
            'Dirichlet(1,1,1,1)',
            f"{results['fraction_preserved']*100:.1f}%",
            f"{ci_lo*100:.1f}%",
            f"{ci_hi*100:.1f}%",
            failure_analysis['greed_failures'] if failure_analysis else 0,
            failure_analysis['fear_failures'] if failure_analysis else 0,
        ]
    })
    summary.to_csv('../results/mc_weight_robustness_summary.csv', index=False)

    print("\n" + "="*70)
    print("OUTPUT FILES")
    print("="*70)
    print("  - results/mc_weight_robustness_results.csv (detailed)")
    print("  - results/mc_weight_robustness_summary.csv (paper table)")

    # Key finding for paper
    print("\n" + "="*70)
    print("KEY FINDING FOR PAPER")
    print("="*70)
    print(f"""
The extremity premium holds in {pct:.1f}% of {results['n_simulations']} Monte Carlo
weight configurations drawn from Dirichlet(1,1,1,1), confirming qualitative
robustness to weight specification (95% CI: [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]).
""")

    return results


if __name__ == "__main__":
    main()
