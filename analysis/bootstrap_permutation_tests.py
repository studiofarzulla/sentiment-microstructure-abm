"""
Bootstrap and Permutation Tests for the Extremity Premium

Provides statistical inference on:
1. Bootstrap CIs on extreme vs neutral uncertainty gap
2. Permutation test: is the pattern non-random?

Author: Murad Farzulla
Date: January 2026
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


def load_data():
    """Load and prepare data."""
    df_spreads = pd.read_csv('results/real_spread_data.csv', parse_dates=['date'])
    df_sentiment = pd.read_csv('data/datasets/btc_sentiment_daily.csv', parse_dates=['date'])

    df = pd.merge(df_spreads, df_sentiment[['date', 'regime', 'fear_greed_value']],
                  on='date', how='inner')
    df['volatility'] = df['realized_vol'].fillna(df['parkinson_vol'])
    df_clean = df.dropna(subset=['total_uncertainty', 'volatility', 'regime']).copy()

    # Add regime classifications
    df_clean['is_extreme'] = df_clean['regime'].isin(['extreme_greed', 'extreme_fear']).astype(int)
    df_clean['is_neutral'] = (df_clean['regime'] == 'neutral').astype(int)

    return df_clean


def compute_gap(df):
    """Compute extreme - neutral uncertainty gap, controlling for volatility."""
    # Residualize uncertainty by volatility
    X = sm.add_constant(df['volatility'])
    y = df['total_uncertainty']
    model = sm.OLS(y, X).fit()
    df = df.copy()
    df['resid'] = model.resid

    extreme = df[df['is_extreme'] == 1]['resid'].mean()
    neutral = df[df['is_neutral'] == 1]['resid'].mean()

    return extreme - neutral


def bootstrap_confidence_interval(df, n_bootstrap=10000, confidence=0.95):
    """
    Bootstrap 95% CI on extreme-neutral gap.
    """
    print("="*70)
    print(f"BOOTSTRAP CONFIDENCE INTERVALS ({n_bootstrap} resamples)")
    print("="*70)

    # Observed gap
    observed_gap = compute_gap(df)
    print(f"\nObserved gap (extreme - neutral): {observed_gap:.4f}")

    # Bootstrap
    bootstrap_gaps = []
    n = len(df)

    for i in range(n_bootstrap):
        # Resample with replacement
        sample = df.sample(n=n, replace=True)
        gap = compute_gap(sample)
        bootstrap_gaps.append(gap)

    bootstrap_gaps = np.array(bootstrap_gaps)

    # Percentile CI
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_gaps, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_gaps, 100 * (1 - alpha / 2))

    # Bias-corrected and accelerated (BCa) would be better but percentile is fine for now
    print(f"\n{confidence*100:.0f}% Confidence Interval:")
    print(f"  Lower: {ci_lower:.4f}")
    print(f"  Upper: {ci_upper:.4f}")

    # Does CI exclude zero?
    excludes_zero = ci_lower > 0 or ci_upper < 0
    if excludes_zero and ci_lower > 0:
        print(f"\n  ✓ CI EXCLUDES ZERO: Extreme regimes have significantly higher")
        print(f"    uncertainty than neutral regimes (p < {alpha:.2f}).")
    elif excludes_zero and ci_upper < 0:
        print(f"\n  ✓ CI EXCLUDES ZERO: Neutral has significantly higher uncertainty")
        print(f"    (this would reverse the finding).")
    else:
        print(f"\n  ✗ CI INCLUDES ZERO: Not significant at {confidence*100:.0f}% level.")

    # Bootstrap SE and z-score
    bootstrap_se = np.std(bootstrap_gaps)
    z_score = observed_gap / bootstrap_se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-sided

    print(f"\n  Bootstrap SE: {bootstrap_se:.4f}")
    print(f"  Z-score: {z_score:.2f}")
    print(f"  P-value (two-sided): {p_value:.4f}")

    results = {
        'observed_gap': observed_gap,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bootstrap_se': bootstrap_se,
        'z_score': z_score,
        'p_value': p_value,
        'excludes_zero': excludes_zero,
        'n_bootstrap': n_bootstrap
    }

    return results, bootstrap_gaps


def permutation_test(df, n_permutations=10000):
    """
    Permutation test: Under the null, regime labels are unrelated to uncertainty.
    Shuffle regime labels and compute gap each time.
    """
    print("\n" + "="*70)
    print(f"PERMUTATION TEST ({n_permutations} permutations)")
    print("="*70)

    # Observed gap
    observed_gap = compute_gap(df)
    print(f"\nObserved gap: {observed_gap:.4f}")

    # Permutation distribution
    permuted_gaps = []
    n = len(df)

    for i in range(n_permutations):
        # Shuffle regime labels
        df_perm = df.copy()
        df_perm['is_extreme'] = np.random.permutation(df['is_extreme'].values)
        df_perm['is_neutral'] = np.random.permutation(df['is_neutral'].values)

        gap = compute_gap(df_perm)
        permuted_gaps.append(gap)

    permuted_gaps = np.array(permuted_gaps)

    # P-value: fraction of permutations with gap >= observed (one-sided)
    # For two-sided: use absolute values
    p_value_one_sided = (permuted_gaps >= observed_gap).mean()
    p_value_two_sided = (np.abs(permuted_gaps) >= np.abs(observed_gap)).mean()

    print(f"\nPermutation distribution:")
    print(f"  Mean: {permuted_gaps.mean():.4f}")
    print(f"  Std:  {permuted_gaps.std():.4f}")
    print(f"  Min:  {permuted_gaps.min():.4f}")
    print(f"  Max:  {permuted_gaps.max():.4f}")

    print(f"\nP-values:")
    print(f"  One-sided (extreme > neutral): {p_value_one_sided:.4f}")
    print(f"  Two-sided: {p_value_two_sided:.4f}")

    if p_value_one_sided < 0.05:
        print(f"\n  ✓ PERMUTATION TEST PASSED: The extreme > neutral pattern")
        print(f"    is unlikely to occur by chance (p = {p_value_one_sided:.4f}).")
    else:
        print(f"\n  ✗ Permutation test not significant at 5% level.")

    results = {
        'observed_gap': observed_gap,
        'permuted_mean': permuted_gaps.mean(),
        'permuted_std': permuted_gaps.std(),
        'p_value_one_sided': p_value_one_sided,
        'p_value_two_sided': p_value_two_sided,
        'n_permutations': n_permutations
    }

    return results, permuted_gaps


def bootstrap_by_regime(df, n_bootstrap=10000):
    """
    Bootstrap CIs for each regime's excess uncertainty relative to neutral.
    """
    print("\n" + "="*70)
    print("BOOTSTRAP CIs BY REGIME")
    print("="*70)

    # Residualize
    X = sm.add_constant(df['volatility'])
    y = df['total_uncertainty']
    model = sm.OLS(y, X).fit()
    df = df.copy()
    df['resid'] = model.resid

    regimes = ['extreme_greed', 'extreme_fear', 'fear', 'greed']
    neutral_resid = df[df['regime'] == 'neutral']['resid']

    results = []

    for regime in regimes:
        regime_resid = df[df['regime'] == regime]['resid']
        observed_gap = regime_resid.mean() - neutral_resid.mean()

        # Bootstrap
        bootstrap_gaps = []
        for i in range(n_bootstrap):
            r_sample = np.random.choice(regime_resid, size=len(regime_resid), replace=True)
            n_sample = np.random.choice(neutral_resid, size=len(neutral_resid), replace=True)
            bootstrap_gaps.append(r_sample.mean() - n_sample.mean())

        bootstrap_gaps = np.array(bootstrap_gaps)
        ci_lower = np.percentile(bootstrap_gaps, 2.5)
        ci_upper = np.percentile(bootstrap_gaps, 97.5)
        excludes_zero = ci_lower > 0 or ci_upper < 0

        sig = '✓' if excludes_zero and ci_lower > 0 else ''
        print(f"{regime:15s}: gap={observed_gap:+.4f}, 95% CI=[{ci_lower:+.4f}, {ci_upper:+.4f}] {sig}")

        results.append({
            'regime': regime,
            'gap_vs_neutral': observed_gap,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'excludes_zero': excludes_zero
        })

    return pd.DataFrame(results)


def main():
    print("="*70)
    print("STATISTICAL INFERENCE FOR THE EXTREMITY PREMIUM")
    print("="*70)

    # Load data
    df = load_data()
    print(f"\nData: {len(df)} observations")

    # Bootstrap CI on extreme-neutral gap
    bootstrap_results, bootstrap_gaps = bootstrap_confidence_interval(df, n_bootstrap=10000)

    # Permutation test
    permutation_results, permuted_gaps = permutation_test(df, n_permutations=10000)

    # Bootstrap by regime
    regime_cis = bootstrap_by_regime(df, n_bootstrap=10000)

    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    pd.DataFrame([bootstrap_results]).to_csv('results/bootstrap_extreme_neutral_gap.csv', index=False)
    print("  Saved: results/bootstrap_extreme_neutral_gap.csv")

    pd.DataFrame([permutation_results]).to_csv('results/permutation_test_results.csv', index=False)
    print("  Saved: results/permutation_test_results.csv")

    regime_cis.to_csv('results/bootstrap_regime_cis.csv', index=False)
    print("  Saved: results/bootstrap_regime_cis.csv")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
  Bootstrap 95% CI on (extreme - neutral) gap:
    [{bootstrap_results['ci_lower']:.4f}, {bootstrap_results['ci_upper']:.4f}]
    {'✓ Excludes zero' if bootstrap_results['excludes_zero'] else '✗ Includes zero'}

  Permutation test p-value (one-sided):
    p = {permutation_results['p_value_one_sided']:.4f}
    {'✓ Significant' if permutation_results['p_value_one_sided'] < 0.05 else '✗ Not significant'}

  By-regime CIs (vs neutral):
""")
    for _, row in regime_cis.iterrows():
        sig = '✓' if row['excludes_zero'] and row['ci_lower'] > 0 else ''
        print(f"    {row['regime']:15s}: [{row['ci_lower']:+.4f}, {row['ci_upper']:+.4f}] {sig}")

    return bootstrap_results, permutation_results, regime_cis


if __name__ == '__main__':
    bootstrap_results, permutation_results, regime_cis = main()
