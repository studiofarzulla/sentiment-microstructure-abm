"""
Volatility-Matched Regime Comparison: THE KILLER TEST

Purpose: Kill the "it's just volatility" objection by showing that
neutral regimes have higher uncertainty than directional regimes
EVEN WHEN CONTROLLING FOR VOLATILITY LEVEL.

Method:
1. Bin all days by realized volatility (quintiles)
2. Within each volatility bin, compare mean uncertainty across regimes
3. If neutral > extreme within-bin, the finding is NOT mechanical

Author: Murad Farzulla
Date: January 2026
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def load_and_merge_data():
    """Load spread data and sentiment data, merge on date."""
    # Load datasets
    df_spreads = pd.read_csv('results/real_spread_data.csv', parse_dates=['date'])
    df_sentiment = pd.read_csv('data/datasets/btc_sentiment_daily.csv', parse_dates=['date'])

    # Merge on date
    df = pd.merge(df_spreads, df_sentiment[['date', 'regime', 'fear_greed_value', 'volatility']],
                  on='date', how='inner')

    # If volatility not in sentiment data, use realized_vol from spreads
    if 'volatility' not in df.columns or df['volatility'].isna().all():
        df['volatility'] = df['realized_vol']

    # Fill any missing volatility with parkinson_vol
    df['volatility'] = df['volatility'].fillna(df['parkinson_vol'])

    # Drop rows with missing key variables
    df_clean = df.dropna(subset=['total_uncertainty', 'volatility', 'regime'])

    print(f"Merged dataset: {len(df_clean)} rows with complete data")
    print(f"Regime distribution: {df_clean.regime.value_counts().to_dict()}")

    return df_clean


def bin_by_volatility(df, n_bins=5):
    """Bin days by volatility quintiles."""
    df = df.copy()
    df['vol_quintile'] = pd.qcut(df['volatility'], q=n_bins, labels=False, duplicates='drop')

    print(f"\nVolatility quintile distribution:")
    for q in sorted(df['vol_quintile'].unique()):
        subset = df[df['vol_quintile'] == q]
        print(f"  Q{q}: n={len(subset)}, vol range=[{subset['volatility'].min():.4f}, {subset['volatility'].max():.4f}]")

    return df


def compute_regime_uncertainty_by_quintile(df):
    """Compute mean uncertainty for each regime within each volatility quintile."""
    results = []

    for q in sorted(df['vol_quintile'].unique()):
        q_df = df[df['vol_quintile'] == q]

        for regime in df['regime'].unique():
            regime_df = q_df[q_df['regime'] == regime]
            if len(regime_df) >= 3:  # Need at least 3 observations
                results.append({
                    'vol_quintile': q,
                    'regime': regime,
                    'n': len(regime_df),
                    'mean_uncertainty': regime_df['total_uncertainty'].mean(),
                    'std_uncertainty': regime_df['total_uncertainty'].std(),
                    'mean_volatility': regime_df['volatility'].mean(),
                })

    return pd.DataFrame(results)


def simplify_regimes(df):
    """Collapse to 3 regimes: neutral, bullish (greed + extreme_greed), bearish (fear + extreme_fear)."""
    df = df.copy()
    regime_map = {
        'neutral': 'neutral',
        'greed': 'bullish',
        'extreme_greed': 'bullish',
        'fear': 'bearish',
        'extreme_fear': 'bearish'
    }
    df['regime_simple'] = df['regime'].map(regime_map)
    return df


def run_within_quintile_tests(df):
    """Test if neutral > directional within each volatility quintile."""
    print("\n" + "="*70)
    print("WITHIN-QUINTILE REGIME COMPARISON (THE KILLER TEST)")
    print("="*70)

    df_simple = simplify_regimes(df)

    results = []

    for q in sorted(df_simple['vol_quintile'].unique()):
        q_df = df_simple[df_simple['vol_quintile'] == q]

        neutral = q_df[q_df['regime_simple'] == 'neutral']['total_uncertainty']
        bullish = q_df[q_df['regime_simple'] == 'bullish']['total_uncertainty']
        bearish = q_df[q_df['regime_simple'] == 'bearish']['total_uncertainty']
        directional = q_df[q_df['regime_simple'] != 'neutral']['total_uncertainty']

        print(f"\n--- Volatility Quintile {q} ---")
        print(f"  Mean volatility: {q_df['volatility'].mean():.4f}")
        print(f"  Neutral: n={len(neutral)}, mean={neutral.mean():.4f}" if len(neutral) > 0 else "  Neutral: n=0")
        print(f"  Bullish: n={len(bullish)}, mean={bullish.mean():.4f}" if len(bullish) > 0 else "  Bullish: n=0")
        print(f"  Bearish: n={len(bearish)}, mean={bearish.mean():.4f}" if len(bearish) > 0 else "  Bearish: n=0")

        # Test neutral vs directional
        if len(neutral) >= 3 and len(directional) >= 3:
            t_stat, p_value = stats.ttest_ind(neutral, directional, alternative='greater')
            effect_size = (neutral.mean() - directional.mean()) / np.sqrt(
                ((len(neutral)-1)*neutral.std()**2 + (len(directional)-1)*directional.std()**2) /
                (len(neutral) + len(directional) - 2)
            ) if neutral.std() > 0 and directional.std() > 0 else 0

            sig = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
            print(f"  Neutral vs Directional: t={t_stat:.3f}, p={p_value:.4f} {sig}, Cohen's d={effect_size:.3f}")

            results.append({
                'vol_quintile': q,
                'mean_vol': q_df['volatility'].mean(),
                'neutral_mean': neutral.mean(),
                'neutral_n': len(neutral),
                'directional_mean': directional.mean(),
                'directional_n': len(directional),
                'gap': neutral.mean() - directional.mean(),
                't_stat': t_stat,
                'p_value': p_value,
                'cohens_d': effect_size,
                'significant_005': p_value < 0.05
            })

    return pd.DataFrame(results)


def run_overall_test(df):
    """Test neutral > directional controlling for volatility (regression approach)."""
    from scipy.stats import pearsonr

    print("\n" + "="*70)
    print("OVERALL TEST: NEUTRAL VS DIRECTIONAL (VOLATILITY CONTROLLED)")
    print("="*70)

    df_simple = simplify_regimes(df)

    # Overall means
    neutral = df_simple[df_simple['regime_simple'] == 'neutral']['total_uncertainty']
    bullish = df_simple[df_simple['regime_simple'] == 'bullish']['total_uncertainty']
    bearish = df_simple[df_simple['regime_simple'] == 'bearish']['total_uncertainty']

    print(f"\nOverall Uncertainty by Regime:")
    print(f"  Neutral:  mean={neutral.mean():.4f}, std={neutral.std():.4f}, n={len(neutral)}")
    print(f"  Bullish:  mean={bullish.mean():.4f}, std={bullish.std():.4f}, n={len(bullish)}")
    print(f"  Bearish:  mean={bearish.mean():.4f}, std={bearish.std():.4f}, n={len(bearish)}")

    # T-tests
    print(f"\nPairwise Comparisons:")

    t1, p1 = stats.ttest_ind(neutral, bullish, alternative='greater')
    print(f"  Neutral > Bullish:  t={t1:.3f}, p={p1:.4f} {'***' if p1 < 0.01 else '**' if p1 < 0.05 else '*' if p1 < 0.1 else ''}")

    t2, p2 = stats.ttest_ind(neutral, bearish, alternative='greater')
    print(f"  Neutral > Bearish:  t={t2:.3f}, p={p2:.4f} {'***' if p2 < 0.01 else '**' if p2 < 0.05 else '*' if p2 < 0.1 else ''}")

    # Volatility correlation check
    r_vol, p_vol = pearsonr(df_simple['volatility'], df_simple['total_uncertainty'])
    print(f"\nVolatility-Uncertainty Correlation: r={r_vol:.4f}, p={p_vol:.4f}")

    # Check if volatility differs by regime
    print(f"\nVolatility by Regime (to check if confounded):")
    for reg in ['neutral', 'bullish', 'bearish']:
        reg_vol = df_simple[df_simple['regime_simple'] == reg]['volatility']
        print(f"  {reg}: mean_vol={reg_vol.mean():.4f}")

    return {
        'neutral_mean': neutral.mean(),
        'bullish_mean': bullish.mean(),
        'bearish_mean': bearish.mean(),
        'neutral_vs_bullish_t': t1,
        'neutral_vs_bullish_p': p1,
        'neutral_vs_bearish_t': t2,
        'neutral_vs_bearish_p': p2,
        'vol_uncertainty_r': r_vol,
        'vol_uncertainty_p': p_vol
    }


def run_regression_test(df):
    """Run OLS with regime dummies controlling for volatility."""
    import statsmodels.api as sm

    print("\n" + "="*70)
    print("REGRESSION: Uncertainty ~ Volatility + Regime Dummies")
    print("="*70)

    df_simple = simplify_regimes(df)

    # Create dummy variables
    df_simple['is_neutral'] = (df_simple['regime_simple'] == 'neutral').astype(int)
    df_simple['is_bullish'] = (df_simple['regime_simple'] == 'bullish').astype(int)
    # bearish is reference category

    # Model 1: Just volatility
    X1 = sm.add_constant(df_simple['volatility'])
    y = df_simple['total_uncertainty']
    model1 = sm.OLS(y, X1).fit(cov_type='HC3')

    print(f"\nModel 1 (Just Volatility): R² = {model1.rsquared:.4f}")
    print(f"  Volatility coef: {model1.params['volatility']:.4f}, p={model1.pvalues['volatility']:.4f}")

    # Model 2: Volatility + regime dummies
    X2 = df_simple[['volatility', 'is_neutral', 'is_bullish']]
    X2 = sm.add_constant(X2)
    model2 = sm.OLS(y, X2).fit(cov_type='HC3')

    print(f"\nModel 2 (Volatility + Regime Dummies): R² = {model2.rsquared:.4f}")
    print(f"  Volatility coef: {model2.params['volatility']:.4f}, p={model2.pvalues['volatility']:.4f}")
    print(f"  is_neutral coef: {model2.params['is_neutral']:.4f}, p={model2.pvalues['is_neutral']:.4f}")
    print(f"  is_bullish coef: {model2.params['is_bullish']:.4f}, p={model2.pvalues['is_bullish']:.4f}")
    print(f"  (Reference: bearish)")

    # Delta R²
    delta_r2 = model2.rsquared - model1.rsquared
    print(f"\n  ΔR² from adding regime dummies: {delta_r2:.4f}")

    # Interpret
    print("\n  INTERPRETATION:")
    if model2.pvalues['is_neutral'] < 0.05 and model2.params['is_neutral'] > 0:
        print("  ✓ Neutral regime has SIGNIFICANTLY HIGHER uncertainty than bearish,")
        print("    CONTROLLING FOR VOLATILITY. The finding is NOT mechanical.")
    elif model2.params['is_neutral'] > 0:
        print("  ~ Neutral regime has higher uncertainty than bearish, but not significant at 5%.")
    else:
        print("  ✗ Neutral regime does NOT have higher uncertainty than bearish after vol control.")

    return {
        'model1_r2': model1.rsquared,
        'model2_r2': model2.rsquared,
        'delta_r2': delta_r2,
        'is_neutral_coef': model2.params['is_neutral'],
        'is_neutral_p': model2.pvalues['is_neutral'],
        'is_bullish_coef': model2.params['is_bullish'],
        'is_bullish_p': model2.pvalues['is_bullish'],
    }


def main():
    print("="*70)
    print("VOLATILITY-MATCHED REGIME COMPARISON")
    print("The Ambiguity Premium: Does neutral regime have higher uncertainty")
    print("than directional regimes, controlling for volatility?")
    print("="*70)

    # Load and prepare data
    df = load_and_merge_data()
    df = bin_by_volatility(df, n_bins=5)

    # Run tests
    overall_results = run_overall_test(df)
    quintile_results = run_within_quintile_tests(df)
    regression_results = run_regression_test(df)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: IS THE AMBIGUITY PREMIUM REAL?")
    print("="*70)

    n_significant = quintile_results['significant_005'].sum()
    n_positive_gap = (quintile_results['gap'] > 0).sum()

    print(f"\n1. Within-Quintile Tests:")
    print(f"   - {n_positive_gap}/{len(quintile_results)} quintiles: neutral has higher uncertainty")
    print(f"   - {n_significant}/{len(quintile_results)} quintiles: difference is significant (p<0.05)")

    print(f"\n2. Regression Test (controlling for volatility):")
    if regression_results['is_neutral_p'] < 0.05 and regression_results['is_neutral_coef'] > 0:
        print(f"   ✓ KILLER TEST PASSED: Neutral regime has significantly higher uncertainty")
        print(f"     (coef={regression_results['is_neutral_coef']:.4f}, p={regression_results['is_neutral_p']:.4f})")
        print(f"     even after controlling for volatility.")
    else:
        print(f"   ✗ Killer test inconclusive: is_neutral coef={regression_results['is_neutral_coef']:.4f}, p={regression_results['is_neutral_p']:.4f}")

    # Save results
    quintile_results.to_csv('results/volatility_matched_regime_comparison.csv', index=False)
    print(f"\nResults saved to: results/volatility_matched_regime_comparison.csv")

    return quintile_results, overall_results, regression_results


if __name__ == '__main__':
    quintile_results, overall_results, regression_results = main()
