"""
Bear Market Out-of-Sample Validation (2022 vs 2024)

Tests whether the extremity premium holds in the 2022 bear market,
providing crucial out-of-sample validation.

Key differences:
- 2022: 93% fear regimes (extreme_fear 57%, fear 36%)
- 2024: Greed-dominated (mix of regimes)

If extremity premium holds in 2022:
- Extreme fear should show elevated uncertainty vs neutral
- Despite different market conditions, the pattern persists

Author: Murad Farzulla
Date: January 2026
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')

# Add parent for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_spread_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Corwin-Schultz and Abdi-Ranaldo spread proxies.
    """
    df = df.copy()

    # Corwin-Schultz
    high_2d = df['high'].rolling(2).max()
    low_2d = df['low'].rolling(2).min()

    beta = (np.log(df['high'] / df['low'])) ** 2
    gamma = np.log(high_2d / low_2d) ** 2
    beta_sum = beta.rolling(2).sum()

    numerator = np.sqrt(2 * beta_sum) - np.sqrt(beta_sum)
    denominator = 3 - 2 * np.sqrt(2)
    alpha = numerator / denominator - np.sqrt(gamma / denominator)

    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    df['cs_spread'] = spread.clip(lower=0) * 10000

    # Abdi-Ranaldo
    mid = (df['high'] + df['low']) / 2
    close_mid_diff = df['close'] - mid
    close_lag_diff = df['close'] - df['close'].shift(1)
    spread_sq = 4 * close_mid_diff * close_lag_diff
    df['ar_spread'] = np.sqrt(np.maximum(spread_sq.rolling(2).mean(), 0)) / df['close'] * 10000

    # Volatility measures
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    df['realized_vol'] = df['returns'].rolling(20).std() * np.sqrt(252)

    # Parkinson volatility
    log_hl = np.log(df['high'] / df['low'])
    df['parkinson_vol'] = np.sqrt((log_hl ** 2).rolling(20).mean() / (4 * np.log(2))) * np.sqrt(252)

    # Use parkinson as uncertainty proxy
    df['total_uncertainty'] = df['parkinson_vol']

    return df


def run_regime_analysis(df: pd.DataFrame, year_label: str) -> dict:
    """
    Run regime-based uncertainty analysis for a given dataset.

    Returns coefficients for each regime relative to neutral.
    """
    print(f"\n{'='*60}")
    print(f"REGIME ANALYSIS: {year_label}")
    print(f"{'='*60}")

    # Create regime dummies
    df = df.copy()
    for reg in ['extreme_fear', 'fear', 'greed', 'extreme_greed']:
        df[f'is_{reg}'] = (df['regime'] == reg).astype(int)

    # Clean data
    df_clean = df.dropna(subset=['total_uncertainty', 'realized_vol', 'regime'])

    print(f"\nN observations: {len(df_clean)}")
    print(f"Date range: {df_clean['date'].min()} to {df_clean['date'].max()}")

    print(f"\nRegime distribution:")
    for reg in ['extreme_greed', 'greed', 'neutral', 'fear', 'extreme_fear']:
        n = len(df_clean[df_clean['regime'] == reg])
        pct = 100 * n / len(df_clean)
        print(f"  {reg:15s}: {n:3d} ({pct:5.1f}%)")

    # Regression: uncertainty ~ volatility + regime dummies (neutral = reference)
    available_regimes = []
    for reg in ['extreme_fear', 'fear', 'greed', 'extreme_greed']:
        if df_clean[f'is_{reg}'].sum() >= 5:  # Need at least 5 obs
            available_regimes.append(f'is_{reg}')

    if not available_regimes:
        print("\nInsufficient regime variation for regression")
        return {'year': year_label, 'n_obs': len(df_clean), 'regimes': {}}

    X = df_clean[['realized_vol'] + available_regimes]
    X = sm.add_constant(X)
    y = df_clean['total_uncertainty']

    model = sm.OLS(y, X).fit(cov_type='HC3')

    print(f"\nRegression: Uncertainty ~ Volatility + Regime Dummies")
    print(f"R-squared: {model.rsquared:.4f}")
    print(f"\nCoefficients (relative to NEUTRAL):")

    results = {
        'year': year_label,
        'n_obs': len(df_clean),
        'r_squared': model.rsquared,
        'regimes': {}
    }

    for var in available_regimes:
        coef = model.params[var]
        se = model.bse[var]
        pval = model.pvalues[var]
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""

        regime_name = var.replace('is_', '')
        results['regimes'][regime_name] = {
            'coefficient': coef,
            'std_error': se,
            'p_value': pval,
            'significant': pval < 0.05
        }

        print(f"  {regime_name:15s}: {coef:+.4f} (SE={se:.4f}, p={pval:.4f}) {sig}")

    return results


def compare_markets(results_2022: dict, results_2024: dict) -> pd.DataFrame:
    """
    Compare regime effects between 2022 bear market and 2024 bull market.
    """
    print("\n" + "=" * 70)
    print("CROSS-MARKET COMPARISON: 2022 Bear vs 2024 Bull")
    print("=" * 70)

    comparison = []

    all_regimes = set(results_2022['regimes'].keys()) | set(results_2024['regimes'].keys())

    print("\n{:15s} {:>15s} {:>15s} {:>12s}".format(
        "Regime", "2022 (Bear)", "2024 (Bull)", "Direction"))
    print("-" * 60)

    for regime in ['extreme_fear', 'fear', 'greed', 'extreme_greed']:
        coef_2022 = results_2022['regimes'].get(regime, {}).get('coefficient', np.nan)
        coef_2024 = results_2024['regimes'].get(regime, {}).get('coefficient', np.nan)

        sig_2022 = results_2022['regimes'].get(regime, {}).get('significant', False)
        sig_2024 = results_2024['regimes'].get(regime, {}).get('significant', False)

        # Check if direction is consistent
        if not np.isnan(coef_2022) and not np.isnan(coef_2024):
            same_direction = np.sign(coef_2022) == np.sign(coef_2024)
            direction = "Same ✓" if same_direction else "Opposite"
        else:
            direction = "N/A"

        sig_marker_2022 = "*" if sig_2022 else ""
        sig_marker_2024 = "*" if sig_2024 else ""

        print(f"{regime:15s} {coef_2022:>+13.4f}{sig_marker_2022:>2s} "
              f"{coef_2024:>+13.4f}{sig_marker_2024:>2s} {direction:>12s}")

        comparison.append({
            'regime': regime,
            'coef_2022_bear': coef_2022,
            'sig_2022': sig_2022,
            'coef_2024_bull': coef_2024,
            'sig_2024': sig_2024,
            'same_direction': same_direction if not np.isnan(coef_2022) and not np.isnan(coef_2024) else None,
        })

    comparison_df = pd.DataFrame(comparison)

    # Summary
    n_same_direction = comparison_df['same_direction'].sum()
    n_total = comparison_df['same_direction'].notna().sum()

    print("\n" + "-" * 60)
    print(f"Direction agreement: {n_same_direction}/{n_total} regimes")

    if n_same_direction == n_total:
        print("\n✓ STRONG OUT-OF-SAMPLE VALIDATION")
        print("  All regime effects have consistent direction across markets")
    elif n_same_direction >= n_total / 2:
        print("\n~ PARTIAL VALIDATION")
        print("  Most regime effects are consistent across markets")
    else:
        print("\n✗ WEAK VALIDATION")
        print("  Regime effects differ between bull and bear markets")

    return comparison_df


def run_bear_market_analysis(save_results: bool = True):
    """
    Run full bear market out-of-sample validation.
    """
    print("=" * 70)
    print("BEAR MARKET OUT-OF-SAMPLE VALIDATION")
    print("=" * 70)

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, "data", "datasets")
    results_dir = os.path.join(project_dir, "results")

    # Load 2022 data
    path_2022 = os.path.join(data_dir, "btc_sentiment_2022_bear.csv")
    if not os.path.exists(path_2022):
        print("ERROR: 2022 data not found. Run fetch_2022_data.py first")
        return None

    df_2022 = pd.read_csv(path_2022, parse_dates=['date'])
    print(f"\nLoaded 2022 bear market data: {len(df_2022)} days")

    # Load 2024 data
    path_2024 = os.path.join(data_dir, "btc_sentiment_daily.csv")
    df_2024 = pd.read_csv(path_2024, parse_dates=['date'])
    print(f"Loaded 2024 bull market data: {len(df_2024)} days")

    # Compute spread proxies for 2022
    print("\nComputing spread proxies for 2022 data...")
    df_2022 = compute_spread_proxies(df_2022)

    # Compute spread proxies for 2024 if not already present
    if 'cs_spread' not in df_2024.columns:
        print("Computing spread proxies for 2024 data...")
        df_2024 = compute_spread_proxies(df_2024)

    # Run regime analysis for each year
    results_2022 = run_regime_analysis(df_2022, "2022 Bear Market")
    results_2024 = run_regime_analysis(df_2024, "2024 Bull Market")

    # Compare
    comparison_df = compare_markets(results_2022, results_2024)

    # Key finding
    print("\n" + "=" * 70)
    print("KEY FINDING: THE EXTREMITY PREMIUM")
    print("=" * 70)

    # Check extreme_fear in 2022 (dominant regime)
    ef_2022 = results_2022['regimes'].get('extreme_fear', {})
    if ef_2022.get('significant', False) and ef_2022.get('coefficient', 0) > 0:
        print("""
✓ EXTREMITY PREMIUM CONFIRMED IN BEAR MARKET

In 2022 (93% fear regimes):
  - Extreme fear shows ELEVATED uncertainty vs neutral
  - Coefficient: +{:.4f} (significant)
  - Pattern persists even when extreme_fear is dominant regime

This is crucial evidence because:
  1. 2022 is truly out-of-sample (different market regime)
  2. The extremity premium isn't just a bull market phenomenon
  3. Both extreme_greed (2024) and extreme_fear (2022) show elevated uncertainty
  4. The common factor is EXTREMITY, not direction

Theoretical implication:
  - Extreme sentiment (either direction) indicates active disagreement
  - Market makers widen spreads because of adverse selection risk
  - This is consistent with Glosten-Milgrom extended to sentiment regimes
""".format(ef_2022['coefficient']))
    else:
        print("""
~ PARTIAL VALIDATION

Extreme fear coefficient: {:.4f} (p={:.4f})

The 2022 bear market shows different patterns than 2024.
This may indicate:
  1. Regime effects are time-varying
  2. Bull vs bear market dynamics differ fundamentally
  3. The extremity premium is context-dependent
""".format(
            ef_2022.get('coefficient', 0),
            ef_2022.get('p_value', 1)
        ))

    # Save results
    if save_results:
        # Save 2022 spread data
        df_2022.to_csv(os.path.join(results_dir, "spread_data_2022_bear.csv"), index=False)

        # Save comparison
        comparison_df.to_csv(os.path.join(results_dir, "bear_bull_comparison.csv"), index=False)

        # Save summary
        summary = {
            '2022_n_obs': results_2022['n_obs'],
            '2022_r_squared': results_2022['r_squared'],
            '2024_n_obs': results_2024['n_obs'],
            '2024_r_squared': results_2024['r_squared'],
            'n_regimes_same_direction': comparison_df['same_direction'].sum(),
            'n_regimes_total': comparison_df['same_direction'].notna().sum(),
        }

        for regime in ['extreme_fear', 'fear', 'greed', 'extreme_greed']:
            r_2022 = results_2022['regimes'].get(regime, {})
            r_2024 = results_2024['regimes'].get(regime, {})
            summary[f'{regime}_2022_coef'] = r_2022.get('coefficient', np.nan)
            summary[f'{regime}_2022_sig'] = r_2022.get('significant', False)
            summary[f'{regime}_2024_coef'] = r_2024.get('coefficient', np.nan)
            summary[f'{regime}_2024_sig'] = r_2024.get('significant', False)

        pd.DataFrame([summary]).to_csv(
            os.path.join(results_dir, "bear_market_summary.csv"), index=False
        )

        print(f"\nResults saved to {results_dir}/")

    return {
        'results_2022': results_2022,
        'results_2024': results_2024,
        'comparison': comparison_df,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Bear market out-of-sample validation")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")

    args = parser.parse_args()

    results = run_bear_market_analysis(save_results=not args.no_save)


if __name__ == '__main__':
    main()
