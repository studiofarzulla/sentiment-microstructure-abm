"""
Abdi-Ranaldo (2017) Spread Estimator - Robustness Check

Implements the Abdi & Ranaldo (2017) high-frequency spread estimator as an
alternative to Corwin-Schultz. This provides a robustness check since:

1. AR estimator uses different assumptions (no bid-ask bounce required)
2. AR is generally more efficient for high-frequency data
3. If correlations hold with both estimators, findings are more robust

Reference:
Abdi, F., & Ranaldo, A. (2017). A simple estimation of bid-ask spreads from
daily close, high, and low prices. The Review of Financial Studies, 30(12),
4437-4480.

Author: Murad Farzulla
Date: January 2026
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr

# For loading existing data
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def abdi_ranaldo_spread(df: pd.DataFrame, window: int = 1) -> pd.Series:
    """
    Abdi-Ranaldo (2017) spread estimator.

    The key insight is that if the close is near the ask, it's likely
    positive (buyer-initiated), and if near the bid, negative (seller-initiated).
    The next close will then mean-revert, creating a pattern.

    Formula:
        S^2 = 4 * E[(c_t - m_t) * (c_t - c_{t-1})]

        where: m_t = (h_t + l_t) / 2 (midpoint)
               c_t = close price at time t

    The estimator is unbiased under mild assumptions and more efficient
    than Corwin-Schultz for liquid assets.

    Args:
        df: DataFrame with 'close', 'high', 'low' columns
        window: Rolling window for smoothing (default=1, no smoothing)

    Returns:
        Series of spread estimates in basis points
    """
    # Midpoint
    mid = (df['high'] + df['low']) / 2

    # Close deviation from midpoint
    close_mid_diff = df['close'] - mid

    # Serial close change
    close_lag_diff = df['close'] - df['close'].shift(1)

    # AR estimator: spread^2 = 4 * E[...
    spread_sq = 4 * close_mid_diff * close_lag_diff

    if window > 1:
        # Rolling mean for smoothing
        spread_sq_smooth = spread_sq.rolling(window).mean()
    else:
        spread_sq_smooth = spread_sq

    # Take sqrt (only for positive values to avoid NaN)
    # Negative values indicate the spread estimate is invalid for that period
    spread = np.sqrt(np.maximum(spread_sq_smooth, 0))

    # Convert to relative spread (proportion of price)
    spread_relative = spread / df['close']

    # Convert to basis points
    spread_bps = spread_relative * 10000

    return spread_bps


def abdi_ranaldo_spread_with_variance(df: pd.DataFrame, window: int = 20):
    """
    AR spread with variance estimate for confidence intervals.

    Returns both the spread estimate and its standard error.
    """
    mid = (df['high'] + df['low']) / 2
    close_mid_diff = df['close'] - mid
    close_lag_diff = df['close'] - df['close'].shift(1)

    # Individual spread^2 observations
    spread_sq_obs = 4 * close_mid_diff * close_lag_diff

    # Rolling mean and std
    spread_sq_mean = spread_sq_obs.rolling(window).mean()
    spread_sq_std = spread_sq_obs.rolling(window).std()

    # Point estimate
    spread = np.sqrt(np.maximum(spread_sq_mean, 0))
    spread_bps = (spread / df['close']) * 10000

    # Standard error (delta method approximation)
    # If S^2 ~ N(mu, sigma^2), then S ~ N(sqrt(mu), sigma/(2*sqrt(mu)))
    se_spread_sq = spread_sq_std / np.sqrt(window)
    se_spread = se_spread_sq / (2 * np.sqrt(np.maximum(spread_sq_mean, 1e-10)))
    se_spread_bps = (se_spread / df['close']) * 10000

    return spread_bps, se_spread_bps


def compare_spread_estimators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare Corwin-Schultz and Abdi-Ranaldo spread estimates.

    This is the key robustness check - if both estimators show
    similar correlations with uncertainty, the finding is robust.
    """
    # Import CS estimator from existing code
    from analysis.real_spread_validation import SpreadProxyCalculator

    calc = SpreadProxyCalculator()

    # Compute both spreads
    cs_spread = calc.corwin_schultz_spread(df)
    ar_spread = abdi_ranaldo_spread(df, window=2)  # Slight smoothing for AR

    # Correlation between estimators
    valid_idx = ~(cs_spread.isna() | ar_spread.isna())
    if valid_idx.sum() > 30:
        corr_r, corr_p = pearsonr(cs_spread[valid_idx], ar_spread[valid_idx])
    else:
        corr_r, corr_p = np.nan, np.nan

    # Summary stats
    results = {
        'estimator': ['Corwin-Schultz', 'Abdi-Ranaldo'],
        'mean_bps': [cs_spread.mean(), ar_spread.mean()],
        'median_bps': [cs_spread.median(), ar_spread.median()],
        'std_bps': [cs_spread.std(), ar_spread.std()],
        'min_bps': [cs_spread.min(), ar_spread.min()],
        'max_bps': [cs_spread.max(), ar_spread.max()],
        'valid_obs': [cs_spread.notna().sum(), ar_spread.notna().sum()],
    }

    print("\n" + "=" * 70)
    print("SPREAD ESTIMATOR COMPARISON: Corwin-Schultz vs Abdi-Ranaldo")
    print("=" * 70)

    results_df = pd.DataFrame(results)
    print("\nDescriptive Statistics:")
    print(results_df.to_string(index=False))

    print(f"\nCorrelation between estimators:")
    print(f"  Pearson r = {corr_r:.4f} (p = {corr_p:.4f})")

    if corr_r > 0.8:
        print("  STRONG AGREEMENT: Both estimators highly correlated")
    elif corr_r > 0.5:
        print("  MODERATE AGREEMENT: Estimators reasonably correlated")
    else:
        print("  WEAK AGREEMENT: Estimators diverge significantly")

    return results_df, cs_spread, ar_spread, (corr_r, corr_p)


def uncertainty_correlation_comparison(
    df: pd.DataFrame,
    cs_spread: pd.Series,
    ar_spread: pd.Series
) -> pd.DataFrame:
    """
    Compare uncertainty correlations for both spread estimators.

    This is the key test - do both estimators show similar patterns?
    """
    # Compute uncertainty proxies (or load existing)
    from analysis.real_spread_validation import UncertaintyProxyCalculator

    unc_calc = UncertaintyProxyCalculator()

    # Compute proxies if not already in df
    if 'realized_vol' not in df.columns:
        df['realized_vol'] = unc_calc.realized_volatility(df)
    if 'volume_dispersion' not in df.columns:
        df['volume_dispersion'] = unc_calc.volume_dispersion(df)
    if 'total_uncertainty' not in df.columns:
        df['aleatoric_proxy'] = (df['realized_vol'] +
                                  unc_calc.intraday_range_volatility(df)) / 2
        df['epistemic_proxy'] = (df['volume_dispersion'] +
                                  unc_calc.momentum_uncertainty(df)) / 2
        df['total_uncertainty'] = unc_calc.compute_total_uncertainty(
            df['aleatoric_proxy'], df['epistemic_proxy']
        )

    # Uncertainty columns to test
    unc_cols = ['realized_vol', 'volume_dispersion', 'total_uncertainty']

    results = []

    print("\n" + "=" * 70)
    print("UNCERTAINTY CORRELATION COMPARISON")
    print("=" * 70)
    print("\n{:20s}  {:>15s}  {:>15s}  {:>10s}".format(
        "Uncertainty Proxy", "CS Spread (r)", "AR Spread (r)", "Agreement"))
    print("-" * 70)

    for unc_col in unc_cols:
        # Corwin-Schultz correlation
        valid_cs = ~(cs_spread.isna() | df[unc_col].isna())
        if valid_cs.sum() > 30:
            cs_r, cs_p = pearsonr(cs_spread[valid_cs], df[unc_col][valid_cs])
        else:
            cs_r, cs_p = np.nan, np.nan

        # Abdi-Ranaldo correlation
        valid_ar = ~(ar_spread.isna() | df[unc_col].isna())
        if valid_ar.sum() > 30:
            ar_r, ar_p = pearsonr(ar_spread[valid_ar], df[unc_col][valid_ar])
        else:
            ar_r, ar_p = np.nan, np.nan

        # Check agreement
        same_sign = np.sign(cs_r) == np.sign(ar_r)
        similar_magnitude = abs(cs_r - ar_r) < 0.1
        agreement = "Yes" if same_sign and similar_magnitude else "Partial" if same_sign else "No"

        cs_sig = "***" if cs_p < 0.001 else "**" if cs_p < 0.01 else "*" if cs_p < 0.05 else ""
        ar_sig = "***" if ar_p < 0.001 else "**" if ar_p < 0.01 else "*" if ar_p < 0.05 else ""

        print(f"{unc_col:20s}  {cs_r:>10.3f}{cs_sig:>4s}  {ar_r:>10.3f}{ar_sig:>4s}  {agreement:>10s}")

        results.append({
            'uncertainty_proxy': unc_col,
            'cs_pearson_r': cs_r,
            'cs_p_value': cs_p,
            'ar_pearson_r': ar_r,
            'ar_p_value': ar_p,
            'direction_agreement': same_sign,
            'magnitude_agreement': similar_magnitude,
        })

    results_df = pd.DataFrame(results)

    # Summary
    n_agree = (results_df['direction_agreement'] & results_df['magnitude_agreement']).sum()
    print(f"\nSummary: {n_agree}/{len(results)} uncertainty proxies show full agreement")

    if n_agree == len(results):
        print("ROBUSTNESS CONFIRMED: Both spread estimators show consistent patterns")
    elif n_agree > len(results) / 2:
        print("PARTIAL ROBUSTNESS: Most results consistent across estimators")
    else:
        print("WEAK ROBUSTNESS: Estimators diverge on key correlations")

    return results_df


def run_ar_spread_analysis(save_results: bool = True) -> dict:
    """
    Run full Abdi-Ranaldo spread analysis as robustness check.
    """
    # Load existing processed data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(project_dir, "results")

    spread_data_path = os.path.join(results_dir, "real_spread_data.csv")

    if not os.path.exists(spread_data_path):
        print(f"ERROR: Run real_spread_validation.py first to generate {spread_data_path}")
        return None

    df = pd.read_csv(spread_data_path, index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} observations from {spread_data_path}")

    # Compare spread estimators
    comparison_df, cs_spread, ar_spread, estimator_corr = compare_spread_estimators(df)

    # Add AR spread to dataframe
    df['ar_spread'] = ar_spread

    # Compare uncertainty correlations
    corr_comparison_df = uncertainty_correlation_comparison(df, cs_spread, ar_spread)

    # Save results
    if save_results:
        ar_results_path = os.path.join(results_dir, "abdi_ranaldo_comparison.csv")
        corr_comparison_df.to_csv(ar_results_path, index=False)
        print(f"\nSaved results to: {ar_results_path}")

        # Also save updated spread data with AR spread
        df.to_csv(spread_data_path)
        print(f"Updated {spread_data_path} with AR spread column")

    return {
        'estimator_comparison': comparison_df,
        'estimator_correlation': estimator_corr,
        'uncertainty_correlations': corr_comparison_df,
        'data': df
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Abdi-Ranaldo spread estimator - robustness check"
    )
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results")

    args = parser.parse_args()

    results = run_ar_spread_analysis(save_results=not args.no_save)

    if results is not None:
        print("\n" + "=" * 70)
        print("ABDI-RANALDO ROBUSTNESS CHECK COMPLETE")
        print("=" * 70)
        print("""
Key findings:
1. AR estimator provides alternative spread measurement
2. If CS and AR spreads are highly correlated, both capture similar liquidity
3. If uncertainty correlations agree, the finding is robust to estimation method

For the paper:
- Report both CS and AR spreads in Appendix
- Note agreement/disagreement in Results section
- This addresses the "spread estimation sensitivity" concern
""")


if __name__ == '__main__':
    main()
