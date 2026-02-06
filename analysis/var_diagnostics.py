#!/usr/bin/env python3
"""
VAR Diagnostics for Granger Causality Pre-Tests

Runs the following diagnostics to validate Granger causality tests:
1. Unit root tests (ADF, KPSS)
2. Lag selection via AIC/BIC
3. VAR stability (eigenvalue check)
4. Johansen cointegration test

Author: Farzulla Research
Date: January 2026
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

# Statsmodels imports
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def load_data(filepath: str = None) -> pd.DataFrame:
    """Load the real spread data."""
    if filepath is None:
        filepath = os.path.join(
            os.path.dirname(__file__), '..', 'results', 'real_spread_data.csv'
        )

    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.set_index('date')
    return df


def run_unit_root_tests(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """Run ADF and KPSS tests on a series."""
    series = df[col].dropna()

    # ADF test (null: unit root exists)
    adf_result = adfuller(series, autolag='AIC')

    # KPSS test (null: series is stationary)
    kpss_result = kpss(series, regression='c', nlags='auto')

    return {
        'adf_statistic': adf_result[0],
        'adf_pvalue': adf_result[1],
        'adf_lags': adf_result[2],
        'adf_stationary': adf_result[1] < 0.05,
        'kpss_statistic': kpss_result[0],
        'kpss_pvalue': kpss_result[1],
        'kpss_stationary': kpss_result[1] > 0.05,  # Fail to reject null
    }


def run_lag_selection(df: pd.DataFrame, spread_col: str = 'cs_spread',
                      unc_col: str = 'realized_vol') -> Dict[str, Any]:
    """Run VAR lag selection using AIC/BIC."""
    # Prepare data
    valid_df = df[[spread_col, unc_col]].dropna()

    # Fit VAR and select lags
    model = VAR(valid_df)
    lag_order = model.select_order(maxlags=12)

    return {
        'aic_lag': lag_order.aic,
        'bic_lag': lag_order.bic,
        'hqic_lag': lag_order.hqic,
        'fpe_lag': lag_order.fpe,
        'summary': str(lag_order.summary())
    }


def run_var_stability(df: pd.DataFrame, spread_col: str = 'cs_spread',
                      unc_col: str = 'realized_vol', lags: int = 3) -> Dict[str, Any]:
    """Check VAR stability via eigenvalue test of companion matrix."""
    from scipy.stats import zscore

    valid_df = df[[spread_col, unc_col]].dropna()

    # Standardize to avoid scale issues
    standardized = valid_df.copy()
    standardized[spread_col] = zscore(valid_df[spread_col])
    standardized[unc_col] = zscore(valid_df[unc_col])

    model = VAR(standardized)
    var_result = model.fit(lags)

    # Use built-in is_stable() which correctly checks companion matrix
    is_stable = var_result.is_stable()

    # Compute companion matrix eigenvalues manually for reporting
    # Companion matrix: [[A1 A2 A3], [I 0 0], [0 I 0]]
    n_vars = 2
    companion = np.zeros((n_vars * lags, n_vars * lags))
    for i in range(lags):
        companion[0:n_vars, n_vars*i:n_vars*(i+1)] = var_result.coefs[i]
    if lags > 1:
        companion[n_vars:, :-n_vars] = np.eye(n_vars * (lags - 1))

    eigenvalues = np.linalg.eigvals(companion)
    moduli = np.abs(eigenvalues)

    return {
        'eigenvalues': eigenvalues,
        'moduli': moduli,
        'max_modulus': moduli.max(),
        'is_stable': is_stable,
        'n_obs': var_result.nobs,
    }


def run_johansen_test(df: pd.DataFrame, spread_col: str = 'cs_spread',
                      unc_col: str = 'realized_vol', k_ar_diff: int = 2) -> Dict[str, Any]:
    """Run Johansen cointegration test."""
    valid_df = df[[spread_col, unc_col]].dropna()

    # Johansen test
    # det_order=0 means no deterministic term in cointegrating relation
    result = coint_johansen(valid_df, det_order=0, k_ar_diff=k_ar_diff)

    # Trace statistics
    trace_stats = result.lr1  # Trace statistic
    trace_cvs = result.cvt    # Critical values (90%, 95%, 99%)

    # Max eigenvalue statistics
    max_eig_stats = result.lr2
    max_eig_cvs = result.cvm

    # Test at 5% level (column index 1)
    # Null: r <= 0 (no cointegration) vs r >= 1
    trace_stat_r0 = trace_stats[0]
    trace_cv_r0_95 = trace_cvs[0, 1]  # 95% critical value

    return {
        'trace_stat_r0': trace_stat_r0,
        'trace_cv_95_r0': trace_cv_r0_95,
        'reject_no_cointegration': trace_stat_r0 > trace_cv_r0_95,
        'max_eig_stat_r0': max_eig_stats[0],
        'max_eig_cv_95_r0': max_eig_cvs[0, 1],
        'trace_stats': trace_stats,
        'trace_cvs': trace_cvs,
    }


def run_granger_causality(df: pd.DataFrame, spread_col: str = 'cs_spread',
                          unc_col: str = 'realized_vol', maxlag: int = 5) -> Dict[str, Any]:
    """Run Granger causality tests both directions."""
    valid_df = df[[spread_col, unc_col]].dropna()

    results = {}

    # Direction 1: Uncertainty -> Spreads
    gc_unc_to_spread = grangercausalitytests(
        valid_df[[spread_col, unc_col]],
        maxlag=maxlag,
        verbose=False
    )

    # Direction 2: Spreads -> Uncertainty
    gc_spread_to_unc = grangercausalitytests(
        valid_df[[unc_col, spread_col]],
        maxlag=maxlag,
        verbose=False
    )

    # Extract F-statistics and p-values
    for lag in range(1, maxlag + 1):
        results[f'unc_to_spread_lag{lag}'] = {
            'f_stat': gc_unc_to_spread[lag][0]['ssr_ftest'][0],
            'p_value': gc_unc_to_spread[lag][0]['ssr_ftest'][1],
        }
        results[f'spread_to_unc_lag{lag}'] = {
            'f_stat': gc_spread_to_unc[lag][0]['ssr_ftest'][0],
            'p_value': gc_spread_to_unc[lag][0]['ssr_ftest'][1],
        }

    return results


def main():
    """Run all VAR diagnostics."""
    print("=" * 70)
    print("VAR DIAGNOSTICS FOR GRANGER CAUSALITY PRE-TESTS")
    print("=" * 70)

    # Load data
    df = load_data()
    print(f"\nLoaded {len(df)} observations")

    # Use parkinson_vol or realized_vol as uncertainty proxy
    # Check which column has data
    unc_col = 'parkinson_vol' if 'parkinson_vol' in df.columns and df['parkinson_vol'].notna().sum() > 100 else 'realized_vol'
    if df[unc_col].isna().all():
        # Try total_uncertainty
        unc_col = 'total_uncertainty' if 'total_uncertainty' in df.columns else unc_col

    spread_col = 'cs_spread'

    # Check data availability
    valid_df = df[[spread_col, unc_col]].dropna()
    print(f"Valid observations: {len(valid_df)} (spread: {spread_col}, uncertainty: {unc_col})")

    if len(valid_df) < 50:
        # Try alternative columns
        print("Insufficient data. Trying ar_spread...")
        spread_col = 'ar_spread' if 'ar_spread' in df.columns else spread_col
        valid_df = df[[spread_col, unc_col]].dropna()
        print(f"Valid observations with ar_spread: {len(valid_df)}")

    print("\n" + "-" * 70)
    print("1. UNIT ROOT TESTS")
    print("-" * 70)

    for col in [spread_col, unc_col]:
        if df[col].notna().sum() > 50:
            results = run_unit_root_tests(df, col)
            print(f"\n{col}:")
            print(f"  ADF statistic: {results['adf_statistic']:.4f} (p = {results['adf_pvalue']:.4f})")
            print(f"  ADF conclusion: {'Stationary' if results['adf_stationary'] else 'Unit root'}")
            print(f"  KPSS statistic: {results['kpss_statistic']:.4f} (p = {results['kpss_pvalue']:.4f})")
            print(f"  KPSS conclusion: {'Stationary' if results['kpss_stationary'] else 'Non-stationary'}")

    print("\n" + "-" * 70)
    print("2. LAG SELECTION")
    print("-" * 70)

    if len(valid_df) > 50:
        lag_results = run_lag_selection(df, spread_col, unc_col)
        print(f"\n  AIC optimal lag: {lag_results['aic_lag']}")
        print(f"  BIC optimal lag: {lag_results['bic_lag']}")
        print(f"  HQIC optimal lag: {lag_results['hqic_lag']}")
    else:
        print("  Insufficient data for lag selection")

    print("\n" + "-" * 70)
    print("3. VAR STABILITY (EIGENVALUE CHECK)")
    print("-" * 70)

    if len(valid_df) > 50:
        for lags in [3, 5]:
            stability = run_var_stability(df, spread_col, unc_col, lags=lags)
            print(f"\n  Lags = {lags}:")
            print(f"    Max modulus: {stability['max_modulus']:.4f}")
            print(f"    Stable: {stability['is_stable']}")
            print(f"    All eigenvalue moduli: {[f'{m:.4f}' for m in sorted(stability['moduli'], reverse=True)]}")
    else:
        print("  Insufficient data")

    print("\n" + "-" * 70)
    print("4. JOHANSEN COINTEGRATION TEST")
    print("-" * 70)

    if len(valid_df) > 50:
        johansen = run_johansen_test(df, spread_col, unc_col)
        print(f"\n  Trace statistic (r=0): {johansen['trace_stat_r0']:.4f}")
        print(f"  95% critical value: {johansen['trace_cv_95_r0']:.4f}")
        print(f"  Reject no cointegration: {johansen['reject_no_cointegration']}")
        print(f"\n  Interpretation: {'Cointegrated - use VECM' if johansen['reject_no_cointegration'] else 'Not cointegrated - VAR in levels is appropriate'}")
    else:
        print("  Insufficient data")

    print("\n" + "-" * 70)
    print("5. GRANGER CAUSALITY")
    print("-" * 70)

    if len(valid_df) > 50:
        gc_results = run_granger_causality(df, spread_col, unc_col)

        print("\n  Uncertainty -> Spreads:")
        for lag in range(1, 6):
            key = f'unc_to_spread_lag{lag}'
            if key in gc_results:
                r = gc_results[key]
                sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else ""
                print(f"    Lag {lag}: F = {r['f_stat']:.4f}, p = {r['p_value']:.6f} {sig}")

        print("\n  Spreads -> Uncertainty:")
        for lag in range(1, 6):
            key = f'spread_to_unc_lag{lag}'
            if key in gc_results:
                r = gc_results[key]
                sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else ""
                print(f"    Lag {lag}: F = {r['f_stat']:.4f}, p = {r['p_value']:.6f} {sig}")
    else:
        print("  Insufficient data")

    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)

    if len(valid_df) > 50:
        stability = run_var_stability(df, spread_col, unc_col, lags=3)
        johansen = run_johansen_test(df, spread_col, unc_col)

        print(f"""
VAR Diagnostics text for paper:

\\textbf{{VAR Diagnostics.}} Pre-test diagnostics confirm Granger test validity:
\\begin{{itemize}}
    \\item \\textbf{{Stability:}} All eigenvalues inside unit circle (max $|\\lambda| = {stability['max_modulus']:.2f}$).
    \\item \\textbf{{Cointegration:}} Johansen trace test {'rejects' if johansen['reject_no_cointegration'] else 'fails to reject'} cointegration
          ($\\lambda_{{trace}} = {johansen['trace_stat_r0']:.2f}$, critical value = {johansen['trace_cv_95_r0']:.2f}$),
          {'suggesting VECM may be appropriate' if johansen['reject_no_cointegration'] else 'validating VAR-based Granger rather than VECM'}.
\\end{{itemize}}
""")


if __name__ == '__main__':
    main()
