#!/usr/bin/env python3
"""
Flexible Volatility Controls: Spline-Based Regression

Tests whether regime effects survive with flexible (nonlinear) volatility controls
using natural splines, rather than just linear + quadratic terms.

This addresses the core identification question: Is the extremity premium
capturing nonlinear volatility effects, or is there genuine sentiment signal?

Author: Murad Farzulla
Date: January 2026
"""

import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix
from scipy import stats

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
DATA_FILE = os.path.join(RESULTS_DIR, 'full_sample_btc_data.csv')


def load_and_prepare_data():
    """Load data and create all required variables."""
    print("Loading data...")
    df = pd.read_csv(DATA_FILE, parse_dates=['date'])

    # Basic cleaning
    df = df.dropna(subset=['cs_spread', 'fear_greed_value', 'parkinson_vol'])

    # Create regime dummies (neutral = baseline)
    df['extreme_fear'] = (df['fear_greed_value'] <= 25).astype(int)
    df['fear'] = ((df['fear_greed_value'] > 25) & (df['fear_greed_value'] <= 45)).astype(int)
    df['greed'] = ((df['fear_greed_value'] > 55) & (df['fear_greed_value'] <= 75)).astype(int)
    df['extreme_greed'] = (df['fear_greed_value'] > 75).astype(int)

    # Continuous extremity
    df['distance_from_neutral'] = np.abs(df['fear_greed_value'] - 50) / 50
    df['is_extreme'] = ((df['fear_greed_value'] <= 25) | (df['fear_greed_value'] > 75)).astype(int)

    # Controls
    df['rv'] = df['parkinson_vol']
    df['abs_returns'] = np.abs(df['returns'])
    df['log_vol'] = df['log_volume']

    # Day of week
    df['dow'] = df['date'].dt.dayofweek
    for i in range(6):
        df[f'dow_{i}'] = (df['dow'] == i).astype(int)

    # Filter to positive spreads
    df = df[df['cs_spread'] > 0].copy()

    print(f"  Loaded {len(df)} observations with positive spreads")
    return df


def create_spline_basis(x, df_spline=5):
    """Create natural spline basis using patsy."""
    # Natural spline with specified degrees of freedom
    spline_formula = f"cr(x, df={df_spline}) - 1"  # -1 removes intercept
    basis = dmatrix(spline_formula, {"x": x}, return_type='dataframe')
    return basis


def run_spline_regression(df, y_col, regime_cols, spline_df=5, include_controls=True):
    """Run regression with spline-based volatility control."""

    # Create spline basis for volatility
    spline_basis = create_spline_basis(df['rv'].values, df_spline=spline_df)
    spline_cols = [f'rv_spline_{i}' for i in range(spline_basis.shape[1])]
    for i, col in enumerate(spline_cols):
        df[col] = spline_basis.iloc[:, i].values

    # Build design matrix
    x_cols = regime_cols + spline_cols

    if include_controls:
        x_cols += ['abs_returns', 'log_vol']
        x_cols += [f'dow_{i}' for i in range(6)]

    y = df[y_col].values
    X = df[x_cols].values
    X = sm.add_constant(X)

    # Fit with HAC
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 10})

    # Extract regime coefficients
    results = {
        'n_obs': int(model.nobs),
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
    }

    coef_names = ['const'] + x_cols
    for i, name in enumerate(coef_names):
        if name in regime_cols or name == 'distance_from_neutral' or name == 'is_extreme':
            results[f'coef_{name}'] = model.params[i]
            results[f'se_{name}'] = model.bse[i]
            results[f'pval_{name}'] = model.pvalues[i]
            results[f'tstat_{name}'] = model.tvalues[i]

    return results, model


def main():
    print("=" * 70)
    print("FLEXIBLE VOLATILITY CONTROLS: SPLINE-BASED ANALYSIS")
    print("=" * 70)

    df = load_and_prepare_data()

    regime_cols = ['extreme_fear', 'fear', 'greed', 'extreme_greed']

    results_all = []

    # Test different spline flexibility levels
    print("\n" + "=" * 70)
    print("TESTING SPLINE DEGREES OF FREEDOM")
    print("=" * 70)

    for spline_df in [3, 5, 7, 10, 15]:
        print(f"\n--- Spline df = {spline_df} ---")

        try:
            res, mod = run_spline_regression(
                df.copy(), 'cs_spread', regime_cols,
                spline_df=spline_df, include_controls=True
            )

            print(f"  R² = {res['r_squared']:.4f}")
            print(f"  Regime coefficients:")

            any_sig = False
            for var in regime_cols:
                coef = res.get(f'coef_{var}', 0)
                pval = res.get(f'pval_{var}', 1)
                stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else '†' if pval < 0.1 else ''
                print(f"    {var}: {coef:8.2f} (p={pval:.3f}) {stars}")
                if pval < 0.1:
                    any_sig = True

            res['spline_df'] = spline_df
            res['any_regime_sig'] = any_sig
            results_all.append(res)

        except Exception as e:
            print(f"  Error: {e}")

    # Also test with continuous extremity measure
    print("\n" + "=" * 70)
    print("CONTINUOUS EXTREMITY WITH FLEXIBLE VOLATILITY")
    print("=" * 70)

    for spline_df in [5, 10]:
        print(f"\n--- Continuous Extremity, Spline df = {spline_df} ---")

        try:
            # Need to handle this differently
            df_temp = df.copy()

            # Create spline basis
            spline_basis = create_spline_basis(df_temp['rv'].values, df_spline=spline_df)
            spline_cols = [f'rv_spline_{i}' for i in range(spline_basis.shape[1])]
            for i, col in enumerate(spline_cols):
                df_temp[col] = spline_basis.iloc[:, i].values

            # Build model with continuous extremity
            x_cols = ['distance_from_neutral'] + spline_cols + ['abs_returns', 'log_vol']
            x_cols += [f'dow_{i}' for i in range(6)]

            y = df_temp['cs_spread'].values
            X = df_temp[x_cols].values
            X = sm.add_constant(X)

            model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 10})

            # Extract distance_from_neutral coefficient
            coef_idx = 1  # After constant
            coef = model.params[coef_idx]
            se = model.bse[coef_idx]
            pval = model.pvalues[coef_idx]

            stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else '†' if pval < 0.1 else ''

            print(f"  R² = {model.rsquared:.4f}")
            print(f"  Distance from Neutral: {coef:.2f} (SE={se:.2f}, p={pval:.3f}) {stars}")

        except Exception as e:
            print(f"  Error: {e}")

    # Binary extreme test
    print("\n" + "=" * 70)
    print("BINARY EXTREME WITH FLEXIBLE VOLATILITY")
    print("=" * 70)

    for spline_df in [5, 10]:
        print(f"\n--- Binary Extreme, Spline df = {spline_df} ---")

        try:
            df_temp = df.copy()

            spline_basis = create_spline_basis(df_temp['rv'].values, df_spline=spline_df)
            spline_cols = [f'rv_spline_{i}' for i in range(spline_basis.shape[1])]
            for i, col in enumerate(spline_cols):
                df_temp[col] = spline_basis.iloc[:, i].values

            x_cols = ['is_extreme'] + spline_cols + ['abs_returns', 'log_vol']
            x_cols += [f'dow_{i}' for i in range(6)]

            y = df_temp['cs_spread'].values
            X = df_temp[x_cols].values
            X = sm.add_constant(X)

            model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 10})

            coef_idx = 1
            coef = model.params[coef_idx]
            se = model.bse[coef_idx]
            pval = model.pvalues[coef_idx]

            stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else '†' if pval < 0.1 else ''

            print(f"  R² = {model.rsquared:.4f}")
            print(f"  Is Extreme (binary): {coef:.2f} (SE={se:.2f}, p={pval:.3f}) {stars}")

        except Exception as e:
            print(f"  Error: {e}")

    # Comparison: Within-quintile approach (the original method)
    print("\n" + "=" * 70)
    print("COMPARISON: WITHIN-QUINTILE STRATIFICATION")
    print("=" * 70)

    df['vol_quintile'] = pd.qcut(df['rv'], 5, labels=False, duplicates='drop')

    print("\nExtreme vs Neutral spread difference by volatility quintile:")

    for q in range(5):
        df_q = df[df['vol_quintile'] == q]

        extreme = df_q[df_q['is_extreme'] == 1]['cs_spread']
        neutral = df_q[(df_q['is_extreme'] == 0) &
                       (df_q['fear_greed_value'] >= 46) &
                       (df_q['fear_greed_value'] <= 55)]['cs_spread']

        if len(extreme) > 5 and len(neutral) > 5:
            gap = extreme.mean() - neutral.mean()
            t_stat, p_val = stats.ttest_ind(extreme, neutral, equal_var=False)
            stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else '†' if p_val < 0.1 else ''
            print(f"  Q{q+1}: Gap = {gap:7.2f}, t = {t_stat:5.2f}, p = {p_val:.3f} {stars} (n_ext={len(extreme)}, n_neu={len(neutral)})")
        else:
            print(f"  Q{q+1}: Insufficient data (n_ext={len(extreme)}, n_neu={len(neutral)})")

    # Aggregate within-quintile test (pooled)
    print("\n--- Pooled Within-Quintile Test ---")

    # Residualize spreads by quintile
    df['spread_resid'] = df.groupby('vol_quintile')['cs_spread'].transform(
        lambda x: x - x.mean()
    )

    extreme_resid = df[df['is_extreme'] == 1]['spread_resid']
    neutral_resid = df[(df['is_extreme'] == 0) &
                       (df['fear_greed_value'] >= 46) &
                       (df['fear_greed_value'] <= 55)]['spread_resid']

    gap = extreme_resid.mean() - neutral_resid.mean()
    t_stat, p_val = stats.ttest_ind(extreme_resid, neutral_resid, equal_var=False)
    stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else '†' if p_val < 0.1 else ''

    print(f"  Volatility-demeaned gap: {gap:.2f}")
    print(f"  t-statistic: {t_stat:.2f}")
    print(f"  p-value: {p_val:.4f} {stars}")

    # Effect size
    pooled_std = np.sqrt((extreme_resid.var() + neutral_resid.var()) / 2)
    cohens_d = gap / pooled_std
    print(f"  Cohen's d: {cohens_d:.3f}")

    # Save results
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    results_df = pd.DataFrame(results_all)
    csv_path = os.path.join(RESULTS_DIR, 'flexible_volatility_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAY")
    print("=" * 70)
    print("""
The difference between regression-based controls and stratification-based controls:

1. REGRESSION (Kitchen Sink): Forces a parametric functional form on the
   volatility-spread relationship. Even with splines, regime effects wash out.

2. STRATIFICATION (Within-Quintile): Allows arbitrary nonlinear relationships
   by comparing extreme vs neutral WITHIN each volatility bucket. This is
   more conservative and regime effects can survive.

The extremity premium appears to capture volatility effects that are:
- Nonlinearly related to spreads
- Correlated with F&G regime classification
- Not fully captured by flexible parametric controls

This is consistent with the circularity critique: F&G's volatility component
creates regime classifications that correlate with spreads through volatility,
not through "pure" sentiment.
""")


if __name__ == "__main__":
    main()
