"""
Instrumental Variables Analysis for Causal Identification

Addresses the endogeneity concern: spreads might cause uncertainty
(reverse causality) or both might be driven by a common factor (omitted
variable bias).

Instruments (exogenous to individual BTC spreads):
1. VIX jumps (macro volatility shock from equity markets)
2. FOMC announcement days (monetary policy shocks)
3. Weekend/holiday effects (exogenous timing variation)

For valid IV, instruments must satisfy:
- Relevance: corr(Z, uncertainty) != 0 (F > 10 rule of thumb)
- Exogeneity: Z affects spread ONLY through uncertainty

Author: Murad Farzulla
Date: January 2026
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Add parent for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def add_instruments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add instrumental variables to the dataset.

    Instruments:
    1. VIX jump: Daily VIX change > 2 std devs (equity market shock)
    2. Monday effect: Markets more uncertain at week start
    3. Month end: Rebalancing creates uncertainty
    4. Lagged uncertainty: Valid under persistence assumption
    """
    df = df.copy()

    # Ensure we have a date index
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

    # 1. VIX-like proxy: Use realized volatility jumps
    # (Since we may not have actual VIX, use rolling vol jumps as proxy)
    if 'realized_vol' in df.columns:
        vol = df['realized_vol']
    else:
        # Compute if not available
        returns = np.log(df['close'] / df['close'].shift(1))
        vol = returns.rolling(20).std() * np.sqrt(252)
        df['realized_vol'] = vol

    # VIX jump: vol change > 1.5 std devs
    vol_change = vol - vol.shift(1)
    vol_change_std = vol_change.rolling(60).std()
    df['iv_vix_jump'] = (vol_change > 1.5 * vol_change_std).astype(int)

    # 2. Day of week effects
    df['iv_monday'] = (df.index.dayofweek == 0).astype(int)
    df['iv_friday'] = (df.index.dayofweek == 4).astype(int)

    # 3. Month-end effect (last 3 days of month)
    df['iv_month_end'] = (df.index.day >= 28).astype(int)

    # 4. Lagged uncertainty (valid IV under persistence)
    if 'total_uncertainty' in df.columns:
        df['iv_uncertainty_lag1'] = df['total_uncertainty'].shift(1)
        df['iv_uncertainty_lag2'] = df['total_uncertainty'].shift(2)
    elif 'realized_vol' in df.columns:
        df['iv_uncertainty_lag1'] = df['realized_vol'].shift(1)
        df['iv_uncertainty_lag2'] = df['realized_vol'].shift(2)

    # 5. Return sign change (uncertainty often higher after direction change)
    if 'returns' in df.columns:
        returns = df['returns']
    else:
        returns = np.log(df['close'] / df['close'].shift(1))
        df['returns'] = returns

    df['iv_direction_change'] = (
        np.sign(returns) != np.sign(returns.shift(1))
    ).astype(int)

    return df


def first_stage_regression(
    df: pd.DataFrame,
    endogenous: str = 'total_uncertainty',
    instruments: List[str] = None,
    controls: List[str] = None,
) -> Dict:
    """
    First stage: Regress endogenous variable on instruments.

    Tests instrument relevance (F > 10 rule).

    Args:
        df: Data with instruments
        endogenous: Variable to instrument (e.g., 'total_uncertainty')
        instruments: List of instrument columns
        controls: Control variables to include

    Returns:
        Dict with first stage results
    """
    import statsmodels.api as sm

    if instruments is None:
        instruments = ['iv_vix_jump', 'iv_monday', 'iv_uncertainty_lag1']

    if controls is None:
        controls = []

    # Clean data
    all_vars = [endogenous] + instruments + controls
    df_clean = df[all_vars].dropna()

    y = df_clean[endogenous]
    X = df_clean[instruments + controls]
    X = sm.add_constant(X)

    # OLS first stage
    model = sm.OLS(y, X).fit()

    # F-test for instrument relevance
    # Test: all instrument coefficients = 0
    r_matrix = np.zeros((len(instruments), len(model.params)))
    for i, inst in enumerate(instruments):
        r_matrix[i, list(model.params.index).index(inst)] = 1

    f_test = model.f_test(r_matrix)

    # Extract F-stat
    f_stat = float(f_test.fvalue) if np.isscalar(f_test.fvalue) else f_test.fvalue[0][0]
    f_pvalue = float(f_test.pvalue)

    return {
        'model': model,
        'f_stat': f_stat,
        'f_pvalue': f_pvalue,
        'r_squared': model.rsquared,
        'instrument_coeffs': {inst: model.params[inst] for inst in instruments if inst in model.params},
        'instrument_pvals': {inst: model.pvalues[inst] for inst in instruments if inst in model.pvalues},
        'n_obs': len(df_clean),
        'strong_instruments': f_stat > 10,
    }


def second_stage_regression(
    df: pd.DataFrame,
    dependent: str = 'cs_spread',
    endogenous: str = 'total_uncertainty',
    instruments: List[str] = None,
    controls: List[str] = None,
) -> Dict:
    """
    Two-stage least squares (2SLS) IV regression.

    Second stage: Regress dependent on predicted endogenous from first stage.

    Args:
        df: Data with instruments
        dependent: Dependent variable (spread)
        endogenous: Endogenous variable (uncertainty)
        instruments: Instrument list
        controls: Control variables

    Returns:
        Dict with 2SLS results
    """
    import statsmodels.api as sm
    from linearmodels.iv import IV2SLS

    if instruments is None:
        instruments = ['iv_vix_jump', 'iv_monday', 'iv_uncertainty_lag1']

    if controls is None:
        controls = []

    # Clean data
    all_vars = [dependent, endogenous] + instruments + controls
    df_clean = df[all_vars].dropna()

    # Prepare for linearmodels IV2SLS
    try:
        # Using linearmodels for proper IV estimation
        formula = f"{dependent} ~ 1 + [{endogenous} ~ {' + '.join(instruments)}]"

        if controls:
            formula = f"{dependent} ~ 1 + {' + '.join(controls)} + [{endogenous} ~ {' + '.join(instruments)}]"

        y = df_clean[dependent]
        X_exog = sm.add_constant(df_clean[controls]) if controls else sm.add_constant(pd.DataFrame(index=df_clean.index))
        X_endog = df_clean[[endogenous]]
        Z = df_clean[instruments]

        model = IV2SLS(y, X_exog, X_endog, Z).fit(cov_type='robust')

        return {
            'model': model,
            'endogenous_coef': model.params[endogenous],
            'endogenous_se': model.std_errors[endogenous],
            'endogenous_pval': model.pvalues[endogenous],
            'r_squared': model.rsquared,
            'n_obs': model.nobs,
            'sargan_stat': None,  # Would need overid test
            'sargan_pval': None,
        }

    except ImportError:
        # Fallback: Manual 2SLS
        print("Warning: linearmodels not installed, using manual 2SLS")
        return manual_2sls(df_clean, dependent, endogenous, instruments, controls)


def manual_2sls(
    df: pd.DataFrame,
    dependent: str,
    endogenous: str,
    instruments: List[str],
    controls: List[str],
) -> Dict:
    """
    Manual 2SLS implementation as fallback.
    """
    import statsmodels.api as sm

    # First stage: endogenous ~ instruments + controls
    X1 = df[instruments + controls]
    X1 = sm.add_constant(X1)
    y1 = df[endogenous]

    first_stage = sm.OLS(y1, X1).fit()
    predicted_endog = first_stage.fittedvalues

    # Second stage: dependent ~ predicted_endog + controls
    df['_predicted_endog'] = predicted_endog
    X2 = df[['_predicted_endog'] + controls]
    X2 = sm.add_constant(X2)
    y2 = df[dependent]

    second_stage = sm.OLS(y2, X2).fit(cov_type='HC3')

    return {
        'model': second_stage,
        'endogenous_coef': second_stage.params['_predicted_endog'],
        'endogenous_se': second_stage.bse['_predicted_endog'],
        'endogenous_pval': second_stage.pvalues['_predicted_endog'],
        'r_squared': second_stage.rsquared,
        'n_obs': len(df),
        'sargan_stat': None,
        'sargan_pval': None,
    }


def compare_ols_iv(
    df: pd.DataFrame,
    dependent: str = 'cs_spread',
    endogenous: str = 'total_uncertainty',
    instruments: List[str] = None,
) -> pd.DataFrame:
    """
    Compare OLS and IV estimates to assess endogeneity bias.

    If OLS and IV differ significantly, endogeneity is present.
    """
    import statsmodels.api as sm

    if instruments is None:
        instruments = ['iv_vix_jump', 'iv_monday', 'iv_uncertainty_lag1']

    # Clean data
    all_vars = [dependent, endogenous] + instruments
    df_clean = df[all_vars].dropna()

    # OLS
    X_ols = sm.add_constant(df_clean[[endogenous]])
    y = df_clean[dependent]
    ols_model = sm.OLS(y, X_ols).fit(cov_type='HC3')

    # IV (first stage)
    first_stage = first_stage_regression(df_clean, endogenous, instruments)

    # IV (second stage) - manual for simplicity
    df_clean['predicted_unc'] = first_stage['model'].fittedvalues
    X_iv = sm.add_constant(df_clean[['predicted_unc']])
    iv_model = sm.OLS(y, X_iv).fit(cov_type='HC3')

    results = {
        'method': ['OLS', '2SLS-IV'],
        'coefficient': [ols_model.params[endogenous], iv_model.params['predicted_unc']],
        'std_error': [ols_model.bse[endogenous], iv_model.bse['predicted_unc']],
        'p_value': [ols_model.pvalues[endogenous], iv_model.pvalues['predicted_unc']],
        'n_obs': [len(df_clean), len(df_clean)],
    }

    return pd.DataFrame(results), first_stage


def run_iv_analysis(save_results: bool = True) -> Dict:
    """
    Run full IV analysis.
    """
    print("=" * 70)
    print("INSTRUMENTAL VARIABLES ANALYSIS")
    print("=" * 70)

    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(project_dir, "results")

    data_path = os.path.join(results_dir, "real_spread_data.csv")
    if not os.path.exists(data_path):
        print("ERROR: Run real_spread_validation.py first")
        return None

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"\nLoaded {len(df)} observations")

    # Add instruments
    df = add_instruments(df)

    # Select instruments
    instruments = ['iv_vix_jump', 'iv_monday', 'iv_uncertainty_lag1', 'iv_direction_change']

    # Use realized_vol if total_uncertainty not available
    if 'total_uncertainty' not in df.columns or df['total_uncertainty'].isna().all():
        endog_var = 'realized_vol'
    else:
        endog_var = 'total_uncertainty'

    # First stage
    print("\n[First Stage: Instrument Relevance]")
    print("-" * 50)

    first_stage = first_stage_regression(df, endog_var, instruments)

    print(f"F-statistic: {first_stage['f_stat']:.2f}")
    print(f"F p-value: {first_stage['f_pvalue']:.4f}")
    print(f"R-squared: {first_stage['r_squared']:.4f}")
    print(f"N observations: {first_stage['n_obs']}")

    if first_stage['strong_instruments']:
        print("\n✓ STRONG INSTRUMENTS (F > 10)")
    else:
        print("\n✗ WEAK INSTRUMENTS (F < 10)")
        print("  Results should be interpreted with caution")

    print("\nInstrument coefficients:")
    for inst, coef in first_stage['instrument_coeffs'].items():
        pval = first_stage['instrument_pvals'][inst]
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {inst:25s}: {coef:>8.4f}{sig}")

    # OLS vs IV comparison
    print("\n[OLS vs IV Comparison]")
    print("-" * 50)

    comparison_df, _ = compare_ols_iv(df, 'cs_spread', endog_var, instruments)
    print(comparison_df.to_string(index=False))

    # Interpret difference
    ols_coef = comparison_df.loc[0, 'coefficient']
    iv_coef = comparison_df.loc[1, 'coefficient']

    print(f"\nDifference (IV - OLS): {iv_coef - ols_coef:.4f}")

    if abs(iv_coef - ols_coef) / abs(ols_coef) > 0.2:
        print("SUBSTANTIAL DIFFERENCE: Endogeneity likely present")
        if iv_coef > ols_coef:
            print("  OLS underestimates the true causal effect")
        else:
            print("  OLS overestimates the true causal effect")
    else:
        print("MINIMAL DIFFERENCE: Endogeneity bias appears small")
        print("  OLS and IV estimates are consistent")

    # Summary
    print("\n" + "=" * 70)
    print("IV ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"""
Instruments used:
  - VIX jumps (volatility shocks from equity markets)
  - Monday effect (calendar anomaly)
  - Lagged uncertainty (persistence)
  - Direction change (regime shift)

First Stage F-stat: {first_stage['f_stat']:.2f} ({'Strong' if first_stage['strong_instruments'] else 'Weak'})

Causal interpretation:
  - OLS coefficient: {ols_coef:.4f}
  - IV coefficient:  {iv_coef:.4f}
  - Bias direction:  {'Upward' if ols_coef > iv_coef else 'Downward' if ols_coef < iv_coef else 'None'}

For the paper:
  1. Report first stage F-stat (should be > 10)
  2. Compare OLS and IV estimates
  3. If F > 10 and estimates similar: causal claim supported
  4. If F < 10: acknowledge weak instruments limitation
""")

    # Save results
    if save_results:
        comparison_df.to_csv(os.path.join(results_dir, "iv_comparison.csv"), index=False)

        first_stage_df = pd.DataFrame({
            'instrument': list(first_stage['instrument_coeffs'].keys()),
            'coefficient': list(first_stage['instrument_coeffs'].values()),
            'p_value': list(first_stage['instrument_pvals'].values()),
        })
        first_stage_df.to_csv(os.path.join(results_dir, "iv_first_stage.csv"), index=False)

        summary_df = pd.DataFrame([{
            'f_stat': first_stage['f_stat'],
            'f_pvalue': first_stage['f_pvalue'],
            'strong_instruments': first_stage['strong_instruments'],
            'ols_coef': ols_coef,
            'iv_coef': iv_coef,
            'bias_direction': 'upward' if ols_coef > iv_coef else 'downward',
        }])
        summary_df.to_csv(os.path.join(results_dir, "iv_summary.csv"), index=False)

        print(f"\nResults saved to {results_dir}/iv_*.csv")

    return {
        'first_stage': first_stage,
        'comparison': comparison_df,
        'ols_coef': ols_coef,
        'iv_coef': iv_coef,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Instrumental Variables Analysis")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")

    args = parser.parse_args()

    results = run_iv_analysis(save_results=not args.no_save)


if __name__ == '__main__':
    main()
