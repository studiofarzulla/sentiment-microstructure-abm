"""
Threshold Sensitivity Analysis for Extremity Premium

Tests whether the extremity premium is robust to different
threshold definitions for "extreme" sentiment regimes.

Baseline: 25/75 thresholds (current paper)
Alternatives: 20/80 (stricter), 15/85 (very strict), 30/70 (looser)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

def classify_regime_with_threshold(fg_value, extreme_low, extreme_high):
    """Classify regime with custom extreme thresholds."""
    if pd.isna(fg_value):
        return 'unknown'
    elif fg_value <= extreme_low:
        return 'extreme_fear'
    elif fg_value <= 45:
        return 'fear'
    elif fg_value <= 55:
        return 'neutral'
    elif fg_value <= extreme_high:
        return 'greed'
    else:
        return 'extreme_greed'

def run_regression_for_threshold(df, extreme_low, extreme_high, control_vol=False):
    """Run the regression for a given threshold.

    Args:
        control_vol: If True, include volatility control. If False, just test
                     regime effects on uncertainty (appropriate when uncertainty
                     IS volatility, as in ETH analysis).
    """
    # Reclassify regimes
    df = df.copy()
    df['regime'] = df['fear_greed_value'].apply(
        lambda x: classify_regime_with_threshold(x, extreme_low, extreme_high)
    )

    # Filter valid data
    required_cols = ['total_uncertainty', 'regime']
    if control_vol:
        required_cols.append('volatility')
    df_valid = df.dropna(subset=required_cols)
    df_valid = df_valid[df_valid['regime'] != 'unknown']

    # Create dummies (neutral = baseline)
    for reg in ['extreme_fear', 'fear', 'greed', 'extreme_greed']:
        df_valid[f'is_{reg}'] = (df_valid['regime'] == reg).astype(int)

    # Run regression - with or without volatility control
    if control_vol:
        X = df_valid[['volatility', 'is_extreme_fear', 'is_fear', 'is_greed', 'is_extreme_greed']]
    else:
        X = df_valid[['is_extreme_fear', 'is_fear', 'is_greed', 'is_extreme_greed']]
    X = sm.add_constant(X)
    y = df_valid['total_uncertainty']

    model = sm.OLS(y, X).fit(cov_type='HC3')

    # Extract coefficients
    results = {
        'extreme_low': extreme_low,
        'extreme_high': extreme_high,
        'threshold_label': f'{extreme_low}/{extreme_high}',
        'n_extreme_fear': (df_valid['regime'] == 'extreme_fear').sum(),
        'n_extreme_greed': (df_valid['regime'] == 'extreme_greed').sum(),
        'n_neutral': (df_valid['regime'] == 'neutral').sum(),
        'n_total': len(df_valid),
        'r_squared': model.rsquared,
    }

    for var in ['is_extreme_fear', 'is_extreme_greed']:
        regime_name = var.replace('is_', '')
        if var in model.params:
            results[f'{regime_name}_coef'] = model.params[var]
            results[f'{regime_name}_se'] = model.bse[var]
            results[f'{regime_name}_pval'] = model.pvalues[var]
            results[f'{regime_name}_sig'] = model.pvalues[var] < 0.05

    return results

def test_threshold_sensitivity(data_path: str = None):
    """Test extremity premium across different thresholds."""

    base_dir = Path(__file__).parent.parent

    # Use ETH data which has fear_greed_value column
    if data_path is None:
        data_path = base_dir / 'results' / 'eth_spread_data.csv'

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} observations")

    # Ensure we have the required columns
    if 'fear_greed_value' not in df.columns:
        raise ValueError("Data must contain 'fear_greed_value' column")

    # Use parkinson_vol as uncertainty proxy (total_uncertainty may be constant)
    # For ETH, we use Parkinson volatility as the dependent variable
    if 'parkinson_vol' in df.columns:
        df['total_uncertainty'] = df['parkinson_vol']
        df['volatility'] = df['parkinson_vol']  # Use same for control (testing regime effect)
    elif 'volatility' in df.columns:
        df['total_uncertainty'] = df['volatility']
    else:
        raise ValueError("No volatility column found")

    print(f"  F&G range: {df['fear_greed_value'].min():.0f} - {df['fear_greed_value'].max():.0f}")

    # Define threshold configurations to test
    thresholds = [
        (15, 85),   # Very strict
        (20, 80),   # Strict
        (25, 75),   # Baseline (current)
        (30, 70),   # Loose
    ]

    results = []
    for low, high in thresholds:
        print(f"\nTesting threshold: {low}/{high}...")
        # Don't control for volatility when volatility IS the DV (ETH case)
        result = run_regression_for_threshold(df, low, high, control_vol=False)
        results.append(result)
        print(f"  Extreme fear: n={result['n_extreme_fear']}, coef={result.get('extreme_fear_coef', 'N/A'):.4f}")
        print(f"  Extreme greed: n={result['n_extreme_greed']}, coef={result.get('extreme_greed_coef', 'N/A'):.4f}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Check if premium is preserved
    results_df['premium_preserved'] = (
        (results_df['extreme_fear_sig'] | results_df['extreme_greed_sig']) &
        (results_df['extreme_fear_coef'] > 0) &
        (results_df['extreme_greed_coef'] > 0)
    )

    print("\n" + "="*70)
    print("THRESHOLD SENSITIVITY SUMMARY")
    print("="*70)

    print("\n{:<12} {:>8} {:>8} {:>12} {:>12} {:>10}".format(
        "Threshold", "N_ExtF", "N_ExtG", "ExtF_Coef", "ExtG_Coef", "Premium?"
    ))
    print("-"*70)

    for _, row in results_df.iterrows():
        sig_f = "***" if row.get('extreme_fear_pval', 1) < 0.001 else "**" if row.get('extreme_fear_pval', 1) < 0.01 else "*" if row.get('extreme_fear_pval', 1) < 0.05 else ""
        sig_g = "***" if row.get('extreme_greed_pval', 1) < 0.001 else "**" if row.get('extreme_greed_pval', 1) < 0.01 else "*" if row.get('extreme_greed_pval', 1) < 0.05 else ""

        coef_f = f"{row.get('extreme_fear_coef', 0):+.4f}{sig_f}"
        coef_g = f"{row.get('extreme_greed_coef', 0):+.4f}{sig_g}"

        print("{:<12} {:>8} {:>8} {:>12} {:>12} {:>10}".format(
            row['threshold_label'],
            row['n_extreme_fear'],
            row['n_extreme_greed'],
            coef_f,
            coef_g,
            "Yes" if row['premium_preserved'] else "No"
        ))

    # Save results
    output_path = Path(__file__).parent.parent / 'results' / 'threshold_sensitivity.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    return results_df

if __name__ == '__main__':
    results = test_threshold_sensitivity()
