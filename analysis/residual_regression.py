#!/usr/bin/env python3
"""
Residual-on-Residual Regression Analysis

Addresses AI reviewer critique #4: "The correlation between CS spreads and uncertainty
may be mechanical since both contain volatility."

Method:
1. Regress CS spreads on realized volatility → get spread residuals
2. Regress uncertainty index on realized volatility → get uncertainty residuals
3. Test if spread residuals correlate with uncertainty residuals

Also addresses critique #6: Add trading volume control.

Author: Murad Farzulla
Date: January 2026
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load the empirical data."""
    df = pd.read_csv('results/real_spread_data.csv', parse_dates=['date'])
    df_sentiment = pd.read_csv('data/datasets/btc_sentiment_daily.csv', parse_dates=['date'])

    df = pd.merge(df, df_sentiment[['date', 'regime', 'fear_greed_value']],
                  on='date', how='inner')

    # Compute volatility if not present
    if 'realized_vol' not in df.columns:
        df['realized_vol'] = df['parkinson_vol']

    df['volatility'] = df['realized_vol'].fillna(df['parkinson_vol'])

    return df.dropna(subset=['cs_spread', 'total_uncertainty', 'volatility']).copy()


def residual_regression(df):
    """
    Run residual-on-residual regression to isolate non-volatility uncertainty channel.

    Method:
    1. spread_resid = CS_spread - β₀ - β₁*volatility
    2. unc_resid = total_uncertainty - γ₀ - γ₁*volatility
    3. Test: corr(spread_resid, unc_resid)
    """
    print("="*70)
    print("RESIDUAL-ON-RESIDUAL REGRESSION")
    print("Isolating uncertainty channel from mechanical volatility")
    print("="*70)

    # Step 1: Regress spreads on volatility
    X_vol = sm.add_constant(df['volatility'])

    model_spread = sm.OLS(df['cs_spread'], X_vol).fit()
    spread_resid = model_spread.resid

    print(f"\nStep 1: Regress CS spreads on realized volatility")
    print(f"  R² = {model_spread.rsquared:.4f}")
    print(f"  Vol coefficient = {model_spread.params['volatility']:.6f}")
    print(f"  Vol t-stat = {model_spread.tvalues['volatility']:.2f}")

    # Step 2: Regress uncertainty on volatility
    model_unc = sm.OLS(df['total_uncertainty'], X_vol).fit()
    unc_resid = model_unc.resid

    print(f"\nStep 2: Regress uncertainty on realized volatility")
    print(f"  R² = {model_unc.rsquared:.4f}")
    print(f"  Vol coefficient = {model_unc.params['volatility']:.6f}")
    print(f"  Vol t-stat = {model_unc.tvalues['volatility']:.2f}")

    # Step 3: Correlate residuals
    r_resid, p_resid = stats.pearsonr(spread_resid, unc_resid)

    print(f"\nStep 3: Correlation of residuals (volatility-purged)")
    print(f"  r = {r_resid:.4f}")
    print(f"  p = {p_resid:.6f}")

    # Compare to raw correlation
    r_raw, p_raw = stats.pearsonr(df['cs_spread'], df['total_uncertainty'])

    print(f"\n★ Comparison:")
    print(f"  Raw correlation:      r = {r_raw:.4f} (p = {p_raw:.6f})")
    print(f"  Residual correlation: r = {r_resid:.4f} (p = {p_resid:.6f})")
    print(f"  Reduction: {100*(1 - abs(r_resid)/abs(r_raw)):.1f}%")

    if p_resid < 0.05:
        print(f"\n✓ Residual correlation SIGNIFICANT at p < 0.05")
        print(f"  Non-volatility uncertainty channel EXISTS")
    else:
        print(f"\n✗ Residual correlation NOT significant")
        print(f"  Uncertainty-spread relationship may be purely mechanical")

    results = {
        'raw_r': r_raw,
        'raw_p': p_raw,
        'residual_r': r_resid,
        'residual_p': p_resid,
        'spread_vol_r2': model_spread.rsquared,
        'unc_vol_r2': model_unc.rsquared,
    }

    # Also check regime effects on uncertainty residuals
    print(f"\n" + "-"*70)
    print("REGIME EFFECTS ON VOLATILITY-PURGED UNCERTAINTY")
    print("-"*70)

    df['unc_resid'] = unc_resid

    regime_means = df.groupby('regime')['unc_resid'].agg(['mean', 'std', 'count'])
    regime_means = regime_means.sort_values('mean', ascending=False)

    neutral_mean = regime_means.loc['neutral', 'mean'] if 'neutral' in regime_means.index else 0

    print(f"\n{'Regime':<15} {'Mean Resid':>12} {'Gap vs Neutral':>15}")
    print("-"*45)
    for regime in regime_means.index:
        mean = regime_means.loc[regime, 'mean']
        gap = mean - neutral_mean
        print(f"{regime:<15} {mean:>+12.4f} {gap:>+15.4f}")

    # Test extreme vs neutral on residuals
    if 'extreme_greed' in df['regime'].values and 'neutral' in df['regime'].values:
        extreme_greed_resid = df[df['regime'] == 'extreme_greed']['unc_resid']
        neutral_resid = df[df['regime'] == 'neutral']['unc_resid']
        t_eg, p_eg = stats.ttest_ind(extreme_greed_resid, neutral_resid)
        print(f"\n★ Extreme greed vs neutral (on residuals):")
        print(f"  t = {t_eg:.3f}, p = {p_eg:.4f}")

    if 'extreme_fear' in df['regime'].values and 'neutral' in df['regime'].values:
        extreme_fear_resid = df[df['regime'] == 'extreme_fear']['unc_resid']
        neutral_resid = df[df['regime'] == 'neutral']['unc_resid']
        t_ef, p_ef = stats.ttest_ind(extreme_fear_resid, neutral_resid)
        print(f"★ Extreme fear vs neutral (on residuals):")
        print(f"  t = {t_ef:.3f}, p = {p_ef:.4f}")

    return results, spread_resid, unc_resid


def volume_controlled_regression(df):
    """
    Add trading volume to regime regression.

    Addresses critique #6: "Extreme sentiment correlates with volume,
    volume affects spreads."
    """
    print("\n" + "="*70)
    print("VOLUME-CONTROLLED REGIME REGRESSION")
    print("="*70)

    # Check if volume exists
    if 'volume' not in df.columns:
        print("Warning: Volume not in dataset, computing from raw data...")
        # Try to compute from returns or use a proxy
        if 'quote_volume' in df.columns:
            df['volume'] = df['quote_volume']
        else:
            print("No volume data available - skipping volume control")
            return None

    # Log-transform volume
    df['log_volume'] = np.log(df['volume'] + 1)

    # Create regime dummies
    df_clean = df.dropna(subset=['cs_spread', 'total_uncertainty', 'volatility', 'log_volume', 'regime'])

    regime_dummies = pd.get_dummies(df_clean['regime'], prefix='regime', drop_first=True)

    # Ensure all numeric
    regime_dummies = regime_dummies.astype(float)

    # Model 1: Without volume
    X1 = sm.add_constant(pd.concat([
        df_clean[['volatility']].astype(float),
        regime_dummies
    ], axis=1))

    model1 = sm.OLS(df_clean['total_uncertainty'].astype(float), X1).fit(cov_type='HC1')

    print("\nModel 1: Uncertainty ~ Volatility + Regime Dummies")
    print(f"  R² = {model1.rsquared:.4f}")

    # Model 2: With volume
    X2 = sm.add_constant(pd.concat([
        df_clean[['volatility', 'log_volume']].astype(float),
        regime_dummies
    ], axis=1))

    model2 = sm.OLS(df_clean['total_uncertainty'].astype(float), X2).fit(cov_type='HC1')

    print("\nModel 2: Uncertainty ~ Volatility + Log(Volume) + Regime Dummies")
    print(f"  R² = {model2.rsquared:.4f}")
    print(f"  Volume coefficient = {model2.params['log_volume']:.6f}")
    print(f"  Volume t-stat = {model2.tvalues['log_volume']:.2f}")
    print(f"  Volume p-value = {model2.pvalues['log_volume']:.4f}")

    # Compare regime coefficients
    print("\n★ Regime coefficients comparison (vs neutral baseline):")
    print(f"{'Regime':<20} {'Without Vol':<15} {'With Vol':<15} {'Change':}")

    regime_results = []
    for col in regime_dummies.columns:
        regime_name = col.replace('regime_', '')
        coef1 = model1.params[col]
        coef2 = model2.params[col]
        pval2 = model2.pvalues[col]
        change = 100 * (coef2 - coef1) / abs(coef1) if coef1 != 0 else 0

        print(f"{regime_name:<20} {coef1:>+.4f}       {coef2:>+.4f}       {change:>+.1f}%")

        regime_results.append({
            'regime': regime_name,
            'coef_no_volume': coef1,
            'coef_with_volume': coef2,
            'pvalue_with_volume': pval2,
            'change_pct': change
        })

    return pd.DataFrame(regime_results)


def main():
    print("Loading data...")
    df = load_data()
    print(f"Dataset: {len(df)} observations")

    # Run residual regression
    residual_results, spread_resid, unc_resid = residual_regression(df)

    # Save residual results
    pd.DataFrame([residual_results]).to_csv('results/residual_regression_results.csv', index=False)

    # Run volume-controlled regression if volume available
    volume_results = volume_controlled_regression(df)

    if volume_results is not None:
        volume_results.to_csv('results/volume_controlled_regime_results.csv', index=False)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n✓ Results saved to results/residual_regression_results.csv")
    if volume_results is not None:
        print(f"✓ Results saved to results/volume_controlled_regime_results.csv")

    return residual_results


if __name__ == "__main__":
    main()
