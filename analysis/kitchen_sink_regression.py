#!/usr/bin/env python3
"""
Kitchen Sink Regression Table

Addresses reviewer request for comprehensive regression:
spread ~ regime_dummies + RV + RV² + |returns| + log_volume + day_FE + month_FE

Outputs:
- results/kitchen_sink_regression.csv (coefficients)
- results/kitchen_sink_regression.tex (LaTeX table)

Author: Murad Farzulla
Date: January 2026
"""

import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from scipy import stats

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
DATA_FILE = os.path.join(RESULTS_DIR, 'full_sample_btc_data.csv')

os.makedirs(RESULTS_DIR, exist_ok=True)


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
    # neutral is baseline (46-55)

    # Continuous extremity measure (distance from neutral=50, normalized 0-1)
    df['distance_from_neutral'] = np.abs(df['fear_greed_value'] - 50) / 50

    # Extreme dummy (any extreme regime)
    df['is_extreme'] = ((df['fear_greed_value'] <= 25) | (df['fear_greed_value'] > 75)).astype(int)

    # Volatility and controls
    df['rv'] = df['parkinson_vol']  # Realized volatility proxy
    df['rv_sq'] = df['rv'] ** 2  # RV squared for nonlinearity
    df['abs_returns'] = np.abs(df['returns'])
    df['log_vol'] = df['log_volume']

    # Intraday range (high-low as fraction of close)
    # Already have this implicitly in parkinson_vol, but let's be explicit
    # df['range'] = (df['high'] - df['low']) / df['close']  # if needed

    # Day of week dummies
    df['dow'] = df['date'].dt.dayofweek
    for i in range(6):  # Mon=0 through Sat=5, Sun=6 is baseline
        df[f'dow_{i}'] = (df['dow'] == i).astype(int)

    # Month dummies
    df['month'] = df['date'].dt.month
    for i in range(1, 12):  # Jan=1 through Nov=11, Dec=12 is baseline
        df[f'month_{i}'] = (df['month'] == i).astype(int)

    # Year fixed effects (for extended sample spanning multiple years)
    df['year'] = df['date'].dt.year
    years = sorted(df['year'].unique())
    for y in years[:-1]:  # Last year is baseline
        df[f'year_{y}'] = (df['year'] == y).astype(int)

    print(f"  Loaded {len(df)} observations")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Regime distribution:")
    print(f"    Extreme Fear: {df['extreme_fear'].sum()}")
    print(f"    Fear: {df['fear'].sum()}")
    print(f"    Neutral: {len(df) - df['extreme_fear'].sum() - df['fear'].sum() - df['greed'].sum() - df['extreme_greed'].sum()}")
    print(f"    Greed: {df['greed'].sum()}")
    print(f"    Extreme Greed: {df['extreme_greed'].sum()}")

    return df


def run_regression(df, y_col, x_cols, model_name, use_hac=True):
    """Run OLS regression with HAC standard errors."""
    y = df[y_col].values
    X = df[x_cols].values
    X = sm.add_constant(X)

    model = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 10} if use_hac else None)

    # Extract results
    results = {
        'model': model_name,
        'n_obs': int(model.nobs),
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'f_stat': model.fvalue,
        'f_pvalue': model.f_pvalue,
    }

    # Coefficients
    coef_names = ['const'] + x_cols
    for i, name in enumerate(coef_names):
        results[f'coef_{name}'] = model.params[i]
        results[f'se_{name}'] = model.bse[i]
        results[f'pval_{name}'] = model.pvalues[i]

    return results, model


def format_coef(coef, se, pval, decimals=3):
    """Format coefficient with significance stars."""
    stars = ''
    if pval < 0.001:
        stars = '***'
    elif pval < 0.01:
        stars = '**'
    elif pval < 0.05:
        stars = '*'
    elif pval < 0.1:
        stars = '†'

    return f"{coef:.{decimals}f}{stars}", f"({se:.{decimals}f})"


def main():
    print("=" * 60)
    print("KITCHEN SINK REGRESSION TABLE")
    print("=" * 60)

    df = load_and_prepare_data()

    # Define dependent variable (spread in basis points for interpretability)
    # CS spread is already in price units, convert to bps relative to price
    # Actually, from the data, cs_spread looks like it's already computed
    # Let's use it as-is but note units
    df['spread_bps'] = df['cs_spread']  # Already in appropriate units

    # Remove rows with zero/missing spreads for cleaner regression
    df_reg = df[df['spread_bps'] > 0].copy()
    print(f"\nUsing {len(df_reg)} observations with positive spreads")
    print(f"  Dropped {len(df) - len(df_reg)} zero/negative spread days")

    # =========================================================================
    # Model Specifications
    # =========================================================================

    results_list = []

    # Model 1: Baseline - Just regime dummies
    print("\n--- Model 1: Regime Dummies Only ---")
    x_cols_1 = ['extreme_fear', 'fear', 'greed', 'extreme_greed']
    res1, mod1 = run_regression(df_reg, 'spread_bps', x_cols_1, 'M1: Regimes Only')
    results_list.append(res1)
    print(f"  R² = {res1['r_squared']:.4f}")

    # Model 2: + Volatility controls
    print("\n--- Model 2: + Volatility ---")
    x_cols_2 = ['extreme_fear', 'fear', 'greed', 'extreme_greed', 'rv', 'rv_sq']
    res2, mod2 = run_regression(df_reg, 'spread_bps', x_cols_2, 'M2: + Volatility')
    results_list.append(res2)
    print(f"  R² = {res2['r_squared']:.4f}, ΔR² = {res2['r_squared'] - res1['r_squared']:.4f}")

    # Model 3: + Returns and Volume
    print("\n--- Model 3: + Returns & Volume ---")
    x_cols_3 = ['extreme_fear', 'fear', 'greed', 'extreme_greed', 'rv', 'rv_sq',
                'abs_returns', 'log_vol']
    res3, mod3 = run_regression(df_reg, 'spread_bps', x_cols_3, 'M3: + Returns/Volume')
    results_list.append(res3)
    print(f"  R² = {res3['r_squared']:.4f}, ΔR² = {res3['r_squared'] - res2['r_squared']:.4f}")

    # Model 4: + Day-of-Week FE
    print("\n--- Model 4: + Day FE ---")
    dow_cols = [f'dow_{i}' for i in range(6)]
    x_cols_4 = ['extreme_fear', 'fear', 'greed', 'extreme_greed', 'rv', 'rv_sq',
                'abs_returns', 'log_vol'] + dow_cols
    res4, mod4 = run_regression(df_reg, 'spread_bps', x_cols_4, 'M4: + Day FE')
    results_list.append(res4)
    print(f"  R² = {res4['r_squared']:.4f}, ΔR² = {res4['r_squared'] - res3['r_squared']:.4f}")

    # Model 5: Full Kitchen Sink (+ Month FE + Year FE)
    print("\n--- Model 5: Full Kitchen Sink ---")
    month_cols = [f'month_{i}' for i in range(1, 12)]
    year_cols = [f'year_{y}' for y in sorted(df_reg['year'].unique())[:-1]]
    x_cols_5 = ['extreme_fear', 'fear', 'greed', 'extreme_greed', 'rv', 'rv_sq',
                'abs_returns', 'log_vol'] + dow_cols + month_cols + year_cols
    res5, mod5 = run_regression(df_reg, 'spread_bps', x_cols_5, 'M5: Kitchen Sink')
    results_list.append(res5)
    print(f"  R² = {res5['r_squared']:.4f}, ΔR² = {res5['r_squared'] - res4['r_squared']:.4f}")

    # Model 6: Continuous Extremity (alternative spec)
    print("\n--- Model 6: Continuous Extremity ---")
    x_cols_6 = ['distance_from_neutral', 'rv', 'rv_sq', 'abs_returns', 'log_vol'] + dow_cols
    res6, mod6 = run_regression(df_reg, 'spread_bps', x_cols_6, 'M6: Continuous')
    results_list.append(res6)
    print(f"  R² = {res6['r_squared']:.4f}")

    # Model 7: Extreme vs Non-Extreme (binary)
    print("\n--- Model 7: Binary Extreme ---")
    x_cols_7 = ['is_extreme', 'rv', 'rv_sq', 'abs_returns', 'log_vol'] + dow_cols
    res7, mod7 = run_regression(df_reg, 'spread_bps', x_cols_7, 'M7: Binary Extreme')
    results_list.append(res7)
    print(f"  R² = {res7['r_squared']:.4f}")

    # =========================================================================
    # Save Results
    # =========================================================================

    # CSV output
    results_df = pd.DataFrame(results_list)
    csv_path = os.path.join(RESULTS_DIR, 'kitchen_sink_regression.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # =========================================================================
    # Generate LaTeX Table
    # =========================================================================

    print("\n" + "=" * 60)
    print("GENERATING LATEX TABLE")
    print("=" * 60)

    # Key variables to show in main table
    key_vars = ['extreme_fear', 'fear', 'greed', 'extreme_greed',
                'distance_from_neutral', 'is_extreme',
                'rv', 'rv_sq', 'abs_returns', 'log_vol']

    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Comprehensive Regression: Spread Determinants}")
    latex.append(r"\label{tab:kitchen_sink}")
    latex.append(r"\small")
    latex.append(r"\begin{tabular}{@{}lccccccc@{}}")
    latex.append(r"\toprule")
    latex.append(r"& \textbf{(1)} & \textbf{(2)} & \textbf{(3)} & \textbf{(4)} & \textbf{(5)} & \textbf{(6)} & \textbf{(7)} \\")
    latex.append(r"& Regimes & +Vol & +Ret/Vol & +Day FE & Kitchen & Continuous & Binary \\")
    latex.append(r"\midrule")

    # Regime dummies section
    latex.append(r"\multicolumn{8}{l}{\textit{Sentiment Regime (Neutral = baseline)}} \\")

    for var in ['extreme_fear', 'fear', 'greed', 'extreme_greed']:
        var_label = var.replace('_', ' ').title()
        row = f"{var_label}"
        for i, res in enumerate(results_list):
            coef_key = f'coef_{var}'
            se_key = f'se_{var}'
            pval_key = f'pval_{var}'
            if coef_key in res:
                coef, se = format_coef(res[coef_key], res[se_key], res[pval_key])
                row += f" & {coef}"
            else:
                row += " & ---"
        row += r" \\"
        latex.append(row)

        # SE row
        se_row = ""
        for i, res in enumerate(results_list):
            se_key = f'se_{var}'
            if se_key in res:
                se_row += f" & ({res[se_key]:.3f})"
            else:
                se_row += " &"
        se_row += r" \\"
        latex.append(se_row)

    latex.append(r"\addlinespace")
    latex.append(r"\multicolumn{8}{l}{\textit{Alternative Extremity Measures}} \\")

    # Continuous distance
    var = 'distance_from_neutral'
    row = "Distance from Neutral"
    for i, res in enumerate(results_list):
        coef_key = f'coef_{var}'
        if coef_key in res:
            coef, se = format_coef(res[coef_key], res[f'se_{var}'], res[f'pval_{var}'])
            row += f" & {coef}"
        else:
            row += " & ---"
    row += r" \\"
    latex.append(row)
    se_row = ""
    for i, res in enumerate(results_list):
        se_key = f'se_{var}'
        if se_key in res:
            se_row += f" & ({res[se_key]:.3f})"
        else:
            se_row += " &"
    se_row += r" \\"
    latex.append(se_row)

    # Binary extreme
    var = 'is_extreme'
    row = "Extreme (binary)"
    for i, res in enumerate(results_list):
        coef_key = f'coef_{var}'
        if coef_key in res:
            coef, se = format_coef(res[coef_key], res[f'se_{var}'], res[f'pval_{var}'])
            row += f" & {coef}"
        else:
            row += " & ---"
    row += r" \\"
    latex.append(row)
    se_row = ""
    for i, res in enumerate(results_list):
        se_key = f'se_{var}'
        if se_key in res:
            se_row += f" & ({res[se_key]:.3f})"
        else:
            se_row += " &"
    se_row += r" \\"
    latex.append(se_row)

    latex.append(r"\addlinespace")
    latex.append(r"\multicolumn{8}{l}{\textit{Volatility Controls}} \\")

    # Volatility
    for var, label in [('rv', 'Realized Volatility'), ('rv_sq', 'RV$^2$')]:
        row = label
        for i, res in enumerate(results_list):
            coef_key = f'coef_{var}'
            if coef_key in res:
                coef, se = format_coef(res[coef_key], res[f'se_{var}'], res[f'pval_{var}'])
                row += f" & {coef}"
            else:
                row += " & ---"
        row += r" \\"
        latex.append(row)
        se_row = ""
        for i, res in enumerate(results_list):
            se_key = f'se_{var}'
            if se_key in res:
                se_row += f" & ({res[se_key]:.3f})"
            else:
                se_row += " &"
        se_row += r" \\"
        latex.append(se_row)

    latex.append(r"\addlinespace")
    latex.append(r"\multicolumn{8}{l}{\textit{Additional Controls}} \\")

    for var, label in [('abs_returns', '$|$Returns$|$'), ('log_vol', 'Log(Volume)')]:
        row = label
        for i, res in enumerate(results_list):
            coef_key = f'coef_{var}'
            if coef_key in res:
                coef, se = format_coef(res[coef_key], res[f'se_{var}'], res[f'pval_{var}'])
                row += f" & {coef}"
            else:
                row += " & ---"
        row += r" \\"
        latex.append(row)
        se_row = ""
        for i, res in enumerate(results_list):
            se_key = f'se_{var}'
            if se_key in res:
                se_row += f" & ({res[se_key]:.3f})"
            else:
                se_row += " &"
        se_row += r" \\"
        latex.append(se_row)

    latex.append(r"\midrule")
    latex.append(r"\multicolumn{8}{l}{\textit{Fixed Effects}} \\")
    latex.append(r"Day-of-Week FE & No & No & No & Yes & Yes & Yes & Yes \\")
    latex.append(r"Month FE & No & No & No & No & Yes & No & No \\")
    latex.append(r"Year FE & No & No & No & No & Yes & No & No \\")

    latex.append(r"\midrule")
    latex.append(r"\multicolumn{8}{l}{\textit{Model Statistics}} \\")

    # R-squared
    row = "$R^2$"
    for res in results_list:
        row += f" & {res['r_squared']:.3f}"
    row += r" \\"
    latex.append(row)

    # Adj R-squared
    row = "Adj. $R^2$"
    for res in results_list:
        row += f" & {res['adj_r_squared']:.3f}"
    row += r" \\"
    latex.append(row)

    # N
    row = "$N$"
    for res in results_list:
        row += f" & {res['n_obs']:,}"
    row += r" \\"
    latex.append(row)

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\begin{tablenotes}")
    latex.append(r"\small")
    latex.append(r"\item \textit{Notes:} Dependent variable is Corwin-Schultz spread proxy. HAC (Newey-West) standard errors in parentheses with 10 lags. Neutral sentiment regime (F\&G 46--55) is the omitted baseline. $^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$, $^{\dagger}p<0.10$.")
    latex.append(r"\end{tablenotes}")
    latex.append(r"\end{table}")

    latex_str = '\n'.join(latex)

    tex_path = os.path.join(RESULTS_DIR, 'kitchen_sink_regression.tex')
    with open(tex_path, 'w') as f:
        f.write(latex_str)
    print(f"Saved: {tex_path}")

    # =========================================================================
    # Print Summary
    # =========================================================================

    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    # Model 5 (kitchen sink) results for regime dummies
    print("\nModel 5 (Kitchen Sink) - Regime Coefficients:")
    for var in ['extreme_fear', 'fear', 'greed', 'extreme_greed']:
        coef = res5.get(f'coef_{var}', 'N/A')
        se = res5.get(f'se_{var}', 'N/A')
        pval = res5.get(f'pval_{var}', 'N/A')
        if isinstance(coef, float):
            stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            print(f"  {var}: {coef:.3f} ({se:.3f}){stars}")

    # Coefficient stability check
    print("\nCoefficient Stability (Extreme Fear across models):")
    for i, res in enumerate(results_list[:5], 1):
        coef = res.get('coef_extreme_fear', 'N/A')
        if isinstance(coef, float):
            print(f"  Model {i}: {coef:.3f}")

    # Incremental R²
    print("\nIncremental R² from Adding Controls:")
    print(f"  Regimes only: {res1['r_squared']:.4f}")
    print(f"  + Volatility: +{res2['r_squared'] - res1['r_squared']:.4f}")
    print(f"  + Ret/Volume: +{res3['r_squared'] - res2['r_squared']:.4f}")
    print(f"  + Day FE:     +{res4['r_squared'] - res3['r_squared']:.4f}")
    print(f"  + Month/Year: +{res5['r_squared'] - res4['r_squared']:.4f}")
    print(f"  Final R²:     {res5['r_squared']:.4f}")

    # F-test for regime dummies (joint significance)
    print("\nJoint F-test for Regime Dummies in Kitchen Sink:")
    # Extract from model
    regime_indices = [1, 2, 3, 4]  # const=0, then regime dummies
    r_matrix = np.zeros((4, len(mod5.params)))
    for i, idx in enumerate(regime_indices):
        r_matrix[i, idx] = 1
    try:
        f_test = mod5.f_test(r_matrix)
        print(f"  F({int(f_test.df_num)}, {int(f_test.df_denom)}) = {f_test.fvalue[0][0]:.2f}, p = {f_test.pvalue:.4f}")
    except:
        print("  (F-test computation failed)")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
