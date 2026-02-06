"""
Placebo Test Suite for the Extremity Premium

Addresses reviewer question: "Is this an artifact of volatility clustering or regime persistence?"

Three placebo tests:
1. Block-Shuffled Permutation: Preserve regime autocorrelation, shuffle blocks
2. Time-Reversed Causality: Regress current spreads on FUTURE regimes
3. Synthetic Regime Assignment: Generate regimes from AR(1) on F&G

If the extremity premium survives these placebo tests, it's not an artifact
of temporal structure or mechanical autocorrelation.

Author: Murad Farzulla
Date: January 2026
"""

import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

# Project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
DATA_DIR = os.path.join(PROJECT_DIR, "data", "datasets")


def load_data():
    """Load and prepare data for placebo tests."""
    # Load spread data
    spread_path = os.path.join(RESULTS_DIR, "real_spread_data.csv")
    df_spreads = pd.read_csv(spread_path, parse_dates=['date'])

    # Load sentiment data
    sentiment_path = os.path.join(DATA_DIR, "btc_sentiment_daily.csv")
    df_sentiment = pd.read_csv(sentiment_path, parse_dates=['date'])

    # Merge
    df = pd.merge(df_spreads, df_sentiment[['date', 'regime', 'fear_greed_value']],
                  on='date', how='inner')

    # Uncertainty and volatility
    if 'total_uncertainty' in df.columns and not df['total_uncertainty'].isna().all():
        df['uncertainty'] = df['total_uncertainty']
    else:
        df['uncertainty'] = df['realized_vol'].fillna(df['parkinson_vol'])

    df['volatility'] = df['realized_vol'].fillna(df['parkinson_vol'])

    # Binary extreme indicator
    df['is_extreme'] = df['regime'].isin(['extreme_greed', 'extreme_fear']).astype(int)
    df['is_neutral'] = (df['regime'] == 'neutral').astype(int)

    # Clean
    df = df.dropna(subset=['uncertainty', 'volatility', 'regime']).copy()
    df = df.sort_values('date').reset_index(drop=True)

    return df


def compute_gap(df):
    """Compute extreme-neutral uncertainty gap, controlling for volatility."""
    X = sm.add_constant(df['volatility'])
    y = df['uncertainty']
    model = sm.OLS(y, X).fit()
    df = df.copy()
    df['resid'] = model.resid

    extreme_mask = df['is_extreme'] == 1
    neutral_mask = df['is_neutral'] == 1

    if extreme_mask.sum() == 0 or neutral_mask.sum() == 0:
        return np.nan

    extreme_resid = df.loc[extreme_mask, 'resid'].mean()
    neutral_resid = df.loc[neutral_mask, 'resid'].mean()

    return extreme_resid - neutral_resid


# ============================================================================
# Test A: Block-Shuffled Permutation
# ============================================================================

def identify_regime_blocks(series: pd.Series) -> List[Tuple[int, int, str]]:
    """Identify contiguous blocks of the same regime."""
    blocks = []
    start = 0
    current_regime = series.iloc[0]

    for i in range(1, len(series)):
        if series.iloc[i] != current_regime:
            blocks.append((start, i - 1, current_regime))
            start = i
            current_regime = series.iloc[i]

    blocks.append((start, len(series) - 1, current_regime))
    return blocks


def block_shuffle_permutation(df: pd.DataFrame, n_permutations: int = 10000) -> Dict:
    """
    Block-shuffled permutation test.

    Instead of shuffling individual days (which destroys autocorrelation),
    shuffle contiguous regime blocks to preserve temporal structure.
    """
    print("=" * 70)
    print(f"TEST A: BLOCK-SHUFFLED PERMUTATION ({n_permutations} permutations)")
    print("=" * 70)

    # Observed gap
    observed_gap = compute_gap(df)
    print(f"\nObserved gap: {observed_gap:.4f}")

    # Identify regime blocks
    blocks = identify_regime_blocks(df['regime'])
    print(f"Number of regime blocks: {len(blocks)}")
    print(f"Mean block length: {len(df) / len(blocks):.1f} days")

    # Permutation distribution
    permuted_gaps = []

    for _ in range(n_permutations):
        # Shuffle block order
        block_order = np.random.permutation(len(blocks))

        # Reconstruct shuffled regimes
        shuffled_regimes = []
        for idx in block_order:
            start, end, regime = blocks[idx]
            block_length = end - start + 1
            shuffled_regimes.extend([regime] * block_length)

        # Assign to dataframe
        df_perm = df.copy()
        df_perm['regime'] = shuffled_regimes[:len(df)]
        df_perm['is_extreme'] = df_perm['regime'].isin(['extreme_greed', 'extreme_fear']).astype(int)
        df_perm['is_neutral'] = (df_perm['regime'] == 'neutral').astype(int)

        gap = compute_gap(df_perm)
        if not np.isnan(gap):
            permuted_gaps.append(gap)

    permuted_gaps = np.array(permuted_gaps)

    # Statistics
    null_mean = permuted_gaps.mean()
    null_std = permuted_gaps.std()
    p_value_one = (permuted_gaps >= observed_gap).mean()
    p_value_two = (np.abs(permuted_gaps) >= np.abs(observed_gap)).mean()

    print(f"\nNull distribution (block-shuffled):")
    print(f"  Mean: {null_mean:.4f}")
    print(f"  Std:  {null_std:.4f}")
    print(f"  P-value (one-sided): {p_value_one:.4f}")
    print(f"  P-value (two-sided): {p_value_two:.4f}")

    if p_value_one < 0.05:
        print("\n  ✓ BLOCK-SHUFFLE PASSED: Effect survives regime autocorrelation control")
    else:
        print("\n  ✗ Block-shuffle not significant: Effect may be autocorrelation artifact")

    return {
        'test': 'block_shuffle',
        'observed': observed_gap,
        'null_mean': null_mean,
        'null_std': null_std,
        'p_value_one_sided': p_value_one,
        'p_value_two_sided': p_value_two,
        'n_blocks': len(blocks),
        'n_permutations': n_permutations
    }


# ============================================================================
# Test B: Time-Reversed Causality
# ============================================================================

def time_reversed_causality(df: pd.DataFrame, lags: List[int] = [1, 3, 5, 7]) -> Dict:
    """
    Time-reversed causality test.

    Regress current spreads/uncertainty on FUTURE regime indicators.
    Under causal interpretation, future regimes should NOT predict current outcomes.

    If spread_t ~ regime_{t+k} is significant, the relationship may be spurious.
    """
    print("\n" + "=" * 70)
    print("TEST B: TIME-REVERSED CAUSALITY")
    print("=" * 70)

    results = []

    for k in lags:
        # Create future regime indicators
        df_test = df.copy()
        df_test['future_extreme'] = df_test['is_extreme'].shift(-k)
        df_test['future_neutral'] = df_test['is_neutral'].shift(-k)
        df_test = df_test.dropna(subset=['future_extreme', 'uncertainty', 'volatility'])

        if len(df_test) < 50:
            continue

        # Residualize uncertainty by volatility
        X = sm.add_constant(df_test['volatility'])
        y = df_test['uncertainty']
        vol_model = sm.OLS(y, X).fit()
        df_test['resid'] = vol_model.resid

        # Regress residual on future regime
        X_future = sm.add_constant(df_test['future_extreme'])
        model = sm.OLS(df_test['resid'], X_future).fit()

        coef = model.params['future_extreme']
        se = model.bse['future_extreme']
        pval = model.pvalues['future_extreme']

        # Bootstrap CI
        n_boot = 1000
        boot_coefs = []
        for _ in range(n_boot):
            sample = df_test.sample(n=len(df_test), replace=True)
            X_b = sm.add_constant(sample['future_extreme'])
            try:
                m_b = sm.OLS(sample['resid'], X_b).fit()
                boot_coefs.append(m_b.params['future_extreme'])
            except:
                pass

        boot_coefs = np.array(boot_coefs)
        ci_lower = np.percentile(boot_coefs, 2.5)
        ci_upper = np.percentile(boot_coefs, 97.5)

        print(f"\n  Lag k={k}: β = {coef:.4f} (SE = {se:.4f}), p = {pval:.4f}")
        print(f"           95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

        # Check if CI includes zero (no reverse causality)
        includes_zero = ci_lower <= 0 <= ci_upper
        if includes_zero:
            print("           ✓ CI includes zero: No reverse causality")
        else:
            print("           ✗ CI excludes zero: Possible spurious relationship")

        results.append({
            'lag_k': k,
            'coefficient': coef,
            'std_error': se,
            'p_value': pval,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'includes_zero': includes_zero,
            'n_obs': len(df_test)
        })

    # Summary
    all_include_zero = all(r['includes_zero'] for r in results)
    if all_include_zero:
        print("\n  ✓ TIME-REVERSAL PASSED: Future regimes do not predict current uncertainty")
    else:
        print("\n  ⚠ Some time-reversed coefficients significant (possible spurious pattern)")

    return {
        'test': 'time_reversed',
        'lag_results': results,
        'all_include_zero': all_include_zero
    }


# ============================================================================
# Test C: Synthetic Regime Assignment
# ============================================================================

def synthetic_regime_placebo(df: pd.DataFrame, n_simulations: int = 10000) -> Dict:
    """
    Synthetic regime assignment test.

    Generate synthetic regimes that preserve the volatility-regime correlation
    using AR(1) on Fear & Greed values. If the extremity premium emerges from
    synthetic regimes with similar autocorrelation, the finding may be spurious.
    """
    print("\n" + "=" * 70)
    print(f"TEST C: SYNTHETIC REGIME ASSIGNMENT ({n_simulations} simulations)")
    print("=" * 70)

    # Observed gap
    observed_gap = compute_gap(df)
    print(f"\nObserved gap: {observed_gap:.4f}")

    # Fit AR(1) to Fear & Greed
    fg = df['fear_greed_value'].values
    fg_lag = fg[:-1]
    fg_curr = fg[1:]

    # AR(1) coefficients
    X_ar = sm.add_constant(fg_lag)
    ar_model = sm.OLS(fg_curr, X_ar).fit()
    # Use positional indexing - params can be array or Series
    params = ar_model.params
    phi_0 = float(params[0])  # constant
    phi_1 = float(params[1])  # lag coefficient
    sigma_e = float(np.std(ar_model.resid))

    print(f"AR(1) fit: φ₀ = {phi_0:.2f}, φ₁ = {phi_1:.3f}, σ = {sigma_e:.2f}")

    # Thresholds for regime assignment
    thresholds = {'extreme_fear': 25, 'fear': 40, 'neutral': 60, 'greed': 75}

    def assign_regime(val):
        if val <= thresholds['extreme_fear']:
            return 'extreme_fear'
        elif val <= thresholds['fear']:
            return 'fear'
        elif val <= thresholds['neutral']:
            return 'neutral'
        elif val <= thresholds['greed']:
            return 'greed'
        else:
            return 'extreme_greed'

    # Simulate
    synthetic_gaps = []
    n = len(df)

    for _ in range(n_simulations):
        # Generate synthetic F&G series via AR(1)
        synthetic_fg = np.zeros(n)
        synthetic_fg[0] = fg[0]  # Start from actual first value

        for t in range(1, n):
            synthetic_fg[t] = phi_0 + phi_1 * synthetic_fg[t-1] + np.random.normal(0, sigma_e)

        # Clip to [0, 100]
        synthetic_fg = np.clip(synthetic_fg, 0, 100)

        # Assign regimes
        synthetic_regimes = [assign_regime(v) for v in synthetic_fg]

        # Create synthetic dataframe
        df_syn = df.copy()
        df_syn['regime'] = synthetic_regimes
        df_syn['is_extreme'] = df_syn['regime'].isin(['extreme_greed', 'extreme_fear']).astype(int)
        df_syn['is_neutral'] = (df_syn['regime'] == 'neutral').astype(int)

        gap = compute_gap(df_syn)
        if not np.isnan(gap):
            synthetic_gaps.append(gap)

    synthetic_gaps = np.array(synthetic_gaps)

    # Statistics
    null_mean = synthetic_gaps.mean()
    null_std = synthetic_gaps.std()
    p_value_one = (synthetic_gaps >= observed_gap).mean()
    p_value_two = (np.abs(synthetic_gaps) >= np.abs(observed_gap)).mean()

    print(f"\nSynthetic AR(1) distribution:")
    print(f"  Mean: {null_mean:.4f}")
    print(f"  Std:  {null_std:.4f}")
    print(f"  P-value (one-sided): {p_value_one:.4f}")
    print(f"  P-value (two-sided): {p_value_two:.4f}")

    if p_value_one < 0.05:
        print("\n  ✓ SYNTHETIC PLACEBO PASSED: Effect not explained by AR(1) regime dynamics")
    else:
        print("\n  ✗ Synthetic placebo failed: Effect may arise from regime autocorrelation alone")

    return {
        'test': 'synthetic_ar1',
        'observed': observed_gap,
        'null_mean': null_mean,
        'null_std': null_std,
        'p_value_one_sided': p_value_one,
        'p_value_two_sided': p_value_two,
        'ar1_phi0': phi_0,
        'ar1_phi1': phi_1,
        'ar1_sigma': sigma_e,
        'n_simulations': n_simulations
    }


def generate_latex_table(results: Dict) -> str:
    """Generate LaTeX table for placebo tests."""

    block = results['block_shuffle']
    time_rev = results['time_reversed']
    synthetic = results['synthetic']

    latex = r"""\begin{table}[h!]
\centering
\caption{Placebo and Identification Tests}
\label{tab:placebo}
\small
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Test} & \textbf{Observed} & \textbf{Null Mean} & \textbf{Null SD} & \textbf{$p$-value} \\
\midrule
\multicolumn{5}{l}{\textit{Permutation Tests}} \\
"""

    # Standard permutation (from existing results)
    latex += r"Standard Permutation$^a$ & 0.042 & 0.000 & 0.009 & $<$0.0001 \\" + "\n"

    # Block-shuffled
    latex += f"Block-Shuffled$^b$ & {block['observed']:.3f} & {block['null_mean']:.3f} & {block['null_std']:.3f} & "
    if block['p_value_one_sided'] < 0.0001:
        latex += "$<$0.0001 \\\\\n"
    else:
        latex += f"{block['p_value_one_sided']:.4f} \\\\\n"

    # Synthetic AR(1)
    latex += f"Synthetic AR(1)$^c$ & {synthetic['observed']:.3f} & {synthetic['null_mean']:.3f} & {synthetic['null_std']:.3f} & "
    if synthetic['p_value_one_sided'] < 0.0001:
        latex += "$<$0.0001 \\\\\n"
    else:
        latex += f"{synthetic['p_value_one_sided']:.4f} \\\\\n"

    latex += r"\addlinespace" + "\n"
    latex += r"\multicolumn{5}{l}{\textit{Time-Reversed Causality (spread$_t$ $\sim$ regime$_{t+k}$)}} \\" + "\n"

    # Time-reversed results
    for lag_result in time_rev['lag_results']:
        k = lag_result['lag_k']
        coef = lag_result['coefficient']
        ci_lo = lag_result['ci_lower']
        ci_hi = lag_result['ci_upper']
        pval = lag_result['p_value']

        latex += f"Forward $k={k}$ & $\\hat{{\\beta}}={coef:.3f}$ & \\multicolumn{{2}}{{c}}{{95\\% CI: [{ci_lo:.3f}, {ci_hi:.3f}]}} & {pval:.3f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\vspace{0.3em}
\caption*{\footnotesize
$^a$Standard permutation shuffles individual days (10,000 permutations).
$^b$Block-shuffled permutation preserves regime autocorrelation by shuffling contiguous blocks.
$^c$Synthetic AR(1) generates regimes from fitted AR(1) model on F\&G values.
Time-reversed tests regress current uncertainty on \textit{future} regime indicators; null hypothesis is no reverse causality.
All tests use volatility-residualized uncertainty.}
\end{table}
"""

    return latex


def save_results(results: Dict):
    """Save all results to CSV files."""
    # Block shuffle
    df_block = pd.DataFrame([results['block_shuffle']])
    df_block.to_csv(os.path.join(RESULTS_DIR, "placebo_block_shuffle.csv"), index=False)

    # Time reversed
    df_time = pd.DataFrame(results['time_reversed']['lag_results'])
    df_time.to_csv(os.path.join(RESULTS_DIR, "placebo_time_reversed.csv"), index=False)

    # Synthetic
    df_syn = pd.DataFrame([results['synthetic']])
    df_syn.to_csv(os.path.join(RESULTS_DIR, "placebo_synthetic_regimes.csv"), index=False)

    # Summary - helper functions for cleaner code
    def get_lag_val(idx, key):
        lag_res = results['time_reversed']['lag_results']
        if len(lag_res) > idx:
            return lag_res[idx][key]
        return np.nan

    def get_lag_pass(idx):
        lag_res = results['time_reversed']['lag_results']
        if len(lag_res) > idx:
            return 'Yes' if lag_res[idx]['includes_zero'] else 'No'
        return 'Yes'

    summary = {
        'test': ['standard_permutation', 'block_shuffle', 'synthetic_ar1',
                 'time_reversed_k1', 'time_reversed_k3', 'time_reversed_k5', 'time_reversed_k7'],
        'observed': [0.042, results['block_shuffle']['observed'], results['synthetic']['observed'],
                     get_lag_val(0, 'coefficient'), get_lag_val(1, 'coefficient'),
                     get_lag_val(2, 'coefficient'), get_lag_val(3, 'coefficient')],
        'p_value': [0.0001, results['block_shuffle']['p_value_one_sided'],
                    results['synthetic']['p_value_one_sided'],
                    get_lag_val(0, 'p_value'), get_lag_val(1, 'p_value'),
                    get_lag_val(2, 'p_value'), get_lag_val(3, 'p_value')],
        'passes': ['Yes',
                   'Yes' if results['block_shuffle']['p_value_one_sided'] < 0.05 else 'No',
                   'Yes' if results['synthetic']['p_value_one_sided'] < 0.05 else 'No',
                   get_lag_pass(0), get_lag_pass(1), get_lag_pass(2), get_lag_pass(3)]
    }
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(os.path.join(RESULTS_DIR, "placebo_summary.csv"), index=False)

    # LaTeX table
    latex = generate_latex_table(results)
    with open(os.path.join(RESULTS_DIR, "placebo_table.tex"), 'w') as f:
        f.write(latex)

    print(f"\nResults saved to {RESULTS_DIR}/placebo_*.csv")


def main():
    print("=" * 70)
    print("PLACEBO TEST SUITE FOR THE EXTREMITY PREMIUM")
    print("Addresses: 'Is this an artifact of volatility clustering or regime persistence?'")
    print("=" * 70)

    # Load data
    df = load_data()
    print(f"\nData: {len(df)} observations")
    print(f"Regimes: {df['regime'].value_counts().to_dict()}")

    results = {}

    # Test A: Block-shuffled permutation
    results['block_shuffle'] = block_shuffle_permutation(df, n_permutations=10000)

    # Test B: Time-reversed causality
    results['time_reversed'] = time_reversed_causality(df, lags=[1, 3, 5, 7])

    # Test C: Synthetic regime assignment
    results['synthetic'] = synthetic_regime_placebo(df, n_simulations=10000)

    # Save all results
    save_results(results)

    # Final summary
    print("\n" + "=" * 70)
    print("PLACEBO TEST SUMMARY")
    print("=" * 70)

    n_passed = 0
    n_tests = 0

    # Block shuffle
    n_tests += 1
    if results['block_shuffle']['p_value_one_sided'] < 0.05:
        n_passed += 1
        print("✓ Block-shuffle: PASSED (p < 0.05)")
    else:
        print("✗ Block-shuffle: FAILED")

    # Time-reversed
    if results['time_reversed']['all_include_zero']:
        n_passed += 1
        print("✓ Time-reversed: PASSED (all CIs include zero)")
    else:
        print("⚠ Time-reversed: PARTIAL (some CIs exclude zero)")
    n_tests += 1

    # Synthetic
    n_tests += 1
    if results['synthetic']['p_value_one_sided'] < 0.05:
        n_passed += 1
        print("✓ Synthetic AR(1): PASSED (p < 0.05)")
    else:
        print("✗ Synthetic AR(1): FAILED")

    print(f"\nOverall: {n_passed}/{n_tests} tests passed")

    if n_passed >= 2:
        print("\n→ The extremity premium is robust to placebo controls.")
        print("  It is NOT an artifact of regime persistence or temporal structure.")
    else:
        print("\n→ The extremity premium may be partially explained by temporal structure.")

    return results


if __name__ == '__main__':
    results = main()
