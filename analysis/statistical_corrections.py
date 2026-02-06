"""
Statistical Corrections for Multiple Comparisons

Implements:
1. Bonferroni correction for family-wise error rate (FWER)
2. Benjamini-Hochberg FDR correction
3. Effect size interpretation with economic context

This addresses reviewer concerns about multiple testing in regime comparisons.

Author: Murad Farzulla
Date: January 2026
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Tuple, Optional

# For loading project data
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """
    Bonferroni correction for multiple comparisons.

    Controls the family-wise error rate (FWER) - the probability of making
    at least one Type I error across all tests.

    Conservative but widely accepted in finance literature.

    Args:
        p_values: List of p-values from individual tests
        alpha: Desired significance level (default 0.05)

    Returns:
        List of (corrected_p, is_significant) tuples
    """
    n = len(p_values)
    alpha_corrected = alpha / n

    results = []
    for p in p_values:
        # Corrected p-value (multiply by number of tests)
        p_corrected = min(p * n, 1.0)
        is_sig = p_corrected < alpha

        results.append((p_corrected, is_sig))

    return results


def benjamini_hochberg_fdr(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """
    Benjamini-Hochberg FDR correction.

    Controls the false discovery rate (FDR) - the expected proportion of
    false positives among all rejections.

    Less conservative than Bonferroni, often preferred in exploratory research.

    Args:
        p_values: List of p-values from individual tests
        alpha: Desired FDR level (default 0.05)

    Returns:
        List of (adjusted_p, is_significant) tuples
    """
    n = len(p_values)
    indexed = list(enumerate(p_values))

    # Sort by p-value (ascending)
    sorted_indexed = sorted(indexed, key=lambda x: x[1])

    # BH adjusted p-values
    adjusted = [None] * n
    prev_adj = 1.0

    # Process in reverse order for adjusted p-values
    for rank in range(n, 0, -1):
        orig_idx, p = sorted_indexed[rank - 1]
        adj_p = min(p * n / rank, prev_adj)
        adjusted[orig_idx] = adj_p
        prev_adj = adj_p

    # Determine significance
    results = [(adj_p, adj_p < alpha) for adj_p in adjusted]

    return results


def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """
    Holm-Bonferroni step-down procedure.

    More powerful than standard Bonferroni while still controlling FWER.
    """
    n = len(p_values)
    indexed = list(enumerate(p_values))

    # Sort by p-value
    sorted_indexed = sorted(indexed, key=lambda x: x[1])

    adjusted = [None] * n
    prev_adj = 0.0

    for rank, (orig_idx, p) in enumerate(sorted_indexed, 1):
        adj_p = max(p * (n - rank + 1), prev_adj)
        adj_p = min(adj_p, 1.0)
        adjusted[orig_idx] = adj_p
        prev_adj = adj_p

    results = [(adj_p, adj_p < alpha) for adj_p in adjusted]

    return results


def apply_corrections_to_regime_analysis(
    results_df: pd.DataFrame,
    p_col: str = 'p_value',
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Apply multiple testing corrections to regime comparison results.

    Args:
        results_df: DataFrame with regime comparison results
        p_col: Column name containing p-values
        alpha: Significance level

    Returns:
        DataFrame with corrected p-values and significance flags
    """
    p_values = results_df[p_col].tolist()

    # Apply all three corrections
    bonf = bonferroni_correction(p_values, alpha)
    bh = benjamini_hochberg_fdr(p_values, alpha)
    holm = holm_bonferroni_correction(p_values, alpha)

    results_df = results_df.copy()

    results_df['p_bonferroni'] = [x[0] for x in bonf]
    results_df['sig_bonferroni'] = [x[1] for x in bonf]

    results_df['p_bh_fdr'] = [x[0] for x in bh]
    results_df['sig_bh_fdr'] = [x[1] for x in bh]

    results_df['p_holm'] = [x[0] for x in holm]
    results_df['sig_holm'] = [x[1] for x in holm]

    return results_df


def compute_effect_sizes(
    coef: float,
    se: float,
    baseline_spread: float = 50.0,
    sample_size: int = 100
) -> Dict[str, float]:
    """
    Compute effect sizes for regime coefficients with economic interpretation.

    Args:
        coef: Regression coefficient (change in uncertainty)
        se: Standard error of coefficient
        baseline_spread: Typical spread in bps (for economic interpretation)
        sample_size: Number of observations

    Returns:
        Dict with various effect size measures
    """
    # Cohen's d (standardized effect size)
    # For regression: d = coef / pooled_sd (approx as 2 * se * sqrt(n))
    # This is a rough approximation
    cohens_d = coef / (se * np.sqrt(sample_size) / 2) if se > 0 else 0

    # Percentage change interpretation
    # If baseline spread is 50 bps and coef = 0.05, then
    # change = 0.05 * spread_sensitivity
    # Assuming 1% change in uncertainty -> ~0.5 bps change in spread
    spread_impact_bps = coef * baseline_spread  # Proportional impact

    # 95% confidence interval
    ci_low = coef - 1.96 * se
    ci_high = coef + 1.96 * se

    effect_interpretation = (
        "negligible" if abs(cohens_d) < 0.2 else
        "small" if abs(cohens_d) < 0.5 else
        "medium" if abs(cohens_d) < 0.8 else
        "large"
    )

    return {
        'cohens_d': cohens_d,
        'effect_interpretation': effect_interpretation,
        'spread_impact_bps': spread_impact_bps,
        'ci_95_low': ci_low,
        'ci_95_high': ci_high,
        'pct_change_from_baseline': (spread_impact_bps / baseline_spread) * 100
    }


def economic_interpretation(
    regime_coefs: Dict[str, Tuple[float, float]],
    baseline_spread: float = 50.0,
    trading_volume_daily: float = 1e9
) -> pd.DataFrame:
    """
    Economic interpretation of regime effects on spreads.

    Args:
        regime_coefs: Dict of {regime: (coefficient, std_error)}
        baseline_spread: Typical spread in bps
        trading_volume_daily: Daily trading volume in USD

    Returns:
        DataFrame with economic interpretation
    """
    results = []

    print("\n" + "=" * 70)
    print("ECONOMIC INTERPRETATION OF REGIME EFFECTS")
    print("=" * 70)
    print(f"\nBaseline assumptions:")
    print(f"  - Typical spread: {baseline_spread:.1f} bps")
    print(f"  - Daily volume: ${trading_volume_daily/1e9:.1f}B")

    for regime, (coef, se) in regime_coefs.items():
        # Spread change in bps
        spread_change_bps = coef * baseline_spread

        # Total spread in regime
        spread_in_regime = baseline_spread + spread_change_bps

        # Transaction cost per trade
        # Spread is round-trip cost, so half for one-way
        cost_per_trade_bps = spread_in_regime / 2

        # Daily transaction cost increase (assuming volume constant)
        # Cost = volume * (spread / 10000)
        daily_cost_baseline = trading_volume_daily * (baseline_spread / 10000 / 2)
        daily_cost_regime = trading_volume_daily * (spread_in_regime / 10000 / 2)
        daily_cost_increase = daily_cost_regime - daily_cost_baseline

        # Annualized cost
        annual_cost_increase = daily_cost_increase * 365

        results.append({
            'regime': regime,
            'coefficient': coef,
            'std_error': se,
            'spread_change_bps': spread_change_bps,
            'spread_in_regime_bps': spread_in_regime,
            'pct_spread_increase': (spread_change_bps / baseline_spread) * 100,
            'daily_cost_increase_usd': daily_cost_increase,
            'annual_cost_increase_usd': annual_cost_increase,
        })

        sig = "***" if (abs(coef) / se) > 3.3 else "**" if (abs(coef) / se) > 2.6 else "*" if (abs(coef) / se) > 2.0 else ""

        print(f"\n{regime}:{sig}")
        print(f"  Uncertainty coefficient: {coef:+.4f} (SE={se:.4f})")
        print(f"  Spread change: {spread_change_bps:+.2f} bps ({spread_change_bps/baseline_spread*100:+.1f}%)")
        print(f"  Total spread in regime: {spread_in_regime:.2f} bps")
        print(f"  Daily cost increase: ${daily_cost_increase:,.0f}")
        print(f"  Annual cost increase: ${annual_cost_increase:,.0f}")

    results_df = pd.DataFrame(results)

    # Summary
    print("\n" + "-" * 70)
    print("KEY TAKEAWAY:")
    max_regime = results_df.loc[results_df['spread_change_bps'].idxmax(), 'regime']
    max_annual = results_df['annual_cost_increase_usd'].max()
    print(f"  {max_regime} regime has highest spread impact")
    print(f"  Market-wide annual transaction cost increase: ${max_annual:,.0f}")

    return results_df


def run_statistical_corrections():
    """
    Run statistical corrections on existing regime analysis results.
    """
    # Load existing regime results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(project_dir, "results")

    # Load extremity premium regression results
    regression_path = os.path.join(results_dir, "extremity_premium_regression.csv")

    if not os.path.exists(regression_path):
        print("ERROR: Run extremity_premium_analysis.py first")
        return None

    df = pd.read_csv(regression_path)
    print(f"Loaded {len(df)} regime comparisons from {regression_path}")

    print("\n" + "=" * 70)
    print("MULTIPLE TESTING CORRECTIONS FOR REGIME COMPARISONS")
    print("=" * 70)

    print(f"\nNumber of comparisons: {len(df)}")
    print(f"Original alpha: 0.05")
    print(f"Bonferroni-corrected alpha: {0.05 / len(df):.4f}")

    # Apply corrections
    df_corrected = apply_corrections_to_regime_analysis(df)

    # Print results
    print("\n{:15s} {:>10s} {:>12s} {:>12s} {:>12s} {:>8s}".format(
        "Regime", "p (raw)", "p (Bonf)", "p (BH-FDR)", "p (Holm)", "Survives"))
    print("-" * 75)

    for _, row in df_corrected.iterrows():
        sig_raw = "*" if row['p_value'] < 0.05 else ""
        sig_bonf = "✓" if row['sig_bonferroni'] else ""
        sig_bh = "✓" if row['sig_bh_fdr'] else ""
        sig_holm = "✓" if row['sig_holm'] else ""

        survives = "All" if row['sig_bonferroni'] else "FDR" if row['sig_bh_fdr'] else "None"

        print(f"{row['regime']:15s} {row['p_value']:>9.4f}{sig_raw} "
              f"{row['p_bonferroni']:>10.4f}{sig_bonf:>2s} "
              f"{row['p_bh_fdr']:>10.4f}{sig_bh:>2s} "
              f"{row['p_holm']:>10.4f}{sig_holm:>2s} "
              f"{survives:>8s}")

    # Summary
    n_survive_bonf = df_corrected['sig_bonferroni'].sum()
    n_survive_bh = df_corrected['sig_bh_fdr'].sum()
    n_survive_holm = df_corrected['sig_holm'].sum()

    print("\n" + "-" * 75)
    print(f"Results surviving corrections:")
    print(f"  Bonferroni (strictest):  {n_survive_bonf}/{len(df)} regimes")
    print(f"  Holm-Bonferroni:         {n_survive_holm}/{len(df)} regimes")
    print(f"  Benjamini-Hochberg FDR:  {n_survive_bh}/{len(df)} regimes")

    # Effect size interpretation
    print("\n" + "=" * 70)
    print("EFFECT SIZE INTERPRETATION")
    print("=" * 70)

    for _, row in df_corrected.iterrows():
        effects = compute_effect_sizes(
            row['coefficient'],
            row['std_error'],
            baseline_spread=50.0,
            sample_size=200  # Approximate
        )

        print(f"\n{row['regime']}:")
        print(f"  Cohen's d: {effects['cohens_d']:.3f} ({effects['effect_interpretation']})")
        print(f"  Spread impact: {effects['spread_impact_bps']:+.2f} bps")
        print(f"  95% CI: [{effects['ci_95_low']:.4f}, {effects['ci_95_high']:.4f}]")

    # Save corrected results
    output_path = os.path.join(results_dir, "extremity_premium_corrected.csv")
    df_corrected.to_csv(output_path, index=False)
    print(f"\nSaved corrected results to: {output_path}")

    # Economic interpretation using the coefficients
    regime_coefs = {}
    for _, row in df_corrected.iterrows():
        regime_coefs[row['regime']] = (row['coefficient'], row['std_error'])

    econ_df = economic_interpretation(regime_coefs)
    econ_path = os.path.join(results_dir, "extremity_premium_economic.csv")
    econ_df.to_csv(econ_path, index=False)
    print(f"Saved economic interpretation to: {econ_path}")

    return df_corrected, econ_df


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Statistical corrections for multiple comparisons"
    )
    args = parser.parse_args()

    results = run_statistical_corrections()

    if results is not None:
        print("\n" + "=" * 70)
        print("STATISTICAL CORRECTIONS COMPLETE")
        print("=" * 70)
        print("""
For the paper:
1. Report corrected p-values in Table X (regime comparisons)
2. Note which findings survive Bonferroni vs BH-FDR correction
3. Include effect sizes (Cohen's d) for practical significance
4. Report economic interpretation in Discussion section

Key methodological note:
- Bonferroni is conservative but widely accepted
- If finding survives Bonferroni, it's robust to multiple testing
- BH-FDR is acceptable for exploratory analysis
""")


if __name__ == '__main__':
    main()
