#!/usr/bin/env python3
"""
Run complete analysis pipeline for Sentiment-Microstructure ABM paper.

Generates all tables and figures for Section 7 (Preliminary Results).
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from datetime import datetime

# Import analysis modules
from analysis.statistical_analysis import (
    compute_return_statistics,
    jarque_bera_test,
    adf_test,
    kpss_test,
    compute_acf,
    ljung_box_test,
    volatility_clustering_test,
    compute_correlation_matrix,
    run_all_diagnostics,
)
from analysis.regime_analysis import (
    compute_regime_statistics,
    compute_regime_transitions,
    compute_regime_durations,
    analyze_regime_switching_dynamics,
)
from analysis.figure_generation import (
    plot_return_distribution,
    plot_acf_comparison,
    plot_regime_dynamics,
    plot_uncertainty_decomposition,
    set_paper_style,
)


def load_simulation_data() -> pd.DataFrame:
    """Load the simulation data from CSV."""
    csv_path = project_root / 'demo' / 'preliminary_results.csv'
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} observations from {csv_path}")
    return df


def generate_latex_table(df: pd.DataFrame, caption: str, label: str,
                          columns: list = None, fmt: dict = None) -> str:
    """Generate LaTeX table from DataFrame."""
    if columns:
        df = df[columns]

    latex = df.to_latex(
        float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else str(x),
        escape=False,
        caption=caption,
        label=label,
        bold_rows=True,
    )
    return latex


def run_full_analysis():
    """Run complete analysis pipeline."""
    print("=" * 60)
    print("SENTIMENT-MICROSTRUCTURE ABM: FULL ANALYSIS PIPELINE")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)

    # Load data
    df = load_simulation_data()

    # Setup output directories
    figures_dir = project_root / 'paper' / 'figures'
    tables_dir = project_root / 'paper' / 'tables'
    results_dir = project_root / 'analysis' / 'results'

    for d in [figures_dir, tables_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    results = {}

    # ========================================
    # 1. RETURN DISTRIBUTION ANALYSIS
    # ========================================
    print("\n" + "=" * 40)
    print("1. RETURN DISTRIBUTION ANALYSIS")
    print("=" * 40)

    return_stats = compute_return_statistics(df)
    results['return_statistics'] = return_stats

    print(f"\nReturn Statistics:")
    print(f"  N observations: {return_stats['n_obs']}")
    print(f"  Mean return: {return_stats['mean']:.6f}")
    print(f"  Std deviation: {return_stats['std']:.6f}")
    print(f"  Skewness: {return_stats['skewness']:.4f}")
    print(f"  Excess Kurtosis: {return_stats['kurtosis']:.4f}")
    print(f"  Min/Max: {return_stats['min']:.6f} / {return_stats['max']:.6f}")

    # Jarque-Bera test
    jb_result = jarque_bera_test(df['log_return'])
    results['jarque_bera'] = jb_result
    print(f"\nJarque-Bera Test for Normality:")
    print(f"  Statistic: {jb_result['statistic']:.4f}")
    print(f"  p-value: {jb_result['p_value']:.4f}")
    print(f"  Is Normal (p > 0.05): {jb_result['is_normal']}")

    # Generate Figure 2: Return Distribution
    print("\nGenerating Figure 2: Return Distribution...")
    fig2 = plot_return_distribution(
        df,
        save_path=str(figures_dir / 'return_distribution.pdf')
    )

    # ========================================
    # 2. TIME-SERIES DIAGNOSTICS
    # ========================================
    print("\n" + "=" * 40)
    print("2. TIME-SERIES DIAGNOSTICS")
    print("=" * 40)

    # ADF test on spreads
    adf_spread = adf_test(df['spread_bps'])
    results['adf_spread'] = adf_spread
    print(f"\nADF Test on Spread (stationarity):")
    print(f"  ADF Statistic: {adf_spread['adf_statistic']:.4f}")
    print(f"  p-value: {adf_spread['p_value']:.4f}")
    print(f"  Is Stationary (p < 0.05): {adf_spread['is_stationary']}")

    # KPSS test on spreads
    kpss_spread = kpss_test(df['spread_bps'])
    results['kpss_spread'] = kpss_spread
    print(f"\nKPSS Test on Spread (stationarity):")
    print(f"  KPSS Statistic: {kpss_spread['kpss_statistic']:.4f}")
    print(f"  p-value: {kpss_spread['p_value']:.4f}")
    print(f"  Is Stationary (p > 0.05): {kpss_spread['is_stationary']}")

    # Ljung-Box tests
    print("\nLjung-Box Test for Autocorrelation (Returns):")
    lb_returns = ljung_box_test(df['log_return'])
    results['ljung_box_returns'] = lb_returns.to_dict()
    print(lb_returns.to_string(index=False))

    print("\nLjung-Box Test for Autocorrelation (|Returns|):")
    lb_abs_returns = ljung_box_test(df['log_return'].abs())
    results['ljung_box_abs_returns'] = lb_abs_returns.to_dict()
    print(lb_abs_returns.to_string(index=False))

    # Volatility clustering test
    vol_cluster = volatility_clustering_test(df)
    results['volatility_clustering'] = vol_cluster
    print(f"\nVolatility Clustering Analysis:")
    print(f"  ACF(returns, lag=10): {vol_cluster['acf_returns_lag10']:.4f}")
    print(f"  ACF(|returns|, lag=10): {vol_cluster['acf_abs_returns_lag10']:.4f}")
    print(f"  Clustering Ratio: {vol_cluster['clustering_ratio_lag10']:.4f}")
    print(f"  Has Volatility Clustering: {vol_cluster['has_volatility_clustering']}")

    # Generate Figure 3: ACF Comparison
    print("\nGenerating Figure 3: ACF Comparison...")
    fig3 = plot_acf_comparison(
        df,
        save_path=str(figures_dir / 'acf_comparison.pdf')
    )

    # ========================================
    # 3. REGIME ANALYSIS
    # ========================================
    print("\n" + "=" * 40)
    print("3. REGIME-CONDITIONAL ANALYSIS")
    print("=" * 40)

    regime_stats = compute_regime_statistics(df)
    results['regime_statistics'] = regime_stats.to_dict()

    print("\nRegime Statistics:")
    print(regime_stats.to_string())

    # Transition matrix
    transitions = compute_regime_transitions(df)
    results['regime_transitions'] = transitions.to_dict()
    print("\nRegime Transition Matrix P(t+1 | t):")
    print(transitions.to_string())

    # Duration analysis
    durations = compute_regime_durations(df)
    results['regime_durations'] = durations
    print("\nRegime Duration Statistics:")
    for regime, stats in durations.items():
        print(f"\n  {regime.upper()}:")
        print(f"    Episodes: {stats['n_episodes']}")
        print(f"    Mean duration: {stats['mean_duration']:.1f} steps")
        print(f"    Std duration: {stats['std_duration']:.1f} steps")
        print(f"    Max duration: {stats['max_duration']} steps")

    # Generate Figure 4: Regime Dynamics
    print("\nGenerating Figure 4: Regime Dynamics...")
    fig4 = plot_regime_dynamics(
        df,
        save_path=str(figures_dir / 'regime_dynamics.pdf')
    )

    # ========================================
    # 4. UNCERTAINTY DECOMPOSITION
    # ========================================
    print("\n" + "=" * 40)
    print("4. UNCERTAINTY DECOMPOSITION")
    print("=" * 40)

    # Correlation analysis
    corr_matrix = compute_correlation_matrix(df, [
        'sentiment', 'epistemic_uncertainty', 'aleatoric_uncertainty',
        'total_uncertainty', 'spread_bps', 'log_return', 'inventory'
    ])
    results['correlation_matrix'] = corr_matrix.to_dict()

    print("\nCorrelation Matrix:")
    print(corr_matrix.to_string())

    # Key correlations
    print("\nKey Correlations:")
    print(f"  Epistemic ↔ Spread: {corr_matrix.loc['epistemic_uncertainty', 'spread_bps']:.4f}")
    print(f"  Aleatoric ↔ Spread: {corr_matrix.loc['aleatoric_uncertainty', 'spread_bps']:.4f}")
    print(f"  Total Unc ↔ Spread: {corr_matrix.loc['total_uncertainty', 'spread_bps']:.4f}")
    print(f"  Sentiment ↔ Spread: {corr_matrix.loc['sentiment', 'spread_bps']:.4f}")

    # Uncertainty ratio analysis
    epi_ratio = df['epistemic_uncertainty'].mean() / df['total_uncertainty'].mean()
    ale_ratio = df['aleatoric_uncertainty'].mean() / df['total_uncertainty'].mean()

    results['uncertainty_ratios'] = {
        'epistemic_ratio': epi_ratio,
        'aleatoric_ratio': ale_ratio,
        'mean_epistemic': df['epistemic_uncertainty'].mean(),
        'mean_aleatoric': df['aleatoric_uncertainty'].mean(),
        'mean_total': df['total_uncertainty'].mean(),
    }

    print(f"\nUncertainty Decomposition:")
    print(f"  Mean Epistemic: {df['epistemic_uncertainty'].mean():.4f}")
    print(f"  Mean Aleatoric: {df['aleatoric_uncertainty'].mean():.4f}")
    print(f"  Mean Total: {df['total_uncertainty'].mean():.4f}")
    print(f"  Epistemic Ratio: {epi_ratio:.2%}")
    print(f"  Aleatoric Ratio: {ale_ratio:.2%}")

    # Generate Figure 5: Uncertainty Decomposition
    print("\nGenerating Figure 5: Uncertainty Decomposition...")
    fig5 = plot_uncertainty_decomposition(
        df,
        save_path=str(figures_dir / 'uncertainty_decomposition.pdf')
    )

    # ========================================
    # 5. GENERATE LATEX TABLES
    # ========================================
    print("\n" + "=" * 40)
    print("5. GENERATING LATEX TABLES")
    print("=" * 40)

    # Table 2: Summary Statistics
    summary_df = pd.DataFrame({
        'Statistic': ['Mean', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis'],
        'Log Return': [
            f"{return_stats['mean']:.6f}",
            f"{return_stats['std']:.6f}",
            f"{return_stats['min']:.6f}",
            f"{return_stats['max']:.6f}",
            f"{return_stats['skewness']:.3f}",
            f"{return_stats['kurtosis']:.3f}",
        ],
        'Spread (bps)': [
            f"{df['spread_bps'].mean():.4f}",
            f"{df['spread_bps'].std():.4f}",
            f"{df['spread_bps'].min():.4f}",
            f"{df['spread_bps'].max():.4f}",
            f"{df['spread_bps'].skew():.3f}",
            f"{df['spread_bps'].kurtosis():.3f}",
        ],
        'Sentiment': [
            f"{df['sentiment'].mean():.4f}",
            f"{df['sentiment'].std():.4f}",
            f"{df['sentiment'].min():.4f}",
            f"{df['sentiment'].max():.4f}",
            f"{df['sentiment'].skew():.3f}",
            f"{df['sentiment'].kurtosis():.3f}",
        ],
    })

    with open(tables_dir / 'table2_summary_stats.tex', 'w') as f:
        f.write(summary_df.to_latex(index=False, escape=False, caption='Summary Statistics', label='tab:summary_stats'))
    print(f"  Saved: {tables_dir / 'table2_summary_stats.tex'}")

    # Table 3: Regime Statistics
    regime_table = regime_stats.reset_index()
    regime_table.columns = ['Regime', 'N Obs', '% Time', 'Mean Spread', 'Std Spread',
                            'Median Spread', 'Mean Return', 'Volatility',
                            'Mean Sent', 'Sent Std', 'Mean Epi', 'Mean Ale',
                            'Mean Total', 'Mean Inv', 'Inv Std', 'Max |Inv|']

    # Select key columns
    regime_table_slim = regime_table[['Regime', 'N Obs', '% Time', 'Mean Spread',
                                       'Volatility', 'Mean Sent', 'Mean Total']].copy()

    with open(tables_dir / 'table3_regime_stats.tex', 'w') as f:
        f.write(regime_table_slim.to_latex(index=False, escape=False,
                float_format='%.4f', caption='Regime-Conditional Statistics',
                label='tab:regime_stats'))
    print(f"  Saved: {tables_dir / 'table3_regime_stats.tex'}")

    # Table 4: Time-Series Diagnostics
    diag_df = pd.DataFrame({
        'Test': ['ADF (Spread)', 'KPSS (Spread)', 'Jarque-Bera (Returns)'],
        'Statistic': [
            f"{adf_spread['adf_statistic']:.4f}",
            f"{kpss_spread['kpss_statistic']:.4f}",
            f"{jb_result['statistic']:.4f}",
        ],
        'p-value': [
            f"{adf_spread['p_value']:.4f}",
            f"{kpss_spread['p_value']:.4f}",
            f"{jb_result['p_value']:.4f}",
        ],
        'Conclusion': [
            'Stationary' if adf_spread['is_stationary'] else 'Non-stationary',
            'Stationary' if kpss_spread['is_stationary'] else 'Non-stationary',
            'Non-normal' if not jb_result['is_normal'] else 'Normal',
        ],
    })

    with open(tables_dir / 'table4_diagnostics.tex', 'w') as f:
        f.write(diag_df.to_latex(index=False, escape=False,
                caption='Time-Series Diagnostic Tests', label='tab:diagnostics'))
    print(f"  Saved: {tables_dir / 'table4_diagnostics.tex'}")

    # Table 5: Correlation Matrix
    corr_slim = corr_matrix.loc[
        ['sentiment', 'epistemic_uncertainty', 'aleatoric_uncertainty', 'spread_bps'],
        ['sentiment', 'epistemic_uncertainty', 'aleatoric_uncertainty', 'spread_bps']
    ]

    with open(tables_dir / 'table5_correlation.tex', 'w') as f:
        f.write(corr_slim.to_latex(escape=False, float_format='%.3f',
                caption='Correlation Matrix (Key Variables)', label='tab:correlation'))
    print(f"  Saved: {tables_dir / 'table5_correlation.tex'}")

    # Table 6: Regime Transitions
    with open(tables_dir / 'table6_transitions.tex', 'w') as f:
        f.write(transitions.to_latex(escape=False, float_format='%.3f',
                caption='Regime Transition Probabilities $P(\\text{regime}_{t+1} | \\text{regime}_t)$',
                label='tab:transitions'))
    print(f"  Saved: {tables_dir / 'table6_transitions.tex'}")

    # ========================================
    # 6. SAVE FULL RESULTS
    # ========================================
    print("\n" + "=" * 40)
    print("6. SAVING FULL RESULTS")
    print("=" * 40)

    # Convert non-serializable items
    results_serializable = {}
    for k, v in results.items():
        if isinstance(v, pd.DataFrame):
            results_serializable[k] = v.to_dict()
        elif isinstance(v, dict):
            results_serializable[k] = v
        else:
            results_serializable[k] = str(v)

    with open(results_dir / 'full_analysis_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2, default=str)
    print(f"  Saved: {results_dir / 'full_analysis_results.json'}")

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nGenerated Figures:")
    print(f"  - {figures_dir / 'return_distribution.pdf'}")
    print(f"  - {figures_dir / 'acf_comparison.pdf'}")
    print(f"  - {figures_dir / 'regime_dynamics.pdf'}")
    print(f"  - {figures_dir / 'uncertainty_decomposition.pdf'}")
    print(f"\nGenerated Tables:")
    print(f"  - {tables_dir / 'table2_summary_stats.tex'}")
    print(f"  - {tables_dir / 'table3_regime_stats.tex'}")
    print(f"  - {tables_dir / 'table4_diagnostics.tex'}")
    print(f"  - {tables_dir / 'table5_correlation.tex'}")
    print(f"  - {tables_dir / 'table6_transitions.tex'}")

    return results


if __name__ == '__main__':
    results = run_full_analysis()
