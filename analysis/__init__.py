"""
Analysis module for Sentiment-Microstructure ABM.

Provides statistical analysis, regime analysis, sensitivity analysis,
ablation studies, and figure generation for paper results.
"""

from .statistical_analysis import (
    compute_return_statistics,
    adf_test,
    kpss_test,
    compute_acf,
    ljung_box_test,
    volatility_clustering_test,
    jarque_bera_test,
)

from .regime_analysis import (
    compute_regime_statistics,
    compute_regime_transitions,
    compute_regime_durations,
)

from .figure_generation import (
    plot_return_distribution,
    plot_acf_comparison,
    plot_regime_dynamics,
    plot_sensitivity_heatmap,
    FARZULLA_BURGUNDY,
)

from .ablation_analysis import (
    run_full_ablation_study,
    AblationConfig,
    AblationRunner,
    AblationExporter,
)

__all__ = [
    # Statistical analysis
    'compute_return_statistics',
    'adf_test',
    'kpss_test',
    'compute_acf',
    'ljung_box_test',
    'volatility_clustering_test',
    'jarque_bera_test',
    # Regime analysis
    'compute_regime_statistics',
    'compute_regime_transitions',
    'compute_regime_durations',
    # Figure generation
    'plot_return_distribution',
    'plot_acf_comparison',
    'plot_regime_dynamics',
    'plot_sensitivity_heatmap',
    'FARZULLA_BURGUNDY',
    # Ablation analysis
    'run_full_ablation_study',
    'AblationConfig',
    'AblationRunner',
    'AblationExporter',
]
