"""
Figure generation for Sentiment-Microstructure ABM paper.

Publication-quality figures using Farzulla Research color scheme.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as scipy_stats
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Farzulla Research color scheme
FARZULLA_BURGUNDY = '#800020'
FARZULLA_BLUE = '#3498db'
FARZULLA_GREEN = '#2ecc71'
FARZULLA_PURPLE = '#9b59b6'
FARZULLA_RED = '#e74c3c'
FARZULLA_GRAY = '#95a5a6'

REGIME_COLORS = {
    'bullish': '#2ecc71',   # Green
    'neutral': '#95a5a6',   # Gray
    'bearish': '#e74c3c',   # Red
}


def set_paper_style():
    """Set matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.figsize': (8, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def plot_return_distribution(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    return_col: str = 'log_return'
) -> plt.Figure:
    """
    Generate return distribution figure with histogram and Q-Q plot.

    Panel A: Histogram with normal overlay
    Panel B: Q-Q plot against normal distribution

    Args:
        df: DataFrame with return data
        save_path: Path to save figure (PDF)
        return_col: Name of return column

    Returns:
        matplotlib Figure
    """
    set_paper_style()

    returns = df[return_col].dropna()
    if returns.iloc[0] == 0:
        returns = returns.iloc[1:]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Histogram with normal overlay
    ax1 = axes[0]

    # Histogram
    n, bins, patches = ax1.hist(
        returns, bins=50, density=True, alpha=0.7,
        color=FARZULLA_BURGUNDY, edgecolor='white', linewidth=0.5,
        label='Simulated Returns'
    )

    # Fit normal distribution
    mu, std = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    normal_pdf = scipy_stats.norm.pdf(x, mu, std)
    ax1.plot(x, normal_pdf, color=FARZULLA_BLUE, linewidth=2,
             label=f'Normal ($\\mu$={mu:.2e}, $\\sigma$={std:.2e})')

    ax1.set_xlabel('Log Return')
    ax1.set_ylabel('Density')
    ax1.set_title('A) Return Distribution')
    ax1.legend(loc='upper right')

    # Add statistics annotation
    skew = scipy_stats.skew(returns)
    kurt = scipy_stats.kurtosis(returns)
    jb_stat, jb_pval = scipy_stats.jarque_bera(returns)

    stats_text = (f'Skewness: {skew:.3f}\n'
                  f'Excess Kurtosis: {kurt:.3f}\n'
                  f'J-B p-value: {jb_pval:.3f}')
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel B: Q-Q Plot
    ax2 = axes[1]

    # Q-Q plot
    (osm, osr), (slope, intercept, r) = scipy_stats.probplot(returns, dist='norm')
    ax2.scatter(osm, osr, c=FARZULLA_BURGUNDY, alpha=0.5, s=10, label='Sample Quantiles')
    ax2.plot(osm, slope * np.array(osm) + intercept, color=FARZULLA_BLUE,
             linewidth=2, label=f'Normal Line ($R^2$={r**2:.3f})')

    ax2.set_xlabel('Theoretical Quantiles (Normal)')
    ax2.set_ylabel('Sample Quantiles')
    ax2.set_title('B) Q-Q Plot')
    ax2.legend(loc='lower right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    return fig


def plot_acf_comparison(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    return_col: str = 'log_return',
    nlags: int = 30
) -> plt.Figure:
    """
    Generate ACF comparison figure for returns vs |returns|.

    Panel A: ACF of returns (should decay quickly)
    Panel B: ACF of |returns| (should persist - volatility clustering)

    Args:
        df: DataFrame with return data
        save_path: Path to save figure
        return_col: Name of return column
        nlags: Number of lags to plot

    Returns:
        matplotlib Figure
    """
    from statsmodels.tsa.stattools import acf

    set_paper_style()

    returns = df[return_col].dropna()
    if returns.iloc[0] == 0:
        returns = returns.iloc[1:]

    abs_returns = returns.abs()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Compute ACFs
    acf_returns, confint_returns = acf(returns, nlags=nlags, alpha=0.05, fft=True)
    acf_abs_returns, confint_abs = acf(abs_returns, nlags=nlags, alpha=0.05, fft=True)

    lags = np.arange(nlags + 1)

    # Panel A: ACF of returns
    ax1 = axes[0]

    # Confidence bounds
    lower = confint_returns[:, 0] - acf_returns
    upper = confint_returns[:, 1] - acf_returns
    ax1.fill_between(lags, lower, upper, color=FARZULLA_GRAY, alpha=0.3,
                     label='95% CI')

    # ACF bars
    markerline, stemlines, baseline = ax1.stem(lags, acf_returns, basefmt=' ')
    plt.setp(stemlines, color=FARZULLA_BURGUNDY, linewidth=1.5)
    plt.setp(markerline, color=FARZULLA_BURGUNDY, markersize=4)

    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Autocorrelation')
    ax1.set_title('A) ACF of Returns')
    ax1.set_xlim(-0.5, nlags + 0.5)

    # Panel B: ACF of |returns|
    ax2 = axes[1]

    # Confidence bounds
    lower_abs = confint_abs[:, 0] - acf_abs_returns
    upper_abs = confint_abs[:, 1] - acf_abs_returns
    ax2.fill_between(lags, lower_abs, upper_abs, color=FARZULLA_GRAY, alpha=0.3,
                     label='95% CI')

    # ACF bars
    markerline, stemlines, baseline = ax2.stem(lags, acf_abs_returns, basefmt=' ')
    plt.setp(stemlines, color=FARZULLA_BURGUNDY, linewidth=1.5)
    plt.setp(markerline, color=FARZULLA_BURGUNDY, markersize=4)

    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.axhline(y=0.1, color=FARZULLA_RED, linewidth=1, linestyle='--',
                label='Volatility clustering threshold (0.1)')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Autocorrelation')
    ax2.set_title('B) ACF of |Returns| (Volatility)')
    ax2.set_xlim(-0.5, nlags + 0.5)
    ax2.legend(loc='upper right', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    return fig


def plot_regime_dynamics(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    max_steps: int = 500
) -> plt.Figure:
    """
    Generate regime dynamics visualization.

    Panel A: Sentiment time series with regime shading
    Panel B: Spread time series with regime shading
    Panel C: Regime duration histogram

    Args:
        df: DataFrame with simulation data
        save_path: Path to save figure
        max_steps: Maximum steps to plot (for readability)

    Returns:
        matplotlib Figure
    """
    set_paper_style()

    # Use subset for readability
    plot_df = df.iloc[:max_steps].copy()

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # Panel A: Sentiment with regime shading
    ax1 = axes[0]

    ax1.plot(plot_df['step'], plot_df['sentiment'], color=FARZULLA_BURGUNDY,
             linewidth=0.8, alpha=0.8)

    # Add regime shading
    _add_regime_shading(ax1, plot_df)

    ax1.axhline(y=0.2, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax1.axhline(y=-0.2, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_ylabel('Sentiment Score')
    ax1.set_title('A) Sentiment Dynamics with Regime Identification')
    ax1.set_xlim(0, max_steps)

    # Add legend
    patches = [mpatches.Patch(color=REGIME_COLORS[r], alpha=0.2, label=r.capitalize())
               for r in ['bullish', 'neutral', 'bearish']]
    ax1.legend(handles=patches, loc='upper right', ncol=3)

    # Panel B: Spread with regime shading
    ax2 = axes[1]

    ax2.plot(plot_df['step'], plot_df['spread_bps'], color=FARZULLA_BLUE,
             linewidth=0.8, alpha=0.8)

    _add_regime_shading(ax2, plot_df)

    ax2.set_ylabel('Spread (bps)')
    ax2.set_title('B) Bid-Ask Spread Response to Sentiment Regimes')
    ax2.set_xlim(0, max_steps)

    # Panel C: Regime durations
    ax3 = axes[2]

    # Compute regime durations
    regime_changes = df['regime'] != df['regime'].shift(1)
    regime_changes.iloc[0] = True
    episode_starts = df.index[regime_changes].tolist()
    episode_starts.append(len(df))

    durations = {r: [] for r in ['bullish', 'neutral', 'bearish']}
    for i in range(len(episode_starts) - 1):
        start_idx = episode_starts[i]
        end_idx = episode_starts[i + 1]
        regime = df['regime'].iloc[start_idx]
        duration = end_idx - start_idx
        durations[regime].append(duration)

    # Plot histograms
    for i, (regime, color) in enumerate(REGIME_COLORS.items()):
        if durations[regime]:
            ax3.hist(durations[regime], bins=20, alpha=0.6, color=color,
                     label=f'{regime.capitalize()} (n={len(durations[regime])})',
                     edgecolor='white')

    ax3.set_xlabel('Duration (timesteps)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('C) Distribution of Regime Durations')
    ax3.legend(loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    return fig


def _add_regime_shading(ax, df):
    """Add regime background shading to an axis."""
    regime_changes = df['regime'] != df['regime'].shift(1)
    regime_changes.iloc[0] = True
    change_points = df.index[regime_changes].tolist()
    change_points.append(len(df) - 1)

    for i in range(len(change_points) - 1):
        start = df['step'].iloc[change_points[i]]
        end = df['step'].iloc[change_points[i + 1]] if change_points[i + 1] < len(df) else df['step'].iloc[-1]
        regime = df['regime'].iloc[change_points[i]]
        color = REGIME_COLORS.get(regime, FARZULLA_GRAY)
        ax.axvspan(start, end, alpha=0.15, color=color)


def plot_sensitivity_heatmap(
    sweep_results: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate sensitivity analysis heatmaps.

    2x2 grid of heatmaps showing how key metrics vary with parameters.

    Args:
        sweep_results: DataFrame with parameter sweep results
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    set_paper_style()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    metrics = [
        ('sentiment_spread_corr', 'A) Sentiment-Spread Correlation'),
        ('epistemic_spread_corr', 'B) Epistemic-Spread Correlation'),
        ('mean_spread_bps', 'C) Mean Spread (bps)'),
        ('inventory_volatility', 'D) Inventory Volatility'),
    ]

    for ax, (metric, title) in zip(axes.flat, metrics):
        if metric in sweep_results.columns:
            pivot = sweep_results.pivot_table(
                values=metric,
                index='sentiment_sensitivity',
                columns='epistemic_sensitivity',
                aggfunc='mean'
            )

            im = ax.imshow(pivot.values, cmap='RdYlBu_r', aspect='auto',
                           origin='lower')
            ax.set_xlabel('Epistemic Sensitivity')
            ax.set_ylabel('Sentiment Sensitivity')
            ax.set_title(title)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.ax.tick_params(labelsize=8)

            # Set tick labels
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f'{x:.1f}' for x in pivot.columns], fontsize=8)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f'{y:.1f}' for y in pivot.index], fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    return fig


def plot_uncertainty_decomposition(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate uncertainty decomposition visualization.

    Panel A: Epistemic vs Aleatoric contribution over time
    Panel B: Scatter plot of uncertainty vs spread by type

    Args:
        df: DataFrame with simulation data
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    set_paper_style()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Uncertainty over time (subsample for readability)
    ax1 = axes[0]
    plot_df = df.iloc[::4]  # Every 4th point

    ax1.fill_between(plot_df['step'], 0, plot_df['epistemic_uncertainty'],
                     color=FARZULLA_BLUE, alpha=0.5, label='Epistemic')
    ax1.fill_between(plot_df['step'], plot_df['epistemic_uncertainty'],
                     plot_df['total_uncertainty'],
                     color=FARZULLA_RED, alpha=0.5, label='Aleatoric')

    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Uncertainty')
    ax1.set_title('A) Uncertainty Decomposition Over Time')
    ax1.legend(loc='upper right')

    # Panel B: Scatter of uncertainty vs spread
    ax2 = axes[1]

    ax2.scatter(df['epistemic_uncertainty'], df['spread_bps'],
                alpha=0.3, s=10, c=FARZULLA_BLUE, label='Epistemic')
    ax2.scatter(df['aleatoric_uncertainty'], df['spread_bps'],
                alpha=0.3, s=10, c=FARZULLA_RED, label='Aleatoric')

    # Add trend lines
    z_epi = np.polyfit(df['epistemic_uncertainty'], df['spread_bps'], 1)
    z_ale = np.polyfit(df['aleatoric_uncertainty'], df['spread_bps'], 1)

    x_range = np.linspace(0, df['total_uncertainty'].max(), 100)
    ax2.plot(x_range, np.polyval(z_epi, x_range), color=FARZULLA_BLUE,
             linewidth=2, linestyle='--')
    ax2.plot(x_range, np.polyval(z_ale, x_range), color=FARZULLA_RED,
             linewidth=2, linestyle='--')

    ax2.set_xlabel('Uncertainty')
    ax2.set_ylabel('Spread (bps)')
    ax2.set_title('B) Spread Response to Uncertainty Components')
    ax2.legend(loc='upper left')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    return fig
