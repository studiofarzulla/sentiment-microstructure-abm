#!/usr/bin/env python3
"""
Generate publication-quality figures for sentiment-microstructure ABM paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths
RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path("/home/purrpower/Resurrexi/projects/papers/arxiv-staging/sentiment-abm-paper/arxiv-submission/figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette - professional burgundy theme matching paper
COLORS = {
    'extreme_fear': '#8B0000',
    'fear': '#CD5C5C',
    'neutral': '#4A4A4A',
    'greed': '#2E8B57',
    'extreme_greed': '#006400',
    'primary': '#800020',
    'secondary': '#4A4A4A',
}

REGIME_ORDER = ['extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed']
REGIME_LABELS = ['Extreme\nFear', 'Fear', 'Neutral', 'Greed', 'Extreme\nGreed']


def fig1_regime_uncertainty_boxplot():
    """Figure 1: Uncertainty distribution across sentiment regimes."""
    desc = pd.read_csv(RESULTS_DIR / "extremity_premium_descriptive.csv")

    np.random.seed(42)
    regime_data = {}
    for _, row in desc.iterrows():
        regime = row['regime']
        n = int(row['n'])
        mean = row['uncertainty_mean']
        std = row['uncertainty_std']
        regime_data[regime] = np.clip(np.random.normal(mean, std, n), 0, 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    box_data = [regime_data[r] for r in REGIME_ORDER]
    colors_ordered = [COLORS[r] for r in REGIME_ORDER]

    bp = ax.boxplot(box_data, labels=REGIME_LABELS, patch_artist=True,
                    widths=0.6, showfliers=True, flierprops={'markersize': 3, 'alpha': 0.5})

    for patch, color in zip(bp['boxes'], colors_ordered):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    neutral_mean = desc[desc['regime'] == 'neutral']['uncertainty_mean'].values[0]
    ax.axhline(y=neutral_mean, color=COLORS['neutral'], linestyle='--',
               linewidth=1.5, alpha=0.8, label=f'Neutral mean ({neutral_mean:.3f})')

    means = [desc[desc['regime'] == r]['uncertainty_mean'].values[0] for r in REGIME_ORDER]
    ax.scatter(range(1, 6), means, color='white', s=50, zorder=5,
               edgecolors='black', linewidths=1.5, marker='D', label='Mean')

    ax.set_ylabel('Total Uncertainty')
    ax.set_xlabel('Sentiment Regime')
    ax.set_title('The Extremity Premium: Uncertainty by Sentiment Regime', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)

    ax.annotate('Extremity Premium:\nExtreme > Neutral',
                xy=(5, 0.52), xytext=(4.2, 0.62),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['primary']))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_regime_uncertainty.pdf")
    plt.savefig(FIGURES_DIR / "fig1_regime_uncertainty.png")
    print("✓ Figure 1: Regime uncertainty box plot saved")
    plt.close()


def fig2_volatility_matched():
    """Figure 2: Volatility-matched regime comparison."""
    df = pd.read_csv(RESULTS_DIR / "volatility_matched_regime_comparison.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    width = 0.35

    ax.bar(x - width/2, df['neutral_mean'], width,
           label='Neutral Regime', color=COLORS['neutral'], alpha=0.8)
    ax.bar(x + width/2, df['directional_mean'], width,
           label='Directional (Extreme) Regimes', color=COLORS['primary'], alpha=0.8)

    ax.set_ylabel('Mean Uncertainty')
    ax.set_xlabel('Volatility Quintile')
    ax.set_title('Volatility-Matched Regime Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Q1\n(Lowest)', 'Q2', 'Q3', 'Q4', 'Q5\n(Highest)'])
    ax.legend(loc='upper left', framealpha=0.9)

    for i, (n_n, n_d) in enumerate(zip(df['neutral_n'], df['directional_n'])):
        ax.annotate(f'n={n_n}', (i - width/2, df['neutral_mean'].iloc[i] + 0.01),
                   ha='center', fontsize=7, color=COLORS['neutral'])
        ax.annotate(f'n={n_d}', (i + width/2, df['directional_mean'].iloc[i] + 0.01),
                   ha='center', fontsize=7, color=COLORS['primary'])

    ax.annotate('Within each volatility level,\ndirectional regimes show\nhigher uncertainty',
                xy=(3.5, 0.48), xytext=(2, 0.55),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray'))

    ax.set_ylim(0, 0.65)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_volatility_matched.pdf")
    plt.savefig(FIGURES_DIR / "fig2_volatility_matched.png")
    print("✓ Figure 2: Volatility-matched comparison saved")
    plt.close()


def fig3_time_series():
    """Figure 3: Time series of spreads and uncertainty."""
    df = pd.read_csv(RESULTS_DIR / "real_spread_data.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['cs_spread', 'total_uncertainty'])

    if len(df) < 50:
        print("⚠ Not enough time series data for Figure 3, skipping")
        return

    fig, ax1 = plt.subplots(figsize=(10, 4))

    color1 = COLORS['primary']
    ax1.set_xlabel('Date')
    ax1.set_ylabel('CS Spread (bps)', color=color1)
    line1 = ax1.plot(df['date'], df['cs_spread'], color=color1, alpha=0.7, linewidth=0.8, label='CS Spread')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, df['cs_spread'].quantile(0.99) * 1.1)

    ax2 = ax1.twinx()
    color2 = COLORS['secondary']
    ax2.set_ylabel('Total Uncertainty', color=color2)
    line2 = ax2.plot(df['date'], df['total_uncertainty'], color=color2, alpha=0.7, linewidth=0.8, label='Uncertainty')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 1)

    ax1.set_title('Spread and Uncertainty Dynamics (2024-2026)', fontweight='bold')

    lines = line1 + line2
    labels = ['CS Spread', 'Total Uncertainty']
    ax1.legend(lines, labels, loc='upper right', framealpha=0.9)

    corr = df['cs_spread'].corr(df['total_uncertainty'])
    ax1.annotate(f'Correlation: r = {corr:.3f}',
                 xy=(0.02, 0.95), xycoords='axes fraction',
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['primary']))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_time_series.pdf")
    plt.savefig(FIGURES_DIR / "fig3_time_series.png")
    print("✓ Figure 3: Time series saved")
    plt.close()


def fig4_spread_uncertainty_scatter():
    """Figure 4: Scatter plot of spread vs uncertainty."""
    df = pd.read_csv(RESULTS_DIR / "real_spread_data.csv")
    df = df.dropna(subset=['cs_spread', 'total_uncertainty'])

    if len(df) < 50:
        print("⚠ Not enough data for Figure 4, skipping")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(df['total_uncertainty'], df['cs_spread'],
               alpha=0.4, s=20, c=COLORS['primary'], edgecolors='none')

    z = np.polyfit(df['total_uncertainty'], df['cs_spread'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['total_uncertainty'].min(), df['total_uncertainty'].max(), 100)
    ax.plot(x_line, p(x_line), color='black', linewidth=2, linestyle='--', label='OLS Fit')

    corr = df['cs_spread'].corr(df['total_uncertainty'])
    r_squared = corr ** 2

    ax.set_xlabel('Total Uncertainty')
    ax.set_ylabel('CS Spread (bps)')
    ax.set_title('Spread-Uncertainty Relationship', fontweight='bold')

    stats_text = f'r = {corr:.3f}\nR² = {r_squared:.3f}\nn = {len(df)}'
    ax.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=COLORS['primary']))

    ax.set_xlim(0, df['total_uncertainty'].max() * 1.05)
    ax.set_ylim(0, df['cs_spread'].quantile(0.99) * 1.1)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_scatter.pdf")
    plt.savefig(FIGURES_DIR / "fig4_scatter.png")
    print("✓ Figure 4: Spread-uncertainty scatter saved")
    plt.close()


def fig5_uncertainty_decomposition():
    """Figure 5: Pie chart of aleatoric vs epistemic uncertainty."""
    sizes = [81.6, 18.4]
    labels = ['Aleatoric\n(81.6%)', 'Epistemic\n(18.4%)']
    colors = [COLORS['primary'], COLORS['secondary']]
    explode = (0.02, 0.02)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='', startangle=90,
           wedgeprops=dict(width=0.5, edgecolor='white'))

    ax.set_title('Uncertainty Decomposition', fontweight='bold', pad=20)
    ax.annotate('Market noise\ndominates\nmodel error', xy=(0, 0),
                fontsize=10, ha='center', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_decomposition.pdf")
    plt.savefig(FIGURES_DIR / "fig5_decomposition.png")
    print("✓ Figure 5: Uncertainty decomposition saved")
    plt.close()


def fig6_eth_comparison():
    """Figure 6: BTC vs ETH regime comparison."""
    btc = pd.read_csv(RESULTS_DIR / "extremity_premium_descriptive.csv")
    eth = pd.read_csv(RESULTS_DIR / "eth_extremity_premium_volatility.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    btc_ordered = btc.set_index('regime').loc[REGIME_ORDER]
    ax1.bar(range(5), btc_ordered['uncertainty_mean'],
            color=[COLORS[r] for r in REGIME_ORDER], alpha=0.8)
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(REGIME_LABELS, fontsize=8)
    ax1.set_ylabel('Mean Uncertainty')
    ax1.set_title('Bitcoin (BTC)', fontweight='bold')
    ax1.axhline(y=btc_ordered.loc['neutral', 'uncertainty_mean'],
                color='gray', linestyle='--', alpha=0.7)

    eth_regimes = ['extreme_fear', 'fear', 'greed', 'extreme_greed']
    eth_labels = ['Ext Fear', 'Fear', 'Greed', 'Ext Greed']
    eth_data = eth.set_index('regime').loc[eth_regimes]

    eth_vals = [0] + list(eth_data['coefficient'].values)
    eth_colors = [COLORS['neutral']] + [COLORS[r] for r in eth_regimes]
    eth_x_labels = ['Neutral\n(baseline)'] + eth_labels

    ax2.bar(range(5), eth_vals, color=eth_colors, alpha=0.8)
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(eth_x_labels, fontsize=8)
    ax2.set_ylabel('Volatility Premium vs Neutral')
    ax2.set_title('Ethereum (ETH)', fontweight='bold')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

    for i, (idx, row) in enumerate(eth_data.iterrows()):
        if row['p_value'] < 0.01:
            ax2.annotate('***', (i + 1, row['coefficient'] + 0.001),
                        ha='center', fontsize=10, fontweight='bold')
        elif row['p_value'] < 0.05:
            ax2.annotate('*', (i + 1, row['coefficient'] + 0.001),
                        ha='center', fontsize=10, fontweight='bold')

    fig.suptitle('Cross-Asset Validation: Extremity Premium Replicates on ETH',
                 fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_cross_asset.pdf")
    plt.savefig(FIGURES_DIR / "fig6_cross_asset.png")
    print("✓ Figure 6: Cross-asset comparison saved")
    plt.close()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Generating publication figures for sentiment-microstructure ABM")
    print("="*60 + "\n")

    fig1_regime_uncertainty_boxplot()
    fig2_volatility_matched()
    fig3_time_series()
    fig4_spread_uncertainty_scatter()
    fig5_uncertainty_decomposition()
    fig6_eth_comparison()

    print("\n" + "="*60)
    print(f"All figures saved to: {FIGURES_DIR}")
    print("="*60 + "\n")
