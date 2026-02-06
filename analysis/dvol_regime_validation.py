"""
DVOL-Based Regime Validation

Replicates extremity premium analysis using Deribit DVOL instead of F&G Index.
This validates that the finding isn't an artifact of F&G construction.

Key insight:
- High DVOL = market panic (fear) - implied volatility spikes during uncertainty
- Low DVOL = complacency (greed) - low implied volatility during calm/euphoria

This is an INDEPENDENT measure from F&G because:
1. DVOL is pure options-derived implied volatility
2. F&G includes volatility as only ONE of 7 components (25% weight)
3. F&G uses other signals: momentum, social, dominance, etc.
4. DVOL reflects institutional derivative positioning, not retail sentiment

Author: Murad Farzulla
Date: January 2026
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

from data_ingestion.dvol_fetcher import DeribitDVOLFetcher


def fetch_dvol_data(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch DVOL historical data from Deribit."""
    fetcher = DeribitDVOLFetcher()

    print(f"Fetching DVOL data from {start_date.date()} to {end_date.date()}...")

    dvol_df = fetcher.fetch_historical_dvol(
        start_time=start_date,
        end_time=end_date,
        resolution="1D"
    )

    if len(dvol_df) == 0:
        print("⚠ No DVOL data from API, attempting cache/fallback...")
        # Try to load from cache if it exists
        cache_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data', 'cache', 'dvol_historical.csv'
        )
        if os.path.exists(cache_path):
            dvol_df = pd.read_csv(cache_path, parse_dates=['timestamp'])
            print(f"  Loaded {len(dvol_df)} records from cache")

    return dvol_df


def create_dvol_regimes(dvol_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create quintile-based sentiment regimes from DVOL.

    Mapping:
    - Q1 (lowest DVOL): Extreme Greed (complacency, low implied vol)
    - Q2: Greed
    - Q3: Neutral
    - Q4: Fear
    - Q5 (highest DVOL): Extreme Fear (panic, high implied vol)
    """
    df = dvol_df.copy()

    # Compute quintile boundaries
    quantiles = df['dvol'].quantile([0.2, 0.4, 0.6, 0.8]).values

    def classify_regime(dvol):
        if dvol <= quantiles[0]:
            return 'extreme_greed'  # Lowest DVOL = complacency
        elif dvol <= quantiles[1]:
            return 'greed'
        elif dvol <= quantiles[2]:
            return 'neutral'
        elif dvol <= quantiles[3]:
            return 'fear'
        else:
            return 'extreme_fear'  # Highest DVOL = panic

    df['dvol_regime'] = df['dvol'].apply(classify_regime)

    print(f"\nDVOL quintile boundaries:")
    print(f"  Extreme Greed: DVOL ≤ {quantiles[0]:.1f}%")
    print(f"  Greed: DVOL {quantiles[0]:.1f}% - {quantiles[1]:.1f}%")
    print(f"  Neutral: DVOL {quantiles[1]:.1f}% - {quantiles[2]:.1f}%")
    print(f"  Fear: DVOL {quantiles[2]:.1f}% - {quantiles[3]:.1f}%")
    print(f"  Extreme Fear: DVOL > {quantiles[3]:.1f}%")

    return df


def load_uncertainty_data() -> pd.DataFrame:
    """Load the existing uncertainty/spread data."""
    # Primary location: project root results/
    results_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'results', 'real_spread_data.csv'
    )

    if not os.path.exists(results_path):
        print(f"⚠ Results file not found: {results_path}")
        print("  Attempting to load from alternative location...")

        # Try analysis/results
        alt_path = os.path.join(
            os.path.dirname(__file__),
            'results', 'real_spread_data.csv'
        )
        if os.path.exists(alt_path):
            results_path = alt_path
        else:
            raise FileNotFoundError("Cannot find real_spread_data.csv")

    df = pd.read_csv(results_path, parse_dates=['date'])
    print(f"Loaded uncertainty data: {len(df)} days")

    return df


def merge_dvol_with_uncertainty(dvol_df: pd.DataFrame, uncertainty_df: pd.DataFrame) -> pd.DataFrame:
    """Merge DVOL regimes with uncertainty data on date."""
    # Standardize date columns
    dvol_df = dvol_df.copy()
    dvol_df['date'] = pd.to_datetime(dvol_df['timestamp']).dt.normalize()

    uncertainty_df = uncertainty_df.copy()
    uncertainty_df['date'] = pd.to_datetime(uncertainty_df['date']).dt.normalize()

    # Merge
    merged = pd.merge(
        uncertainty_df,
        dvol_df[['date', 'dvol', 'dvol_regime']],
        on='date',
        how='inner'
    )

    print(f"\nMerged dataset: {len(merged)} observations")
    print(f"Date range: {merged['date'].min().date()} to {merged['date'].max().date()}")

    return merged


def descriptive_by_dvol_regime(df: pd.DataFrame):
    """Compute descriptive stats by DVOL regime."""
    print("\n" + "="*70)
    print("DVOL REGIME DESCRIPTIVE STATISTICS")
    print("="*70)

    results = []
    regime_order = ['extreme_greed', 'greed', 'neutral', 'fear', 'extreme_fear']

    for regime in regime_order:
        subset = df[df['dvol_regime'] == regime]
        if len(subset) == 0:
            continue

        results.append({
            'dvol_regime': regime,
            'n': len(subset),
            'dvol_mean': subset['dvol'].mean(),
            'dvol_std': subset['dvol'].std(),
            'uncertainty_mean': subset['total_uncertainty'].mean(),
            'uncertainty_std': subset['total_uncertainty'].std(),
            'volatility_mean': subset['realized_vol'].mean() if 'realized_vol' in subset else np.nan,
        })

    results_df = pd.DataFrame(results)

    print("\nRaw statistics by DVOL regime:")
    print(results_df.to_string(index=False))

    # Compute vs neutral difference
    neutral_mean = results_df[results_df['dvol_regime'] == 'neutral']['uncertainty_mean'].values[0]

    print(f"\n★ Uncertainty relative to neutral ({neutral_mean:.4f}):")
    for _, row in results_df.iterrows():
        diff = row['uncertainty_mean'] - neutral_mean
        pct_diff = 100 * diff / neutral_mean
        print(f"  {row['dvol_regime']:15s}: {row['uncertainty_mean']:.4f} ({pct_diff:+.1f}% vs neutral)")

    return results_df


def regression_dvol_regimes(df: pd.DataFrame):
    """
    Run OLS regression: Uncertainty ~ Volatility + DVOL Regime Dummies

    This tests whether DVOL-based regimes predict uncertainty
    independently of realized volatility.
    """
    print("\n" + "="*70)
    print("REGRESSION: Uncertainty ~ Volatility + DVOL Regime Dummies")
    print("="*70)

    df = df.copy()

    # Create dummy variables (neutral = reference)
    for reg in ['extreme_fear', 'fear', 'greed', 'extreme_greed']:
        df[f'is_{reg}'] = (df['dvol_regime'] == reg).astype(int)

    # Prepare variables
    df['volatility'] = df['realized_vol'].fillna(df['parkinson_vol'] if 'parkinson_vol' in df else np.nan)
    df_clean = df.dropna(subset=['total_uncertainty', 'volatility'])

    print(f"Clean observations: {len(df_clean)}")

    # Model with regime dummies
    X = df_clean[['volatility', 'is_extreme_fear', 'is_fear', 'is_greed', 'is_extreme_greed']]
    X = sm.add_constant(X)
    y = df_clean['total_uncertainty']

    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})

    print("\nRegression results (Newey-West HAC SEs, 5 lags):")
    print(f"  R² = {model.rsquared:.3f}")
    print(f"  Adj. R² = {model.rsquared_adj:.3f}")
    print()

    # Extract coefficients
    results = []
    for var in ['is_extreme_fear', 'is_fear', 'is_greed', 'is_extreme_greed', 'volatility']:
        coef = model.params.get(var, np.nan)
        se = model.bse.get(var, np.nan)
        pval = model.pvalues.get(var, np.nan)
        results.append({
            'variable': var,
            'coefficient': coef,
            'std_error': se,
            'p_value': pval,
            'significant': pval < 0.05
        })

    for r in results:
        sig = '***' if r['p_value'] < 0.001 else '**' if r['p_value'] < 0.01 else '*' if r['p_value'] < 0.05 else ''
        print(f"  {r['variable']:18s}: {r['coefficient']:+.4f} (SE={r['std_error']:.4f}, p={r['p_value']:.4f}){sig}")

    return model, pd.DataFrame(results)


def test_extremity_premium(df: pd.DataFrame):
    """
    Test the core hypothesis: Do extreme DVOL regimes show higher uncertainty?

    Pool extreme greed + extreme fear vs neutral, compute effect size.
    """
    print("\n" + "="*70)
    print("EXTREMITY PREMIUM TEST (DVOL-BASED)")
    print("="*70)

    df = df.copy()
    df['volatility'] = df['realized_vol'].fillna(df.get('parkinson_vol', np.nan))
    df = df.dropna(subset=['total_uncertainty', 'volatility', 'dvol_regime'])

    # Create extreme indicator
    df['is_extreme'] = df['dvol_regime'].isin(['extreme_greed', 'extreme_fear'])
    df['is_neutral'] = df['dvol_regime'] == 'neutral'

    extreme = df[df['is_extreme']]['total_uncertainty']
    neutral = df[df['is_neutral']]['total_uncertainty']

    if len(extreme) == 0 or len(neutral) == 0:
        print("⚠ Insufficient data for extremity test")
        return None

    # Two-sample t-test
    t_stat, p_value = stats.ttest_ind(extreme, neutral)

    # Cohen's d
    pooled_std = np.sqrt((extreme.std()**2 + neutral.std()**2) / 2)
    cohens_d = (extreme.mean() - neutral.mean()) / pooled_std

    print(f"\nPooled extreme (greed + fear) vs neutral:")
    print(f"  Extreme mean: {extreme.mean():.4f} (n={len(extreme)})")
    print(f"  Neutral mean: {neutral.mean():.4f} (n={len(neutral)})")
    print(f"  Difference: {extreme.mean() - neutral.mean():+.4f}")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value (two-sided): {p_value:.6f}")
    print(f"  Cohen's d: {cohens_d:.3f}")

    # Effect size interpretation
    if abs(cohens_d) >= 0.8:
        effect = "LARGE"
    elif abs(cohens_d) >= 0.5:
        effect = "MEDIUM"
    elif abs(cohens_d) >= 0.2:
        effect = "SMALL"
    else:
        effect = "NEGLIGIBLE"

    print(f"  Effect size: {effect}")

    return {
        'extreme_mean': extreme.mean(),
        'neutral_mean': neutral.mean(),
        'difference': extreme.mean() - neutral.mean(),
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'effect_size': effect,
        'n_extreme': len(extreme),
        'n_neutral': len(neutral),
    }


def compare_fg_dvol_regimes(df: pd.DataFrame, fg_sentiment_path: str = None):
    """
    Compare F&G-based and DVOL-based regime classifications.

    Computes:
    1. Concordance rate on extreme classifications
    2. Pattern correlation between regime coefficients
    """
    print("\n" + "="*70)
    print("CROSS-VALIDATION: F&G vs DVOL REGIME AGREEMENT")
    print("="*70)

    # Load F&G sentiment data
    if fg_sentiment_path is None:
        fg_sentiment_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data', 'datasets', 'btc_sentiment_daily.csv'
        )

    if not os.path.exists(fg_sentiment_path):
        print(f"⚠ F&G data not found: {fg_sentiment_path}")
        return None

    fg_df = pd.read_csv(fg_sentiment_path, parse_dates=['date'])
    fg_df['date'] = pd.to_datetime(fg_df['date']).dt.normalize()

    # Merge with DVOL data
    df = df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.normalize()

    merged = pd.merge(df, fg_df[['date', 'regime']], on='date', how='inner')
    merged = merged.rename(columns={'regime': 'fg_regime'})

    print(f"Merged for comparison: {len(merged)} days")

    # Concordance: both extreme or both non-extreme
    merged['fg_extreme'] = merged['fg_regime'].isin(['extreme_greed', 'extreme_fear'])
    merged['dvol_extreme'] = merged['dvol_regime'].isin(['extreme_greed', 'extreme_fear'])

    concordance = (merged['fg_extreme'] == merged['dvol_extreme']).mean()
    print(f"\nExtreme vs non-extreme concordance: {100*concordance:.1f}%")

    # Count specific agreement types
    both_extreme = ((merged['fg_extreme']) & (merged['dvol_extreme'])).sum()
    both_non = ((~merged['fg_extreme']) & (~merged['dvol_extreme'])).sum()
    fg_only = ((merged['fg_extreme']) & (~merged['dvol_extreme'])).sum()
    dvol_only = ((~merged['fg_extreme']) & (merged['dvol_extreme'])).sum()

    print(f"  Both classify as extreme: {both_extreme} days ({100*both_extreme/len(merged):.1f}%)")
    print(f"  Both classify as non-extreme: {both_non} days ({100*both_non/len(merged):.1f}%)")
    print(f"  F&G extreme, DVOL non-extreme: {fg_only} days ({100*fg_only/len(merged):.1f}%)")
    print(f"  DVOL extreme, F&G non-extreme: {dvol_only} days ({100*dvol_only/len(merged):.1f}%)")

    # Direction agreement for extreme days
    extreme_days = merged[(merged['fg_extreme']) | (merged['dvol_extreme'])]
    if len(extreme_days) > 0:
        # Check if they agree on direction (both fear or both greed type)
        merged['fg_fear_type'] = merged['fg_regime'].isin(['fear', 'extreme_fear'])
        merged['dvol_fear_type'] = merged['dvol_regime'].isin(['fear', 'extreme_fear'])
        direction_agree = (merged['fg_fear_type'] == merged['dvol_fear_type']).mean()
        print(f"\nDirection agreement (fear vs greed type): {100*direction_agree:.1f}%")

    return {
        'concordance': concordance,
        'both_extreme': both_extreme,
        'both_non_extreme': both_non,
        'fg_only_extreme': fg_only,
        'dvol_only_extreme': dvol_only,
        'n_days': len(merged),
    }


def generate_latex_table(desc_df: pd.DataFrame, regression_results: pd.DataFrame,
                         extremity_results: dict) -> str:
    """Generate LaTeX table for paper."""

    # Build coefficient dict from regression
    coef_dict = {}
    for _, row in regression_results.iterrows():
        var = row['variable']
        if var.startswith('is_'):
            regime = var.replace('is_', '')
            coef_dict[regime] = {
                'coef': row['coefficient'],
                'p': row['p_value']
            }

    latex = r"""
\begin{table}[h!]
\centering
\caption{DVOL-Based Regime Uncertainty}
\label{tab:dvol_regimes}
\small
\begin{tabular}{@{}lccccc@{}}
\toprule
\textbf{DVOL Regime} & \textbf{N} & \textbf{Mean Unc.} & \textbf{$\Delta$ vs Neutral} & \textbf{Coef.} & \textbf{p-value} \\
\midrule
"""

    # Get neutral baseline
    neutral_row = desc_df[desc_df['dvol_regime'] == 'neutral'].iloc[0]
    neutral_mean = neutral_row['uncertainty_mean']

    # Add rows
    regime_order = ['extreme_greed', 'greed', 'neutral', 'fear', 'extreme_fear']
    for regime in regime_order:
        row = desc_df[desc_df['dvol_regime'] == regime]
        if len(row) == 0:
            continue
        row = row.iloc[0]

        n = int(row['n'])
        mean_unc = row['uncertainty_mean']

        if regime == 'neutral':
            delta = "---"
            coef = "---"
            pval = "---"
        else:
            pct_diff = 100 * (mean_unc - neutral_mean) / neutral_mean
            delta = f"{pct_diff:+.1f}\\%"

            if regime in coef_dict:
                coef = f"{coef_dict[regime]['coef']:.3f}"
                p = coef_dict[regime]['p']
                pval = f"{p:.3f}" if p >= 0.001 else "$<$0.001"
            else:
                coef = "---"
                pval = "---"

        # Format regime name
        regime_fmt = regime.replace('_', ' ').title()
        if regime == 'neutral':
            regime_fmt += " (ref)"

        latex += f"{regime_fmt} & {n} & {mean_unc:.3f} & {delta} & {coef} & {pval} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\vspace{0.3em}
"""

    # Add notes with effect size
    d_value = extremity_results['cohens_d'] if extremity_results else "---"
    if isinstance(d_value, float):
        d_str = f"{d_value:.2f}"
    else:
        d_str = str(d_value)

    latex += r"""\caption*{\footnotesize \textit{Notes:} DVOL quintile classification from Deribit implied
volatility data (Jan 2024--Jan 2026). Coefficients from regression with Parkinson volatility
control. Extreme regimes pooled: $d = """ + d_str + r"""$.}
\end{table}
"""

    return latex


def main():
    """Run full DVOL regime validation analysis."""
    print("="*70)
    print("DVOL REGIME VALIDATION ANALYSIS")
    print("Alternative Sentiment Proxy for Extremity Premium")
    print("="*70)

    # Date range matching main analysis
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2026, 1, 10)

    # 1. Fetch DVOL data
    dvol_df = fetch_dvol_data(start_date, end_date)

    if len(dvol_df) == 0:
        print("\n⚠ ERROR: Could not fetch DVOL data")
        print("  Check network connection and Deribit API status")
        return None

    print(f"\nDVOL data summary:")
    print(f"  Range: {dvol_df['dvol'].min():.1f}% - {dvol_df['dvol'].max():.1f}%")
    print(f"  Mean: {dvol_df['dvol'].mean():.1f}%")
    print(f"  Observations: {len(dvol_df)}")

    # 2. Create DVOL regimes
    dvol_df = create_dvol_regimes(dvol_df)

    # 3. Load uncertainty data
    uncertainty_df = load_uncertainty_data()

    # 4. Merge
    merged_df = merge_dvol_with_uncertainty(dvol_df, uncertainty_df)

    if len(merged_df) == 0:
        print("\n⚠ ERROR: No overlapping dates between DVOL and uncertainty data")
        return None

    # 5. Descriptive statistics
    desc_df = descriptive_by_dvol_regime(merged_df)

    # 6. Regression analysis
    model, reg_results = regression_dvol_regimes(merged_df)

    # 7. Extremity premium test
    extremity_results = test_extremity_premium(merged_df)

    # 8. Cross-validation with F&G
    concordance_results = compare_fg_dvol_regimes(merged_df)

    # 9. Generate LaTeX table
    print("\n" + "="*70)
    print("LATEX TABLE FOR PAPER")
    print("="*70)
    latex_table = generate_latex_table(desc_df, reg_results, extremity_results)
    print(latex_table)

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)

    # Save merged data
    merged_df.to_csv(os.path.join(output_dir, 'dvol_regime_data.csv'), index=False)

    # Save LaTeX
    with open(os.path.join(output_dir, 'dvol_regime_table.tex'), 'w') as f:
        f.write(latex_table)

    print(f"\n✓ Results saved to {output_dir}/")

    # Return summary
    return {
        'n_observations': len(merged_df),
        'date_range': (merged_df['date'].min(), merged_df['date'].max()),
        'extremity_cohens_d': extremity_results['cohens_d'] if extremity_results else None,
        'extremity_p_value': extremity_results['p_value'] if extremity_results else None,
        'concordance_with_fg': concordance_results['concordance'] if concordance_results else None,
        'latex_table': latex_table,
    }


if __name__ == '__main__':
    results = main()
