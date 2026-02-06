#!/usr/bin/env python3
"""
Tick Data Validation Analysis
Computes effective spreads from tick data and correlates with CS spread measures.
"""

import gzip
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path('/home/purrpower/Resurrexi/projects/papers/working/no-structure/old/sentiment-microstructure-abm/arxiv-submission')
TRADES_DIR = PROJECT_ROOT / 'data' / 'lob' / 'btcusdt' / 'trades'
SPREAD_DATA_PATH = Path('/home/purrpower/Resurrexi/projects/papers/working/no-structure/old/sentiment-microstructure-abm/results/real_spread_data.csv')
OUTPUT_PATH = PROJECT_ROOT / 'results' / 'tick_validation_results.json'

def load_trade_file(filepath):
    """Load a compressed trade CSV file."""
    try:
        with gzip.open(filepath, 'rt') as f:
            df = pd.read_csv(f)
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def compute_effective_spread(trades_df):
    """
    Compute effective spread from tick data.

    Effective spread = 2 * |price - midpoint| / midpoint

    For tick data without explicit quotes, we estimate midpoint using
    rolling average of recent prices and compute spread relative to it.

    Alternative: Use tick direction to infer bid/ask hits:
    - Buy (taker buys) = hit the ask -> price > midpoint
    - Sell (taker sells) = hit the bid -> price < midpoint
    """
    if trades_df is None or len(trades_df) == 0:
        return None

    # Ensure price column exists
    if 'price' not in trades_df.columns:
        print(f"Columns available: {trades_df.columns.tolist()}")
        return None

    df = trades_df.copy()

    # Convert price to numeric
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['price'])

    if len(df) < 100:
        return None

    # Estimate midpoint using rolling median (more robust than mean)
    # Use 50-tick window as proxy for microprice
    df['midpoint'] = df['price'].rolling(window=50, min_periods=25).median()
    df = df.dropna(subset=['midpoint'])

    if len(df) == 0:
        return None

    # Effective spread calculation
    # Traditional: spread = 2 * |trade_price - midpoint| / midpoint
    df['eff_spread'] = 2 * np.abs(df['price'] - df['midpoint']) / df['midpoint']

    # Also compute directional spread using side info if available
    if 'side' in df.columns:
        # Signed spread: positive for buys (above mid), negative for sells (below mid)
        df['signed_spread'] = np.where(
            df['side'].str.upper() == 'BUY',
            (df['price'] - df['midpoint']) / df['midpoint'],
            (df['midpoint'] - df['price']) / df['midpoint']
        )

    # Aggregate statistics
    results = {
        'n_trades': len(df),
        'mean_eff_spread_bps': df['eff_spread'].mean() * 10000,  # in basis points
        'median_eff_spread_bps': df['eff_spread'].median() * 10000,
        'std_eff_spread_bps': df['eff_spread'].std() * 10000,
        'p25_eff_spread_bps': df['eff_spread'].quantile(0.25) * 10000,
        'p75_eff_spread_bps': df['eff_spread'].quantile(0.75) * 10000,
    }

    if 'signed_spread' in df.columns:
        results['mean_signed_spread_bps'] = df['signed_spread'].mean() * 10000

    # Volume-weighted spread if size available
    if 'size' in df.columns:
        df['size'] = pd.to_numeric(df['size'], errors='coerce')
        df_valid = df.dropna(subset=['size'])
        if len(df_valid) > 0:
            vwap_spread = np.average(df_valid['eff_spread'], weights=df_valid['size'])
            results['vwap_eff_spread_bps'] = vwap_spread * 10000

    return results

def extract_date_from_filename(filename):
    """Extract date from filename like BTCUSDT20251012_trades.csv.gz"""
    # Format: BTCUSDT + YYYYMMDD + _trades.csv.gz
    date_str = filename.replace('BTCUSDT', '').replace('_trades.csv.gz', '')
    try:
        return pd.to_datetime(date_str, format='%Y%m%d').strftime('%Y-%m-%d')
    except:
        return None

def main():
    print("="*60)
    print("Tick Data Validation Analysis")
    print("="*60)

    # Get list of trade files
    trade_files = sorted(TRADES_DIR.glob('BTCUSDT*_trades.csv.gz'))
    print(f"\nFound {len(trade_files)} trade files")

    # Sample 15 files spread across the date range
    n_sample = min(15, len(trade_files))
    step = max(1, len(trade_files) // n_sample)
    sampled_files = trade_files[::step][:n_sample]

    print(f"Sampling {len(sampled_files)} files for analysis")

    # Process each file
    daily_spreads = []

    for filepath in sampled_files:
        filename = filepath.name
        date = extract_date_from_filename(filename)

        if date is None:
            print(f"  Skipping {filename} - could not parse date")
            continue

        print(f"  Processing {filename}...", end=" ")

        trades_df = load_trade_file(filepath)
        if trades_df is None:
            print("FAILED to load")
            continue

        spread_stats = compute_effective_spread(trades_df)
        if spread_stats is None:
            print("FAILED to compute spread")
            continue

        spread_stats['date'] = date
        daily_spreads.append(spread_stats)
        print(f"OK - {spread_stats['n_trades']:,} trades, mean spread: {spread_stats['mean_eff_spread_bps']:.2f} bps")

    if len(daily_spreads) == 0:
        print("\nNo valid spread data computed!")
        return

    # Create DataFrame of daily tick spreads
    tick_df = pd.DataFrame(daily_spreads)
    tick_df['date'] = pd.to_datetime(tick_df['date'])
    tick_df = tick_df.sort_values('date')

    print(f"\n{'='*60}")
    print("Tick-Based Spread Summary")
    print("="*60)
    print(f"Date range: {tick_df['date'].min()} to {tick_df['date'].max()}")
    print(f"Days analyzed: {len(tick_df)}")
    print(f"\nEffective Spread (basis points):")
    print(f"  Mean:   {tick_df['mean_eff_spread_bps'].mean():.2f}")
    print(f"  Median: {tick_df['median_eff_spread_bps'].mean():.2f}")
    print(f"  Std:    {tick_df['mean_eff_spread_bps'].std():.2f}")

    # Load CS spread data
    print(f"\n{'='*60}")
    print("Loading CS Spread Data")
    print("="*60)

    cs_df = pd.read_csv(SPREAD_DATA_PATH)
    cs_df['date'] = pd.to_datetime(cs_df['date'])

    print(f"CS data date range: {cs_df['date'].min()} to {cs_df['date'].max()}")
    print(f"CS data rows: {len(cs_df)}")

    # Merge datasets
    merged = pd.merge(tick_df, cs_df, on='date', how='inner')
    print(f"\nMatched dates: {len(merged)}")

    if len(merged) < 3:
        print("WARNING: Not enough matched dates for meaningful correlation analysis")

    # Correlation analysis
    print(f"\n{'='*60}")
    print("Correlation Analysis")
    print("="*60)

    correlations = {}

    # Key variables to correlate with tick spread
    tick_spread_col = 'mean_eff_spread_bps'
    cs_columns = ['cs_spread', 'total_uncertainty', 'aleatoric_proxy', 'epistemic_proxy',
                  'parkinson_vol', 'realized_vol', 'amihud']

    for col in cs_columns:
        if col in merged.columns:
            # Drop NaN for this specific comparison
            valid = merged[[tick_spread_col, col]].dropna()

            if len(valid) >= 3:
                # Pearson correlation
                r, p = stats.pearsonr(valid[tick_spread_col], valid[col])
                # Spearman (rank) correlation - more robust
                rho, p_spearman = stats.spearmanr(valid[tick_spread_col], valid[col])

                correlations[col] = {
                    'pearson_r': r,
                    'pearson_p': p,
                    'spearman_rho': rho,
                    'spearman_p': p_spearman,
                    'n_obs': len(valid)
                }

                print(f"\n{col}:")
                print(f"  Pearson r = {r:.4f} (p = {p:.4f})")
                print(f"  Spearman rho = {rho:.4f} (p = {p_spearman:.4f})")
                print(f"  N = {len(valid)}")
            else:
                print(f"\n{col}: Insufficient data (n={len(valid)})")

    # Prepare results
    results = {
        'analysis_info': {
            'n_files_sampled': len(sampled_files),
            'n_days_analyzed': len(tick_df),
            'n_matched_days': len(merged),
            'tick_data_date_range': [str(tick_df['date'].min()), str(tick_df['date'].max())],
            'cs_data_date_range': [str(cs_df['date'].min()), str(cs_df['date'].max())],
        },
        'tick_spread_summary': {
            'mean_spread_bps': float(tick_df['mean_eff_spread_bps'].mean()),
            'median_spread_bps': float(tick_df['median_eff_spread_bps'].mean()),
            'std_spread_bps': float(tick_df['mean_eff_spread_bps'].std()),
            'min_spread_bps': float(tick_df['mean_eff_spread_bps'].min()),
            'max_spread_bps': float(tick_df['mean_eff_spread_bps'].max()),
        },
        'daily_tick_spreads': tick_df[['date', 'n_trades', 'mean_eff_spread_bps', 'median_eff_spread_bps']].to_dict('records'),
        'correlations': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                            for kk, vv in v.items()}
                        for k, v in correlations.items()},
        'interpretation': {}
    }

    # Convert dates to strings for JSON
    for item in results['daily_tick_spreads']:
        item['date'] = str(item['date'])[:10]

    # Add interpretation
    if 'total_uncertainty' in correlations:
        corr = correlations['total_uncertainty']
        if corr['spearman_p'] < 0.05:
            direction = "positive" if corr['spearman_rho'] > 0 else "negative"
            results['interpretation']['uncertainty_correlation'] = (
                f"Significant {direction} correlation (rho={corr['spearman_rho']:.3f}, p={corr['spearman_p']:.4f}) "
                f"between tick-based effective spread and CS model total uncertainty measure. "
                f"This {'supports' if corr['spearman_rho'] > 0 else 'contradicts'} the hypothesis that "
                f"higher market uncertainty is associated with wider spreads."
            )
        else:
            results['interpretation']['uncertainty_correlation'] = (
                f"No significant correlation (rho={corr['spearman_rho']:.3f}, p={corr['spearman_p']:.4f}) "
                f"found between tick-based spreads and uncertainty. Sample size may be insufficient."
            )

    if 'cs_spread' in correlations:
        corr = correlations['cs_spread']
        results['interpretation']['cs_spread_correlation'] = (
            f"Correlation with Corwin-Schultz spread estimate: rho={corr['spearman_rho']:.3f} "
            f"(p={corr['spearman_p']:.4f}). "
            f"{'Strong agreement' if corr['spearman_rho'] > 0.7 else 'Moderate agreement' if corr['spearman_rho'] > 0.4 else 'Weak agreement' if corr['spearman_rho'] > 0 else 'Disagreement'} "
            f"between tick-based and OHLC-based spread measures."
        )

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {OUTPUT_PATH}")
    print("="*60)

    # Print key findings
    print("\nKEY FINDINGS:")
    for key, interp in results['interpretation'].items():
        print(f"\n{key.upper()}:")
        print(f"  {interp}")

if __name__ == '__main__':
    main()
