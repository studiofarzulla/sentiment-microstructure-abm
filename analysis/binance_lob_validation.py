#!/usr/bin/env python3
"""
Binance Multi-Exchange LOB Validation

Downloads Binance historical trade data and calculates daily spread estimates
for comparison with Bybit-derived LOB spreads and CS estimates.

This addresses the "single-exchange" limitation noted in the paper by validating
that the CS spread correlation holds across exchanges.

Usage:
    python binance_lob_validation.py --download --days 60
    python binance_lob_validation.py --analyze
    python binance_lob_validation.py --all

Author: Murad Farzulla
Date: January 2026
"""

import os
import sys
import zipfile
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import warnings

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from scipy import stats

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'binance'
RESULTS_DIR = PROJECT_ROOT / 'results'

BINANCE_S3_BASE = "https://data.binance.vision/data/spot/daily/trades"
SYMBOL = "BTCUSDT"


# =============================================================================
# Data Download
# =============================================================================

def download_binance_trades(days: int = 60) -> List[Path]:
    """Download Binance historical trade data."""
    print("=" * 60)
    print("DOWNLOADING BINANCE TRADE DATA")
    print("=" * 60)

    output_dir = DATA_DIR / SYMBOL.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    end_date = datetime.now() - timedelta(days=2)  # Data has 1-2 day lag
    start_date = end_date - timedelta(days=days)

    print(f"Date range: {start_date.date()} to {end_date.date()}")

    downloaded = []
    current = start_date

    while current <= end_date:
        date_str = current.strftime('%Y-%m-%d')
        filename = f"{SYMBOL}-trades-{date_str}.zip"
        url = f"{BINANCE_S3_BASE}/{SYMBOL}/{filename}"
        output_path = output_dir / filename

        if output_path.exists():
            print(f"  Already exists: {filename}")
            downloaded.append(output_path)
            current += timedelta(days=1)
            continue

        try:
            response = requests.get(url, stream=True, timeout=60)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"  Downloaded: {filename}")
                downloaded.append(output_path)
            else:
                print(f"  Not found: {filename} (HTTP {response.status_code})")
        except Exception as e:
            print(f"  Error: {filename} - {e}")

        current += timedelta(days=1)

    print(f"\nDownloaded {len(downloaded)} files")
    return downloaded


# =============================================================================
# Data Processing
# =============================================================================

def parse_binance_trades(filepath: Path) -> pd.DataFrame:
    """
    Parse Binance trade data from zip file.

    Binance trade format:
    - id: trade id
    - price: trade price
    - qty: quantity
    - quoteQty: quote quantity
    - time: timestamp (ms)
    - isBuyerMaker: True if buyer is maker (i.e., this is a sell)
    - isBestMatch: best match flag
    """
    try:
        with zipfile.ZipFile(filepath, 'r') as z:
            # Get the CSV file name inside
            csv_name = z.namelist()[0]

            # Read with explicit column names
            with z.open(csv_name) as f:
                df = pd.read_csv(
                    f,
                    names=['id', 'price', 'qty', 'quoteQty', 'time',
                           'isBuyerMaker', 'isBestMatch'],
                    dtype={
                        'id': 'int64',
                        'price': 'float64',
                        'qty': 'float64',
                        'quoteQty': 'float64',
                        'time': 'int64',
                        'isBuyerMaker': 'bool',
                        'isBestMatch': 'bool'
                    }
                )

        # Convert timestamp (Binance uses microseconds, not milliseconds)
        df['timestamp'] = pd.to_datetime(df['time'], unit='us')

        # isBuyerMaker = True means buyer was passive (maker)
        # so the aggressor was a SELL -> this hit the BID
        # isBuyerMaker = False means buyer was aggressive (taker)
        # so the aggressor was a BUY -> this hit the ASK
        df['side'] = np.where(df['isBuyerMaker'], 'sell', 'buy')

        return df

    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return pd.DataFrame()


def calculate_daily_spread(df: pd.DataFrame, date: str) -> Dict:
    """
    Calculate daily effective spread from trade data.

    Method:
    We use the EFFECTIVE SPREAD approach, standard in market microstructure:

    1. For each trade, compute a rolling VWAP as the "fair value" estimate
    2. Effective spread = 2 * |trade_price - midpoint|
    3. Sign by trade direction: buys should be above midpoint, sells below

    For high-frequency data, we use 100-trade rolling midpoint as reference.
    This captures actual transaction costs better than quoted spread proxies.

    Alternative approach also computed: consecutive opposing trade spread.
    """
    if df.empty or len(df) < 1000:
        return None

    df = df.set_index('timestamp').sort_index()

    # Method 1: Effective spread using rolling midpoint
    # Use VWAP as fair value proxy
    df['vwap_100'] = (df['price'] * df['qty']).rolling(100, min_periods=50).sum() / \
                     df['qty'].rolling(100, min_periods=50).sum()

    # Alternative: simple rolling median as midpoint
    df['mid_100'] = df['price'].rolling(100, min_periods=50).median()

    # Effective half-spread = |price - mid| (full effective spread = 2x this)
    df['effective_halfspread'] = np.abs(df['price'] - df['mid_100'])
    df['effective_spread_bps'] = (df['effective_halfspread'] / df['mid_100']) * 10000 * 2

    # Filter valid observations
    valid = df['effective_spread_bps'].dropna()

    # Remove outliers (trades during price jumps)
    q99 = valid.quantile(0.99)
    valid = valid[valid < q99]

    if len(valid) < 500:
        return None

    # Method 2: Trade sequence spread (consecutive opposing trades)
    # When a buy follows a sell, spread ≈ buy_price - sell_price (if positive)
    df['prev_price'] = df['price'].shift(1)
    df['prev_side'] = df['side'].shift(1)

    # Buy following sell: spread = buy_price - prev_sell_price
    buy_after_sell = df[(df['side'] == 'buy') & (df['prev_side'] == 'sell')]
    trade_seq_spread = buy_after_sell['price'] - buy_after_sell['prev_price']
    trade_seq_spread_bps = (trade_seq_spread / buy_after_sell['price']) * 10000

    # Filter valid (positive, reasonable)
    trade_seq_valid = trade_seq_spread_bps[(trade_seq_spread_bps > 0) &
                                           (trade_seq_spread_bps < 100)]

    return {
        'date': date,
        'spread_mean': valid.mean(),
        'spread_median': valid.median(),
        'spread_std': valid.std(),
        'n_obs': len(valid),
        'n_trades': len(df),
        # Also store trade-sequence spread if available
        'seq_spread_mean': trade_seq_valid.mean() if len(trade_seq_valid) > 100 else np.nan,
        'seq_spread_median': trade_seq_valid.median() if len(trade_seq_valid) > 100 else np.nan
    }


def process_all_files() -> pd.DataFrame:
    """Process all downloaded Binance trade files."""
    print("\n" + "=" * 60)
    print("PROCESSING BINANCE TRADE DATA")
    print("=" * 60)

    input_dir = DATA_DIR / SYMBOL.lower()

    if not input_dir.exists():
        print(f"No data directory: {input_dir}")
        return pd.DataFrame()

    files = sorted(input_dir.glob('*.zip'))
    print(f"Found {len(files)} files")

    results = []

    for filepath in tqdm(files, desc="Processing"):
        # Extract date from filename: BTCUSDT-trades-2025-12-01.zip
        date_str = filepath.stem.split('-trades-')[1]

        df = parse_binance_trades(filepath)

        if df.empty:
            continue

        daily = calculate_daily_spread(df, date_str)

        if daily:
            results.append(daily)

    if not results:
        print("No valid data processed")
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    result_df['date'] = pd.to_datetime(result_df['date'])
    result_df = result_df.set_index('date').sort_index()

    print(f"\nProcessed {len(result_df)} days of Binance data")
    print(f"Date range: {result_df.index.min()} to {result_df.index.max()}")
    print(f"Mean spread: {result_df['spread_mean'].mean():.2f} bps")

    # Save
    output_path = RESULTS_DIR / 'binance_daily_spreads.csv'
    result_df.to_csv(output_path)
    print(f"Saved: {output_path}")

    return result_df


# =============================================================================
# Analysis
# =============================================================================

def load_bybit_spreads() -> pd.DataFrame:
    """Load Bybit LOB spreads from previous analysis."""
    path = RESULTS_DIR / 'lob_daily_spreads_90d.csv'

    if not path.exists():
        # Try btcusdt specific file
        path = RESULTS_DIR / 'lob_daily_spreads_btcusdt.csv'

    if not path.exists():
        print(f"Bybit spread file not found")
        return pd.DataFrame()

    df = pd.read_csv(path)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    elif df.columns[0] == 'Unnamed: 0':
        df.index = pd.to_datetime(df['Unnamed: 0'])
        df = df.drop(columns=['Unnamed: 0'])
    else:
        # First column is date
        df.index = pd.to_datetime(df.iloc[:, 0])
        df = df.iloc[:, 1:]

    return df


def load_cs_spreads() -> pd.DataFrame:
    """Load Corwin-Schultz spread estimates."""
    path = RESULTS_DIR / 'real_spread_data.csv'

    if not path.exists():
        print(f"CS spread file not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

    return df


def run_analysis():
    """Run multi-exchange validation analysis."""
    print("\n" + "=" * 60)
    print("MULTI-EXCHANGE LOB VALIDATION")
    print("=" * 60)

    # Load all data
    binance_df = pd.read_csv(RESULTS_DIR / 'binance_daily_spreads.csv',
                             index_col=0, parse_dates=True)
    bybit_df = load_bybit_spreads()
    cs_df = load_cs_spreads()

    print(f"\nData loaded:")
    print(f"  Binance: {len(binance_df)} days")
    print(f"  Bybit: {len(bybit_df)} days")
    print(f"  CS spreads: {len(cs_df)} days")

    results = {}

    # 1. Binance vs CS correlation
    print("\n--- Binance vs CS Spreads ---")
    common_dates = binance_df.index.intersection(cs_df.index)

    if len(common_dates) >= 20:
        binance_aligned = binance_df.loc[common_dates, 'spread_mean']

        # Find CS spread column
        cs_col = None
        for col in cs_df.columns:
            if 'cs_spread' in col.lower() or col == 'spread':
                cs_col = col
                break

        if cs_col is None:
            cs_col = cs_df.columns[0]  # Fallback to first column

        cs_aligned = cs_df.loc[common_dates, cs_col]

        # Drop NaN and filter CS > 0 (CS formula returns 0 when negative)
        valid = ~(binance_aligned.isna() | cs_aligned.isna()) & (cs_aligned > 0)
        binance_valid = binance_aligned[valid]
        cs_valid = cs_aligned[valid]

        if len(binance_valid) >= 20:
            pearson_r, pearson_p = stats.pearsonr(binance_valid, cs_valid)
            spearman_r, spearman_p = stats.spearmanr(binance_valid, cs_valid)

            print(f"  N = {len(binance_valid)}")
            print(f"  Pearson: {pearson_r:.3f} (p = {pearson_p:.4f})")
            print(f"  Spearman: {spearman_r:.3f} (p = {spearman_p:.4f})")
            print(f"  Binance mean: {binance_valid.mean():.2f} bps")
            print(f"  CS mean: {cs_valid.mean():.2f} bps")

            results['binance_vs_cs'] = {
                'n': len(binance_valid),
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'binance_mean': binance_valid.mean(),
                'cs_mean': cs_valid.mean()
            }
    else:
        print(f"  Insufficient overlap: {len(common_dates)} days")

    # 2. Binance vs Bybit correlation (cross-exchange validation)
    print("\n--- Binance vs Bybit Spreads ---")
    common_dates = binance_df.index.intersection(bybit_df.index)

    if len(common_dates) >= 20:
        binance_aligned = binance_df.loc[common_dates, 'spread_mean']
        bybit_aligned = bybit_df.loc[common_dates, 'spread_mean']

        valid = ~(binance_aligned.isna() | bybit_aligned.isna())
        binance_valid = binance_aligned[valid]
        bybit_valid = bybit_aligned[valid]

        if len(binance_valid) >= 20:
            pearson_r, pearson_p = stats.pearsonr(binance_valid, bybit_valid)
            spearman_r, spearman_p = stats.spearmanr(binance_valid, bybit_valid)

            print(f"  N = {len(binance_valid)}")
            print(f"  Pearson: {pearson_r:.3f} (p = {pearson_p:.4f})")
            print(f"  Spearman: {spearman_r:.3f} (p = {spearman_p:.4f})")
            print(f"  Binance mean: {binance_valid.mean():.2f} bps")
            print(f"  Bybit mean: {bybit_valid.mean():.2f} bps")

            results['binance_vs_bybit'] = {
                'n': len(binance_valid),
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'binance_mean': binance_valid.mean(),
                'bybit_mean': bybit_valid.mean()
            }
    else:
        print(f"  Insufficient overlap: {len(common_dates)} days")

    # 3. Bybit vs CS (sanity check - should match previous analysis)
    print("\n--- Bybit vs CS Spreads (verification) ---")
    common_dates = bybit_df.index.intersection(cs_df.index)

    if len(common_dates) >= 20:
        bybit_aligned = bybit_df.loc[common_dates, 'spread_mean']
        cs_col = None
        for col in cs_df.columns:
            if 'cs_spread' in col.lower() or col == 'spread':
                cs_col = col
                break
        if cs_col is None:
            cs_col = cs_df.columns[0]
        cs_aligned = cs_df.loc[common_dates, cs_col]

        # Filter CS > 0 (consistent with original 90d validation)
        valid = ~(bybit_aligned.isna() | cs_aligned.isna()) & (cs_aligned > 0)
        bybit_valid = bybit_aligned[valid]
        cs_valid = cs_aligned[valid]

        if len(bybit_valid) >= 20:
            spearman_r, spearman_p = stats.spearmanr(bybit_valid, cs_valid)
            print(f"  N = {len(bybit_valid)}, Spearman: {spearman_r:.3f} (p = {spearman_p:.4f}) [filters CS > 0]")

            results['bybit_vs_cs'] = {
                'n': len(bybit_valid),
                'spearman_r': spearman_r,
                'spearman_p': spearman_p
            }

    # Generate LaTeX table entry
    print("\n" + "=" * 60)
    print("LATEX TABLE ENTRY FOR PAPER")
    print("=" * 60)

    if 'binance_vs_cs' in results:
        r = results['binance_vs_cs']
        p_star = ""
        if r['spearman_p'] < 0.01:
            p_star = "***"
        elif r['spearman_p'] < 0.05:
            p_star = "**"
        elif r['spearman_p'] < 0.10:
            p_star = "*"

        print(f"""
\\multicolumn{{5}}{{l}}{{\\textit{{Panel D: Multi-Exchange Validation (Binance)}}}} \\\\
Binance LOB vs.\\ CS Spread & {r['n']} & {r['pearson_r']:.3f} & {r['spearman_r']:.3f}{p_star} & {r['pearson_p']:.3f} / {r['spearman_p']:.3f} \\\\
""")

    if 'binance_vs_bybit' in results:
        r = results['binance_vs_bybit']
        p_star = ""
        if r['spearman_p'] < 0.01:
            p_star = "***"
        elif r['spearman_p'] < 0.05:
            p_star = "**"
        elif r['spearman_p'] < 0.10:
            p_star = "*"

        print(f"""Binance LOB vs.\\ Bybit LOB & {r['n']} & {r['pearson_r']:.3f} & {r['spearman_r']:.3f}{p_star} & {r['pearson_p']:.3f} / {r['spearman_p']:.3f} \\\\
""")

    # Save results
    import json
    results_path = RESULTS_DIR / 'binance_validation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results: {results_path}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Binance Multi-Exchange LOB Validation')
    parser.add_argument('--download', action='store_true', help='Download Binance trade data')
    parser.add_argument('--process', action='store_true', help='Process downloaded data')
    parser.add_argument('--analyze', action='store_true', help='Run analysis')
    parser.add_argument('--all', action='store_true', help='Do everything')
    parser.add_argument('--days', type=int, default=60, help='Days of data to download')

    args = parser.parse_args()

    # Create directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        args.download = True
        args.process = True
        args.analyze = True

    if args.download:
        download_binance_trades(args.days)

    if args.process:
        process_all_files()

    if args.analyze:
        run_analysis()

    if not any([args.download, args.process, args.analyze, args.all]):
        parser.print_help()
        print("\nExample usage:")
        print("  python binance_lob_validation.py --all --days 60")
        print("  python binance_lob_validation.py --download --days 30")
        print("  python binance_lob_validation.py --analyze")


if __name__ == '__main__':
    main()
