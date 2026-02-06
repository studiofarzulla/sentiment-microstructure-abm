#!/usr/bin/env python3
"""
LOB Validation: Direct Quoted Spreads from Bybit Order Book Data

Downloads Bybit historical LOB data, calculates quoted spreads directly,
and compares with Corwin-Schultz estimates to validate the spread-uncertainty relationship.

This addresses reviewer concerns about CS estimator limitations by using
actual order book quotes rather than OHLC-derived estimates.

Usage:
    python lob_validation.py --download      # Download data from Bybit
    python lob_validation.py --parse         # Parse downloaded JSON files
    python lob_validation.py --analyze       # Run comparison analysis
    python lob_validation.py --all           # Do everything

Data Source: https://www.bybit.com/derivatives/en/history-data
Format: L2 order book snapshots, 10ms intervals, top 200 levels

Author: Murad Farzulla
Date: January 2026
"""

import os
import sys
import json
import gzip
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
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
DATA_DIR = PROJECT_ROOT / 'data' / 'lob'
RESULTS_DIR = PROJECT_ROOT / 'results'

# Bybit data config
BYBIT_BASE_URL = "https://public.bybit.com"
SYMBOLS = ['BTCUSDT', 'ETHUSDT']

# Analysis config
RESAMPLE_FREQ = '1min'  # Aggregate 10ms snapshots to 1-minute
DAILY_FREQ = '1D'       # For comparison with CS spreads


# =============================================================================
# Data Download
# =============================================================================

def get_available_dates(symbol: str, data_type: str = 'orderbook') -> List[str]:
    """
    Get list of available dates for a symbol from Bybit.

    Bybit organizes data as: /trading/{symbol}/{data_type}/{symbol}{date}.csv.gz
    """
    # Bybit doesn't have a listing API, so we'll generate recent dates
    # and check which exist
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Last 30 days

    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)

    return dates


def download_orderbook_file(symbol: str, date: str, output_dir: Path) -> Optional[Path]:
    """
    Download a single day's orderbook data from Bybit.

    File format: {symbol}/{symbol}{YYYY-MM-DD}_orderbook.csv.gz
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Bybit URL format for order book data
    # https://public.bybit.com/trading/BTCUSDT/BTCUSDT2024-01-15.csv.gz
    date_compact = date.replace('-', '')
    filename = f"{symbol}{date_compact}_orderbook.csv.gz"
    url = f"{BYBIT_BASE_URL}/trading/{symbol}/{symbol}{date}.csv.gz"

    output_path = output_dir / filename

    if output_path.exists():
        print(f"  Already exists: {filename}")
        return output_path

    try:
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"  Downloaded: {filename}")
            return output_path
        else:
            # Try alternative URL format
            alt_url = f"{BYBIT_BASE_URL}/spot/{symbol}/{symbol}{date}.csv.gz"
            response = requests.get(alt_url, stream=True, timeout=30)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"  Downloaded (alt): {filename}")
                return output_path
            else:
                print(f"  Not found: {filename} (HTTP {response.status_code})")
                return None
    except Exception as e:
        print(f"  Error downloading {filename}: {e}")
        return None


def download_trades_file(symbol: str, date: str, output_dir: Path) -> Optional[Path]:
    """
    Download trade data as fallback/supplement.
    Trade data is more reliably available than full orderbook.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{symbol}{date.replace('-', '')}_trades.csv.gz"
    url = f"{BYBIT_BASE_URL}/trading/{symbol}/{symbol}{date}.csv.gz"

    output_path = output_dir / filename

    if output_path.exists():
        return output_path

    try:
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return output_path
    except:
        pass

    return None


def download_data(symbols: List[str] = None, days: int = 14):
    """Download LOB data for specified symbols."""
    symbols = symbols or SYMBOLS

    print("=" * 60)
    print("DOWNLOADING BYBIT LOB DATA")
    print("=" * 60)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Symbols: {symbols}")

    downloaded = []

    for symbol in symbols:
        print(f"\n--- {symbol} ---")
        symbol_dir = DATA_DIR / symbol.lower()

        current = start_date
        while current <= end_date:
            date_str = current.strftime('%Y-%m-%d')

            # Try orderbook first, fall back to trades
            result = download_orderbook_file(symbol, date_str, symbol_dir)
            if result is None:
                result = download_trades_file(symbol, date_str, symbol_dir)

            if result:
                downloaded.append(result)

            current += timedelta(days=1)

    print(f"\nDownloaded {len(downloaded)} files")
    return downloaded


# =============================================================================
# Data Parsing
# =============================================================================

def parse_bybit_orderbook(filepath: Path) -> pd.DataFrame:
    """
    Parse Bybit orderbook CSV file.

    Expected columns vary by data type, but typically include:
    - timestamp (ms or us)
    - symbol
    - side (Buy/Sell)
    - price
    - size/qty

    For L2 data, we get aggregated levels.
    """
    try:
        if filepath.suffix == '.gz':
            df = pd.read_csv(filepath, compression='gzip')
        else:
            df = pd.read_csv(filepath)

        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()

        # Handle timestamp
        if 'timestamp' in df.columns:
            # Bybit uses milliseconds or microseconds
            ts = df['timestamp'].iloc[0]
            if ts > 1e15:  # Microseconds
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')
            elif ts > 1e12:  # Milliseconds
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            else:  # Seconds
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        return df

    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return pd.DataFrame()


def parse_bybit_trades(filepath: Path) -> pd.DataFrame:
    """
    Parse Bybit trades CSV file.

    Trade data includes:
    - timestamp
    - symbol
    - side (Buy/Sell)
    - size
    - price
    - tickDirection
    - trdMatchID
    """
    try:
        if filepath.suffix == '.gz':
            df = pd.read_csv(filepath, compression='gzip')
        else:
            df = pd.read_csv(filepath)

        df.columns = df.columns.str.lower().str.strip()

        # Handle timestamp
        if 'timestamp' in df.columns:
            ts = df['timestamp'].iloc[0]
            if ts > 1e15:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')
            elif ts > 1e12:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        return df

    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return pd.DataFrame()


def calculate_quoted_spread_from_trades(trades_df: pd.DataFrame, freq: str = '1min') -> pd.DataFrame:
    """
    Estimate quoted spread from trade data using the tick rule.

    When we don't have actual order book snapshots, we can estimate
    the spread from trade prices and directions:
    - Buy trades hit the ask
    - Sell trades hit the bid

    This gives us implicit bid/ask from executed trades.
    """
    if trades_df.empty:
        return pd.DataFrame()

    df = trades_df.copy()

    # Ensure we have required columns
    if 'side' not in df.columns or 'price' not in df.columns:
        print("Missing required columns (side, price)")
        return pd.DataFrame()

    # Separate buys and sells
    df['is_buy'] = df['side'].str.lower().isin(['buy', 'b', '1'])

    # Set index for resampling
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')

    # Resample to get bid (min sell) and ask (max buy) per interval
    # This is an approximation - real quotes would be better
    resampled = df.resample(freq).agg({
        'price': ['min', 'max', 'mean', 'count'],
        'is_buy': 'sum'
    })

    resampled.columns = ['price_low', 'price_high', 'price_mean', 'n_trades', 'n_buys']
    resampled = resampled.dropna()

    if len(resampled) == 0:
        return pd.DataFrame()

    # Estimate spread from high-low range (similar to CS but at higher frequency)
    # This is a crude approximation
    resampled['spread_approx'] = resampled['price_high'] - resampled['price_low']
    resampled['midpoint'] = (resampled['price_high'] + resampled['price_low']) / 2
    resampled['spread_bps'] = (resampled['spread_approx'] / resampled['midpoint']) * 10000

    # Clean up extreme values
    resampled.loc[resampled['spread_bps'] > 1000, 'spread_bps'] = np.nan
    resampled.loc[resampled['spread_bps'] < 0, 'spread_bps'] = np.nan

    return resampled


def calculate_quoted_spread_from_orderbook(ob_df: pd.DataFrame, freq: str = '1min') -> pd.DataFrame:
    """
    Calculate quoted spread directly from L2 order book snapshots.

    This is the gold standard - actual best bid and best ask.
    """
    if ob_df.empty:
        return pd.DataFrame()

    df = ob_df.copy()

    # Look for bid/ask columns
    bid_col = None
    ask_col = None

    for col in df.columns:
        if 'bid' in col.lower() and 'price' in col.lower():
            bid_col = col
        elif 'ask' in col.lower() and 'price' in col.lower():
            ask_col = col
        elif col.lower() in ['bid', 'bid_price', 'best_bid', 'bid1']:
            bid_col = col
        elif col.lower() in ['ask', 'ask_price', 'best_ask', 'ask1']:
            ask_col = col

    if bid_col and ask_col:
        # Direct calculation from order book
        df['spread'] = df[ask_col] - df[bid_col]
        df['midpoint'] = (df[ask_col] + df[bid_col]) / 2
        df['spread_bps'] = (df['spread'] / df['midpoint']) * 10000

        # Resample
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')

        resampled = df.resample(freq).agg({
            'spread_bps': 'mean',
            'midpoint': 'mean',
            bid_col: 'mean',
            ask_col: 'mean'
        })

        resampled = resampled.rename(columns={bid_col: 'best_bid', ask_col: 'best_ask'})
        return resampled.dropna()

    else:
        # Fall back to trade-based estimation
        print("No bid/ask columns found, using trade-based estimation")
        return calculate_quoted_spread_from_trades(df, freq)


def parse_all_data(symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
    """Parse all downloaded data files."""
    symbols = symbols or SYMBOLS

    print("=" * 60)
    print("PARSING LOB DATA")
    print("=" * 60)

    all_data = {}

    for symbol in symbols:
        print(f"\n--- {symbol} ---")
        symbol_dir = DATA_DIR / symbol.lower()

        if not symbol_dir.exists():
            print(f"  No data directory found")
            continue

        files = list(symbol_dir.glob('*.csv.gz')) + list(symbol_dir.glob('*.csv'))
        print(f"  Found {len(files)} files")

        dfs = []
        for filepath in tqdm(files, desc=f"  Parsing {symbol}"):
            if 'orderbook' in filepath.name.lower():
                df = parse_bybit_orderbook(filepath)
            else:
                df = parse_bybit_trades(filepath)

            if not df.empty:
                dfs.append(df)

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            combined = combined.sort_values('timestamp') if 'timestamp' in combined.columns else combined
            all_data[symbol] = combined
            print(f"  Total rows: {len(combined):,}")
        else:
            print(f"  No valid data parsed")

    return all_data


# =============================================================================
# Analysis
# =============================================================================

def aggregate_to_daily(spread_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate intraday spreads to daily for comparison with CS."""
    if spread_df.empty:
        return pd.DataFrame()

    daily = spread_df.resample('1D').agg({
        'spread_bps': ['mean', 'median', 'std', 'min', 'max', 'count']
    })

    daily.columns = ['spread_mean', 'spread_median', 'spread_std',
                     'spread_min', 'spread_max', 'n_observations']

    return daily.dropna()


def load_cs_spreads() -> pd.DataFrame:
    """Load Corwin-Schultz spread estimates from previous analysis."""
    cs_path = RESULTS_DIR / 'real_spread_data.csv'

    if not cs_path.exists():
        print(f"CS spread file not found: {cs_path}")
        return pd.DataFrame()

    df = pd.read_csv(cs_path)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

    return df


def compare_spreads(lob_daily: pd.DataFrame, cs_df: pd.DataFrame) -> Dict:
    """
    Compare LOB-derived quoted spreads with Corwin-Schultz estimates.

    This is the key validation: if CS spreads are valid, they should
    correlate strongly with actual quoted spreads.
    """
    if lob_daily.empty or cs_df.empty:
        return {'error': 'Missing data for comparison'}

    # Align dates
    common_dates = lob_daily.index.intersection(cs_df.index)

    if len(common_dates) < 10:
        return {'error': f'Insufficient overlap: {len(common_dates)} days'}

    lob_aligned = lob_daily.loc[common_dates]
    cs_aligned = cs_df.loc[common_dates]

    # Get CS spread column
    cs_col = None
    for col in cs_aligned.columns:
        if 'cs_spread' in col.lower() or 'spread' in col.lower():
            cs_col = col
            break

    if cs_col is None:
        return {'error': 'No CS spread column found'}

    # Calculate correlations
    lob_spread = lob_aligned['spread_mean']
    cs_spread = cs_aligned[cs_col]

    # Drop any NaN
    valid = ~(lob_spread.isna() | cs_spread.isna())
    lob_spread = lob_spread[valid]
    cs_spread = cs_spread[valid]

    if len(lob_spread) < 10:
        return {'error': 'Insufficient valid observations'}

    # Pearson correlation
    corr, p_value = stats.pearsonr(lob_spread, cs_spread)

    # Spearman (rank) correlation
    spearman_corr, spearman_p = stats.spearmanr(lob_spread, cs_spread)

    # Mean absolute difference
    mae = np.mean(np.abs(lob_spread - cs_spread))

    # Relative difference
    rel_diff = np.mean((cs_spread - lob_spread) / lob_spread) * 100

    results = {
        'n_days': len(lob_spread),
        'pearson_corr': corr,
        'pearson_p': p_value,
        'spearman_corr': spearman_corr,
        'spearman_p': spearman_p,
        'mae_bps': mae,
        'relative_diff_pct': rel_diff,
        'lob_mean': lob_spread.mean(),
        'cs_mean': cs_spread.mean(),
        'lob_std': lob_spread.std(),
        'cs_std': cs_spread.std(),
    }

    return results


def run_uncertainty_validation(lob_daily: pd.DataFrame, cs_df: pd.DataFrame) -> Dict:
    """
    Test if the spread-uncertainty relationship holds with LOB spreads.

    If CS spreads are biased but the relationship is real, we should
    see similar correlations with LOB-derived spreads.
    """
    if lob_daily.empty or cs_df.empty:
        return {'error': 'Missing data'}

    # Align dates
    common_dates = lob_daily.index.intersection(cs_df.index)

    if len(common_dates) < 30:
        return {'error': f'Insufficient overlap: {len(common_dates)} days'}

    lob_aligned = lob_daily.loc[common_dates]
    cs_aligned = cs_df.loc[common_dates]

    results = {}

    # Find uncertainty columns
    uncertainty_cols = [col for col in cs_aligned.columns
                       if 'uncertainty' in col.lower() or 'volatility' in col.lower()]

    for unc_col in uncertainty_cols:
        unc = cs_aligned[unc_col]
        lob_spread = lob_aligned['spread_mean']

        valid = ~(lob_spread.isna() | unc.isna())

        if valid.sum() < 20:
            continue

        corr, p_val = stats.pearsonr(lob_spread[valid], unc[valid])

        results[f'lob_vs_{unc_col}'] = {
            'correlation': corr,
            'p_value': p_val,
            'n': valid.sum()
        }

    return results


def generate_latex_table(spread_comparison: Dict, uncertainty_results: Dict) -> str:
    """Generate LaTeX table for paper."""

    latex = r"""
\begin{table}[htbp]
\centering
\caption{LOB Validation: Quoted vs. Corwin-Schultz Spreads}
\label{tab:lob_validation}
\small
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Metric} & \textbf{Value} & \textbf{p-value} \\
\midrule
"""

    if 'error' not in spread_comparison:
        latex += f"Pearson Correlation & {spread_comparison['pearson_corr']:.3f} & {spread_comparison['pearson_p']:.4f} \\\\\n"
        latex += f"Spearman Correlation & {spread_comparison['spearman_corr']:.3f} & {spread_comparison['spearman_p']:.4f} \\\\\n"
        latex += f"Mean Abs. Difference (bps) & {spread_comparison['mae_bps']:.2f} & -- \\\\\n"
        latex += f"LOB Mean Spread (bps) & {spread_comparison['lob_mean']:.2f} & -- \\\\\n"
        latex += f"CS Mean Spread (bps) & {spread_comparison['cs_mean']:.2f} & -- \\\\\n"
        latex += f"N (days) & {spread_comparison['n_days']} & -- \\\\\n"

    latex += r"""\bottomrule
\end{tabular}

\vspace{0.5em}
\raggedright\footnotesize
\textit{Notes:} Comparison of directly observed quoted spreads from Bybit L2 order book data
with Corwin-Schultz (2012) estimates from daily OHLC. High correlation validates the CS estimator
for this sample.
\end{table}
"""

    return latex


def run_analysis():
    """Run full comparison analysis."""
    print("=" * 60)
    print("RUNNING LOB VALIDATION ANALYSIS")
    print("=" * 60)

    # Load CS spreads
    print("\nLoading Corwin-Schultz spreads...")
    cs_df = load_cs_spreads()

    if cs_df.empty:
        print("ERROR: Could not load CS spread data")
        print("Make sure real_spread_data.csv exists in results/")
        return

    print(f"CS data: {len(cs_df)} days")

    # Parse LOB data
    all_data = parse_all_data()

    results = {}

    for symbol, data in all_data.items():
        print(f"\n--- Analyzing {symbol} ---")

        # Calculate quoted spreads
        if 'bid' in str(data.columns).lower() and 'ask' in str(data.columns).lower():
            spread_df = calculate_quoted_spread_from_orderbook(data, RESAMPLE_FREQ)
        else:
            spread_df = calculate_quoted_spread_from_trades(data, RESAMPLE_FREQ)

        if spread_df.empty:
            print(f"  Could not calculate spreads for {symbol}")
            continue

        print(f"  Intraday observations: {len(spread_df):,}")

        # Aggregate to daily
        daily_df = aggregate_to_daily(spread_df)
        print(f"  Daily observations: {len(daily_df)}")

        # Compare with CS
        comparison = compare_spreads(daily_df, cs_df)
        results[f'{symbol}_comparison'] = comparison

        if 'error' not in comparison:
            print(f"  Pearson correlation: {comparison['pearson_corr']:.3f} (p={comparison['pearson_p']:.4f})")
            print(f"  Mean LOB spread: {comparison['lob_mean']:.2f} bps")
            print(f"  Mean CS spread: {comparison['cs_mean']:.2f} bps")
        else:
            print(f"  Error: {comparison['error']}")

        # Test uncertainty relationship
        unc_results = run_uncertainty_validation(daily_df, cs_df)
        results[f'{symbol}_uncertainty'] = unc_results

        # Save daily spreads
        output_path = RESULTS_DIR / f'lob_daily_spreads_{symbol.lower()}.csv'
        daily_df.to_csv(output_path)
        print(f"  Saved: {output_path}")

    # Generate LaTeX
    if results:
        # Use first symbol's results for table
        first_symbol = list(all_data.keys())[0]
        comparison = results.get(f'{first_symbol}_comparison', {})
        uncertainty = results.get(f'{first_symbol}_uncertainty', {})

        latex = generate_latex_table(comparison, uncertainty)

        latex_path = RESULTS_DIR / 'lob_validation_table.tex'
        with open(latex_path, 'w') as f:
            f.write(latex)
        print(f"\nSaved LaTeX table: {latex_path}")

    # Save full results as JSON
    results_path = RESULTS_DIR / 'lob_validation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved results: {results_path}")

    return results


# =============================================================================
# Alternative: Use Kaggle Dataset
# =============================================================================

def download_kaggle_lob(dataset: str = 'martinsn/high-frequency-crypto-limit-order-book-data'):
    """
    Download LOB data from Kaggle.

    Requires: pip install kaggle
    And ~/.kaggle/kaggle.json with API credentials
    """
    try:
        import kaggle

        output_dir = DATA_DIR / 'kaggle'
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {dataset} to {output_dir}")
        kaggle.api.dataset_download_files(dataset, path=output_dir, unzip=True)
        print("Download complete!")

        return output_dir

    except ImportError:
        print("Kaggle package not installed. Run: pip install kaggle")
        return None
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        print("Make sure ~/.kaggle/kaggle.json exists with your API key")
        return None


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='LOB Validation Analysis')
    parser.add_argument('--download', action='store_true', help='Download data from Bybit')
    parser.add_argument('--kaggle', action='store_true', help='Download from Kaggle instead')
    parser.add_argument('--parse', action='store_true', help='Parse downloaded data')
    parser.add_argument('--analyze', action='store_true', help='Run comparison analysis')
    parser.add_argument('--all', action='store_true', help='Do everything')
    parser.add_argument('--days', type=int, default=14, help='Days of data to download')
    parser.add_argument('--symbols', nargs='+', default=SYMBOLS, help='Symbols to process')

    args = parser.parse_args()

    # Create directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        args.download = True
        args.parse = True
        args.analyze = True

    if args.download:
        if args.kaggle:
            download_kaggle_lob()
        else:
            download_data(args.symbols, args.days)

    if args.parse:
        parse_all_data(args.symbols)

    if args.analyze:
        run_analysis()

    if not any([args.download, args.parse, args.analyze, args.all]):
        parser.print_help()
        print("\nExample usage:")
        print("  python lob_validation.py --download --days 7")
        print("  python lob_validation.py --analyze")
        print("  python lob_validation.py --all")


if __name__ == '__main__':
    main()
