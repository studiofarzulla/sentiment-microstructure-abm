#!/usr/bin/env python3
"""
Real Spread Validation: Empirical Test of Uncertainty-Spread Correlation

This script addresses the reviewer concern about tautological spread-uncertainty
correlation in the ABM simulation. We validate the finding against real Binance
BTC/USDT market data.

Reviewer concern: "The spread-uncertainty correlation is derived from ABM where
spreads are mechanically widened by sigma_total - result is at risk of tautology"

Approach:
1. Fetch historical Binance BTC/USDT order book depth data
2. Compute realized spreads (quoted and effective)
3. Construct uncertainty proxies from market observables:
   - Aleatoric: Realized volatility, VIX correlation, intraday range
   - Epistemic: Volume dispersion, sentiment disagreement, price momentum uncertainty
4. Test correlation between real spreads and uncertainty measures
5. Compare with simulation correlations for validation

Author: Farzulla Research
Date: January 2026
"""

import os
import sys
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import requests
from scipy import stats
from scipy.stats import pearsonr, spearmanr

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class ValidationResults:
    """Container for validation results."""
    real_correlations: Dict[str, Tuple[float, float]]  # (corr, p-value)
    simulation_correlations: Dict[str, Tuple[float, float]]
    n_observations: int
    date_range: Tuple[str, str]
    spread_stats: Dict[str, float]
    uncertainty_stats: Dict[str, Dict[str, float]]


class BinanceSpreadFetcher:
    """
    Fetch historical spread data from Binance public API.

    Uses multiple data sources:
    1. Klines (OHLCV) for price/volume data
    2. Aggregated trades for execution data
    3. Historical order book depth (via archived data if available)
    """

    BASE_URL = "https://api.binance.com/api/v3"
    DATA_URL = "https://data.binance.vision"

    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Research Client)'
        })

    def fetch_klines(
        self,
        interval: str = "1d",
        start_date: str = "2024-01-01",
        end_date: Optional[str] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch OHLCV klines data.

        Args:
            interval: Kline interval (1m, 5m, 1h, 1d, etc.)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to now
            limit: Max candles per request

        Returns:
            DataFrame with OHLCV data
        """
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.now().timestamp() * 1000) if end_date is None else \
                 int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

        all_data = []
        current_start = start_ts

        while current_start < end_ts:
            params = {
                "symbol": self.symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ts,
                "limit": limit
            }

            try:
                response = self.session.get(f"{self.BASE_URL}/klines", params=params)
                response.raise_for_status()
                data = response.json()

                if not data:
                    break

                all_data.extend(data)
                # Move to next batch
                current_start = data[-1][0] + 1

            except requests.RequestException as e:
                print(f"Error fetching klines: {e}")
                break

        if not all_data:
            return pd.DataFrame()

        # Parse klines
        df = pd.DataFrame(all_data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # Convert types
        df['date'] = pd.to_datetime(df['open_time'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                    'taker_buy_base', 'taker_buy_quote']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['trades'] = pd.to_numeric(df['trades'], errors='coerce').astype(int)

        # Select columns
        df = df[['date', 'open', 'high', 'low', 'close', 'volume',
                 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']]
        df.set_index('date', inplace=True)

        return df

    def fetch_24hr_ticker_history(self, days: int = 365) -> pd.DataFrame:
        """
        Fetch 24hr ticker data for historical spread approximation.

        Note: Binance doesn't provide historical order book snapshots via API.
        We use bid/ask from 24hr ticker when available, otherwise approximate.
        """
        # Current ticker gives us live spread
        try:
            response = self.session.get(
                f"{self.BASE_URL}/ticker/24hr",
                params={"symbol": self.symbol}
            )
            response.raise_for_status()
            ticker = response.json()

            # Extract current spread info
            bid = float(ticker.get('bidPrice', 0))
            ask = float(ticker.get('askPrice', 0))

            print(f"Current Binance {self.symbol} spread:")
            print(f"  Bid: {bid:.2f}, Ask: {ask:.2f}")
            print(f"  Spread: {ask - bid:.2f} ({(ask - bid) / ((bid + ask) / 2) * 10000:.2f} bps)")

            return ticker

        except requests.RequestException as e:
            print(f"Error fetching 24hr ticker: {e}")
            return {}

    def fetch_recent_trades(self, limit: int = 1000) -> pd.DataFrame:
        """
        Fetch recent trades for realized spread computation.
        """
        try:
            response = self.session.get(
                f"{self.BASE_URL}/trades",
                params={"symbol": self.symbol, "limit": limit}
            )
            response.raise_for_status()
            trades = response.json()

            df = pd.DataFrame(trades)
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df['price'] = pd.to_numeric(df['price'])
            df['qty'] = pd.to_numeric(df['qty'])

            return df

        except requests.RequestException as e:
            print(f"Error fetching trades: {e}")
            return pd.DataFrame()


class SpreadProxyCalculator:
    """
    Calculate spread proxies from OHLCV data.

    Since historical order book snapshots aren't available via Binance API,
    we use established spread proxies from microstructure literature:

    1. High-Low spread proxy (Corwin-Schultz)
    2. Roll measure (serial covariance)
    3. Effective spread from price impact
    4. Volume-weighted spread proxy
    """

    @staticmethod
    def corwin_schultz_spread(df: pd.DataFrame) -> pd.Series:
        """
        Corwin-Schultz (2012) high-low spread estimator.

        Based on the idea that the high-low range reflects both volatility
        and bid-ask spread. Two-day high-low ratio used to separate them.

        Formula:
        spread = 2(exp(alpha) - 1) / (1 + exp(alpha))
        where alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma/(3 - 2*sqrt(2)))
        """
        # High-low ratio components
        high_2d = df['high'].rolling(2).max()
        low_2d = df['low'].rolling(2).min()

        beta = (np.log(df['high'] / df['low'])) ** 2
        gamma = np.log(high_2d / low_2d) ** 2

        # Sum over 2 consecutive days
        beta_sum = beta.rolling(2).sum()

        # Corwin-Schultz alpha
        numerator = np.sqrt(2 * beta_sum) - np.sqrt(beta_sum)
        denominator = 3 - 2 * np.sqrt(2)

        alpha = numerator / denominator - np.sqrt(gamma / denominator)

        # Spread estimate (set negative values to 0)
        spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        spread = spread.clip(lower=0)

        # Convert to basis points (relative to price)
        spread_bps = spread * 10000

        return spread_bps

    @staticmethod
    def roll_measure(df: pd.DataFrame) -> pd.Series:
        """
        Roll (1984) effective spread estimator.

        Based on serial covariance of price changes. Under market efficiency,
        the bid-ask bounce creates negative serial correlation.

        Spread = 2 * sqrt(-cov(r_t, r_{t-1})) if cov < 0, else 0
        """
        returns = np.log(df['close'] / df['close'].shift(1))

        # Rolling covariance
        window = 20
        cov = returns.rolling(window).cov(returns.shift(1))

        # Roll measure (only defined when covariance is negative)
        roll_spread = np.where(cov < 0, 2 * np.sqrt(-cov), 0)
        roll_spread_bps = pd.Series(roll_spread, index=df.index) * 10000

        return roll_spread_bps

    @staticmethod
    def amihud_illiquidity(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Amihud (2002) illiquidity ratio.

        Measures price impact: |return| / dollar volume
        Higher values indicate wider effective spreads.
        """
        returns = np.abs(np.log(df['close'] / df['close'].shift(1)))
        dollar_volume = df['quote_volume']

        # Avoid division by zero
        illiquidity = returns / (dollar_volume + 1e-10)

        # Rolling average
        amihud = illiquidity.rolling(window).mean()

        # Scale to comparable units (multiply by typical dollar volume)
        amihud_scaled = amihud * 1e9  # Scale factor for interpretability

        return amihud_scaled

    @staticmethod
    def range_based_volatility(df: pd.DataFrame) -> pd.Series:
        """
        Parkinson (1980) range-based volatility estimator.

        More efficient than close-to-close for high-frequency data.
        """
        log_hl = np.log(df['high'] / df['low'])
        parkinson = (1 / (4 * np.log(2))) * (log_hl ** 2)
        volatility = np.sqrt(parkinson.rolling(20).mean()) * np.sqrt(252)  # Annualized

        return volatility


class UncertaintyProxyCalculator:
    """
    Construct uncertainty proxies from market observables.

    We decompose into epistemic and aleatoric components using
    observable market characteristics:

    Aleatoric (irreducible market noise):
    - Realized volatility
    - Intraday range / overnight gaps
    - Volume volatility

    Epistemic (information uncertainty):
    - Sentiment dispersion (if available)
    - Volume concentration
    - Price momentum uncertainty
    - News/event clustering (if available)
    """

    @staticmethod
    def realized_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Standard realized volatility as aleatoric uncertainty proxy.
        """
        returns = np.log(df['close'] / df['close'].shift(1))
        rv = returns.rolling(window).std() * np.sqrt(252)  # Annualized
        return rv

    @staticmethod
    def intraday_range_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Intraday range as volatility/uncertainty proxy.
        Captures within-day price uncertainty.
        """
        intraday_range = (df['high'] - df['low']) / df['close']
        range_vol = intraday_range.rolling(window).mean()
        return range_vol

    @staticmethod
    def volume_dispersion(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Volume dispersion as epistemic uncertainty proxy.

        High volume variance suggests disagreement among traders,
        which is a form of epistemic uncertainty.
        """
        vol_mean = df['volume'].rolling(window).mean()
        vol_std = df['volume'].rolling(window).std()
        dispersion = vol_std / (vol_mean + 1e-10)  # CV of volume
        return dispersion

    @staticmethod
    def momentum_uncertainty(df: pd.DataFrame, short: int = 5, long: int = 20) -> pd.Series:
        """
        Momentum uncertainty as epistemic uncertainty proxy.

        When short and long momentum signals conflict, there's
        uncertainty about the true direction.
        """
        short_mom = df['close'].pct_change(short)
        long_mom = df['close'].pct_change(long)

        # Disagreement: when short and long momentum have opposite signs
        # or very different magnitudes
        momentum_diff = np.abs(short_mom - long_mom)
        momentum_unc = momentum_diff.rolling(5).mean()

        return momentum_unc

    @staticmethod
    def order_flow_imbalance_proxy(df: pd.DataFrame) -> pd.Series:
        """
        Order flow imbalance from taker buy/sell data.

        Binance provides taker buy volume; we can infer sell volume.
        """
        taker_buy_ratio = df['taker_buy_base'] / (df['volume'] + 1e-10)
        # Imbalance: deviation from 0.5 (balanced)
        imbalance = np.abs(taker_buy_ratio - 0.5)

        return imbalance

    @staticmethod
    def trading_intensity_uncertainty(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Trading intensity uncertainty from trade count variability.

        High variability in number of trades suggests uncertain
        market participation.
        """
        trades_mean = df['trades'].rolling(window).mean()
        trades_std = df['trades'].rolling(window).std()
        intensity_cv = trades_std / (trades_mean + 1e-10)

        return intensity_cv

    @staticmethod
    def compute_total_uncertainty(
        aleatoric: pd.Series,
        epistemic: pd.Series,
        weight_aleatoric: float = 0.5
    ) -> pd.Series:
        """
        Combine aleatoric and epistemic into total uncertainty.
        """
        # Normalize both to [0, 1] range
        aleat_norm = (aleatoric - aleatoric.min()) / (aleatoric.max() - aleatoric.min() + 1e-10)
        epist_norm = (epistemic - epistemic.min()) / (epistemic.max() - epistemic.min() + 1e-10)

        total = weight_aleatoric * aleat_norm + (1 - weight_aleatoric) * epist_norm

        return total


def load_sentiment_data(data_path: str) -> pd.DataFrame:
    """
    Load existing sentiment data from the project.
    """
    df = pd.read_csv(data_path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    return df


def merge_datasets(
    klines: pd.DataFrame,
    sentiment: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Merge klines with sentiment data if available.
    """
    df = klines.copy()

    if sentiment is not None:
        # Align on date index
        df = df.join(sentiment[['fear_greed_value', 'macro_sentiment', 'regime']], how='left')

    return df


def compute_correlations(
    df: pd.DataFrame,
    spread_col: str,
    uncertainty_cols: List[str]
) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute Pearson and Spearman correlations with p-values.

    Returns dict of {column: (pearson_r, pearson_p, spearman_r, spearman_p)}
    """
    results = {}

    spread = df[spread_col].dropna()

    for col in uncertainty_cols:
        # Align indices
        unc = df[col].dropna()
        common_idx = spread.index.intersection(unc.index)

        if len(common_idx) < 30:
            results[col] = (np.nan, np.nan, np.nan, np.nan)
            continue

        x = spread.loc[common_idx]
        y = unc.loc[common_idx]

        # Pearson
        pearson_r, pearson_p = pearsonr(x, y)

        # Spearman (robust to outliers)
        spearman_r, spearman_p = spearmanr(x, y)

        results[col] = (pearson_r, pearson_p, spearman_r, spearman_p)

    return results


def load_simulation_results(
    sim_path: str = "simulation/real_sentiment_results.csv"
) -> pd.DataFrame:
    """
    Load ABM simulation results for comparison.
    """
    # Find the file relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    full_path = os.path.join(project_dir, sim_path)

    if not os.path.exists(full_path):
        print(f"Warning: Simulation results not found at {full_path}")
        return pd.DataFrame()

    df = pd.read_csv(full_path)
    return df


def compute_simulation_correlations(sim_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """
    Compute spread-uncertainty correlations from simulation data.
    """
    results = {}

    if sim_df.empty or 'spread_bps' not in sim_df.columns:
        return results

    spread = sim_df['spread_bps']

    # Epistemic
    if 'epistemic_uncertainty' in sim_df.columns:
        r, p = pearsonr(spread, sim_df['epistemic_uncertainty'])
        results['sim_epistemic'] = (r, p)

    # Aleatoric
    if 'aleatoric_uncertainty' in sim_df.columns:
        r, p = pearsonr(spread, sim_df['aleatoric_uncertainty'])
        results['sim_aleatoric'] = (r, p)

    # Total
    if 'epistemic_uncertainty' in sim_df.columns and 'aleatoric_uncertainty' in sim_df.columns:
        total = sim_df['epistemic_uncertainty'] + sim_df['aleatoric_uncertainty']
        r, p = pearsonr(spread, total)
        results['sim_total'] = (r, p)

    return results


def print_validation_report(
    real_corr: Dict[str, Tuple],
    sim_corr: Dict[str, Tuple],
    spread_stats: Dict[str, float],
    n_obs: int,
    date_range: Tuple[str, str]
):
    """
    Print formatted validation report.
    """
    print("\n" + "=" * 80)
    print("REAL SPREAD VALIDATION: UNCERTAINTY-SPREAD CORRELATION")
    print("=" * 80)

    print(f"\n[Data Summary]")
    print(f"  Date Range: {date_range[0]} to {date_range[1]}")
    print(f"  Observations: {n_obs}")
    print(f"  Mean Spread (C-S): {spread_stats.get('cs_mean', np.nan):.2f} bps")
    print(f"  Median Spread (C-S): {spread_stats.get('cs_median', np.nan):.2f} bps")
    print(f"  Spread Std: {spread_stats.get('cs_std', np.nan):.2f} bps")

    print(f"\n[Real Market Correlations - Corwin-Schultz Spread]")
    print("-" * 70)
    print(f"{'Uncertainty Proxy':<30} {'Pearson r':>10} {'p-value':>10} {'Spearman r':>10} {'p-value':>10}")
    print("-" * 70)

    for col, (pr, pp, sr, sp) in real_corr.items():
        sig_p = "***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else ""
        sig_s = "***" if sp < 0.001 else "**" if sp < 0.01 else "*" if sp < 0.05 else ""
        print(f"{col:<30} {pr:>9.3f}{sig_p:<1} {pp:>10.4f} {sr:>9.3f}{sig_s:<1} {sp:>10.4f}")

    print("\n[Simulation Correlations (for comparison)]")
    print("-" * 50)
    for col, (r, p) in sim_corr.items():
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {col:<25} r = {r:>7.3f}{sig}  (p = {p:.4f})")

    print("\n[Interpretation]")
    print("-" * 70)

    # Key comparison: total uncertainty
    real_total = real_corr.get('realized_vol', (0, 1, 0, 1))
    sim_total = sim_corr.get('sim_total', (0, 1))

    if real_total[0] > 0.3 and real_total[1] < 0.05:
        print("  * STRONG VALIDATION: Real market shows significant positive")
        print("    correlation between spreads and volatility/uncertainty proxies.")
        print(f"    (Real: r={real_total[0]:.3f}, Sim: r={sim_total[0]:.3f})")
    elif real_total[0] > 0.1 and real_total[1] < 0.05:
        print("  * MODERATE VALIDATION: Real market shows weak but significant")
        print("    correlation, supporting the simulation findings directionally.")
    else:
        print("  * WEAK/NO VALIDATION: Real market correlation is weak or")
        print("    not statistically significant. Simulation finding may be")
        print("    partially tautological or regime-dependent.")

    print("\n[Methodology Notes]")
    print("  - Corwin-Schultz spread proxy used (OHLCV-based)")
    print("  - Realized volatility as primary aleatoric uncertainty proxy")
    print("  - Volume dispersion as epistemic uncertainty proxy")
    print("  - Binance BTC/USDT spot market data")
    print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05")


def create_comparison_table(
    real_corr: Dict[str, Tuple],
    sim_corr: Dict[str, Tuple]
) -> pd.DataFrame:
    """
    Create comparison table for paper inclusion.
    """
    rows = []

    # Map real proxies to simulation equivalents
    mapping = {
        'realized_vol': 'sim_total',
        'range_vol': 'sim_aleatoric',
        'volume_dispersion': 'sim_epistemic',
    }

    for real_name, sim_name in mapping.items():
        if real_name in real_corr and sim_name in sim_corr:
            rp, rp_pval, rs, rs_pval = real_corr[real_name]
            sp, sp_pval = sim_corr[sim_name]

            rows.append({
                'Uncertainty Measure': real_name.replace('_', ' ').title(),
                'Real Pearson r': f"{rp:.3f}",
                'Real p-value': f"{rp_pval:.4f}",
                'Simulation r': f"{sp:.3f}",
                'Sim p-value': f"{sp_pval:.4f}",
                'Direction Match': "Yes" if np.sign(rp) == np.sign(sp) else "No"
            })

    return pd.DataFrame(rows)


def run_validation(
    start_date: str = "2024-01-01",
    end_date: Optional[str] = None,
    save_results: bool = True
) -> ValidationResults:
    """
    Run full validation pipeline.
    """
    print("\n[1/6] Fetching Binance BTC/USDT klines data...")
    fetcher = BinanceSpreadFetcher("BTCUSDT")
    klines = fetcher.fetch_klines(interval="1d", start_date=start_date, end_date=end_date)

    if klines.empty:
        print("ERROR: Failed to fetch klines data")
        return None

    print(f"  Fetched {len(klines)} daily candles")

    # Get current live spread for reference
    print("\n[2/6] Fetching current live spread for reference...")
    fetcher.fetch_24hr_ticker_history()

    print("\n[3/6] Computing spread proxies...")
    spread_calc = SpreadProxyCalculator()
    klines['cs_spread'] = spread_calc.corwin_schultz_spread(klines)
    klines['roll_spread'] = spread_calc.roll_measure(klines)
    klines['amihud'] = spread_calc.amihud_illiquidity(klines)
    klines['parkinson_vol'] = spread_calc.range_based_volatility(klines)

    print(f"  Corwin-Schultz spread: mean={klines['cs_spread'].mean():.2f} bps")
    print(f"  Roll spread: mean={klines['roll_spread'].mean():.2f} bps")

    print("\n[4/6] Computing uncertainty proxies...")
    unc_calc = UncertaintyProxyCalculator()
    klines['realized_vol'] = unc_calc.realized_volatility(klines)
    klines['range_vol'] = unc_calc.intraday_range_volatility(klines)
    klines['volume_dispersion'] = unc_calc.volume_dispersion(klines)
    klines['momentum_unc'] = unc_calc.momentum_uncertainty(klines)
    klines['flow_imbalance'] = unc_calc.order_flow_imbalance_proxy(klines)
    klines['intensity_unc'] = unc_calc.trading_intensity_uncertainty(klines)

    # Composite uncertainty measures
    klines['aleatoric_proxy'] = (klines['realized_vol'] + klines['range_vol']) / 2
    klines['epistemic_proxy'] = (klines['volume_dispersion'] + klines['momentum_unc']) / 2
    klines['total_uncertainty'] = unc_calc.compute_total_uncertainty(
        klines['aleatoric_proxy'], klines['epistemic_proxy']
    )

    print("\n[5/6] Loading simulation results for comparison...")
    sim_df = load_simulation_results()
    sim_corr = compute_simulation_correlations(sim_df)

    print("\n[6/6] Computing correlations...")
    uncertainty_cols = [
        'realized_vol', 'range_vol', 'volume_dispersion',
        'momentum_unc', 'flow_imbalance', 'intensity_unc',
        'aleatoric_proxy', 'epistemic_proxy', 'total_uncertainty'
    ]

    real_corr = compute_correlations(klines, 'cs_spread', uncertainty_cols)

    # Drop NaN for stats
    valid_df = klines.dropna(subset=['cs_spread'])

    spread_stats = {
        'cs_mean': valid_df['cs_spread'].mean(),
        'cs_median': valid_df['cs_spread'].median(),
        'cs_std': valid_df['cs_spread'].std(),
        'cs_min': valid_df['cs_spread'].min(),
        'cs_max': valid_df['cs_spread'].max(),
    }

    date_range = (
        klines.index.min().strftime("%Y-%m-%d"),
        klines.index.max().strftime("%Y-%m-%d")
    )

    # Print report
    print_validation_report(real_corr, sim_corr, spread_stats, len(valid_df), date_range)

    # Create comparison table
    comparison_table = create_comparison_table(real_corr, sim_corr)
    print("\n[Comparison Table for Paper]")
    print(comparison_table.to_string(index=False))

    # Save results
    if save_results:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        results_dir = os.path.join(project_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        # Save processed data
        klines.to_csv(os.path.join(results_dir, "real_spread_data.csv"))
        comparison_table.to_csv(os.path.join(results_dir, "spread_validation_comparison.csv"), index=False)

        # Save detailed correlations
        corr_data = []
        for col, (pr, pp, sr, sp) in real_corr.items():
            corr_data.append({
                'proxy': col,
                'pearson_r': pr,
                'pearson_p': pp,
                'spearman_r': sr,
                'spearman_p': sp
            })
        pd.DataFrame(corr_data).to_csv(
            os.path.join(results_dir, "real_spread_correlations.csv"), index=False
        )

        print(f"\n[Results saved to {results_dir}/]")

    return ValidationResults(
        real_correlations=real_corr,
        simulation_correlations=sim_corr,
        n_observations=len(valid_df),
        date_range=date_range,
        spread_stats=spread_stats,
        uncertainty_stats={}
    )


def run_robustness_tests(df: pd.DataFrame, spread_col: str = 'cs_spread') -> Dict[str, Any]:
    """
    Run additional robustness tests for the spread-uncertainty relationship.

    Tests include:
    1. Granger causality (does uncertainty predict spreads?)
    2. Regime-conditional correlations
    3. Rolling correlation stability
    4. Newey-West HAC standard errors
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    results = {}

    print("\n" + "=" * 80)
    print("ROBUSTNESS TESTS")
    print("=" * 80)

    # 1. Granger Causality Test
    print("\n[Granger Causality: Does Uncertainty Predict Spreads?]")
    print("-" * 60)

    valid_df = df[[spread_col, 'realized_vol']].dropna()

    try:
        gc_results = grangercausalitytests(
            valid_df[[spread_col, 'realized_vol']],
            maxlag=5,
            verbose=False
        )

        # Extract F-test p-values
        gc_pvals = {lag: gc_results[lag][0]['ssr_ftest'][1] for lag in range(1, 6)}
        results['granger_pvals'] = gc_pvals

        print(f"  Lag | F-test p-value | Significant?")
        print(f"  ----|----------------|-------------")
        for lag, pval in gc_pvals.items():
            sig = "Yes ***" if pval < 0.001 else "Yes **" if pval < 0.01 else "Yes *" if pval < 0.05 else "No"
            print(f"   {lag}  |     {pval:.4f}     |    {sig}")

        # Interpretation
        min_pval = min(gc_pvals.values())
        if min_pval < 0.05:
            print(f"\n  RESULT: Volatility Granger-causes spreads (min p={min_pval:.4f})")
            print("  This supports a causal interpretation, not just tautology.")
        else:
            print(f"\n  RESULT: No Granger causality detected (min p={min_pval:.4f})")

    except Exception as e:
        print(f"  Granger test failed: {e}")
        results['granger_pvals'] = {}

    # 2. Regime-Conditional Correlations
    print("\n[Regime-Conditional Correlations]")
    print("-" * 60)

    if 'regime' not in df.columns:
        # Create volatility regimes based on realized vol percentiles
        vol_33 = df['realized_vol'].quantile(0.33)
        vol_67 = df['realized_vol'].quantile(0.67)
        df['vol_regime'] = pd.cut(
            df['realized_vol'],
            bins=[-np.inf, vol_33, vol_67, np.inf],
            labels=['low_vol', 'mid_vol', 'high_vol']
        )
        regime_col = 'vol_regime'
    else:
        regime_col = 'regime'

    regime_corrs = {}
    for regime in df[regime_col].dropna().unique():
        regime_data = df[df[regime_col] == regime][[spread_col, 'realized_vol']].dropna()
        if len(regime_data) > 30:
            r, p = pearsonr(regime_data[spread_col], regime_data['realized_vol'])
            regime_corrs[str(regime)] = {'r': r, 'p': p, 'n': len(regime_data)}
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {str(regime):<15}: r = {r:>6.3f}{sig}  (n={len(regime_data)}, p={p:.4f})")

    results['regime_correlations'] = regime_corrs

    # 3. Rolling Correlation Stability
    print("\n[Rolling Correlation Stability (60-day window)]")
    print("-" * 60)

    valid_df = df[[spread_col, 'realized_vol']].dropna()
    rolling_corr = valid_df[spread_col].rolling(60).corr(valid_df['realized_vol'])

    results['rolling_corr_stats'] = {
        'mean': rolling_corr.mean(),
        'std': rolling_corr.std(),
        'min': rolling_corr.min(),
        'max': rolling_corr.max(),
        'pct_positive': (rolling_corr > 0).mean() * 100
    }

    print(f"  Mean rolling correlation:  {results['rolling_corr_stats']['mean']:.3f}")
    print(f"  Std of rolling correlation: {results['rolling_corr_stats']['std']:.3f}")
    print(f"  Range: [{results['rolling_corr_stats']['min']:.3f}, {results['rolling_corr_stats']['max']:.3f}]")
    print(f"  % of time positive: {results['rolling_corr_stats']['pct_positive']:.1f}%")

    if results['rolling_corr_stats']['pct_positive'] > 80:
        print("  RESULT: Correlation is stable and consistently positive.")
    elif results['rolling_corr_stats']['pct_positive'] > 50:
        print("  RESULT: Correlation is positive on average but variable.")
    else:
        print("  RESULT: Correlation is unstable and often negative.")

    # 4. OLS Regression with HAC Standard Errors
    print("\n[OLS Regression with Newey-West HAC Errors]")
    print("-" * 60)

    try:
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tools import add_constant

        valid_df = df[[spread_col, 'realized_vol', 'volume_dispersion']].dropna()

        X = add_constant(valid_df[['realized_vol', 'volume_dispersion']])
        y = valid_df[spread_col]

        # OLS with HAC (Newey-West) standard errors
        model = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 10})

        print(f"  {'Variable':<20} {'Coef':>10} {'HAC SE':>10} {'t-stat':>10} {'p-value':>10}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

        for var in ['const', 'realized_vol', 'volume_dispersion']:
            coef = model.params[var]
            se = model.bse[var]
            t = model.tvalues[var]
            p = model.pvalues[var]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {var:<20} {coef:>10.3f} {se:>10.3f} {t:>10.3f} {p:>9.4f}{sig}")

        print(f"\n  R-squared: {model.rsquared:.4f}")
        print(f"  Adj. R-squared: {model.rsquared_adj:.4f}")
        print(f"  F-statistic: {model.fvalue:.2f} (p={model.f_pvalue:.4f})")

        results['ols_results'] = {
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_stat': model.fvalue,
            'f_pval': model.f_pvalue,
            'coefficients': dict(model.params),
            'pvalues': dict(model.pvalues)
        }

    except Exception as e:
        print(f"  OLS regression failed: {e}")
        results['ols_results'] = {}

    return results


def main():
    """
    Main entry point.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate ABM spread-uncertainty correlation against real Binance data"
    )
    parser.add_argument(
        "--start-date", "-s",
        default="2024-01-01",
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", "-e",
        default=None,
        help="End date (YYYY-MM-DD), defaults to today"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to files"
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="Run additional robustness tests"
    )

    args = parser.parse_args()

    results = run_validation(
        start_date=args.start_date,
        end_date=args.end_date,
        save_results=not args.no_save
    )

    # Run robustness tests if requested
    if args.robustness:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        results_file = os.path.join(project_dir, "results", "real_spread_data.csv")

        if os.path.exists(results_file):
            df = pd.read_csv(results_file, index_col=0, parse_dates=True)
            robustness_results = run_robustness_tests(df)

    return results


if __name__ == "__main__":
    main()
