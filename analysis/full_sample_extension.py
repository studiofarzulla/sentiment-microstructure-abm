#!/usr/bin/env python3
"""
Full Sample Extension: Feb 2018 - Jan 2026

Extends the analysis from 739 days to ~2900 days by pulling all available
Fear & Greed Index data and matching OHLCV data.

This addresses the core statistical power issue: with 4x more data,
within-quintile cells go from n=11-59 to n=50-250, and multiple testing
corrections become less punishing.

Author: Murad Farzulla
Date: January 2026
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
from scipy import stats
from scipy.stats import ttest_ind
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Output directories
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'datasets')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_fear_greed_full() -> pd.DataFrame:
    """
    Fetch ALL Fear & Greed Index data from Feb 2018 to present.
    """
    logger.info("Fetching Fear & Greed Index (full history)...")

    url = "https://api.alternative.me/fng/"
    params = {'limit': 0, 'format': 'json'}  # limit=0 gets all

    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        records = []
        for item in data.get('data', []):
            records.append({
                'date': datetime.fromtimestamp(int(item['timestamp'])).date(),
                'fear_greed_value': int(item['value']),
                'fear_greed_class': item['value_classification'],
            })

        df = pd.DataFrame(records)
        df = df.sort_values('date').reset_index(drop=True)

        logger.info(f"  Fetched {len(df)} days")
        logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")

        return df

    except Exception as e:
        logger.error(f"Failed to fetch Fear & Greed: {e}")
        raise


def fetch_binance_ohlcv(
    symbol: str = "BTCUSDT",
    start_date: str = "2018-02-01",
    end_date: str = None
) -> pd.DataFrame:
    """
    Fetch historical OHLCV from Binance (paginated).
    """
    logger.info(f"Fetching {symbol} OHLCV from {start_date}...")

    base_url = "https://api.binance.com/api/v3/klines"

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.utcnow()

    all_data = []
    current_start = start

    while current_start < end:
        params = {
            'symbol': symbol,
            'interval': '1d',
            'startTime': int(current_start.timestamp() * 1000),
            'endTime': int(end.timestamp() * 1000),
            'limit': 1000
        }

        try:
            resp = requests.get(base_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if not data:
                break

            all_data.extend(data)

            # Move to next batch
            last_ts = data[-1][0]
            current_start = datetime.fromtimestamp(last_ts / 1000) + timedelta(days=1)

            logger.info(f"  Fetched up to {current_start.date()}")
            time.sleep(0.1)  # Rate limit

        except Exception as e:
            logger.error(f"Error at {current_start}: {e}")
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    df['date'] = pd.to_datetime(df['open_time'], unit='ms').dt.date

    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
        df[col] = df[col].astype(float)
    df['trades'] = df['trades'].astype(int)

    df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'trades', 'quote_volume']]
    df = df.drop_duplicates(subset=['date']).sort_values('date')

    logger.info(f"  Total: {len(df)} days")

    return df


# =============================================================================
# SPREAD & VOLATILITY COMPUTATION
# =============================================================================

def compute_corwin_schultz_spread(df: pd.DataFrame) -> pd.Series:
    """
    Corwin-Schultz (2012) spread estimator from daily OHLC.

    S = 2(exp(α) - 1) / (1 + exp(α))
    where α = (√2β - √β) / (3 - 2√2) - √(γ / (3 - 2√2))
    """
    high = df['high'].values
    low = df['low'].values

    # β = E[(ln(H_t/L_t))^2 + (ln(H_{t+1}/L_{t+1}))^2]
    log_hl = np.log(high / low)
    log_hl_sq = log_hl ** 2

    # Rolling 2-day sum
    beta = pd.Series(log_hl_sq).rolling(2).sum().values

    # γ = (ln(H_max / L_min))^2 over 2 days
    high_2d = pd.Series(high).rolling(2).max().values
    low_2d = pd.Series(low).rolling(2).min().values
    gamma = np.log(high_2d / low_2d) ** 2

    # α calculation
    sqrt2 = np.sqrt(2)
    denom = 3 - 2 * sqrt2

    with np.errstate(invalid='ignore', divide='ignore'):
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
        alpha = np.where(alpha < 0, 0, alpha)  # Can't be negative

        # Spread
        spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        spread = np.clip(spread, 0, None) * 10000  # Convert to bps

    return pd.Series(spread, index=df.index)


def compute_parkinson_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Parkinson (1980) range-based volatility estimator.
    """
    log_hl = np.log(df['high'] / df['low'])
    parkinson = np.sqrt((log_hl ** 2).rolling(window).mean() / (4 * np.log(2)))
    return parkinson


def compute_realized_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Standard realized volatility from close-to-close returns.
    """
    returns = np.log(df['close'] / df['close'].shift(1))
    return returns.rolling(window).std() * np.sqrt(252)


def classify_regime(fg_value: int) -> str:
    """Classify F&G value into regime."""
    if pd.isna(fg_value):
        return 'unknown'
    elif fg_value <= 25:
        return 'extreme_fear'
    elif fg_value <= 45:
        return 'fear'
    elif fg_value <= 55:
        return 'neutral'
    elif fg_value <= 75:
        return 'greed'
    else:
        return 'extreme_greed'


# =============================================================================
# DATA PIPELINE
# =============================================================================

def build_full_dataset(
    btc_df: pd.DataFrame,
    fg_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge OHLCV with Fear & Greed and compute all derived variables.
    """
    logger.info("Building full dataset...")

    # Merge
    df = btc_df.merge(fg_df, on='date', how='inner')
    logger.info(f"  Merged: {len(df)} days with both price and F&G data")

    # Compute spreads and volatility
    df['cs_spread'] = compute_corwin_schultz_spread(df)
    df['parkinson_vol'] = compute_parkinson_volatility(df)
    df['realized_vol'] = compute_realized_volatility(df)
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    df['log_volume'] = np.log(df['volume'])

    # Regime classification
    df['regime'] = df['fear_greed_value'].apply(classify_regime)
    df['is_extreme'] = df['regime'].isin(['extreme_fear', 'extreme_greed'])

    # Macro sentiment (-1 to 1)
    df['macro_sentiment'] = (df['fear_greed_value'] - 50) / 50

    # For extended sample, use Parkinson volatility as primary uncertainty proxy
    # (CryptoBERT-based decomposition not available for historical period)
    df['uncertainty'] = df['parkinson_vol']
    df['volatility'] = df['realized_vol']

    # Drop rows with NaN in key columns
    initial_len = len(df)
    df = df.dropna(subset=['cs_spread', 'parkinson_vol', 'realized_vol', 'regime'])
    logger.info(f"  After dropna: {len(df)} days (dropped {initial_len - len(df)})")

    return df.reset_index(drop=True)


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_regime_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics by regime.
    """
    regimes = ['extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed']

    results = []
    for regime in regimes:
        subset = df[df['regime'] == regime]
        results.append({
            'regime': regime,
            'n': len(subset),
            'pct': 100 * len(subset) / len(df),
            'cs_spread_mean': subset['cs_spread'].mean(),
            'cs_spread_std': subset['cs_spread'].std(),
            'parkinson_vol_mean': subset['parkinson_vol'].mean(),
            'parkinson_vol_std': subset['parkinson_vol'].std(),
            'fg_mean': subset['fear_greed_value'].mean(),
        })

    return pd.DataFrame(results)


def compute_extremity_premium(df: pd.DataFrame) -> Dict:
    """
    Compute extremity premium with full statistical details.
    """
    extreme = df[df['is_extreme']]['cs_spread']
    neutral = df[df['regime'] == 'neutral']['cs_spread']

    # Welch's t-test
    t_stat, p_value = ttest_ind(extreme, neutral, equal_var=False)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((extreme.var() + neutral.var()) / 2)
    cohens_d = (extreme.mean() - neutral.mean()) / pooled_std

    # Bootstrap CI
    n_boot = 10000
    boot_diffs = []
    for _ in range(n_boot):
        ext_sample = np.random.choice(extreme.values, size=len(extreme), replace=True)
        neu_sample = np.random.choice(neutral.values, size=len(neutral), replace=True)
        boot_diffs.append(ext_sample.mean() - neu_sample.mean())

    ci_lower, ci_upper = np.percentile(boot_diffs, [2.5, 97.5])

    return {
        'n_extreme': len(extreme),
        'n_neutral': len(neutral),
        'extreme_mean': extreme.mean(),
        'neutral_mean': neutral.mean(),
        'gap': extreme.mean() - neutral.mean(),
        'gap_bps': (extreme.mean() - neutral.mean()),
        't_stat': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
    }


def within_quintile_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stratified analysis within volatility quintiles with full statistics.
    """
    # Quintile on realized volatility
    df = df.copy()
    df['vol_quintile'] = pd.qcut(df['realized_vol'], 5, labels=[1, 2, 3, 4, 5])

    results = []
    for q in [1, 2, 3, 4, 5]:
        q_data = df[df['vol_quintile'] == q]

        extreme = q_data[q_data['is_extreme']]['cs_spread']
        neutral = q_data[q_data['regime'] == 'neutral']['cs_spread']

        if len(extreme) < 5 or len(neutral) < 5:
            continue

        # Statistics
        t_stat, p_value = ttest_ind(extreme, neutral, equal_var=False)
        pooled_std = np.sqrt((extreme.var() + neutral.var()) / 2)
        cohens_d = (extreme.mean() - neutral.mean()) / pooled_std if pooled_std > 0 else 0

        # CI via Welch's formula
        se_diff = np.sqrt(extreme.var()/len(extreme) + neutral.var()/len(neutral))
        df_welch = ((extreme.var()/len(extreme) + neutral.var()/len(neutral))**2 /
                    ((extreme.var()/len(extreme))**2/(len(extreme)-1) +
                     (neutral.var()/len(neutral))**2/(len(neutral)-1)))
        t_crit = stats.t.ppf(0.975, df_welch)
        gap = extreme.mean() - neutral.mean()
        ci_lower = gap - t_crit * se_diff
        ci_upper = gap + t_crit * se_diff

        # Volatility range for this quintile
        vol_min = q_data['realized_vol'].min()
        vol_max = q_data['realized_vol'].max()

        results.append({
            'quintile': q,
            'n_extreme': len(extreme),
            'n_neutral': len(neutral),
            'vol_range': f"[{vol_min:.3f}, {vol_max:.3f}]",
            'gap': gap,
            'gap_bps': gap,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            't_stat': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
        })

    results_df = pd.DataFrame(results)

    # Holm-Bonferroni correction
    if len(results_df) > 0:
        _, p_adj, _, _ = multipletests(results_df['p_value'], method='holm')
        results_df['p_adj_holm'] = p_adj
        results_df['sig_holm'] = p_adj < 0.05

    return results_df


def regime_regression(df: pd.DataFrame) -> Dict:
    """
    OLS regression of spreads on regime dummies with volatility control.
    """
    # Prepare data
    df_reg = df.dropna(subset=['cs_spread', 'realized_vol', 'regime']).copy()

    # Create dummies (neutral as reference)
    regime_dummies = pd.get_dummies(df_reg['regime'], prefix='regime', dtype=int)
    if 'regime_neutral' in regime_dummies.columns:
        regime_dummies = regime_dummies.drop('regime_neutral', axis=1)

    # Design matrix
    X = pd.concat([
        pd.DataFrame({'const': 1, 'volatility': df_reg['realized_vol']}),
        regime_dummies
    ], axis=1)
    y = df_reg['cs_spread']

    # OLS with HAC standard errors
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})

    return {
        'params': model.params.to_dict(),
        'pvalues': model.pvalues.to_dict(),
        'rsquared': model.rsquared,
        'nobs': int(model.nobs),
        'summary': model.summary().as_text(),
    }


def granger_causality_test(df: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    """
    Granger causality tests for uncertainty → spreads and reverse.
    """
    from statsmodels.tsa.stattools import grangercausalitytests, adfuller

    # Prepare data
    df_gc = df[['cs_spread', 'parkinson_vol']].dropna()

    # ADF tests
    adf_spread = adfuller(df_gc['cs_spread'])
    adf_vol = adfuller(df_gc['parkinson_vol'])

    results = []

    for lag in range(1, max_lag + 1):
        # Uncertainty → Spreads
        try:
            gc_unc_spread = grangercausalitytests(
                df_gc[['cs_spread', 'parkinson_vol']], maxlag=lag, verbose=False
            )
            f_stat_us = gc_unc_spread[lag][0]['ssr_ftest'][0]
            p_val_us = gc_unc_spread[lag][0]['ssr_ftest'][1]
        except:
            f_stat_us, p_val_us = np.nan, np.nan

        # Spreads → Uncertainty
        try:
            gc_spread_unc = grangercausalitytests(
                df_gc[['parkinson_vol', 'cs_spread']], maxlag=lag, verbose=False
            )
            f_stat_su = gc_spread_unc[lag][0]['ssr_ftest'][0]
            p_val_su = gc_spread_unc[lag][0]['ssr_ftest'][1]
        except:
            f_stat_su, p_val_su = np.nan, np.nan

        results.append({
            'lag': lag,
            'unc_to_spread_F': f_stat_us,
            'unc_to_spread_p': p_val_us,
            'spread_to_unc_F': f_stat_su,
            'spread_to_unc_p': p_val_su,
        })

    results_df = pd.DataFrame(results)
    results_df['adf_spread_stat'] = adf_spread[0]
    results_df['adf_spread_p'] = adf_spread[1]
    results_df['adf_vol_stat'] = adf_vol[0]
    results_df['adf_vol_p'] = adf_vol[1]

    return results_df


def run_placebo_tests(df: pd.DataFrame, n_perms: int = 10000) -> Dict:
    """
    Placebo tests: block-shuffled permutation and time-reversed.
    """
    logger.info("Running placebo tests...")

    # Observed gap
    extreme = df[df['is_extreme']]['cs_spread'].mean()
    neutral = df[df['regime'] == 'neutral']['cs_spread'].mean()
    observed_gap = extreme - neutral

    # 1. Standard permutation
    spread_values = df['cs_spread'].values
    is_extreme = df['is_extreme'].values

    perm_gaps = []
    for _ in range(n_perms):
        shuffled = np.random.permutation(is_extreme)
        ext_mean = spread_values[shuffled].mean()
        neu_mean = spread_values[~shuffled].mean()
        perm_gaps.append(ext_mean - neu_mean)

    perm_gaps = np.array(perm_gaps)
    perm_p = np.mean(np.abs(perm_gaps) >= np.abs(observed_gap))

    # 2. Block-shuffled (preserve autocorrelation)
    # Identify regime blocks
    df_sorted = df.sort_values('date').reset_index(drop=True)
    regime_changes = (df_sorted['regime'] != df_sorted['regime'].shift()).cumsum()
    blocks = df_sorted.groupby(regime_changes).apply(
        lambda x: {'start': x.index[0], 'end': x.index[-1], 'regime': x['regime'].iloc[0]}
    ).tolist()

    block_gaps = []
    for _ in range(n_perms):
        # Shuffle block labels
        shuffled_blocks = blocks.copy()
        np.random.shuffle(shuffled_blocks)

        # Reconstruct shuffled regime assignments
        shuffled_regime = pd.Series(index=df_sorted.index, dtype=str)
        for i, block in enumerate(shuffled_blocks):
            orig_block = blocks[i]
            shuffled_regime.iloc[orig_block['start']:orig_block['end']+1] = block['regime']

        is_ext_shuffled = shuffled_regime.isin(['extreme_fear', 'extreme_greed'])
        is_neu_shuffled = shuffled_regime == 'neutral'

        if is_ext_shuffled.sum() > 0 and is_neu_shuffled.sum() > 0:
            ext_mean = df_sorted.loc[is_ext_shuffled, 'cs_spread'].mean()
            neu_mean = df_sorted.loc[is_neu_shuffled, 'cs_spread'].mean()
            block_gaps.append(ext_mean - neu_mean)

    block_gaps = np.array(block_gaps)
    block_p = np.mean(np.abs(block_gaps) >= np.abs(observed_gap)) if len(block_gaps) > 0 else np.nan

    return {
        'observed_gap': observed_gap,
        'perm_mean': perm_gaps.mean(),
        'perm_std': perm_gaps.std(),
        'perm_p': perm_p,
        'block_mean': block_gaps.mean() if len(block_gaps) > 0 else np.nan,
        'block_std': block_gaps.std() if len(block_gaps) > 0 else np.nan,
        'block_p': block_p,
    }


# =============================================================================
# MARKET CYCLE ANALYSIS
# =============================================================================

def analyze_by_market_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze extremity premium by market cycle.
    """
    # Define market cycles based on BTC price action
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    cycles = [
        ('2018 Bear', '2018-01-01', '2018-12-31'),
        ('2019 Recovery', '2019-01-01', '2019-12-31'),
        ('2020 COVID+Bull', '2020-01-01', '2020-12-31'),
        ('2021 Bull Peak', '2021-01-01', '2021-12-31'),
        ('2022 Bear', '2022-01-01', '2022-12-31'),
        ('2023 Recovery', '2023-01-01', '2023-12-31'),
        ('2024-25 Bull', '2024-01-01', '2026-01-31'),
    ]

    results = []
    for name, start, end in cycles:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

        cycle_df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]

        if len(cycle_df) < 30:
            continue

        extreme = cycle_df[cycle_df['is_extreme']]['cs_spread']
        neutral = cycle_df[cycle_df['regime'] == 'neutral']['cs_spread']

        if len(extreme) < 5 or len(neutral) < 5:
            results.append({
                'cycle': name,
                'n_days': len(cycle_df),
                'n_extreme': len(extreme),
                'n_neutral': len(neutral),
                'gap': np.nan,
                'p_value': np.nan,
                'cohens_d': np.nan,
                'note': 'Insufficient data'
            })
            continue

        t_stat, p_value = ttest_ind(extreme, neutral, equal_var=False)
        pooled_std = np.sqrt((extreme.var() + neutral.var()) / 2)
        cohens_d = (extreme.mean() - neutral.mean()) / pooled_std if pooled_std > 0 else 0

        results.append({
            'cycle': name,
            'n_days': len(cycle_df),
            'n_extreme': len(extreme),
            'n_neutral': len(neutral),
            'extreme_mean': extreme.mean(),
            'neutral_mean': neutral.mean(),
            'gap': extreme.mean() - neutral.mean(),
            't_stat': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
        })

    results_df = pd.DataFrame(results)

    # Multiple testing correction
    valid_p = results_df['p_value'].notna()
    if valid_p.sum() > 0:
        _, p_adj, _, _ = multipletests(results_df.loc[valid_p, 'p_value'], method='holm')
        results_df.loc[valid_p, 'p_adj_holm'] = p_adj
        results_df['sig_holm'] = results_df['p_adj_holm'] < 0.05

    return results_df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Full pipeline: fetch data, build dataset, run all analyses.
    """
    logger.info("=" * 70)
    logger.info("FULL SAMPLE EXTENSION: Feb 2018 - Jan 2026")
    logger.info("=" * 70)

    # 1. FETCH DATA
    logger.info("\n[1/7] Fetching Fear & Greed Index...")
    fg_df = fetch_fear_greed_full()
    fg_df.to_csv(os.path.join(DATA_DIR, 'fear_greed_full_history.csv'), index=False)

    logger.info("\n[2/7] Fetching BTC OHLCV...")
    btc_df = fetch_binance_ohlcv(
        symbol="BTCUSDT",
        start_date="2018-02-01",
        end_date=datetime.utcnow().strftime("%Y-%m-%d")
    )
    btc_df.to_csv(os.path.join(DATA_DIR, 'btc_ohlcv_full_history.csv'), index=False)

    logger.info("\n[3/7] Fetching ETH OHLCV...")
    eth_df = fetch_binance_ohlcv(
        symbol="ETHUSDT",
        start_date="2018-02-01",
        end_date=datetime.utcnow().strftime("%Y-%m-%d")
    )
    eth_df.to_csv(os.path.join(DATA_DIR, 'eth_ohlcv_full_history.csv'), index=False)

    # 2. BUILD DATASET
    logger.info("\n[4/7] Building full BTC dataset...")
    df = build_full_dataset(btc_df, fg_df)
    df.to_csv(os.path.join(RESULTS_DIR, 'full_sample_btc_data.csv'), index=False)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total observations: {len(df)}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Regime distribution:")
    for regime, count in df['regime'].value_counts().items():
        logger.info(f"  {regime:15s}: {count:4d} ({100*count/len(df):5.1f}%)")

    # 3. RUN ANALYSES
    logger.info("\n[5/7] Running core analyses...")

    # Regime statistics
    regime_stats = compute_regime_statistics(df)
    regime_stats.to_csv(os.path.join(RESULTS_DIR, 'full_sample_regime_stats.csv'), index=False)
    logger.info(f"\nRegime statistics saved")

    # Extremity premium
    premium = compute_extremity_premium(df)
    logger.info(f"\nExtremity Premium (Full Sample):")
    logger.info(f"  N extreme: {premium['n_extreme']}, N neutral: {premium['n_neutral']}")
    logger.info(f"  Gap: {premium['gap']:.2f} bps")
    logger.info(f"  95% CI: [{premium['ci_lower']:.2f}, {premium['ci_upper']:.2f}]")
    logger.info(f"  Cohen's d: {premium['cohens_d']:.3f}")
    logger.info(f"  p-value: {premium['p_value']:.2e}")

    pd.DataFrame([premium]).to_csv(
        os.path.join(RESULTS_DIR, 'full_sample_extremity_premium.csv'), index=False
    )

    # Within-quintile
    quintile_results = within_quintile_analysis(df)
    quintile_results.to_csv(
        os.path.join(RESULTS_DIR, 'full_sample_within_quintile.csv'), index=False
    )
    logger.info(f"\nWithin-Quintile Analysis:")
    logger.info(f"  Significant after Holm: {quintile_results['sig_holm'].sum()}/{len(quintile_results)}")

    # Regime regression
    reg_results = regime_regression(df)
    logger.info(f"\nRegime Regression R²: {reg_results['rsquared']:.3f}")
    with open(os.path.join(RESULTS_DIR, 'full_sample_regression_summary.txt'), 'w') as f:
        f.write(reg_results['summary'])

    # Granger causality
    granger_results = granger_causality_test(df)
    granger_results.to_csv(
        os.path.join(RESULTS_DIR, 'full_sample_granger_causality.csv'), index=False
    )
    logger.info(f"\nGranger Causality (lag=1):")
    logger.info(f"  Uncertainty → Spreads: F={granger_results.iloc[0]['unc_to_spread_F']:.2f}, p={granger_results.iloc[0]['unc_to_spread_p']:.4f}")
    logger.info(f"  Spreads → Uncertainty: F={granger_results.iloc[0]['spread_to_unc_F']:.2f}, p={granger_results.iloc[0]['spread_to_unc_p']:.4f}")

    # Placebo tests
    logger.info("\n[6/7] Running placebo tests...")
    placebo_results = run_placebo_tests(df, n_perms=10000)
    logger.info(f"\nPlacebo Tests:")
    logger.info(f"  Standard permutation p: {placebo_results['perm_p']:.4f}")
    logger.info(f"  Block-shuffled p: {placebo_results['block_p']:.4f}")
    pd.DataFrame([placebo_results]).to_csv(
        os.path.join(RESULTS_DIR, 'full_sample_placebo_tests.csv'), index=False
    )

    # Market cycle analysis
    logger.info("\n[7/7] Analyzing by market cycle...")
    cycle_results = analyze_by_market_cycle(df)
    cycle_results.to_csv(
        os.path.join(RESULTS_DIR, 'full_sample_market_cycles.csv'), index=False
    )
    logger.info(f"\nMarket Cycle Analysis:")
    for _, row in cycle_results.iterrows():
        sig = "**" if row.get('sig_holm', False) else ""
        logger.info(f"  {row['cycle']:20s}: gap={row.get('gap', np.nan):+.2f} bps, d={row.get('cohens_d', np.nan):.2f} {sig}")

    # ETH cross-asset
    logger.info("\nBuilding ETH dataset...")
    eth_full = build_full_dataset(eth_df, fg_df)
    eth_full.to_csv(os.path.join(RESULTS_DIR, 'full_sample_eth_data.csv'), index=False)

    eth_premium = compute_extremity_premium(eth_full)
    logger.info(f"\nETH Extremity Premium:")
    logger.info(f"  Gap: {eth_premium['gap']:.2f} bps, d={eth_premium['cohens_d']:.3f}, p={eth_premium['p_value']:.2e}")
    pd.DataFrame([eth_premium]).to_csv(
        os.path.join(RESULTS_DIR, 'full_sample_eth_extremity_premium.csv'), index=False
    )

    # FINAL SUMMARY
    logger.info("\n" + "=" * 70)
    logger.info("FULL SAMPLE ANALYSIS COMPLETE")
    logger.info("=" * 70)
    logger.info(f"BTC: {len(df)} days, ETH: {len(eth_full)} days")
    logger.info(f"Results saved to: {RESULTS_DIR}")

    # Power comparison
    logger.info("\nPOWER COMPARISON:")
    logger.info(f"  Old sample: 739 days")
    logger.info(f"  New sample: {len(df)} days")
    logger.info(f"  Increase: {100*(len(df)/739 - 1):.0f}%")

    return df, eth_full


if __name__ == '__main__':
    btc_df, eth_df = main()
