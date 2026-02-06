"""
ETH Cross-Asset Validation for the Extremity Premium

Tests whether the extremity premium (extreme sentiment → higher uncertainty)
holds for ETH as well as BTC.

Uses the same Fear & Greed Index (market-wide crypto sentiment).

Author: Murad Farzulla
Date: January 2026
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False


def fetch_eth_data_yfinance(start_date, end_date):
    """Fetch ETH data from Yahoo Finance."""
    print("Fetching ETH data from Yahoo Finance...")
    eth = yf.Ticker("ETH-USD")
    df = eth.history(start=start_date, end=end_date)
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={'date': 'date'})
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    return df[['date', 'open', 'high', 'low', 'close', 'volume']]


def fetch_eth_data_ccxt(start_date, end_date):
    """Fetch ETH data from Binance via ccxt."""
    print("Fetching ETH data from Binance via ccxt...")
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv('ETH/USDT', '1d',
                                  since=int(pd.Timestamp(start_date).timestamp() * 1000),
                                  limit=1000)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df[['date', 'open', 'high', 'low', 'close', 'volume']]


def fetch_eth_data(start_date='2024-01-01', end_date='2026-01-10'):
    """Fetch ETH OHLCV data from available source."""
    if HAS_YFINANCE:
        return fetch_eth_data_yfinance(start_date, end_date)
    elif HAS_CCXT:
        return fetch_eth_data_ccxt(start_date, end_date)
    else:
        raise ImportError("Neither yfinance nor ccxt is installed. "
                          "Install with: pip install yfinance")


def compute_spread_metrics(df):
    """Compute Corwin-Schultz spread and realized volatility."""
    df = df.copy()

    # Parkinson volatility
    df['parkinson_vol'] = np.sqrt(np.log(df['high'] / df['low'])**2 / (4 * np.log(2)))

    # Realized volatility (rolling 5-day)
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    df['realized_vol'] = df['returns'].rolling(5).std() * np.sqrt(252)

    # Corwin-Schultz spread (simplified)
    beta = np.log(df['high'] / df['low'])**2
    df['beta'] = beta
    df['gamma'] = np.log(df[['high']].rolling(2).max()['high'] /
                        df[['low']].rolling(2).min()['low'])**2

    # CS formula
    alpha = (np.sqrt(2 * df['beta']) - np.sqrt(df['beta'])) / (3 - 2 * np.sqrt(2))
    alpha = alpha - np.sqrt(df['gamma'] / (3 - 2 * np.sqrt(2)))
    df['cs_spread'] = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    df['cs_spread'] = df['cs_spread'].clip(lower=0)  # Can't be negative

    return df


def compute_uncertainty_proxy(df, df_sentiment):
    """
    Compute total uncertainty proxy for ETH.

    Since we don't have ETH-specific DVOL, we use:
    - Parkinson volatility (35%)
    - Realized volatility (25%)
    - CS spread as proxy for information asymmetry (20%)
    - Sentiment-based uncertainty (20%) - higher for extreme regimes
    """
    df = df.copy()

    # Normalize components
    df['parkinson_norm'] = (df['parkinson_vol'] - df['parkinson_vol'].mean()) / df['parkinson_vol'].std()
    df['realized_norm'] = (df['realized_vol'] - df['realized_vol'].mean()) / df['realized_vol'].std()
    df['cs_norm'] = (df['cs_spread'] - df['cs_spread'].mean()) / df['cs_spread'].std()

    # Merge with sentiment
    df = pd.merge(df, df_sentiment[['date', 'regime', 'fear_greed_value']],
                  on='date', how='inner')

    # Sentiment-based uncertainty: higher for extremes
    df['is_extreme'] = df['regime'].isin(['extreme_greed', 'extreme_fear']).astype(int)
    df['sentiment_uncertainty'] = df['is_extreme'] * 0.5  # Crude proxy

    # Weighted sum (normalized to 0-1 range approximately)
    df['total_uncertainty'] = (
        0.35 * df['parkinson_norm'].clip(-2, 2) / 4 + 0.5 +
        0.25 * df['realized_norm'].clip(-2, 2) / 4 + 0.5 +
        0.20 * df['cs_norm'].clip(-2, 2) / 4 + 0.5 +
        0.20 * df['sentiment_uncertainty']
    ).clip(0, 1)

    # Alternative: just use volatility-based uncertainty
    df['vol_uncertainty'] = (df['parkinson_vol'] + df['realized_vol'].fillna(0)) / 2
    df['volatility'] = df['vol_uncertainty']

    return df


def test_extremity_premium_eth(df):
    """
    Test the extremity premium hypothesis on ETH.

    Use CS spread as the uncertainty proxy (independent of volatility),
    then control for volatility in the regression.
    """
    print("\n" + "="*70)
    print("ETH EXTREMITY PREMIUM TEST")
    print("="*70)

    df = df.copy()
    df = df.dropna(subset=['cs_spread', 'parkinson_vol', 'regime'])

    # Use CS spread as uncertainty proxy
    df['uncertainty'] = df['cs_spread']
    df['volatility'] = df['parkinson_vol']

    # Descriptive stats by regime
    print("\nCS Spread (Uncertainty Proxy) by Regime (ETH):")
    for regime in ['extreme_greed', 'greed', 'neutral', 'fear', 'extreme_fear']:
        subset = df[df['regime'] == regime]
        if len(subset) > 0:
            print(f"  {regime:15s}: n={len(subset):3d}, spread={subset['uncertainty'].mean():.4f}, vol={subset['volatility'].mean():.4f}")

    # Create dummies
    for reg in ['extreme_fear', 'fear', 'greed', 'extreme_greed']:
        df[f'is_{reg}'] = (df['regime'] == reg).astype(int)

    # Regression: CS_Spread ~ Volatility + Regime Dummies
    X = df[['volatility', 'is_extreme_fear', 'is_fear', 'is_greed', 'is_extreme_greed']].dropna()
    y = df.loc[X.index, 'uncertainty']
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit(cov_type='HC3')

    print(f"\nRegression: CS_Spread ~ Volatility + Regime Dummies")
    print(f"R² = {model.rsquared:.4f}")
    print(f"Volatility coef: {model.params['volatility']:.4f} (p={model.pvalues['volatility']:.4f})")
    print(f"\nCoefficients (relative to NEUTRAL, controlling for volatility):")

    results = []
    for var in ['is_extreme_greed', 'is_extreme_fear', 'is_fear', 'is_greed']:
        if var in model.params:
            coef = model.params[var]
            pval = model.pvalues[var]
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            regime = var.replace('is_', '')
            print(f"  {regime:15s}: {coef:+.4f} (p={pval:.4f}) {sig}")
            results.append({
                'regime': regime,
                'coefficient': coef,
                'p_value': pval,
                'significant': pval < 0.05
            })

    return pd.DataFrame(results), model


def compare_btc_eth(btc_results, eth_results):
    """Compare the pattern between BTC and ETH."""
    print("\n" + "="*70)
    print("BTC vs ETH COMPARISON")
    print("="*70)

    print("\nRegression coefficients (relative to neutral):")
    print(f"{'Regime':<15} {'BTC':>10} {'ETH':>10}")
    print("-" * 35)

    btc_dict = dict(zip(btc_results['regime'], btc_results['coefficient']))
    eth_dict = dict(zip(eth_results['regime'], eth_results['coefficient']))

    for regime in ['extreme_greed', 'extreme_fear', 'fear', 'greed']:
        btc_val = btc_dict.get(regime, np.nan)
        eth_val = eth_dict.get(regime, np.nan)
        print(f"{regime:<15} {btc_val:>+10.4f} {eth_val:>+10.4f}")

    # Correlation of regime effects
    btc_vals = [btc_dict.get(r, 0) for r in ['extreme_greed', 'extreme_fear', 'fear', 'greed']]
    eth_vals = [eth_dict.get(r, 0) for r in ['extreme_greed', 'extreme_fear', 'fear', 'greed']]

    if not all(np.isnan(eth_vals)):
        corr, p = stats.pearsonr(btc_vals, eth_vals)
        print(f"\nCorrelation of regime effects (BTC vs ETH): r={corr:.3f}, p={p:.4f}")

        if corr > 0.5:
            print("  ✓ PATTERN REPLICATES: ETH shows similar regime-uncertainty pattern as BTC")
    else:
        print("\n  (Insufficient ETH data for correlation)")


def main():
    print("="*70)
    print("ETH CROSS-ASSET VALIDATION")
    print("Testing the Extremity Premium on Ethereum")
    print("="*70)

    # Load BTC sentiment data (for regime labels)
    df_sentiment = pd.read_csv('data/datasets/btc_sentiment_daily.csv', parse_dates=['date'])
    print(f"\nSentiment data: {len(df_sentiment)} days")

    # Fetch ETH data
    try:
        df_eth = fetch_eth_data('2024-01-01', '2026-01-10')
        print(f"ETH OHLCV data: {len(df_eth)} days")
    except Exception as e:
        print(f"Error fetching ETH data: {e}")
        print("Creating synthetic ETH test...")
        # If we can't fetch, use BTC data as a proxy (for testing)
        return None, None

    # Compute spread metrics
    df_eth = compute_spread_metrics(df_eth)

    # Compute uncertainty proxy
    df_eth = compute_uncertainty_proxy(df_eth, df_sentiment)
    df_eth = df_eth.dropna(subset=['volatility', 'regime'])

    print(f"\nMerged ETH data: {len(df_eth)} days")

    # Test extremity premium on ETH
    eth_results, eth_model = test_extremity_premium_eth(df_eth)

    # Load BTC regression results for comparison
    try:
        btc_results = pd.read_csv('results/extremity_premium_regression.csv')
        compare_btc_eth(btc_results, eth_results)
    except FileNotFoundError:
        print("\nBTC results not found. Run extremity_premium_analysis.py first.")

    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    eth_results.to_csv('results/eth_extremity_premium.csv', index=False)
    print("  Saved: results/eth_extremity_premium.csv")

    df_eth.to_csv('results/eth_spread_data.csv', index=False)
    print("  Saved: results/eth_spread_data.csv")

    return eth_results, df_eth


if __name__ == '__main__':
    eth_results, df_eth = main()
