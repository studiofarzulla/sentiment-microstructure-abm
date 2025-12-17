"""
Statistical analysis functions for Sentiment-Microstructure ABM.

Provides distributional analysis, time-series diagnostics, and
statistical tests for validating stylized facts.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any
from scipy import stats as scipy_stats


def compute_return_statistics(df: pd.DataFrame, return_col: str = 'log_return') -> Dict[str, float]:
    """
    Compute comprehensive return distribution statistics.

    Args:
        df: DataFrame with return data
        return_col: Name of the return column

    Returns:
        Dictionary with distribution statistics
    """
    returns = df[return_col].dropna()
    # Skip first observation if it's 0 (initialization)
    if returns.iloc[0] == 0:
        returns = returns.iloc[1:]

    return {
        'n_obs': len(returns),
        'mean': returns.mean(),
        'std': returns.std(),
        'min': returns.min(),
        'max': returns.max(),
        'median': returns.median(),
        'skewness': scipy_stats.skew(returns),
        'kurtosis': scipy_stats.kurtosis(returns),  # Excess kurtosis
        'iqr': returns.quantile(0.75) - returns.quantile(0.25),
        'q01': returns.quantile(0.01),
        'q05': returns.quantile(0.05),
        'q95': returns.quantile(0.95),
        'q99': returns.quantile(0.99),
    }


def jarque_bera_test(series: pd.Series) -> Dict[str, float]:
    """
    Jarque-Bera test for normality.

    Args:
        series: Time series data

    Returns:
        Dictionary with test statistic and p-value
    """
    series = series.dropna()
    if series.iloc[0] == 0:
        series = series.iloc[1:]

    stat, pval = scipy_stats.jarque_bera(series)
    return {
        'statistic': stat,
        'p_value': pval,
        'is_normal': pval > 0.05,
    }


def adf_test(series: pd.Series, maxlag: int = None, regression: str = 'c') -> Dict[str, Any]:
    """
    Augmented Dickey-Fuller test for stationarity.

    Args:
        series: Time series data
        maxlag: Maximum lag for ADF test (None = auto)
        regression: Type of regression ('c', 'ct', 'ctt', 'n')

    Returns:
        Dictionary with ADF test results
    """
    from statsmodels.tsa.stattools import adfuller

    series = series.dropna()
    result = adfuller(series, maxlag=maxlag, regression=regression, autolag='AIC')

    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'used_lag': result[2],
        'n_obs': result[3],
        'critical_values': result[4],
        'is_stationary': result[1] < 0.05,
    }


def kpss_test(series: pd.Series, regression: str = 'c', nlags: str = 'auto') -> Dict[str, Any]:
    """
    KPSS test for stationarity.

    Null hypothesis: Series is stationary.
    (Opposite of ADF - want p > 0.05 for stationarity)

    Args:
        series: Time series data
        regression: Type of regression ('c' or 'ct')
        nlags: Number of lags ('auto' or int)

    Returns:
        Dictionary with KPSS test results
    """
    from statsmodels.tsa.stattools import kpss

    series = series.dropna()
    stat, pval, lags, crit = kpss(series, regression=regression, nlags=nlags)

    return {
        'kpss_statistic': stat,
        'p_value': pval,
        'used_lags': lags,
        'critical_values': crit,
        'is_stationary': pval > 0.05,  # Fail to reject null = stationary
    }


def compute_acf(
    series: pd.Series,
    nlags: int = 50,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute autocorrelation function with confidence bounds.

    Args:
        series: Time series data
        nlags: Number of lags to compute
        alpha: Significance level for confidence bounds

    Returns:
        (acf_values, confidence_bounds)
    """
    from statsmodels.tsa.stattools import acf

    series = series.dropna()
    if series.iloc[0] == 0:
        series = series.iloc[1:]

    acf_vals, confint = acf(series, nlags=nlags, alpha=alpha, fft=True)

    return acf_vals, confint


def ljung_box_test(
    series: pd.Series,
    lags: List[int] = [5, 10, 20, 50]
) -> pd.DataFrame:
    """
    Ljung-Box test for autocorrelation.

    Null hypothesis: No autocorrelation up to lag k.

    Args:
        series: Time series data
        lags: List of lag values to test

    Returns:
        DataFrame with Q-statistics and p-values for each lag
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    series = series.dropna()
    if len(series) > 0 and series.iloc[0] == 0:
        series = series.iloc[1:]

    results = []
    for lag in lags:
        if lag < len(series):
            lb_result = acorr_ljungbox(series, lags=[lag], return_df=True)
            results.append({
                'lag': lag,
                'lb_stat': lb_result['lb_stat'].values[0],
                'lb_pvalue': lb_result['lb_pvalue'].values[0],
                'has_autocorr': lb_result['lb_pvalue'].values[0] < 0.05,
            })

    return pd.DataFrame(results)


def volatility_clustering_test(df: pd.DataFrame, return_col: str = 'log_return') -> Dict[str, float]:
    """
    Test for volatility clustering by comparing ACF of returns vs |returns|.

    Volatility clustering is indicated by:
    - Low ACF of returns (no autocorrelation in returns)
    - High ACF of |returns| (persistence in volatility)

    Args:
        df: DataFrame with return data
        return_col: Name of return column

    Returns:
        Dictionary with volatility clustering metrics
    """
    returns = df[return_col].dropna()
    if returns.iloc[0] == 0:
        returns = returns.iloc[1:]

    abs_returns = returns.abs()
    squared_returns = returns ** 2

    # Compute ACF at key lags
    acf_returns, _ = compute_acf(returns, nlags=20)
    acf_abs_returns, _ = compute_acf(abs_returns, nlags=20)
    acf_sq_returns, _ = compute_acf(squared_returns, nlags=20)

    return {
        'acf_returns_lag1': acf_returns[1],
        'acf_returns_lag5': acf_returns[5],
        'acf_returns_lag10': acf_returns[10],
        'acf_abs_returns_lag1': acf_abs_returns[1],
        'acf_abs_returns_lag5': acf_abs_returns[5],
        'acf_abs_returns_lag10': acf_abs_returns[10],
        'acf_sq_returns_lag1': acf_sq_returns[1],
        'acf_sq_returns_lag5': acf_sq_returns[5],
        'acf_sq_returns_lag10': acf_sq_returns[10],
        # Volatility clustering = |returns| ACF much higher than returns ACF
        'clustering_ratio_lag10': acf_abs_returns[10] / (abs(acf_returns[10]) + 1e-10),
        'has_volatility_clustering': acf_abs_returns[10] > 0.1 and abs(acf_returns[10]) < 0.1,
    }


def compute_correlation_matrix(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """
    Compute correlation matrix for specified columns.

    Args:
        df: DataFrame
        columns: List of column names (None = all numeric)

    Returns:
        Correlation matrix as DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    return df[columns].corr()


def run_all_diagnostics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run all statistical diagnostics on simulation data.

    Args:
        df: Simulation DataFrame with required columns

    Returns:
        Dictionary with all diagnostic results
    """
    results = {}

    # Return statistics
    results['return_stats'] = compute_return_statistics(df)

    # Normality test
    results['jarque_bera'] = jarque_bera_test(df['log_return'])

    # Stationarity tests
    results['adf_spread'] = adf_test(df['spread_bps'])
    results['adf_returns'] = adf_test(df['log_return'].iloc[1:])
    results['kpss_spread'] = kpss_test(df['spread_bps'])

    # Autocorrelation tests
    results['ljung_box_returns'] = ljung_box_test(df['log_return'])
    results['ljung_box_abs_returns'] = ljung_box_test(df['log_return'].abs())

    # Volatility clustering
    results['volatility_clustering'] = volatility_clustering_test(df)

    # Correlation matrix
    numeric_cols = ['sentiment', 'epistemic_uncertainty', 'aleatoric_uncertainty',
                    'total_uncertainty', 'spread_bps', 'log_return', 'inventory']
    available_cols = [c for c in numeric_cols if c in df.columns]
    results['correlation_matrix'] = compute_correlation_matrix(df, available_cols)

    return results
