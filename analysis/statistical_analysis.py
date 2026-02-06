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


def compare_configurations(
    single_df: pd.DataFrame,
    multi_df: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Compare single-source vs multi-scale configurations with statistical tests.

    Performs:
    - Welch's t-test for volatility and spread differences
    - Bootstrap confidence intervals
    - Effect size (Cohen's d)

    Args:
        single_df: Single-source simulation results
        multi_df: Multi-scale simulation results
        n_bootstrap: Number of bootstrap samples
        seed: Random seed for reproducibility

    Returns:
        Dictionary with comparison statistics
    """
    np.random.seed(seed)

    # Compute returns
    single_prices = single_df['mid_price'].dropna()
    multi_prices = multi_df['mid_price'].dropna()

    single_returns = np.diff(np.log(single_prices))
    multi_returns = np.diff(np.log(multi_prices))

    # Annualization factor (assuming minute-level data)
    ann_factor = np.sqrt(252 * 24 * 60)

    # Volatility comparison
    single_vol = np.std(single_returns) * ann_factor
    multi_vol = np.std(multi_returns) * ann_factor

    # Welch's t-test on returns (testing if distributions differ)
    t_returns, p_returns = scipy_stats.ttest_ind(single_returns, multi_returns, equal_var=False)

    # Spread comparison
    single_spreads = single_df['spread_bps'].dropna()
    multi_spreads = multi_df['spread_bps'].dropna()

    t_spread, p_spread = scipy_stats.ttest_ind(single_spreads, multi_spreads, equal_var=False)

    # Effect size (Cohen's d) for spreads
    pooled_std = np.sqrt((single_spreads.std()**2 + multi_spreads.std()**2) / 2)
    cohens_d = (single_spreads.mean() - multi_spreads.mean()) / pooled_std

    # Bootstrap CI for volatility difference
    def bootstrap_vol_diff(single_ret, multi_ret, n_boot):
        diffs = []
        n_single, n_multi = len(single_ret), len(multi_ret)
        for _ in range(n_boot):
            boot_single = np.random.choice(single_ret, n_single, replace=True)
            boot_multi = np.random.choice(multi_ret, n_multi, replace=True)
            diff = np.std(boot_single) * ann_factor - np.std(boot_multi) * ann_factor
            diffs.append(diff)
        return np.percentile(diffs, [2.5, 97.5])

    vol_diff_ci = bootstrap_vol_diff(single_returns, multi_returns, n_bootstrap)

    # Bootstrap CI for spread difference
    def bootstrap_mean_diff(a, b, n_boot):
        diffs = []
        for _ in range(n_boot):
            boot_a = np.random.choice(a, len(a), replace=True)
            boot_b = np.random.choice(b, len(b), replace=True)
            diffs.append(boot_a.mean() - boot_b.mean())
        return np.percentile(diffs, [2.5, 97.5])

    spread_diff_ci = bootstrap_mean_diff(single_spreads.values, multi_spreads.values, n_bootstrap)

    # Mann-Whitney U test (non-parametric alternative)
    u_spread, p_mann = scipy_stats.mannwhitneyu(single_spreads, multi_spreads, alternative='two-sided')

    return {
        'volatility': {
            'single_source': single_vol,
            'multi_scale': multi_vol,
            'difference': single_vol - multi_vol,
            'reduction_pct': (single_vol - multi_vol) / single_vol * 100,
            'bootstrap_ci_95': vol_diff_ci.tolist(),
            't_statistic': t_returns,
            'p_value': p_returns,
        },
        'spread': {
            'single_source_mean': single_spreads.mean(),
            'multi_scale_mean': multi_spreads.mean(),
            'difference': single_spreads.mean() - multi_spreads.mean(),
            'reduction_pct': (single_spreads.mean() - multi_spreads.mean()) / single_spreads.mean() * 100,
            'cohens_d': cohens_d,
            'bootstrap_ci_95': spread_diff_ci.tolist(),
            't_statistic': t_spread,
            'p_value': p_spread,
            'mann_whitney_u': u_spread,
            'mann_whitney_p': p_mann,
        },
        'sample_sizes': {
            'single_source': len(single_df),
            'multi_scale': len(multi_df),
        }
    }


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
