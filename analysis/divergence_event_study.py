#!/usr/bin/env python3
"""
Divergence Event Study: Empirical Test of the Divergence Hypothesis

This module tests the paper's central hypothesis that retail vs institutional
sentiment divergence has predictive power for future returns and volatility.

The divergence hypothesis (Section 3.2):
    "When retail sentiment (Fear & Greed) diverges significantly from
     institutional sentiment (ASRI-derived), this predicts elevated
     volatility and potential mean reversion as signals converge."

Methods:
    - Event identification: |retail - institutional| > threshold
    - Forward metrics: 1d, 3d, 5d, 7d returns and realized volatility
    - Statistical tests: t-test, Newey-West HAC, bootstrap confidence intervals
    - Predictive power: Who was "right" - retail or institutional?

Author: Farzulla Research
Date: January 2026
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Farzulla Research color scheme
FARZULLA_BURGUNDY = '#800020'
FARZULLA_BLUE = '#3498db'
FARZULLA_GREEN = '#2ecc71'
FARZULLA_PURPLE = '#9b59b6'
FARZULLA_RED = '#e74c3c'
FARZULLA_GRAY = '#95a5a6'
FARZULLA_GOLD = '#f39c12'


@dataclass
class DivergenceEvent:
    """Single divergence event with forward-looking metrics."""

    date: datetime
    divergence: float           # retail - institutional (scaled to [-1, 1])
    retail_sentiment: float     # Fear & Greed scaled to [-1, 1]
    institutional_sentiment: float  # Macro sentiment [-1, 1]
    regime: str                 # Market regime at event time

    # Forward returns (computed after event)
    return_1d: Optional[float] = None
    return_3d: Optional[float] = None
    return_5d: Optional[float] = None
    return_7d: Optional[float] = None

    # Forward volatility (realized vol over window)
    volatility_forward: Optional[float] = None

    # Who was right?
    retail_correct: Optional[bool] = None
    institutional_correct: Optional[bool] = None

    def to_dict(self) -> dict:
        return {
            'date': self.date,
            'divergence': self.divergence,
            'retail_sentiment': self.retail_sentiment,
            'institutional_sentiment': self.institutional_sentiment,
            'regime': self.regime,
            'return_1d': self.return_1d,
            'return_3d': self.return_3d,
            'return_5d': self.return_5d,
            'return_7d': self.return_7d,
            'volatility_forward': self.volatility_forward,
            'retail_correct': self.retail_correct,
            'institutional_correct': self.institutional_correct,
        }


@dataclass
class EventStudyResults:
    """Results from divergence event study."""

    threshold: float
    n_events: int
    n_retail_bullish: int   # Events where retail > institutional
    n_inst_bullish: int     # Events where institutional > retail

    # Average forward returns
    mean_return_1d: float
    mean_return_3d: float
    mean_return_5d: float
    mean_return_7d: float

    # Standard errors (Newey-West adjusted)
    se_return_1d: float
    se_return_3d: float
    se_return_5d: float
    se_return_7d: float

    # t-statistics
    t_stat_1d: float
    t_stat_3d: float
    t_stat_5d: float
    t_stat_7d: float

    # p-values
    p_value_1d: float
    p_value_3d: float
    p_value_5d: float
    p_value_7d: float

    # Volatility metrics
    mean_volatility: float
    median_volatility: float
    unconditional_volatility: float  # For comparison

    # Who was right?
    retail_correct_pct: float
    inst_correct_pct: float

    # Bootstrap confidence intervals
    ci_return_1d: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    ci_return_7d: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))

    events: List[DivergenceEvent] = field(default_factory=list)


def load_sentiment_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load and preprocess BTC sentiment data.

    The data contains:
        - fear_greed_value: 0-100 (retail sentiment proxy)
        - volatility: realized volatility (institutional risk proxy)
        - price_sentiment_corr: correlation between price and sentiment
        - price data for returns calculation

    For the institutional sentiment proxy, we construct it from:
        1. Volatility-implied sentiment: High vol = bearish (risk-off)
        2. Price-sentiment correlation: Negative = retail "wrong"
        3. Sentiment momentum: Lagged, slower-moving signal

    Args:
        data_path: Path to CSV file. If None, uses default location.

    Returns:
        Preprocessed DataFrame with scaled sentiment columns.
    """
    if data_path is None:
        data_path = project_root / 'data' / 'datasets' / 'btc_sentiment_daily.csv'

    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()

    # Scale Fear & Greed (0-100) to [-1, 1]
    # 50 = neutral (0), 0 = extreme fear (-1), 100 = extreme greed (+1)
    df['retail_sentiment'] = (df['fear_greed_value'] - 50) / 50

    # =====================================================
    # CONSTRUCT INSTITUTIONAL SENTIMENT PROXY
    # =====================================================
    # The paper's hypothesis is that institutional sentiment diverges from retail.
    # We construct an institutional proxy using:
    #
    # 1. Volatility-implied sentiment:
    #    - High volatility suggests institutional risk-off positioning
    #    - Scale: vol in [0.005, 0.07] -> sentiment in [1, -1]
    #    - Higher vol = more bearish
    #
    # 2. Price-sentiment correlation adjustment:
    #    - When correlation is highly negative, retail is "wrong"
    #    - Institutional smart money would be contrarian
    #
    # 3. Smoothed/lagged retail sentiment:
    #    - Institutions move slower, less reactive to daily noise

    # Component 1: Volatility-implied sentiment
    # Normalize volatility to [0, 1] range (historical range ~0.005-0.07)
    vol_min, vol_max = 0.005, 0.065
    vol_normalized = df['volatility'].clip(vol_min, vol_max)
    vol_normalized = (vol_normalized - vol_min) / (vol_max - vol_min)
    # High vol = bearish, so invert: vol_sentiment in [-1, 1]
    vol_sentiment = 1 - 2 * vol_normalized
    vol_sentiment = vol_sentiment.fillna(0)

    # Component 2: Contrarian adjustment from price-sentiment correlation
    # When correlation is very negative (<-0.5), retail is "wrong"
    # Institutional would be opposite to retail in these cases
    psc = df['price_sentiment_corr'].fillna(0)
    # Create contrarian weight: high when correlation strongly negative
    contrarian_weight = (-psc).clip(0, 1)  # 0 when corr >= 0, up to 1 when corr = -1

    # Component 3: Smoothed retail sentiment (7-day EMA = slower institutional)
    retail_smooth = df['retail_sentiment'].ewm(span=7, adjust=False).mean()

    # Blend components for institutional sentiment
    # Base: smoothed retail (institutions track same fundamentals)
    # Adjustment: volatility signal (risk management)
    # Contrarian: when retail is clearly wrong, flip
    inst_base = 0.5 * retail_smooth + 0.5 * vol_sentiment

    # Apply contrarian adjustment
    contrarian_adjustment = contrarian_weight * (-df['retail_sentiment'] - inst_base)
    df['institutional_sentiment'] = (inst_base + 0.3 * contrarian_adjustment).clip(-1, 1)

    # Fill any remaining NaN with 0
    df['institutional_sentiment'] = df['institutional_sentiment'].fillna(0)

    # Compute divergence: retail - institutional
    # Positive = retail more bullish than institutional
    # Negative = institutional more bullish than retail
    df['divergence'] = df['retail_sentiment'] - df['institutional_sentiment']

    # Ensure we have returns (some rows may be NaN for first days)
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Compute forward returns for event study
    for horizon in [1, 3, 5, 7]:
        df[f'return_{horizon}d'] = df['close'].shift(-horizon) / df['close'] - 1
        df[f'log_return_{horizon}d'] = np.log(df['close'].shift(-horizon) / df['close'])

    # Forward realized volatility (5-day window)
    df['volatility_5d_forward'] = df['log_returns'].shift(-1).rolling(window=5).std() * np.sqrt(252)

    print(f"Loaded {len(df)} days of sentiment data")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Retail sentiment range: [{df['retail_sentiment'].min():.3f}, {df['retail_sentiment'].max():.3f}]")
    print(f"Institutional sentiment range: [{df['institutional_sentiment'].min():.3f}, {df['institutional_sentiment'].max():.3f}]")
    print(f"Divergence range: [{df['divergence'].min():.3f}, {df['divergence'].max():.3f}]")
    print(f"Divergence > 0.3: {(abs(df['divergence']) > 0.3).sum()} events")
    print(f"Divergence > 0.4: {(abs(df['divergence']) > 0.4).sum()} events")
    print(f"Divergence > 0.5: {(abs(df['divergence']) > 0.5).sum()} events")

    return df


def identify_divergence_events(
    df: pd.DataFrame,
    threshold: float = 0.4,
    min_gap: int = 5,  # Minimum days between events (avoid clustering)
) -> List[DivergenceEvent]:
    """
    Identify divergence events where |retail - institutional| > threshold.

    Args:
        df: DataFrame with sentiment data
        threshold: Absolute divergence threshold
        min_gap: Minimum days between events to avoid clustering

    Returns:
        List of DivergenceEvent objects with forward metrics filled.
    """
    events = []
    last_event_idx = -min_gap  # Allow first event

    for i, (date, row) in enumerate(df.iterrows()):
        # Skip if divergence doesn't exceed threshold
        if abs(row['divergence']) < threshold:
            continue

        # Skip if too close to previous event
        if i - last_event_idx < min_gap:
            continue

        # Skip if we don't have forward data
        if pd.isna(row.get('return_7d', np.nan)):
            continue

        # Determine who was "correct" based on 7-day forward return
        forward_return = row['return_7d']
        retail_bullish = row['retail_sentiment'] > 0
        inst_bullish = row['institutional_sentiment'] > 0
        market_went_up = forward_return > 0

        event = DivergenceEvent(
            date=date,
            divergence=row['divergence'],
            retail_sentiment=row['retail_sentiment'],
            institutional_sentiment=row['institutional_sentiment'],
            regime=row.get('regime', 'unknown'),
            return_1d=row.get('return_1d', np.nan),
            return_3d=row.get('return_3d', np.nan),
            return_5d=row.get('return_5d', np.nan),
            return_7d=row.get('return_7d', np.nan),
            volatility_forward=row.get('volatility_5d_forward', np.nan),
            retail_correct=(retail_bullish == market_went_up),
            institutional_correct=(inst_bullish == market_went_up),
        )

        events.append(event)
        last_event_idx = i

    return events


def newey_west_se(returns: np.ndarray, lags: int = None) -> float:
    """
    Compute Newey-West HAC standard error for autocorrelation-robust inference.

    Args:
        returns: Array of returns
        lags: Number of lags for HAC (default: auto based on sample size)

    Returns:
        Newey-West adjusted standard error.
    """
    n = len(returns)
    if lags is None:
        # Rule of thumb: floor(4 * (n/100)^(2/9))
        lags = int(np.floor(4 * (n / 100) ** (2/9)))

    mean_return = np.mean(returns)
    demean = returns - mean_return

    # Variance term
    var = np.var(demean, ddof=1)

    # Autocovariance terms with Bartlett weights
    autocov_sum = 0
    for j in range(1, lags + 1):
        weight = 1 - j / (lags + 1)  # Bartlett kernel
        autocov = np.sum(demean[j:] * demean[:-j]) / n
        autocov_sum += 2 * weight * autocov

    # HAC variance
    hac_var = var + autocov_sum

    # Standard error of mean
    se = np.sqrt(hac_var / n)

    return se


def bootstrap_ci(
    returns: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for mean return.

    Args:
        returns: Array of returns
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level (0.05 = 95% CI)
        seed: Random seed

    Returns:
        (lower, upper) bounds of CI.
    """
    np.random.seed(seed)
    n = len(returns)

    boot_means = []
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(returns, size=n, replace=True)
        boot_means.append(np.mean(boot_sample))

    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return (lower, upper)


def run_event_study(
    df: pd.DataFrame,
    threshold: float = 0.4,
    n_bootstrap: int = 10000
) -> EventStudyResults:
    """
    Run complete event study for given divergence threshold.

    Args:
        df: DataFrame with sentiment and return data
        threshold: Divergence threshold for event identification
        n_bootstrap: Number of bootstrap samples for CI

    Returns:
        EventStudyResults with all statistical metrics.
    """
    events = identify_divergence_events(df, threshold=threshold)

    if len(events) < 5:
        print(f"Warning: Only {len(events)} events at threshold {threshold}")
        return None

    # Extract return arrays
    returns_1d = np.array([e.return_1d for e in events if not pd.isna(e.return_1d)])
    returns_3d = np.array([e.return_3d for e in events if not pd.isna(e.return_3d)])
    returns_5d = np.array([e.return_5d for e in events if not pd.isna(e.return_5d)])
    returns_7d = np.array([e.return_7d for e in events if not pd.isna(e.return_7d)])
    volatilities = np.array([e.volatility_forward for e in events if not pd.isna(e.volatility_forward)])

    # Unconditional metrics for comparison (annualized)
    # The 'volatility' column in the file is daily vol, so annualize it
    if 'volatility' in df.columns:
        unconditional_vol = df['volatility'].mean() * np.sqrt(252)  # Annualize daily vol
    else:
        unconditional_vol = df['log_returns'].std() * np.sqrt(252)

    # Mean returns
    mean_1d = np.mean(returns_1d)
    mean_3d = np.mean(returns_3d)
    mean_5d = np.mean(returns_5d)
    mean_7d = np.mean(returns_7d)

    # Newey-West standard errors
    se_1d = newey_west_se(returns_1d)
    se_3d = newey_west_se(returns_3d)
    se_5d = newey_west_se(returns_5d)
    se_7d = newey_west_se(returns_7d)

    # t-statistics (testing H0: mean = 0)
    t_1d = mean_1d / se_1d if se_1d > 0 else 0
    t_3d = mean_3d / se_3d if se_3d > 0 else 0
    t_5d = mean_5d / se_5d if se_5d > 0 else 0
    t_7d = mean_7d / se_7d if se_7d > 0 else 0

    # Two-sided p-values
    p_1d = 2 * (1 - scipy_stats.t.cdf(abs(t_1d), df=len(returns_1d) - 1))
    p_3d = 2 * (1 - scipy_stats.t.cdf(abs(t_3d), df=len(returns_3d) - 1))
    p_5d = 2 * (1 - scipy_stats.t.cdf(abs(t_5d), df=len(returns_5d) - 1))
    p_7d = 2 * (1 - scipy_stats.t.cdf(abs(t_7d), df=len(returns_7d) - 1))

    # Bootstrap CIs
    ci_1d = bootstrap_ci(returns_1d, n_bootstrap)
    ci_7d = bootstrap_ci(returns_7d, n_bootstrap)

    # Direction analysis
    n_retail_bullish = sum(1 for e in events if e.divergence > 0)
    n_inst_bullish = len(events) - n_retail_bullish

    # Who was right?
    retail_correct = [e.retail_correct for e in events if e.retail_correct is not None]
    inst_correct = [e.institutional_correct for e in events if e.institutional_correct is not None]

    retail_correct_pct = 100 * sum(retail_correct) / len(retail_correct) if retail_correct else 0
    inst_correct_pct = 100 * sum(inst_correct) / len(inst_correct) if inst_correct else 0

    return EventStudyResults(
        threshold=threshold,
        n_events=len(events),
        n_retail_bullish=n_retail_bullish,
        n_inst_bullish=n_inst_bullish,
        mean_return_1d=mean_1d,
        mean_return_3d=mean_3d,
        mean_return_5d=mean_5d,
        mean_return_7d=mean_7d,
        se_return_1d=se_1d,
        se_return_3d=se_3d,
        se_return_5d=se_5d,
        se_return_7d=se_7d,
        t_stat_1d=t_1d,
        t_stat_3d=t_3d,
        t_stat_5d=t_5d,
        t_stat_7d=t_7d,
        p_value_1d=p_1d,
        p_value_3d=p_3d,
        p_value_5d=p_5d,
        p_value_7d=p_7d,
        mean_volatility=np.mean(volatilities),
        median_volatility=np.median(volatilities),
        unconditional_volatility=unconditional_vol,
        retail_correct_pct=retail_correct_pct,
        inst_correct_pct=inst_correct_pct,
        ci_return_1d=ci_1d,
        ci_return_7d=ci_7d,
        events=events,
    )


def run_directional_analysis(
    df: pd.DataFrame,
    threshold: float = 0.4
) -> Dict[str, Any]:
    """
    Analyze whether direction of divergence matters.

    Split events into:
        - Retail bullish (divergence > 0): Retail more optimistic
        - Institutional bullish (divergence < 0): Institutional more optimistic

    Args:
        df: DataFrame with sentiment data
        threshold: Divergence threshold

    Returns:
        Dictionary with directional analysis results.
    """
    events = identify_divergence_events(df, threshold=threshold)

    retail_bullish = [e for e in events if e.divergence > 0]
    inst_bullish = [e for e in events if e.divergence < 0]

    results = {}

    for name, event_list in [('retail_bullish', retail_bullish), ('inst_bullish', inst_bullish)]:
        if len(event_list) < 3:
            results[name] = {'n': len(event_list), 'insufficient_data': True}
            continue

        returns_7d = np.array([e.return_7d for e in event_list if not pd.isna(e.return_7d)])

        results[name] = {
            'n': len(event_list),
            'mean_return_7d': np.mean(returns_7d) * 100,  # Percentage
            'std_return_7d': np.std(returns_7d) * 100,
            't_stat': np.mean(returns_7d) / (np.std(returns_7d) / np.sqrt(len(returns_7d))),
            'positive_pct': 100 * sum(r > 0 for r in returns_7d) / len(returns_7d),
        }

    # Test if difference between groups is significant
    if len(retail_bullish) >= 3 and len(inst_bullish) >= 3:
        rb_returns = [e.return_7d for e in retail_bullish if not pd.isna(e.return_7d)]
        ib_returns = [e.return_7d for e in inst_bullish if not pd.isna(e.return_7d)]

        t_diff, p_diff = scipy_stats.ttest_ind(rb_returns, ib_returns, equal_var=False)
        results['difference_test'] = {
            't_statistic': t_diff,
            'p_value': p_diff,
            'significant': p_diff < 0.05,
        }

    return results


def generate_event_study_figure(
    df: pd.DataFrame,
    results_by_threshold: Dict[float, EventStudyResults],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate publication-quality event study figure.

    Panel A: Cumulative returns around divergence events
    Panel B: Forward returns by horizon
    Panel C: Volatility comparison
    Panel D: Who was right?

    Args:
        df: Original data
        results_by_threshold: Results for multiple thresholds
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    # Set paper style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Use 0.4 threshold as primary
    primary_threshold = 0.4
    if not results_by_threshold:
        print("Warning: No results to plot")
        return fig

    if primary_threshold in results_by_threshold:
        primary_results = results_by_threshold[primary_threshold]
    else:
        primary_results = list(results_by_threshold.values())[0]

    # Panel A: Cumulative average returns (event study style)
    ax1 = axes[0, 0]

    # Compute CAR for all events
    events = primary_results.events
    event_windows = []

    for event in events:
        date = event.date
        if date not in df.index:
            continue
        idx = df.index.get_loc(date)

        # Get returns from -5 to +10 days
        window_start = max(0, idx - 5)
        window_end = min(len(df) - 1, idx + 10)

        window_returns = df.iloc[window_start:window_end + 1]['log_returns'].values
        # Pad to consistent length
        if len(window_returns) < 16:
            window_returns = np.pad(window_returns, (0, 16 - len(window_returns)),
                                    mode='constant', constant_values=0)
        event_windows.append(window_returns[:16])

    if event_windows:
        event_returns = np.array(event_windows)
        mean_returns = np.nanmean(event_returns, axis=0)
        std_returns = np.nanstd(event_returns, axis=0) / np.sqrt(len(event_windows))

        car = np.cumsum(mean_returns) * 100  # Percentage
        car_se = np.sqrt(np.cumsum(std_returns**2)) * 100

        days = np.arange(-5, 11)
        ax1.fill_between(days, car - 1.96*car_se, car + 1.96*car_se,
                        color=FARZULLA_BURGUNDY, alpha=0.2, label='95% CI')
        ax1.plot(days, car, color=FARZULLA_BURGUNDY, linewidth=2, marker='o',
                markersize=4, label='CAR')
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax1.axhline(y=0, color='black', linewidth=0.5)

    ax1.set_xlabel('Days Relative to Event')
    ax1.set_ylabel('Cumulative Abnormal Return (%)')
    ax1.set_title(f'A) Event Study: CAR around Divergence Events\n(threshold = {primary_results.threshold}, n = {primary_results.n_events})')
    ax1.legend(loc='upper left')

    # Panel B: Forward returns by horizon across thresholds
    ax2 = axes[0, 1]

    horizons = [1, 3, 5, 7]
    thresholds = sorted(results_by_threshold.keys())
    x = np.arange(len(horizons))
    width = 0.25

    colors = [FARZULLA_BURGUNDY, FARZULLA_BLUE, FARZULLA_GREEN]

    for i, threshold in enumerate(thresholds[:3]):  # Max 3 thresholds
        results = results_by_threshold[threshold]
        means = [results.mean_return_1d * 100, results.mean_return_3d * 100,
                 results.mean_return_5d * 100, results.mean_return_7d * 100]
        errors = [results.se_return_1d * 100, results.se_return_3d * 100,
                  results.se_return_5d * 100, results.se_return_7d * 100]

        bars = ax2.bar(x + i*width, means, width, label=f'|div| > {threshold}',
                       color=colors[i % len(colors)], alpha=0.8)
        ax2.errorbar(x + i*width, means, yerr=1.96*np.array(errors),
                    fmt='none', color='black', capsize=3, capthick=1)

    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Horizon (days)')
    ax2.set_ylabel('Mean Return (%)')
    ax2.set_title('B) Forward Returns by Horizon')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([f'{h}d' for h in horizons])
    ax2.legend(loc='best')

    # Panel C: Volatility during divergence vs normal
    ax3 = axes[1, 0]

    vol_divergence = []
    vol_normal = []
    threshold_labels = []

    for threshold, results in sorted(results_by_threshold.items()):
        vol_divergence.append(results.mean_volatility * 100)  # Annualized %
        vol_normal.append(results.unconditional_volatility * 100)
        threshold_labels.append(f'|div| > {threshold}')

    x = np.arange(len(threshold_labels))
    width = 0.35

    ax3.bar(x - width/2, vol_divergence, width, label='During Divergence',
            color=FARZULLA_BURGUNDY, alpha=0.8)
    ax3.bar(x + width/2, vol_normal, width, label='Unconditional',
            color=FARZULLA_GRAY, alpha=0.8)

    ax3.set_xlabel('Divergence Threshold')
    ax3.set_ylabel('Annualized Volatility (%)')
    ax3.set_title('C) Volatility: Divergence Events vs Unconditional')
    ax3.set_xticks(x)
    ax3.set_xticklabels(threshold_labels)
    ax3.legend(loc='best')

    # Panel D: Who was right? (retail vs institutional)
    ax4 = axes[1, 1]

    retail_pcts = []
    inst_pcts = []
    threshold_labels = []

    for threshold, results in sorted(results_by_threshold.items()):
        retail_pcts.append(results.retail_correct_pct)
        inst_pcts.append(results.inst_correct_pct)
        threshold_labels.append(f'|div| > {threshold}')

    x = np.arange(len(threshold_labels))
    width = 0.35

    ax4.bar(x - width/2, retail_pcts, width, label='Retail Correct',
            color=FARZULLA_BLUE, alpha=0.8)
    ax4.bar(x + width/2, inst_pcts, width, label='Institutional Correct',
            color=FARZULLA_GOLD, alpha=0.8)

    ax4.axhline(y=50, color='black', linestyle='--', linewidth=1, alpha=0.7,
                label='Random (50%)')
    ax4.set_xlabel('Divergence Threshold')
    ax4.set_ylabel('Correct Prediction (%)')
    ax4.set_title('D) Predictive Accuracy: Retail vs Institutional')
    ax4.set_xticks(x)
    ax4.set_xticklabels(threshold_labels)
    ax4.legend(loc='best')
    ax4.set_ylim(0, 100)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    return fig


def generate_latex_table(
    results_by_threshold: Dict[float, EventStudyResults],
    save_path: Optional[str] = None
) -> str:
    """
    Generate LaTeX table for paper inclusion.

    Args:
        results_by_threshold: Results dictionary
        save_path: Path to save .tex file

    Returns:
        LaTeX table string.
    """
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Divergence Event Study: Forward Returns}
\label{tab:divergence_event_study}
\begin{tabular}{lccccc}
\toprule
 & \multicolumn{5}{c}{Forward Return (\%)} \\
\cmidrule(lr){2-6}
Threshold & $N$ & 1-day & 3-day & 5-day & 7-day \\
\midrule
"""

    for threshold, results in sorted(results_by_threshold.items()):
        # Format returns with significance stars
        def format_return(mean, pval):
            stars = ''
            if pval < 0.01:
                stars = '***'
            elif pval < 0.05:
                stars = '**'
            elif pval < 0.10:
                stars = '*'
            return f'{mean*100:.2f}{stars}'

        row = f'$|\\text{{div}}| > {threshold}$ & {results.n_events} & '
        row += f'{format_return(results.mean_return_1d, results.p_value_1d)} & '
        row += f'{format_return(results.mean_return_3d, results.p_value_3d)} & '
        row += f'{format_return(results.mean_return_5d, results.p_value_5d)} & '
        row += f'{format_return(results.mean_return_7d, results.p_value_7d)} \\\\\n'

        # Add t-statistics in parentheses
        row += f' & & ({results.t_stat_1d:.2f}) & ({results.t_stat_3d:.2f}) & '
        row += f'({results.t_stat_5d:.2f}) & ({results.t_stat_7d:.2f}) \\\\\n'

        latex += row

    latex += r"""
\bottomrule
\end{tabular}

\begin{tablenotes}
\small
\item \textit{Notes:} Event study results for sentiment divergence (retail minus institutional).
Events identified where absolute divergence exceeds threshold with minimum 5-day gap between events.
Returns are mean forward returns. $t$-statistics in parentheses use Newey-West standard errors.
***, **, * denote significance at 1\%, 5\%, 10\% levels respectively.
\end{tablenotes}
\end{table}
"""

    if save_path:
        with open(save_path, 'w') as f:
            f.write(latex)
        print(f"Saved LaTeX table to: {save_path}")

    return latex


def generate_volatility_table(
    results_by_threshold: Dict[float, EventStudyResults],
    save_path: Optional[str] = None
) -> str:
    """
    Generate LaTeX table for volatility comparison.
    """
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Volatility During Divergence Events}
\label{tab:divergence_volatility}
\begin{tabular}{lcccc}
\toprule
Threshold & $N$ & Mean Vol & Median Vol & Unconditional Vol \\
\midrule
"""

    for threshold, results in sorted(results_by_threshold.items()):
        row = f'$|\\text{{div}}| > {threshold}$ & {results.n_events} & '
        row += f'{results.mean_volatility*100:.1f}\\% & '
        row += f'{results.median_volatility*100:.1f}\\% & '
        row += f'{results.unconditional_volatility*100:.1f}\\% \\\\\n'
        latex += row

    latex += r"""
\bottomrule
\end{tabular}

\begin{tablenotes}
\small
\item \textit{Notes:} Realized volatility (annualized) following divergence events
compared to unconditional volatility. Vol computed as 5-day forward rolling standard
deviation of log returns.
\end{tablenotes}
\end{table}
"""

    if save_path:
        with open(save_path, 'w') as f:
            f.write(latex)
        print(f"Saved volatility table to: {save_path}")

    return latex


def run_full_event_study(
    data_path: Optional[str] = None,
    thresholds: List[float] = [0.3, 0.4, 0.5],
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run complete divergence event study and generate all outputs.

    Args:
        data_path: Path to sentiment data
        thresholds: List of divergence thresholds to test
        output_dir: Directory for outputs (figures, tables)

    Returns:
        Dictionary with all results.
    """
    print("=" * 70)
    print("DIVERGENCE EVENT STUDY: Empirical Test of Divergence Hypothesis")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now().isoformat()}")

    # Setup output directory
    if output_dir is None:
        output_dir = project_root / 'analysis' / 'results'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n" + "-" * 40)
    print("Loading sentiment data...")
    df = load_sentiment_data(data_path)

    # Run event studies for each threshold
    print("\n" + "-" * 40)
    print("Running event studies...")

    results_by_threshold = {}
    for threshold in thresholds:
        print(f"\nThreshold: {threshold}")
        results = run_event_study(df, threshold=threshold)
        if results is not None:
            results_by_threshold[threshold] = results
            print(f"  Events: {results.n_events}")
            print(f"  Mean 7d return: {results.mean_return_7d*100:.3f}% (p={results.p_value_7d:.3f})")
            print(f"  Retail bullish: {results.n_retail_bullish}, Inst bullish: {results.n_inst_bullish}")
            print(f"  Retail correct: {results.retail_correct_pct:.1f}%, Inst correct: {results.inst_correct_pct:.1f}%")

    # Directional analysis
    print("\n" + "-" * 40)
    print("Directional analysis (threshold=0.4)...")
    directional = run_directional_analysis(df, threshold=0.4)

    if 'retail_bullish' in directional and not directional['retail_bullish'].get('insufficient_data'):
        print(f"  When retail more bullish (n={directional['retail_bullish']['n']}):")
        print(f"    Mean 7d return: {directional['retail_bullish']['mean_return_7d']:.2f}%")
        print(f"    Positive outcome: {directional['retail_bullish']['positive_pct']:.1f}%")

    if 'inst_bullish' in directional and not directional['inst_bullish'].get('insufficient_data'):
        print(f"  When institutional more bullish (n={directional['inst_bullish']['n']}):")
        print(f"    Mean 7d return: {directional['inst_bullish']['mean_return_7d']:.2f}%")
        print(f"    Positive outcome: {directional['inst_bullish']['positive_pct']:.1f}%")

    if 'difference_test' in directional:
        print(f"  Difference test: t={directional['difference_test']['t_statistic']:.2f}, "
              f"p={directional['difference_test']['p_value']:.3f}")

    # Generate outputs
    print("\n" + "-" * 40)
    print("Generating outputs...")

    # Figure
    fig_path = output_dir / 'event_study_divergence.pdf'
    generate_event_study_figure(df, results_by_threshold, save_path=str(fig_path))

    # LaTeX tables
    tables_dir = project_root / 'paper' / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)

    generate_latex_table(results_by_threshold,
                        save_path=str(tables_dir / 'table_divergence_event_study.tex'))
    generate_volatility_table(results_by_threshold,
                             save_path=str(tables_dir / 'table_divergence_volatility.tex'))

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Key finding for paper
    primary = results_by_threshold.get(0.4)
    if primary:
        print(f"\nPrimary Results (threshold = 0.4, n = {primary.n_events} events):")
        print(f"  1-day forward return: {primary.mean_return_1d*100:.3f}% (t={primary.t_stat_1d:.2f})")
        print(f"  7-day forward return: {primary.mean_return_7d*100:.3f}% (t={primary.t_stat_7d:.2f})")
        print(f"  Mean volatility during events: {primary.mean_volatility*100:.1f}%")
        print(f"  Unconditional volatility: {primary.unconditional_volatility*100:.1f}%")
        print(f"  Volatility ratio: {primary.mean_volatility/primary.unconditional_volatility:.2f}x")

        # Interpret results
        print("\n  Interpretation:")
        if primary.p_value_7d < 0.05:
            direction = "positive" if primary.mean_return_7d > 0 else "negative"
            print(f"    - Statistically significant {direction} abnormal returns following divergence")
        else:
            print(f"    - No statistically significant abnormal returns (p > 0.05)")

        if primary.mean_volatility > primary.unconditional_volatility * 1.1:
            print(f"    - Elevated volatility during divergence periods (supports hypothesis)")
        else:
            print(f"    - Volatility not significantly elevated during divergence")

        if primary.retail_correct_pct > 55 or primary.inst_correct_pct > 55:
            winner = "Retail" if primary.retail_correct_pct > primary.inst_correct_pct else "Institutional"
            print(f"    - {winner} sentiment shows better predictive accuracy")
        else:
            print(f"    - Neither sentiment source shows clear predictive advantage")

    print("\n" + "=" * 70)
    print("Analysis complete. Outputs saved to:")
    print(f"  Figure: {fig_path}")
    print(f"  Tables: {tables_dir}")
    print("=" * 70)

    return {
        'data': df,
        'results_by_threshold': results_by_threshold,
        'directional_analysis': directional,
    }


if __name__ == '__main__':
    results = run_full_event_study()
