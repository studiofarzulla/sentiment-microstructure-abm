"""
Robustness Tests for Contrarian Signal Finding

Addresses peer review concerns:
1. Statistical tests (t-tests, bootstrap CIs, Newey-West SEs)
2. Out-of-sample validation (train 2024, test 2025-2026)
3. Ablation studies (macro-only, micro-only, fixed weights)
4. Backtest with transaction costs
5. Subperiod analysis

Author: Murad Farzulla
Date: January 2026
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_1samp, ttest_ind
import warnings
warnings.filterwarnings('ignore')
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Load Data
# ============================================================================

def load_data():
    """Load the sentiment-price dataset."""
    data_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'datasets', 'btc_sentiment_daily.csv'
    )
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Ensure we have next-day returns for predictive analysis
    df['next_return'] = df['returns'].shift(-1)
    
    logger.info(f"Loaded {len(df)} days of data")
    return df


# ============================================================================
# 1. Statistical Tests for Contrarian Signal
# ============================================================================

def statistical_tests(df):
    """
    Comprehensive statistical tests for regime-return relationship.
    
    Returns dict with all test results.
    """
    results = {
        'test_date': datetime.now().isoformat(),
        'n_observations': len(df),
    }
    
    logger.info("\n" + "="*70)
    logger.info("1. STATISTICAL TESTS FOR CONTRARIAN SIGNAL")
    logger.info("="*70)
    
    # Define regimes
    regimes = ['extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed']
    
    # ---- A. T-tests: Is mean return different from zero? ----
    logger.info("\n--- A. One-sample t-tests (H0: mean return = 0) ---")
    
    regime_stats = {}
    for regime in regimes:
        mask = df['regime'] == regime
        returns = df.loc[mask, 'returns'].dropna() * 100  # Convert to %
        
        if len(returns) < 3:
            continue
            
        t_stat, p_value = ttest_1samp(returns, 0)
        
        regime_stats[regime] = {
            'n': len(returns),
            'mean_pct': float(returns.mean()),
            'std_pct': float(returns.std()),
            'se_pct': float(returns.std() / np.sqrt(len(returns))),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_5pct': p_value < 0.05,
        }
        
        sig = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
        logger.info(f"  {regime:15s}: mean={returns.mean():+.3f}%, t={t_stat:+.2f}, p={p_value:.4f} {sig}")
    
    results['regime_tstests'] = regime_stats
    
    # ---- B. Two-sample t-test: Extreme fear vs Extreme greed ----
    logger.info("\n--- B. Two-sample t-test: Extreme Fear vs Extreme Greed ---")
    
    fear_returns = df.loc[df['regime'] == 'extreme_fear', 'returns'].dropna() * 100
    greed_returns = df.loc[df['regime'] == 'extreme_greed', 'returns'].dropna() * 100
    
    t_stat, p_value = ttest_ind(fear_returns, greed_returns)
    diff = fear_returns.mean() - greed_returns.mean()
    
    # Cohen's d effect size
    pooled_std = np.sqrt((fear_returns.std()**2 + greed_returns.std()**2) / 2)
    cohens_d = diff / pooled_std
    
    results['fear_vs_greed'] = {
        'mean_diff_pct': float(diff),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'significant_5pct': p_value < 0.05,
    }
    
    sig = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
    logger.info(f"  Difference: {diff:+.3f}%")
    logger.info(f"  t-statistic: {t_stat:.3f}")
    logger.info(f"  p-value: {p_value:.4f} {sig}")
    logger.info(f"  Cohen's d: {cohens_d:.3f} ({'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'} effect)")
    
    # ---- C. Bootstrap Confidence Intervals ----
    logger.info("\n--- C. Bootstrap 95% Confidence Intervals (10,000 resamples) ---")
    
    n_bootstrap = 10000
    bootstrap_results = {}
    
    for regime in regimes:
        returns = df.loc[df['regime'] == regime, 'returns'].dropna() * 100
        if len(returns) < 3:
            continue
        
        # Bootstrap
        boot_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(returns, size=len(returns), replace=True)
            boot_means.append(sample.mean())
        
        ci_lower = np.percentile(boot_means, 2.5)
        ci_upper = np.percentile(boot_means, 97.5)
        
        bootstrap_results[regime] = {
            'mean_pct': float(returns.mean()),
            'ci_lower_pct': float(ci_lower),
            'ci_upper_pct': float(ci_upper),
            'ci_excludes_zero': not (ci_lower <= 0 <= ci_upper),
        }
        
        excludes_zero = "✓" if not (ci_lower <= 0 <= ci_upper) else ""
        logger.info(f"  {regime:15s}: {returns.mean():+.3f}% [{ci_lower:+.3f}%, {ci_upper:+.3f}%] {excludes_zero}")
    
    results['bootstrap_ci'] = bootstrap_results
    
    # ---- D. Newey-West Standard Errors (HAC) ----
    logger.info("\n--- D. Newey-West HAC Standard Errors (autocorrelation robust) ---")
    
    try:
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tools import add_constant
        
        # Create dummy variables for regimes
        df_reg = df.dropna(subset=['returns']).copy()
        df_reg['returns_pct'] = df_reg['returns'] * 100
        
        for regime in regimes:
            df_reg[f'is_{regime}'] = (df_reg['regime'] == regime).astype(int)
        
        # Regression with Newey-West SEs
        X = add_constant(df_reg[[f'is_{r}' for r in regimes[:-1]]])  # Drop one for multicollinearity
        y = df_reg['returns_pct']
        
        model = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
        
        results['newey_west'] = {
            'regression_summary': str(model.summary()),
            'coefficients': {k: float(v) for k, v in model.params.items()},
            'pvalues': {k: float(v) for k, v in model.pvalues.items()},
        }
        
        logger.info("  Newey-West regression completed (see full results in output)")
        
    except ImportError:
        logger.warning("  statsmodels not available for Newey-West")
        results['newey_west'] = None
    
    return results


# ============================================================================
# 2. Out-of-Sample Validation
# ============================================================================

def out_of_sample_validation(df):
    """
    Split data into training (2024) and test (2025-2026) periods.
    Check if contrarian pattern holds out-of-sample.
    """
    results = {}
    
    logger.info("\n" + "="*70)
    logger.info("2. OUT-OF-SAMPLE VALIDATION")
    logger.info("="*70)
    
    # Split by year
    df['year'] = df['date'].dt.year
    
    train = df[df['year'] == 2024].copy()
    test = df[df['year'] >= 2025].copy()
    
    logger.info(f"\n  Training period: 2024 ({len(train)} days)")
    logger.info(f"  Test period: 2025-2026 ({len(test)} days)")
    
    # Check contrarian pattern in each period
    for period_name, period_df in [('train_2024', train), ('test_2025_26', test)]:
        logger.info(f"\n--- {period_name.upper()} ---")
        
        period_stats = {}
        for regime in ['extreme_fear', 'extreme_greed']:
            returns = period_df.loc[period_df['regime'] == regime, 'returns'].dropna() * 100
            
            if len(returns) < 2:
                period_stats[regime] = {'n': len(returns), 'insufficient_data': True}
                continue
            
            t_stat, p_value = ttest_1samp(returns, 0)
            
            period_stats[regime] = {
                'n': int(len(returns)),
                'mean_pct': float(returns.mean()),
                'std_pct': float(returns.std()),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
            }
            
            sig = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
            logger.info(f"  {regime:15s}: n={len(returns):3d}, mean={returns.mean():+.3f}%, t={t_stat:+.2f}, p={p_value:.3f} {sig}")
        
        results[period_name] = period_stats
    
    # Check if pattern is consistent
    train_diff = (results['train_2024'].get('extreme_fear', {}).get('mean_pct', 0) - 
                  results['train_2024'].get('extreme_greed', {}).get('mean_pct', 0))
    test_diff = (results['test_2025_26'].get('extreme_fear', {}).get('mean_pct', 0) - 
                 results['test_2025_26'].get('extreme_greed', {}).get('mean_pct', 0))
    
    results['pattern_consistency'] = {
        'train_diff_pct': float(train_diff),
        'test_diff_pct': float(test_diff),
        'same_direction': (train_diff > 0) == (test_diff > 0),
    }
    
    logger.info(f"\n  Pattern consistency:")
    logger.info(f"    Train (2024): Fear - Greed = {train_diff:+.3f}%")
    logger.info(f"    Test (2025-26): Fear - Greed = {test_diff:+.3f}%")
    logger.info(f"    Same direction: {'✓ YES' if results['pattern_consistency']['same_direction'] else '✗ NO'}")
    
    return results


# ============================================================================
# 3. Ablation Studies
# ============================================================================

def ablation_studies(df):
    """
    Test different configurations:
    - Macro-only (Fear & Greed only)
    - Different regime thresholds
    - Predictive vs contemporaneous
    """
    results = {}
    
    logger.info("\n" + "="*70)
    logger.info("3. ABLATION STUDIES")
    logger.info("="*70)
    
    # ---- A. Alternative Threshold Sensitivity ----
    logger.info("\n--- A. Threshold Sensitivity Analysis ---")
    logger.info("  (Testing different extreme thresholds)")
    
    threshold_results = {}
    for fear_thresh, greed_thresh in [(20, 80), (25, 75), (30, 70), (15, 85)]:
        df_temp = df.copy()
        df_temp['alt_regime'] = 'neutral'
        df_temp.loc[df_temp['fear_greed_value'] < fear_thresh, 'alt_regime'] = 'extreme_fear'
        df_temp.loc[df_temp['fear_greed_value'] > greed_thresh, 'alt_regime'] = 'extreme_greed'
        
        fear_ret = df_temp.loc[df_temp['alt_regime'] == 'extreme_fear', 'returns'].mean() * 100
        greed_ret = df_temp.loc[df_temp['alt_regime'] == 'extreme_greed', 'returns'].mean() * 100
        n_fear = (df_temp['alt_regime'] == 'extreme_fear').sum()
        n_greed = (df_temp['alt_regime'] == 'extreme_greed').sum()
        
        threshold_results[f'fear<{fear_thresh}_greed>{greed_thresh}'] = {
            'n_fear': int(n_fear),
            'n_greed': int(n_greed),
            'fear_return_pct': float(fear_ret) if not np.isnan(fear_ret) else None,
            'greed_return_pct': float(greed_ret) if not np.isnan(greed_ret) else None,
            'contrarian_holds': (fear_ret > greed_ret) if not (np.isnan(fear_ret) or np.isnan(greed_ret)) else None,
        }
        
        contrarian = "✓" if fear_ret > greed_ret else "✗"
        logger.info(f"  Fear<{fear_thresh}, Greed>{greed_thresh}: Fear={fear_ret:+.2f}% (n={n_fear}), Greed={greed_ret:+.2f}% (n={n_greed}) {contrarian}")
    
    results['threshold_sensitivity'] = threshold_results
    
    # ---- B. Predictive vs Contemporaneous ----
    logger.info("\n--- B. Predictive Power: Today's Sentiment → Tomorrow's Return ---")
    
    df_pred = df.dropna(subset=['next_return']).copy()
    
    predictive_results = {}
    for regime in ['extreme_fear', 'extreme_greed']:
        next_returns = df_pred.loc[df_pred['regime'] == regime, 'next_return'] * 100
        same_day_returns = df_pred.loc[df_pred['regime'] == regime, 'returns'] * 100
        
        if len(next_returns) < 3:
            continue
        
        t_next, p_next = ttest_1samp(next_returns, 0)
        t_same, p_same = ttest_1samp(same_day_returns, 0)
        
        predictive_results[regime] = {
            'same_day_mean_pct': float(same_day_returns.mean()),
            'same_day_p': float(p_same),
            'next_day_mean_pct': float(next_returns.mean()),
            'next_day_p': float(p_next),
        }
        
        logger.info(f"  {regime:15s}: Same-day={same_day_returns.mean():+.3f}% (p={p_same:.3f}), Next-day={next_returns.mean():+.3f}% (p={p_next:.3f})")
    
    results['predictive_vs_contemporaneous'] = predictive_results
    
    # ---- C. Rolling Window Stability ----
    logger.info("\n--- C. Rolling 6-Month Window Stability ---")
    
    df_sorted = df.sort_values('date').copy()
    window_size = 180  # ~6 months
    
    rolling_results = []
    for start_idx in range(0, len(df_sorted) - window_size, 30):  # Step by ~1 month
        window = df_sorted.iloc[start_idx:start_idx + window_size]
        
        fear_ret = window.loc[window['regime'] == 'extreme_fear', 'returns'].mean()
        greed_ret = window.loc[window['regime'] == 'extreme_greed', 'returns'].mean()
        
        if not (np.isnan(fear_ret) or np.isnan(greed_ret)):
            rolling_results.append({
                'start_date': str(window['date'].iloc[0].date()),
                'end_date': str(window['date'].iloc[-1].date()),
                'fear_return_pct': float(fear_ret * 100),
                'greed_return_pct': float(greed_ret * 100),
                'contrarian_holds': fear_ret > greed_ret,
            })
    
    n_contrarian = sum(1 for r in rolling_results if r['contrarian_holds'])
    pct_contrarian = n_contrarian / len(rolling_results) * 100 if rolling_results else 0
    
    results['rolling_window'] = {
        'n_windows': len(rolling_results),
        'n_contrarian_holds': n_contrarian,
        'pct_contrarian_holds': float(pct_contrarian),
        'windows': rolling_results,
    }
    
    logger.info(f"  Windows tested: {len(rolling_results)}")
    logger.info(f"  Contrarian pattern holds: {n_contrarian}/{len(rolling_results)} ({pct_contrarian:.1f}%)")
    
    return results


# ============================================================================
# 4. Backtest with Transaction Costs
# ============================================================================

def backtest_with_costs(df):
    """
    Simple contrarian strategy backtest with realistic costs.
    
    Strategy: 
    - Enter long when regime = extreme_fear
    - Exit when regime != extreme_fear OR after max_hold days
    - Include round-trip costs
    """
    results = {}
    
    logger.info("\n" + "="*70)
    logger.info("4. BACKTEST WITH TRANSACTION COSTS")
    logger.info("="*70)
    
    # Parameters
    round_trip_cost_bps = 20  # 10bps entry + 10bps exit
    max_hold_days = 10
    
    df_bt = df.sort_values('date').copy()
    df_bt['position'] = 0
    df_bt['strategy_return'] = 0.0
    
    in_position = False
    entry_idx = None
    hold_days = 0
    
    trades = []
    
    for i in range(len(df_bt)):
        row = df_bt.iloc[i]
        
        if not in_position:
            # Entry condition: extreme fear
            if row['regime'] == 'extreme_fear':
                in_position = True
                entry_idx = i
                hold_days = 0
                df_bt.iloc[i, df_bt.columns.get_loc('position')] = 1
        else:
            hold_days += 1
            df_bt.iloc[i, df_bt.columns.get_loc('position')] = 1
            
            # Exit conditions
            exit_signal = (row['regime'] not in ['extreme_fear', 'fear']) or (hold_days >= max_hold_days)
            
            if exit_signal or i == len(df_bt) - 1:
                # Calculate trade return
                entry_price = df_bt.iloc[entry_idx]['close']
                exit_price = row['close']
                gross_return = (exit_price / entry_price - 1) * 100
                net_return = gross_return - (round_trip_cost_bps / 100)
                
                trades.append({
                    'entry_date': str(df_bt.iloc[entry_idx]['date'].date()),
                    'exit_date': str(row['date'].date()),
                    'hold_days': hold_days,
                    'gross_return_pct': float(gross_return),
                    'net_return_pct': float(net_return),
                })
                
                in_position = False
                entry_idx = None
    
    # Calculate metrics
    if trades:
        gross_returns = [t['gross_return_pct'] for t in trades]
        net_returns = [t['net_return_pct'] for t in trades]
        
        results['strategy_metrics'] = {
            'n_trades': len(trades),
            'avg_hold_days': float(np.mean([t['hold_days'] for t in trades])),
            'gross_return_mean_pct': float(np.mean(gross_returns)),
            'gross_return_std_pct': float(np.std(gross_returns)),
            'net_return_mean_pct': float(np.mean(net_returns)),
            'net_return_std_pct': float(np.std(net_returns)),
            'win_rate_gross': float(sum(1 for r in gross_returns if r > 0) / len(gross_returns)),
            'win_rate_net': float(sum(1 for r in net_returns if r > 0) / len(net_returns)),
            'total_gross_return_pct': float(sum(gross_returns)),
            'total_net_return_pct': float(sum(net_returns)),
            'sharpe_ratio_net': float(np.mean(net_returns) / np.std(net_returns)) if np.std(net_returns) > 0 else 0,
        }
        
        # T-test on net returns
        t_stat, p_value = ttest_1samp(net_returns, 0)
        results['strategy_metrics']['t_statistic'] = float(t_stat)
        results['strategy_metrics']['p_value'] = float(p_value)
        results['strategy_metrics']['significant_5pct'] = p_value < 0.05
        
        results['trades'] = trades
        
        logger.info(f"\n  Strategy: Enter on extreme fear, exit on neutral/greed or {max_hold_days} days")
        logger.info(f"  Transaction costs: {round_trip_cost_bps}bps round-trip")
        logger.info(f"\n  Results:")
        logger.info(f"    Number of trades: {len(trades)}")
        logger.info(f"    Average hold: {results['strategy_metrics']['avg_hold_days']:.1f} days")
        logger.info(f"    Gross return/trade: {results['strategy_metrics']['gross_return_mean_pct']:+.2f}%")
        logger.info(f"    Net return/trade: {results['strategy_metrics']['net_return_mean_pct']:+.2f}%")
        logger.info(f"    Win rate (net): {results['strategy_metrics']['win_rate_net']*100:.1f}%")
        logger.info(f"    Total net return: {results['strategy_metrics']['total_net_return_pct']:+.1f}%")
        logger.info(f"    Sharpe ratio: {results['strategy_metrics']['sharpe_ratio_net']:.2f}")
        logger.info(f"    t-statistic: {t_stat:.2f} (p={p_value:.3f})")
        
        # Compare to buy-and-hold
        bh_return = (df_bt['close'].iloc[-1] / df_bt['close'].iloc[0] - 1) * 100
        results['buy_and_hold_return_pct'] = float(bh_return)
        logger.info(f"\n  Comparison:")
        logger.info(f"    Strategy total: {results['strategy_metrics']['total_net_return_pct']:+.1f}%")
        logger.info(f"    Buy-and-hold: {bh_return:+.1f}%")
        
    else:
        results['strategy_metrics'] = {'n_trades': 0, 'error': 'no_trades'}
        logger.info("  No trades generated")
    
    return results


# ============================================================================
# 5. Summary and Export
# ============================================================================

def run_all_robustness_tests():
    """Run all robustness tests and save results."""
    
    logger.info("="*70)
    logger.info("ROBUSTNESS TESTS FOR PEER REVIEW")
    logger.info("="*70)
    
    # Load data
    df = load_data()
    
    # Run all tests
    all_results = {
        'run_timestamp': datetime.now().isoformat(),
        'data_summary': {
            'n_observations': len(df),
            'date_range': f"{df['date'].min().date()} to {df['date'].max().date()}",
        }
    }
    
    all_results['statistical_tests'] = statistical_tests(df)
    all_results['out_of_sample'] = out_of_sample_validation(df)
    all_results['ablations'] = ablation_studies(df)
    all_results['backtest'] = backtest_with_costs(df)
    
    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'robustness')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'robustness_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"\n{'='*70}")
    logger.info("ROBUSTNESS TESTS COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Results saved to: {output_path}")
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("EXECUTIVE SUMMARY")
    logger.info("="*70)
    
    # Key findings
    fear_greed_test = all_results['statistical_tests']['fear_vs_greed']
    oos = all_results['out_of_sample']['pattern_consistency']
    backtest = all_results['backtest'].get('strategy_metrics', {})
    
    logger.info(f"""
Key Findings:

1. STATISTICAL SIGNIFICANCE
   - Extreme Fear vs Extreme Greed difference: {fear_greed_test['mean_diff_pct']:+.3f}%
   - t-statistic: {fear_greed_test['t_statistic']:.2f}
   - p-value: {fear_greed_test['p_value']:.4f}
   - Effect size (Cohen's d): {fear_greed_test['cohens_d']:.3f}
   - Significant at 5%: {'YES' if fear_greed_test['significant_5pct'] else 'NO'}

2. OUT-OF-SAMPLE VALIDATION
   - Training (2024) difference: {oos['train_diff_pct']:+.3f}%
   - Test (2025-26) difference: {oos['test_diff_pct']:+.3f}%
   - Pattern consistent: {'YES' if oos['same_direction'] else 'NO'}

3. BACKTEST (with 20bps costs)
   - Number of trades: {backtest.get('n_trades', 'N/A')}
   - Net return per trade: {backtest.get('net_return_mean_pct', 0):+.2f}%
   - Win rate: {backtest.get('win_rate_net', 0)*100:.1f}%
   - Total net return: {backtest.get('total_net_return_pct', 0):+.1f}%
   - Statistically significant: {'YES' if backtest.get('significant_5pct', False) else 'NO'}
""")
    
    return all_results


if __name__ == '__main__':
    results = run_all_robustness_tests()
