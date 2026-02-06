"""
Regime Transition Analysis

What happens to uncertainty when markets enter/exit extreme sentiment regimes?

Hypothesis:
- Entering extreme regime → uncertainty RISES
- Exiting extreme regime → uncertainty FALLS

Author: Murad Farzulla
Date: January 2026
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load and prepare data with regime transitions."""
    df_spreads = pd.read_csv('results/real_spread_data.csv', parse_dates=['date'])
    df_sentiment = pd.read_csv('data/datasets/btc_sentiment_daily.csv', parse_dates=['date'])

    df = pd.merge(df_spreads, df_sentiment[['date', 'regime', 'fear_greed_value']],
                  on='date', how='inner')
    df['volatility'] = df['realized_vol'].fillna(df['parkinson_vol'])
    df = df.dropna(subset=['total_uncertainty', 'volatility', 'regime']).copy()
    df = df.sort_values('date').reset_index(drop=True)

    # Classify regimes
    df['is_extreme'] = df['regime'].isin(['extreme_greed', 'extreme_fear']).astype(int)
    df['is_neutral'] = (df['regime'] == 'neutral').astype(int)

    return df


def identify_transitions(df):
    """
    Identify regime transition points.

    Transition types:
    - enter_extreme: non-extreme → extreme
    - exit_extreme: extreme → non-extreme
    - enter_neutral: non-neutral → neutral
    - exit_neutral: neutral → non-neutral
    """
    df = df.copy()

    # Previous day's regime
    df['prev_is_extreme'] = df['is_extreme'].shift(1)
    df['prev_is_neutral'] = df['is_neutral'].shift(1)

    # Transition indicators
    df['enter_extreme'] = ((df['is_extreme'] == 1) & (df['prev_is_extreme'] == 0)).astype(int)
    df['exit_extreme'] = ((df['is_extreme'] == 0) & (df['prev_is_extreme'] == 1)).astype(int)
    df['enter_neutral'] = ((df['is_neutral'] == 1) & (df['prev_is_neutral'] == 0)).astype(int)
    df['exit_neutral'] = ((df['is_neutral'] == 0) & (df['prev_is_neutral'] == 1)).astype(int)

    # Summary
    print("Transition Summary:")
    print(f"  Enter extreme: {df['enter_extreme'].sum()} events")
    print(f"  Exit extreme:  {df['exit_extreme'].sum()} events")
    print(f"  Enter neutral: {df['enter_neutral'].sum()} events")
    print(f"  Exit neutral:  {df['exit_neutral'].sum()} events")

    return df


def analyze_transition_windows(df, window_size=5):
    """
    Analyze uncertainty in windows around transitions.

    For each transition type, compute mean uncertainty in:
    - [-5, -1] days before transition
    - [0] transition day
    - [+1, +5] days after transition
    """
    print(f"\n" + "="*70)
    print(f"TRANSITION WINDOW ANALYSIS (±{window_size} days)")
    print("="*70)

    results = []

    for transition_type in ['enter_extreme', 'exit_extreme', 'enter_neutral', 'exit_neutral']:
        transition_indices = df[df[transition_type] == 1].index.tolist()

        if len(transition_indices) < 3:
            print(f"\n{transition_type}: Insufficient events ({len(transition_indices)})")
            continue

        before_values = []
        after_values = []
        day_values = []

        for idx in transition_indices:
            # Before window
            before_start = max(0, idx - window_size)
            before_end = idx  # exclusive
            before = df.loc[before_start:before_end-1, 'total_uncertainty']
            if len(before) > 0:
                before_values.extend(before.tolist())

            # Transition day
            day_values.append(df.loc[idx, 'total_uncertainty'])

            # After window
            after_start = idx + 1
            after_end = min(len(df) - 1, idx + window_size)
            after = df.loc[after_start:after_end, 'total_uncertainty']
            if len(after) > 0:
                after_values.extend(after.tolist())

        before_mean = np.mean(before_values) if before_values else np.nan
        day_mean = np.mean(day_values) if day_values else np.nan
        after_mean = np.mean(after_values) if after_values else np.nan

        # Test: after vs before
        if len(before_values) >= 3 and len(after_values) >= 3:
            t_stat, p_value = stats.ttest_ind(after_values, before_values)
            change = after_mean - before_mean
        else:
            t_stat, p_value, change = np.nan, np.nan, np.nan

        print(f"\n{transition_type.upper()}:")
        print(f"  Events: {len(transition_indices)}")
        print(f"  Before (mean): {before_mean:.4f}")
        print(f"  Day (mean):    {day_mean:.4f}")
        print(f"  After (mean):  {after_mean:.4f}")
        print(f"  Change (A-B):  {change:+.4f} (t={t_stat:.2f}, p={p_value:.4f})")

        if p_value < 0.05:
            direction = "↑ RISES" if change > 0 else "↓ FALLS"
            print(f"  ✓ Significant: Uncertainty {direction} after {transition_type}")

        results.append({
            'transition_type': transition_type,
            'n_events': len(transition_indices),
            'before_mean': before_mean,
            'day_mean': day_mean,
            'after_mean': after_mean,
            'change': change,
            't_stat': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05 if not np.isnan(p_value) else False
        })

    return pd.DataFrame(results)


def analyze_daily_changes(df):
    """
    Analyze day-over-day uncertainty changes at transitions.
    """
    print("\n" + "="*70)
    print("DAILY CHANGES AT TRANSITIONS")
    print("="*70)

    df = df.copy()
    df['uncertainty_change'] = df['total_uncertainty'].diff()
    df['vol_change'] = df['volatility'].diff()

    results = []

    for transition_type in ['enter_extreme', 'exit_extreme', 'enter_neutral', 'exit_neutral']:
        transition_days = df[df[transition_type] == 1]

        if len(transition_days) < 3:
            continue

        unc_changes = transition_days['uncertainty_change'].dropna()
        vol_changes = transition_days['vol_change'].dropna()

        mean_unc_change = unc_changes.mean()
        mean_vol_change = vol_changes.mean()

        # T-test: is mean change different from zero?
        t_stat, p_value = stats.ttest_1samp(unc_changes, 0)

        print(f"\n{transition_type.upper()}:")
        print(f"  N = {len(unc_changes)}")
        print(f"  Mean uncertainty change: {mean_unc_change:+.4f} (t={t_stat:.2f}, p={p_value:.4f})")
        print(f"  Mean volatility change:  {mean_vol_change:+.4f}")

        sig = '✓' if p_value < 0.05 else ''
        if sig:
            direction = "INCREASES" if mean_unc_change > 0 else "DECREASES"
            print(f"  {sig} Uncertainty significantly {direction} on transition day")

        results.append({
            'transition_type': transition_type,
            'n': len(unc_changes),
            'mean_uncertainty_change': mean_unc_change,
            'mean_vol_change': mean_vol_change,
            't_stat': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        })

    return pd.DataFrame(results)


def analyze_extreme_duration_effects(df):
    """
    Analyze how uncertainty evolves during extreme regime spells.
    Does uncertainty peak on entry and decay, or build over time?
    """
    print("\n" + "="*70)
    print("EXTREME REGIME DURATION EFFECTS")
    print("="*70)

    df = df.copy()

    # Identify extreme regime spells
    df['spell_id'] = (df['is_extreme'] != df['is_extreme'].shift(1)).cumsum()
    extreme_spells = df[df['is_extreme'] == 1].copy()

    # Days within each spell
    extreme_spells['days_in_spell'] = extreme_spells.groupby('spell_id').cumcount() + 1

    print(f"\nExtreme regime spells: {extreme_spells['spell_id'].nunique()}")
    print(f"Total extreme days: {len(extreme_spells)}")

    # Uncertainty by days in spell
    spell_stats = extreme_spells.groupby('days_in_spell').agg({
        'total_uncertainty': ['mean', 'std', 'count']
    }).round(4)
    spell_stats.columns = ['mean', 'std', 'count']

    print("\nUncertainty by days in extreme spell:")
    for day in range(1, min(11, len(spell_stats) + 1)):
        if day in spell_stats.index:
            row = spell_stats.loc[day]
            print(f"  Day {day:2d}: mean={row['mean']:.4f}, n={int(row['count'])}")

    # Correlation: days in spell vs uncertainty
    corr, p_value = stats.pearsonr(extreme_spells['days_in_spell'], extreme_spells['total_uncertainty'])
    print(f"\nCorrelation (days in spell vs uncertainty): r={corr:.3f}, p={p_value:.4f}")

    if p_value < 0.05:
        direction = "builds over time" if corr > 0 else "peaks early and decays"
        print(f"  ✓ Significant: Uncertainty {direction} within extreme spells")

    return extreme_spells, spell_stats


def main():
    print("="*70)
    print("REGIME TRANSITION ANALYSIS")
    print("="*70)

    # Load data
    df = load_data()
    print(f"\nData: {len(df)} observations")

    # Identify transitions
    df = identify_transitions(df)

    # Window analysis
    window_results = analyze_transition_windows(df, window_size=5)

    # Daily changes
    daily_results = analyze_daily_changes(df)

    # Duration effects
    extreme_spells, spell_stats = analyze_extreme_duration_effects(df)

    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    window_results.to_csv('results/regime_transition_windows.csv', index=False)
    print("  Saved: results/regime_transition_windows.csv")

    daily_results.to_csv('results/regime_transition_daily.csv', index=False)
    print("  Saved: results/regime_transition_daily.csv")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
  Key hypothesis tests:
  1. Entering extreme → uncertainty rises
  2. Exiting extreme → uncertainty falls
  3. Entering neutral → uncertainty falls
  4. Exiting neutral → uncertainty rises
""")

    for _, row in window_results.iterrows():
        if row['significant']:
            direction = "↑" if row['change'] > 0 else "↓"
            print(f"  ✓ {row['transition_type']}: {direction} {row['change']:+.4f} (p={row['p_value']:.4f})")
        else:
            print(f"    {row['transition_type']}: {row['change']:+.4f} (p={row['p_value']:.4f})")

    return window_results, daily_results


if __name__ == '__main__':
    window_results, daily_results = main()
