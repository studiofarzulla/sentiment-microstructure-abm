"""
Preliminary Results Demo - Sentiment-Microstructure ABM

Generates preliminary results for PhD supervisor presentation.
Shows:
1. Sentiment analysis with Monte Carlo Dropout uncertainty
2. Market maker quote adjustment based on sentiment
3. Simulated order book dynamics

Author: Murad Farzulla
Date: December 2025
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import the actual sentiment analyzer
# Note: Set USE_SYNTHETIC = True to skip model download and use synthetic data
USE_SYNTHETIC = True  # For reproducible paper results

try:
    if USE_SYNTHETIC:
        raise ImportError("Using synthetic mode")
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch.nn.functional as F
    REAL_ANALYZER = True
except ImportError:
    REAL_ANALYZER = False
    print("Note: Using synthetic sentiment data (for reproducible paper results)")


class CryptoBERTAnalyzer:
    """
    CryptoBERT-based sentiment analyzer with MC Dropout uncertainty.

    Uses ElKulako/cryptobert fine-tuned on 3.2M crypto social media posts.
    Labels: 0=Bearish, 1=Neutral, 2=Bullish
    """

    def __init__(self, model_name: str = "ElKulako/cryptobert", n_mc_samples: int = 20):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.n_mc_samples = n_mc_samples

        # Enable dropout for MC sampling
        self._enable_mc_dropout()
        print(f"Loaded {model_name} with MC Dropout ({n_mc_samples} samples)")

    def _enable_mc_dropout(self):
        """Enable dropout during inference for MC sampling."""
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()  # Keep dropout active

    def analyze(self, text: str) -> Tuple[float, float, float]:
        """
        Analyze sentiment with uncertainty quantification.

        Returns:
            (sentiment_score, epistemic_uncertainty, aleatoric_uncertainty)
            sentiment_score in [-1, 1]: -1=Bearish, 0=Neutral, +1=Bullish
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # MC Dropout sampling
        all_probs = []
        with torch.no_grad():
            for _ in range(self.n_mc_samples):
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                all_probs.append(probs.cpu().numpy())

        all_probs = np.array(all_probs).squeeze()  # (n_samples, 3)

        # Mean prediction
        mean_probs = all_probs.mean(axis=0)  # [bearish, neutral, bullish]

        # Convert to sentiment score [-1, 1]
        # Score = -1*P(bearish) + 0*P(neutral) + 1*P(bullish)
        sentiment = -1 * mean_probs[0] + 0 * mean_probs[1] + 1 * mean_probs[2]

        # Epistemic uncertainty (variance of predictions)
        epistemic = all_probs.var(axis=0).mean()

        # Aleatoric uncertainty (entropy of mean prediction)
        aleatoric = -np.sum(mean_probs * np.log(mean_probs + 1e-10))

        return float(sentiment), float(epistemic), float(aleatoric)


@dataclass
class SentimentResult:
    """Result from sentiment analysis with uncertainty."""
    text: str
    sentiment: float  # [-1, 1]
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    total_uncertainty: float


def get_sample_crypto_texts() -> List[str]:
    """Sample crypto-related texts spanning sentiment range."""
    return [
        # Positive
        "Bitcoin ETF approved! This is huge for institutional adoption. BTC to the moon!",
        "Ethereum staking rewards looking great, the merge was a massive success",
        "Just bought more BTC on this dip, fundamentals are stronger than ever",

        # Neutral
        "Bitcoin trading sideways around 43k, waiting for next move",
        "SEC meeting scheduled for next week to discuss crypto regulations",
        "Crypto market cap holding steady at 1.7 trillion",

        # Negative
        "FTX collapse shows crypto is just a scam, lost everything",
        "Another exchange hack, this is why I don't trust centralized platforms",
        "SEC suing major exchanges, crypto winter is here to stay",

        # Mixed/Uncertain
        "Not sure if this is a bull trap or real recovery, staying cautious",
        "Whale wallets moving big amounts, could go either way",
        "Volume looking weak but price holding, conflicting signals",
    ]


def synthetic_sentiment_analysis(texts: List[str]) -> List[SentimentResult]:
    """Generate synthetic but realistic sentiment results."""
    results = []

    # Predefined sentiment profiles (based on text content patterns)
    for i, text in enumerate(texts):
        text_lower = text.lower()

        # Determine base sentiment from keywords
        positive_words = ['approved', 'huge', 'moon', 'great', 'success', 'bought', 'stronger']
        negative_words = ['collapse', 'scam', 'lost', 'hack', 'suing', 'winter']
        uncertain_words = ['not sure', 'cautious', 'either way', 'conflicting']

        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        unc_count = sum(1 for w in uncertain_words if w in text_lower)

        if pos_count > neg_count:
            base_sentiment = 0.5 + np.random.uniform(0.1, 0.4)
        elif neg_count > pos_count:
            base_sentiment = -0.5 - np.random.uniform(0.1, 0.4)
        else:
            base_sentiment = np.random.uniform(-0.2, 0.2)

        # Higher aleatoric uncertainty for ambiguous texts
        if unc_count > 0:
            aleatoric = np.random.uniform(0.3, 0.5)
        else:
            aleatoric = np.random.uniform(0.1, 0.25)

        # Epistemic uncertainty (model confidence)
        epistemic = np.random.uniform(0.05, 0.15)

        results.append(SentimentResult(
            text=text[:60] + "..." if len(text) > 60 else text,
            sentiment=np.clip(base_sentiment, -1, 1),
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=epistemic + aleatoric
        ))

    return results


def real_sentiment_analysis(texts: List[str]) -> List[SentimentResult]:
    """Run actual MC Dropout sentiment analysis with CryptoBERT."""
    analyzer = CryptoBERTAnalyzer(n_mc_samples=20)
    results = []

    for text in texts:
        score, epistemic, aleatoric = analyzer.analyze(text)
        results.append(SentimentResult(
            text=text[:60] + "..." if len(text) > 60 else text,
            sentiment=score,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=epistemic + aleatoric
        ))

    return results


def simulate_market_maker(
    sentiment_series: np.ndarray,
    epistemic_series: np.ndarray,
    aleatoric_series: np.ndarray,
    regime_series: np.ndarray,
    base_spread: float = 0.001,
    sentiment_sensitivity: float = 0.5,
    epistemic_sensitivity: float = 1.5,
    aleatoric_sensitivity: float = 0.5,
    inventory_aversion: float = 0.0001,
    seed: int = 42
) -> pd.DataFrame:
    """
    Simulate market maker behavior responding to sentiment signals.

    Based on Avellaneda-Stoikov style quoting with sentiment and uncertainty adjustment.
    Spread widening depends more on epistemic (model) uncertainty than aleatoric (data) uncertainty.

    Returns DataFrame with:
    - mid_price: Simulated mid price
    - log_return: Log return from previous step
    - bid: Market maker bid
    - ask: Market maker ask
    - spread: Bid-ask spread
    - spread_bps: Spread in basis points
    - inventory: Market maker inventory
    - regime: Current market regime
    - epistemic_uncertainty: Model uncertainty
    - aleatoric_uncertainty: Data uncertainty
    - total_uncertainty: Sum of uncertainties
    """
    np.random.seed(seed)
    n = len(sentiment_series)

    # Initialize
    mid_price = 100.0  # Arbitrary starting price
    prev_price = mid_price
    inventory = 0.0

    records = []

    for t in range(n):
        sentiment = sentiment_series[t]
        epistemic = epistemic_series[t]
        aleatoric = aleatoric_series[t]
        regime = regime_series[t]

        # 1. Adjust mid price based on sentiment (drift) + volatility
        # Positive sentiment -> price tends to rise
        # Add volatility component based on uncertainty
        volatility = 0.0005 + epistemic * 0.002  # Base vol + uncertainty-driven vol
        price_drift = sentiment * 0.001  # Sentiment drift
        price_noise = np.random.normal(0, volatility)  # Random walk component
        prev_price = mid_price
        mid_price *= (1 + price_drift + price_noise)

        # Calculate log return
        log_return = np.log(mid_price / prev_price) if t > 0 else 0.0

        # 2. Calculate spread
        # Base spread + epistemic premium (model uncertainty matters more) + aleatoric premium
        epistemic_premium = base_spread * epistemic * epistemic_sensitivity
        aleatoric_premium = base_spread * aleatoric * aleatoric_sensitivity
        inventory_skew = -inventory_aversion * inventory  # Mean reversion

        spread = base_spread + epistemic_premium + aleatoric_premium

        # 3. Calculate quotes
        # Sentiment shifts the midpoint of quotes
        sentiment_shift = sentiment * sentiment_sensitivity * spread

        bid = mid_price - spread/2 + inventory_skew + sentiment_shift
        ask = mid_price + spread/2 + inventory_skew + sentiment_shift

        # 4. Simulate order flow (random but sentiment-biased)
        order_prob = 0.3  # Probability of order arrival
        if np.random.random() < order_prob:
            # Buy probability increases with positive sentiment
            buy_prob = 0.5 + sentiment * 0.2
            if np.random.random() < buy_prob:
                # Buy order - hits our ask
                inventory -= 1
            else:
                # Sell order - hits our bid
                inventory += 1

        records.append({
            'step': t,
            'sentiment': sentiment,
            'epistemic_uncertainty': epistemic,
            'aleatoric_uncertainty': aleatoric,
            'total_uncertainty': epistemic + aleatoric,
            'mid_price': mid_price,
            'log_return': log_return,
            'bid': bid,
            'ask': ask,
            'spread': ask - bid,
            'spread_bps': (ask - bid) / mid_price * 10000,
            'inventory': inventory,
            'regime': regime
        })

    return pd.DataFrame(records)


def generate_sentiment_time_series(
    n_steps: int = 2000,
    regime_change_prob: float = 0.01,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic sentiment time series with regime changes.

    Returns:
        sentiment: (n_steps,) sentiment scores in [-1, 1]
        epistemic_uncertainty: (n_steps,) model uncertainty (variance across MC samples)
        aleatoric_uncertainty: (n_steps,) data uncertainty (entropy of predictions)
        total_uncertainty: (n_steps,) epistemic + aleatoric
        regime_labels: (n_steps,) regime strings ('bullish', 'neutral', 'bearish')
    """
    np.random.seed(seed)

    sentiment = np.zeros(n_steps)
    epistemic_uncertainty = np.zeros(n_steps)
    aleatoric_uncertainty = np.zeros(n_steps)
    regime_labels = np.empty(n_steps, dtype=object)

    # Start with neutral sentiment
    current_sentiment = 0.0
    current_regime = 'neutral'  # 'bullish', 'bearish', 'neutral'
    prev_regime = 'neutral'

    for t in range(n_steps):
        prev_regime = current_regime

        # Regime change
        if np.random.random() < regime_change_prob:
            current_regime = np.random.choice(['bullish', 'bearish', 'neutral'])

        # Mean reversion to regime
        if current_regime == 'bullish':
            target = 0.5
        elif current_regime == 'bearish':
            target = -0.5
        else:
            target = 0.0

        # AR(1) with regime-dependent mean
        current_sentiment = 0.9 * current_sentiment + 0.1 * target + np.random.normal(0, 0.1)
        current_sentiment = np.clip(current_sentiment, -1, 1)

        # Epistemic uncertainty: model confidence
        # Higher during regime transitions, lower when sentiment is extreme (model is confident)
        regime_transition = 1.0 if prev_regime != current_regime else 0.0
        base_epistemic = 0.02
        transition_epistemic = regime_transition * 0.08
        sentiment_epistemic = (1.0 - abs(current_sentiment)) * 0.03  # Less confident near neutral
        epistemic = base_epistemic + transition_epistemic + sentiment_epistemic + np.random.uniform(0, 0.02)

        # Aleatoric uncertainty: inherent data noise
        # Higher for ambiguous/neutral sentiment, lower for extreme sentiment
        base_aleatoric = 0.10
        sentiment_aleatoric = (1.0 - abs(current_sentiment)) * 0.15  # More uncertain near neutral
        noise_aleatoric = np.random.uniform(0, 0.05)
        aleatoric = base_aleatoric + sentiment_aleatoric + noise_aleatoric

        sentiment[t] = current_sentiment
        epistemic_uncertainty[t] = epistemic
        aleatoric_uncertainty[t] = aleatoric
        regime_labels[t] = current_regime

    total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty

    return sentiment, epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty, regime_labels


def plot_preliminary_results(sentiment_results: List[SentimentResult],
                            mm_df: pd.DataFrame,
                            save_path: str = None):
    """Generate publication-quality preliminary results figure."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Sentiment Analysis Results (Top Left)
    ax1 = axes[0, 0]
    sentiments = [r.sentiment for r in sentiment_results]
    epistemic = [r.epistemic_uncertainty for r in sentiment_results]
    aleatoric = [r.aleatoric_uncertainty for r in sentiment_results]

    x = np.arange(len(sentiment_results))
    width = 0.6

    colors = ['#2ecc71' if s > 0.2 else '#e74c3c' if s < -0.2 else '#95a5a6' for s in sentiments]
    bars = ax1.bar(x, sentiments, width, color=colors, alpha=0.7, label='Sentiment')
    ax1.errorbar(x, sentiments, yerr=[epistemic, aleatoric], fmt='none',
                 ecolor='black', capsize=3, label='Uncertainty (epistemic/aleatoric)')

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('Sentiment Score')
    ax1.set_xlabel('Sample Text')
    ax1.set_title('A) Sentiment Analysis with MC Dropout Uncertainty')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'T{i+1}' for i in range(len(sentiment_results))], rotation=45)
    ax1.legend(loc='upper right')
    ax1.set_ylim(-1.2, 1.2)

    # 2. Uncertainty Decomposition (Top Right)
    ax2 = axes[0, 1]

    width = 0.35
    x = np.arange(len(sentiment_results))

    ax2.bar(x - width/2, epistemic, width, label='Epistemic (model)', color='#3498db', alpha=0.7)
    ax2.bar(x + width/2, aleatoric, width, label='Aleatoric (data)', color='#e74c3c', alpha=0.7)

    ax2.set_ylabel('Uncertainty')
    ax2.set_xlabel('Sample Text')
    ax2.set_title('B) Uncertainty Decomposition')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'T{i+1}' for i in range(len(sentiment_results))], rotation=45)
    ax2.legend()

    # 3. Market Maker Spread Dynamics (Bottom Left)
    ax3 = axes[1, 0]

    ax3.plot(mm_df['step'], mm_df['spread_bps'], color='#3498db', alpha=0.7, label='Spread (bps)')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(mm_df['step'], mm_df['sentiment'], color='#e74c3c', alpha=0.5, label='Sentiment')
    ax3_twin.fill_between(mm_df['step'],
                          mm_df['sentiment'] - mm_df['total_uncertainty'],
                          mm_df['sentiment'] + mm_df['total_uncertainty'],
                          color='#e74c3c', alpha=0.1)

    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Spread (basis points)', color='#3498db')
    ax3_twin.set_ylabel('Sentiment', color='#e74c3c')
    ax3.set_title('C) Market Maker Spread Response to Sentiment')
    ax3.tick_params(axis='y', labelcolor='#3498db')
    ax3_twin.tick_params(axis='y', labelcolor='#e74c3c')

    # 4. Inventory Dynamics (Bottom Right)
    ax4 = axes[1, 1]

    ax4.plot(mm_df['step'], mm_df['inventory'], color='#9b59b6', alpha=0.7)
    ax4.fill_between(mm_df['step'], 0, mm_df['inventory'],
                     where=mm_df['inventory'] >= 0, color='#2ecc71', alpha=0.3, label='Long')
    ax4.fill_between(mm_df['step'], 0, mm_df['inventory'],
                     where=mm_df['inventory'] < 0, color='#e74c3c', alpha=0.3, label='Short')

    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Inventory')
    ax4.set_title('D) Market Maker Inventory Dynamics')
    ax4.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    plt.savefig(save_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')

    return fig


def compute_statistics(mm_df: pd.DataFrame) -> dict:
    """Compute comprehensive summary statistics."""
    from scipy import stats as scipy_stats

    # Correlations
    sentiment_spread_corr = mm_df['sentiment'].corr(mm_df['spread_bps'])
    epistemic_spread_corr = mm_df['epistemic_uncertainty'].corr(mm_df['spread_bps'])
    aleatoric_spread_corr = mm_df['aleatoric_uncertainty'].corr(mm_df['spread_bps'])
    total_uncertainty_spread_corr = mm_df['total_uncertainty'].corr(mm_df['spread_bps'])

    # Regime analysis using the regime column
    bullish_mask = mm_df['regime'] == 'bullish'
    bearish_mask = mm_df['regime'] == 'bearish'
    neutral_mask = mm_df['regime'] == 'neutral'

    # Return statistics (skip first observation which has 0 return)
    returns = mm_df['log_return'].iloc[1:]
    abs_returns = returns.abs()

    result = {
        # Basic stats
        'n_observations': len(mm_df),
        'mean_spread_bps': mm_df['spread_bps'].mean(),
        'std_spread_bps': mm_df['spread_bps'].std(),
        'median_spread_bps': mm_df['spread_bps'].median(),

        # Correlations
        'sentiment_spread_correlation': sentiment_spread_corr,
        'epistemic_spread_correlation': epistemic_spread_corr,
        'aleatoric_spread_correlation': aleatoric_spread_corr,
        'total_uncertainty_spread_correlation': total_uncertainty_spread_corr,

        # Regime breakdown
        'bullish_periods_pct': bullish_mask.mean() * 100,
        'neutral_periods_pct': neutral_mask.mean() * 100,
        'bearish_periods_pct': bearish_mask.mean() * 100,

        # Regime-conditional spreads
        'mean_spread_bullish': mm_df.loc[bullish_mask, 'spread_bps'].mean() if bullish_mask.any() else np.nan,
        'mean_spread_neutral': mm_df.loc[neutral_mask, 'spread_bps'].mean() if neutral_mask.any() else np.nan,
        'mean_spread_bearish': mm_df.loc[bearish_mask, 'spread_bps'].mean() if bearish_mask.any() else np.nan,

        # Inventory stats
        'inventory_volatility': mm_df['inventory'].std(),
        'max_inventory': mm_df['inventory'].abs().max(),
        'mean_inventory': mm_df['inventory'].mean(),

        # Return distribution
        'return_mean': returns.mean(),
        'return_std': returns.std(),
        'return_skewness': scipy_stats.skew(returns),
        'return_kurtosis': scipy_stats.kurtosis(returns),  # Excess kurtosis

        # Spread distribution
        'spread_skewness': scipy_stats.skew(mm_df['spread_bps']),
        'spread_kurtosis': scipy_stats.kurtosis(mm_df['spread_bps']),

        # Uncertainty stats
        'mean_epistemic': mm_df['epistemic_uncertainty'].mean(),
        'mean_aleatoric': mm_df['aleatoric_uncertainty'].mean(),
        'epistemic_std': mm_df['epistemic_uncertainty'].std(),
        'aleatoric_std': mm_df['aleatoric_uncertainty'].std(),

        # Price stats
        'price_start': mm_df['mid_price'].iloc[0],
        'price_end': mm_df['mid_price'].iloc[-1],
        'price_min': mm_df['mid_price'].min(),
        'price_max': mm_df['mid_price'].max(),
    }

    return result


def main():
    """Run preliminary results demo."""
    print("=" * 60)
    print("Sentiment-Microstructure ABM: Preliminary Results")
    print("=" * 60)
    print()

    # 1. Sentiment Analysis Demo
    print("1. Running sentiment analysis on sample texts...")
    texts = get_sample_crypto_texts()

    if REAL_ANALYZER:
        print("   Using real MC Dropout sentiment analyzer")
        sentiment_results = real_sentiment_analysis(texts)
    else:
        print("   Using synthetic sentiment data (for demo)")
        sentiment_results = synthetic_sentiment_analysis(texts)

    print("\n   Sentiment Analysis Results:")
    print("-" * 80)
    for i, r in enumerate(sentiment_results):
        print(f"   T{i+1}: {r.sentiment:+.3f} ± {r.total_uncertainty:.3f} "
              f"(epist: {r.epistemic_uncertainty:.3f}, aleat: {r.aleatoric_uncertainty:.3f})")
        print(f"       \"{r.text}\"")
    print()

    # 2. Market Maker Simulation
    print("2. Simulating market maker response to sentiment...")
    sentiment_series, epistemic_series, aleatoric_series, total_uncertainty, regime_series = \
        generate_sentiment_time_series(n_steps=2000, seed=42)
    mm_df = simulate_market_maker(
        sentiment_series, epistemic_series, aleatoric_series, regime_series, seed=42
    )
    print(f"   Generated {len(mm_df)} time steps of market maker behavior")
    print()

    # 3. Compute Statistics
    print("3. Computing summary statistics...")
    stats = compute_statistics(mm_df)
    print("\n   Summary Statistics:")
    print("-" * 60)
    print(f"   Observations:                    {stats['n_observations']}")
    print(f"   Mean Spread (bps):               {stats['mean_spread_bps']:.4f}")
    print(f"   Std Spread (bps):                {stats['std_spread_bps']:.4f}")
    print()
    print("   Correlations:")
    print(f"   Sentiment-Spread:                {stats['sentiment_spread_correlation']:.3f}")
    print(f"   Epistemic-Spread:                {stats['epistemic_spread_correlation']:.3f}")
    print(f"   Aleatoric-Spread:                {stats['aleatoric_spread_correlation']:.3f}")
    print(f"   Total Uncertainty-Spread:        {stats['total_uncertainty_spread_correlation']:.3f}")
    print()
    print("   Regime Breakdown:")
    print(f"   Bullish Periods (%):             {stats['bullish_periods_pct']:.1f}%")
    print(f"   Neutral Periods (%):             {stats['neutral_periods_pct']:.1f}%")
    print(f"   Bearish Periods (%):             {stats['bearish_periods_pct']:.1f}%")
    print()
    print("   Regime-Conditional Spreads:")
    print(f"   Mean Spread (Bullish):           {stats['mean_spread_bullish']:.4f} bps")
    print(f"   Mean Spread (Neutral):           {stats['mean_spread_neutral']:.4f} bps")
    print(f"   Mean Spread (Bearish):           {stats['mean_spread_bearish']:.4f} bps")
    print()
    print("   Return Distribution:")
    print(f"   Mean Return:                     {stats['return_mean']:.6f}")
    print(f"   Std Return:                      {stats['return_std']:.6f}")
    print(f"   Skewness:                        {stats['return_skewness']:.3f}")
    print(f"   Excess Kurtosis:                 {stats['return_kurtosis']:.3f}")
    print()
    print(f"   Inventory Volatility:            {stats['inventory_volatility']:.2f}")
    print()

    # 4. Generate Figure
    print("4. Generating preliminary results figure...")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(output_dir, 'preliminary_results.pdf')

    plot_preliminary_results(sentiment_results, mm_df, save_path=fig_path)
    print()

    # 5. Save Results
    print("5. Saving results...")
    results_path = os.path.join(output_dir, 'preliminary_results.csv')
    mm_df.to_csv(results_path, index=False)
    print(f"   Saved simulation data to: {results_path}")

    stats_path = os.path.join(output_dir, 'summary_statistics.txt')
    with open(stats_path, 'w') as f:
        f.write("Sentiment-Microstructure ABM: Preliminary Results\n")
        f.write("=" * 60 + "\n\n")
        f.write("Summary Statistics:\n")
        for k, v in stats.items():
            if isinstance(v, float):
                f.write(f"  {k}: {v:.6f}\n")
            else:
                f.write(f"  {k}: {v}\n")
    print(f"   Saved statistics to: {stats_path}")

    # 6. Save stats as JSON for LaTeX import
    import json
    json_path = os.path.join(output_dir, 'summary_statistics.json')
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"   Saved statistics JSON to: {json_path}")

    print()
    print("=" * 60)
    print("PRELIMINARY RESULTS COMPLETE")
    print("=" * 60)
    print(f"\nOutputs generated in: {output_dir}/")
    print("  - preliminary_results.pdf (figure)")
    print("  - preliminary_results.png (figure)")
    print("  - preliminary_results.csv (simulation data)")
    print("  - summary_statistics.txt (statistics)")
    print("  - summary_statistics.json (statistics for LaTeX)")

    return stats, mm_df, sentiment_results


if __name__ == "__main__":
    main()
