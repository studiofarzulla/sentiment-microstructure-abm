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
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch.nn.functional as F
    REAL_ANALYZER = True
except ImportError:
    REAL_ANALYZER = False
    print("Note: Using synthetic sentiment data (transformers not loaded)")


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


def simulate_market_maker(sentiment_series: np.ndarray,
                          uncertainty_series: np.ndarray,
                          base_spread: float = 0.001,
                          sentiment_sensitivity: float = 0.5,
                          uncertainty_sensitivity: float = 1.0) -> pd.DataFrame:
    """
    Simulate market maker behavior responding to sentiment signals.

    Based on Avellaneda-Stoikov style quoting with sentiment adjustment.

    Returns DataFrame with:
    - mid_price: Simulated mid price
    - bid: Market maker bid
    - ask: Market maker ask
    - spread: Bid-ask spread
    - inventory: Market maker inventory
    """
    n = len(sentiment_series)

    # Initialize
    mid_price = 100.0  # Arbitrary starting price
    inventory = 0.0

    records = []

    for t in range(n):
        sentiment = sentiment_series[t]
        uncertainty = uncertainty_series[t]

        # 1. Adjust mid price based on sentiment (drift)
        # Positive sentiment -> price tends to rise
        price_drift = sentiment * 0.001  # Small drift per step
        mid_price *= (1 + price_drift)

        # 2. Calculate spread
        # Base spread + uncertainty premium + inventory skew
        uncertainty_premium = base_spread * uncertainty * uncertainty_sensitivity
        inventory_skew = -0.0001 * inventory  # Mean reversion

        spread = base_spread + uncertainty_premium

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
            'uncertainty': uncertainty,
            'mid_price': mid_price,
            'bid': bid,
            'ask': ask,
            'spread': ask - bid,
            'spread_bps': (ask - bid) / mid_price * 10000,
            'inventory': inventory
        })

    return pd.DataFrame(records)


def generate_sentiment_time_series(n_steps: int = 500,
                                   regime_change_prob: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic sentiment time series with regime changes.

    Returns (sentiment_series, uncertainty_series)
    """
    sentiment = np.zeros(n_steps)
    uncertainty = np.zeros(n_steps)

    # Start with neutral sentiment, moderate uncertainty
    current_sentiment = 0.0
    current_regime = 'neutral'  # 'bullish', 'bearish', 'neutral'

    for t in range(n_steps):
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

        # Uncertainty increases during regime transitions and extreme sentiment
        base_uncertainty = 0.15
        sentiment_uncertainty = abs(current_sentiment) * 0.1
        noise = np.random.uniform(0, 0.1)

        sentiment[t] = current_sentiment
        uncertainty[t] = base_uncertainty + sentiment_uncertainty + noise

    return sentiment, uncertainty


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
                          mm_df['sentiment'] - mm_df['uncertainty'],
                          mm_df['sentiment'] + mm_df['uncertainty'],
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
    """Compute summary statistics for supervisor presentation."""

    # Correlation between sentiment and spread
    sentiment_spread_corr = mm_df['sentiment'].corr(mm_df['spread_bps'])
    uncertainty_spread_corr = mm_df['uncertainty'].corr(mm_df['spread_bps'])

    # Regime analysis
    bullish_mask = mm_df['sentiment'] > 0.2
    bearish_mask = mm_df['sentiment'] < -0.2

    stats = {
        'n_observations': len(mm_df),
        'mean_spread_bps': mm_df['spread_bps'].mean(),
        'std_spread_bps': mm_df['spread_bps'].std(),
        'sentiment_spread_correlation': sentiment_spread_corr,
        'uncertainty_spread_correlation': uncertainty_spread_corr,
        'bullish_periods_pct': bullish_mask.mean() * 100,
        'bearish_periods_pct': bearish_mask.mean() * 100,
        'mean_spread_bullish': mm_df.loc[bullish_mask, 'spread_bps'].mean() if bullish_mask.any() else np.nan,
        'mean_spread_bearish': mm_df.loc[bearish_mask, 'spread_bps'].mean() if bearish_mask.any() else np.nan,
        'inventory_volatility': mm_df['inventory'].std(),
        'max_inventory': mm_df['inventory'].abs().max(),
    }

    return stats


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
    sentiment_series, uncertainty_series = generate_sentiment_time_series(n_steps=500)
    mm_df = simulate_market_maker(sentiment_series, uncertainty_series)
    print(f"   Generated {len(mm_df)} time steps of market maker behavior")
    print()

    # 3. Compute Statistics
    print("3. Computing summary statistics...")
    stats = compute_statistics(mm_df)
    print("\n   Summary Statistics:")
    print("-" * 50)
    print(f"   Observations:                    {stats['n_observations']}")
    print(f"   Mean Spread (bps):               {stats['mean_spread_bps']:.2f}")
    print(f"   Std Spread (bps):                {stats['std_spread_bps']:.2f}")
    print(f"   Sentiment-Spread Correlation:    {stats['sentiment_spread_correlation']:.3f}")
    print(f"   Uncertainty-Spread Correlation:  {stats['uncertainty_spread_correlation']:.3f}")
    print(f"   Bullish Periods (%):             {stats['bullish_periods_pct']:.1f}%")
    print(f"   Bearish Periods (%):             {stats['bearish_periods_pct']:.1f}%")
    print(f"   Mean Spread (Bullish):           {stats['mean_spread_bullish']:.2f} bps")
    print(f"   Mean Spread (Bearish):           {stats['mean_spread_bearish']:.2f} bps")
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
        f.write("=" * 50 + "\n\n")
        f.write("Summary Statistics:\n")
        for k, v in stats.items():
            f.write(f"  {k}: {v}\n")
    print(f"   Saved statistics to: {stats_path}")

    print()
    print("=" * 60)
    print("PRELIMINARY RESULTS COMPLETE")
    print("=" * 60)
    print(f"\nOutputs generated in: {output_dir}/")
    print("  - preliminary_results.pdf (figure)")
    print("  - preliminary_results.png (figure)")
    print("  - preliminary_results.csv (simulation data)")
    print("  - summary_statistics.txt (statistics)")

    return stats, mm_df, sentiment_results


if __name__ == "__main__":
    main()
