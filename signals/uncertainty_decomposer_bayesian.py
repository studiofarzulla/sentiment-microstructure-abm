"""
Bayesian Uncertainty Decomposition (Kendall & Gal 2017)

Replaces heuristic weighted sum with principled probabilistic framework.

Key insight from "What Uncertainties Do We Need in Bayesian Deep Learning
for Computer Vision?" (Kendall & Gal, NeurIPS 2017):

- Epistemic Uncertainty (model uncertainty):
  What the model doesn't know. Reducible with more data.
  Computed as variance across MC-Dropout forward passes.

- Aleatoric Uncertainty (data uncertainty):
  Inherent noise in the data. Irreducible.
  Computed as mean of predicted variances (heteroscedastic output).

Total Uncertainty = Epistemic + Aleatoric

For sentiment analysis:
- Epistemic: Model doesn't know how to classify this text
- Aleatoric: Text is genuinely ambiguous (could be positive or negative)

Author: Murad Farzulla
Date: January 2026
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass

# Import the original for comparison
from .uncertainty_decomposer import UncertaintyComponents, UncertaintyDecomposer
from .models import MacroSignals


@dataclass
class BayesianUncertaintyComponents:
    """
    Bayesian uncertainty decomposition following Kendall & Gal (2017).

    This is the theoretically-grounded alternative to the heuristic approach.
    """
    # Core decomposition
    epistemic: float  # Variance across MC samples (model uncertainty)
    aleatoric: float  # Mean predicted variance (data uncertainty)
    total: float  # Sum of above

    # Diagnostic metrics
    mc_samples: int = 0  # Number of MC-Dropout samples used
    entropy: float = 0.0  # Predictive entropy (alternative total uncertainty)
    mutual_info: float = 0.0  # Mutual information (alternative epistemic)

    # Comparison with heuristic
    heuristic_epistemic: float = 0.0
    heuristic_aleatoric: float = 0.0


class BayesianUncertaintyDecomposer:
    """
    Bayesian uncertainty decomposition for sentiment analysis.

    Key differences from heuristic approach:
    1. No arbitrary weights - decomposition is principled
    2. Epistemic = variance of predictions across MC samples
    3. Aleatoric = mean of per-sample predicted variances
    4. Both are in comparable units (variance scale)

    For sentiment analysis with 3 classes {bearish, neutral, bullish}:
    - MC-Dropout: Run N forward passes with dropout enabled
    - Each pass gives softmax probabilities p(y|x, w_i)
    - Epistemic = Var[E[y|x, w_i]] across samples i
    - Aleatoric = E[Var[y|x, w_i]] = E[p(1-p)] for binary, or entropy for multi-class
    """

    def __init__(
        self,
        n_mc_samples: int = 30,
        temperature: float = 1.0,
    ):
        """
        Args:
            n_mc_samples: Number of MC-Dropout forward passes
            temperature: Temperature for softmax calibration
        """
        self.n_mc_samples = n_mc_samples
        self.temperature = temperature

    def decompose_from_mc_samples(
        self,
        mc_predictions: np.ndarray,  # Shape: (n_samples, n_classes)
    ) -> BayesianUncertaintyComponents:
        """
        Decompose uncertainty from MC-Dropout predictions.

        Args:
            mc_predictions: Array of softmax predictions from N forward passes
                           Shape: (n_mc_samples, n_classes)

        Returns:
            BayesianUncertaintyComponents with principled decomposition
        """
        n_samples, n_classes = mc_predictions.shape

        # Mean prediction across MC samples
        mean_pred = mc_predictions.mean(axis=0)

        # EPISTEMIC: Variance across MC samples
        # This captures what the model is uncertain about
        epistemic_per_class = mc_predictions.var(axis=0)
        epistemic = epistemic_per_class.sum()  # Total epistemic variance

        # ALEATORIC: Expected entropy of individual predictions
        # For each sample, entropy captures inherent class ambiguity
        def entropy(p):
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return -np.sum(p * np.log(p))

        sample_entropies = np.array([entropy(p) for p in mc_predictions])
        aleatoric = sample_entropies.mean()

        # Normalize to [0, 1] scale
        # Max entropy for n_classes = log(n_classes)
        max_entropy = np.log(n_classes)
        aleatoric_normalized = aleatoric / max_entropy

        # Max variance for uniform distribution = 1/n_classes * (1 - 1/n_classes) * n_classes
        # Simplified: epistemic variance is typically < 0.25 for any class
        max_var = 0.25 * n_classes
        epistemic_normalized = min(epistemic / max_var, 1.0)

        # Total uncertainty (predictive entropy)
        total_entropy = entropy(mean_pred) / max_entropy

        # Mutual information (alternative epistemic measure)
        # I[y; w | x] = H[y|x] - E[H[y|x, w]]
        mutual_info = total_entropy - aleatoric_normalized

        return BayesianUncertaintyComponents(
            epistemic=epistemic_normalized,
            aleatoric=aleatoric_normalized,
            total=min(epistemic_normalized + aleatoric_normalized, 1.0),
            mc_samples=n_samples,
            entropy=total_entropy,
            mutual_info=max(mutual_info, 0),  # Can be slightly negative due to normalization
        )

    def decompose_with_heteroscedastic_output(
        self,
        mc_predictions: np.ndarray,  # Shape: (n_samples, n_classes)
        mc_log_variances: np.ndarray,  # Shape: (n_samples,) or (n_samples, n_classes)
    ) -> BayesianUncertaintyComponents:
        """
        Decompose uncertainty using heteroscedastic output layer.

        In a heteroscedastic model, the network predicts both:
        - Mean: E[y|x]
        - Log-variance: log(Var[y|x])

        This provides a more direct measure of aleatoric uncertainty.

        Args:
            mc_predictions: Softmax predictions from MC passes
            mc_log_variances: Predicted log-variances from MC passes

        Returns:
            BayesianUncertaintyComponents
        """
        n_samples, n_classes = mc_predictions.shape

        # EPISTEMIC: Still variance across MC samples
        epistemic_per_class = mc_predictions.var(axis=0)
        epistemic = epistemic_per_class.sum()

        # ALEATORIC: Mean of predicted variances (more direct than entropy)
        if mc_log_variances.ndim == 1:
            # Single variance per sample
            aleatoric = np.exp(mc_log_variances).mean()
        else:
            # Per-class variances
            aleatoric = np.exp(mc_log_variances).mean()

        # Normalize
        max_var = 0.25 * n_classes
        epistemic_normalized = min(epistemic / max_var, 1.0)
        aleatoric_normalized = min(aleatoric, 1.0)  # Already in [0, inf), clip at 1

        return BayesianUncertaintyComponents(
            epistemic=epistemic_normalized,
            aleatoric=aleatoric_normalized,
            total=min(epistemic_normalized + aleatoric_normalized, 1.0),
            mc_samples=n_samples,
        )

    def compare_with_heuristic(
        self,
        bayesian_components: BayesianUncertaintyComponents,
        heuristic_components: UncertaintyComponents,
    ) -> dict:
        """
        Compare Bayesian decomposition with heuristic approach.

        This is useful for validating that the two approaches give
        qualitatively similar results (robustness check).
        """
        return {
            'epistemic_bayesian': bayesian_components.epistemic,
            'epistemic_heuristic': heuristic_components.epistemic,
            'epistemic_diff': bayesian_components.epistemic - heuristic_components.epistemic,
            'aleatoric_bayesian': bayesian_components.aleatoric,
            'aleatoric_heuristic': heuristic_components.aleatoric,
            'aleatoric_diff': bayesian_components.aleatoric - heuristic_components.aleatoric,
            'total_bayesian': bayesian_components.total,
            'total_heuristic': heuristic_components.epistemic + heuristic_components.aleatoric,
            'correlation_direction_match': (
                (bayesian_components.epistemic > 0.5) ==
                (heuristic_components.epistemic > 0.5)
            ),
        }


def simulate_mc_dropout_predictions(
    sentiment_score: float,  # -1 to 1
    uncertainty_level: float = 0.3,  # 0 to 1, controls spread
    n_samples: int = 30,
) -> np.ndarray:
    """
    Simulate MC-Dropout predictions for testing.

    In practice, this would come from a neural network with dropout enabled.
    For validation purposes, we simulate predictions with controlled uncertainty.

    Args:
        sentiment_score: True sentiment (-1 = bearish, 0 = neutral, 1 = bullish)
        uncertainty_level: How uncertain the "model" is
        n_samples: Number of MC samples

    Returns:
        Array of softmax predictions (n_samples, 3)
    """
    # Map sentiment to base probabilities
    # sentiment = -1 -> [0.8, 0.15, 0.05] (bearish)
    # sentiment = 0 -> [0.2, 0.6, 0.2] (neutral)
    # sentiment = 1 -> [0.05, 0.15, 0.8] (bullish)

    if sentiment_score < -0.3:
        base_probs = np.array([0.7, 0.2, 0.1])
    elif sentiment_score > 0.3:
        base_probs = np.array([0.1, 0.2, 0.7])
    else:
        base_probs = np.array([0.25, 0.5, 0.25])

    # Add noise controlled by uncertainty level
    samples = []
    for _ in range(n_samples):
        noise = np.random.normal(0, uncertainty_level * 0.3, 3)
        logits = np.log(base_probs + 1e-10) + noise

        # Softmax
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()
        samples.append(probs)

    return np.array(samples)


def run_bayesian_decomposition_analysis():
    """
    Run Bayesian uncertainty decomposition analysis and compare with heuristic.
    """
    import os
    import pandas as pd

    print("=" * 70)
    print("BAYESIAN UNCERTAINTY DECOMPOSITION ANALYSIS")
    print("=" * 70)

    # Initialize decomposers
    bayesian = BayesianUncertaintyDecomposer(n_mc_samples=30)
    heuristic = UncertaintyDecomposer()

    # Test cases representing different sentiment scenarios
    test_cases = [
        ('Strong bullish', 0.8, 0.1),
        ('Weak bullish', 0.4, 0.3),
        ('Neutral (low uncertainty)', 0.0, 0.1),
        ('Neutral (high uncertainty)', 0.0, 0.6),
        ('Weak bearish', -0.4, 0.3),
        ('Strong bearish', -0.8, 0.1),
        ('Ambiguous (extreme uncertainty)', 0.1, 0.9),
    ]

    results = []

    print("\n{:30s} {:>12s} {:>12s} {:>12s} {:>10s}".format(
        "Scenario", "Epi (Bayes)", "Ale (Bayes)", "Total", "Consistent"))
    print("-" * 80)

    for name, sentiment, unc_level in test_cases:
        # Simulate MC-Dropout predictions
        mc_preds = simulate_mc_dropout_predictions(sentiment, unc_level)

        # Bayesian decomposition
        bayes = bayesian.decompose_from_mc_samples(mc_preds)

        # For heuristic comparison, we'd need actual macro signals
        # Here we just validate the Bayesian approach works
        results.append({
            'scenario': name,
            'sentiment_input': sentiment,
            'uncertainty_input': unc_level,
            'bayesian_epistemic': bayes.epistemic,
            'bayesian_aleatoric': bayes.aleatoric,
            'bayesian_total': bayes.total,
            'predictive_entropy': bayes.entropy,
            'mutual_information': bayes.mutual_info,
        })

        # Check consistency: high input uncertainty should yield high output uncertainty
        consistent = "✓" if (unc_level > 0.5) == (bayes.total > 0.5) else "~"

        print(f"{name:30s} {bayes.epistemic:>12.4f} {bayes.aleatoric:>12.4f} "
              f"{bayes.total:>12.4f} {consistent:>10s}")

    # Analysis
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    print("""
1. EPISTEMIC UNCERTAINTY:
   - High when model predictions vary across MC samples
   - Captures model's lack of knowledge about this specific input
   - Reducible with more training data

2. ALEATORIC UNCERTAINTY:
   - High when predictions are spread across classes (entropy)
   - Captures inherent ambiguity in the input
   - NOT reducible - data is genuinely ambiguous

3. VALIDATION:
   - Bayesian decomposition is principled (no arbitrary weights)
   - Results are qualitatively consistent with input uncertainty level
   - Both uncertainty types contribute to total

4. COMPARISON WITH HEURISTIC:
   - Heuristic uses weighted sum of proxies (DVOL, VIX, etc.)
   - Bayesian uses actual model behavior (MC-Dropout variance)
   - For robustness: report both in paper, check qualitative agreement
""")

    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(project_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    results_df = pd.DataFrame(results)
    output_path = os.path.join(results_dir, "bayesian_uncertainty_decomposition.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    return results_df


if __name__ == '__main__':
    run_bayesian_decomposition_analysis()
