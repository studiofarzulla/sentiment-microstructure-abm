#!/usr/bin/env python3
"""
Ablation and Sensitivity Analysis Framework

Addresses reviewer concerns about missing ablation/sensitivity analyses on:
1. Market maker delta parameter (uncertainty_sensitivity)
2. Epistemic/aleatoric component weights in uncertainty decomposition
3. Macro/micro blending weights in signal composition

Outputs:
- Table showing how spread-uncertainty correlations change with delta
- Sensitivity heatmaps for weight variations
- LaTeX table for paper (ablation_results.tex)
- JSON summary for programmatic use

Author: Murad Farzulla
Date: January 2026
"""

import sys
from pathlib import Path
import json
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from simulation.market_env import CryptoMarketModel, MarketMakerAgent, InformedTraderAgent, NoiseTraderAgent
from signals.uncertainty_decomposer import UncertaintyDecomposer
from signals.signal_composer import SignalComposer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AblationConfig:
    """Configuration for ablation study."""

    # Simulation parameters
    n_steps: int = 1000
    n_replications: int = 5
    seed_base: int = 42

    # Delta (uncertainty_sensitivity) parameter sweep
    delta_values: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 1.5, 2.0, 2.5])

    # Component weight variations (relative to baseline)
    weight_variations: List[float] = field(default_factory=lambda: [-0.4, -0.2, 0.0, 0.2, 0.4])

    # Macro/micro blending weight sweep
    macro_weights: List[float] = field(default_factory=lambda: [0.0, 0.15, 0.30, 0.45, 0.60, 0.75])

    # Agent configuration
    n_market_makers: int = 2
    n_informed: int = 3
    n_noise: int = 10
    initial_price: float = 100.0


@dataclass
class AblationResult:
    """Results from a single ablation run."""

    config_name: str
    config_params: Dict[str, Any]

    # Core metrics
    spread_uncertainty_corr: float
    spread_epistemic_corr: float
    spread_aleatoric_corr: float

    # Volatility metrics
    return_volatility: float
    return_mean: float
    return_skewness: float
    return_kurtosis: float

    # Regime metrics
    regime_persistence_bullish: float
    regime_persistence_neutral: float
    regime_persistence_bearish: float
    mean_regime_duration: float

    # Spread statistics
    mean_spread_bps: float
    std_spread_bps: float
    median_spread_bps: float

    # Stylized facts
    has_volatility_clustering: bool
    acf_abs_returns_lag10: float
    jarque_bera_stat: float
    jarque_bera_pval: float

    # Metadata
    n_observations: int
    simulation_time_ms: float


# =============================================================================
# Synthetic Sentiment Generator
# =============================================================================

class SyntheticSentimentGenerator:
    """
    Generates synthetic sentiment with configurable uncertainty decomposition.

    Allows varying the epistemic/aleatoric weights and regime dynamics.
    Supports macro/micro blending to test blending weight ablation.
    """

    def __init__(
        self,
        decomposer: UncertaintyDecomposer,
        macro_weight: float = 0.3,
        regime_probs: Dict[str, float] = None,
    ):
        self.decomposer = decomposer
        self.macro_weight = macro_weight
        self.micro_weight = 1.0 - macro_weight
        self.regime_probs = regime_probs or {'bullish': 0.3, 'neutral': 0.4, 'bearish': 0.3}

        # State
        self._current_regime = 'neutral'
        self._regime_duration = 0
        self._min_regime_duration = 20

        # Simulated "institutional" sentiment (macro) - smoother, trend-following
        self._inst_sentiment = 0.0
        self._inst_momentum = 0.0

    def __call__(self, step: int) -> Tuple[float, float, float, str]:
        """Generate sentiment tick for given step."""
        # Update regime with persistence
        self._update_regime(step)

        # Generate retail (micro) sentiment based on regime - more volatile
        if self._current_regime == 'bullish':
            retail_sentiment = 0.4 + np.random.normal(0, 0.2)
        elif self._current_regime == 'bearish':
            retail_sentiment = -0.4 + np.random.normal(0, 0.2)
        else:
            retail_sentiment = np.random.normal(0, 0.25)

        retail_sentiment = np.clip(retail_sentiment, -1, 1)

        # Generate institutional (macro) sentiment - smoother, mean-reverting
        # Institutional sentiment follows regime but with less noise
        if self._current_regime == 'bullish':
            target_inst = 0.3
        elif self._current_regime == 'bearish':
            target_inst = -0.3
        else:
            target_inst = 0.0

        # Mean-revert toward target with momentum
        self._inst_momentum = 0.9 * self._inst_momentum + 0.1 * (target_inst - self._inst_sentiment)
        self._inst_sentiment += self._inst_momentum + np.random.normal(0, 0.02)
        self._inst_sentiment = np.clip(self._inst_sentiment, -1, 1)

        # Blend retail and institutional sentiment
        blended_sentiment = (
            self.micro_weight * retail_sentiment +
            self.macro_weight * self._inst_sentiment
        )
        blended_sentiment = np.clip(blended_sentiment, -1, 1)

        # Generate raw micro uncertainties (before decomposition)
        # Higher macro weight = lower uncertainty (more informed)
        base_epi = 0.05 + abs(blended_sentiment) * 0.03
        base_ale = 0.3 + (1 - abs(blended_sentiment)) * 0.2

        # Macro weight reduces epistemic uncertainty (more informed)
        micro_epistemic = base_epi * (1 - 0.3 * self.macro_weight) + np.random.exponential(0.02)
        micro_aleatoric = base_ale + np.random.exponential(0.05)

        # Apply decomposition weights
        epistemic, aleatoric = self.decomposer.decompose(
            macro_signals=None,
            micro_epistemic=micro_epistemic,
            micro_aleatoric=micro_aleatoric,
            regime=self._current_regime,
        )

        return blended_sentiment, epistemic, aleatoric, self._current_regime

    def _update_regime(self, step: int):
        """Update regime with persistence dynamics."""
        self._regime_duration += 1

        # Only allow regime change after minimum duration
        if self._regime_duration < self._min_regime_duration:
            return

        # Probabilistic regime transition
        if np.random.random() < 0.05:  # 5% chance of regime change
            regimes = list(self.regime_probs.keys())
            weights = list(self.regime_probs.values())
            self._current_regime = np.random.choice(regimes, p=weights)
            self._regime_duration = 0


# =============================================================================
# Ablation Runner
# =============================================================================

class AblationRunner:
    """Runs ablation studies with different parameter configurations."""

    def __init__(self, config: AblationConfig):
        self.config = config
        self.results: List[AblationResult] = []

    def run_delta_ablation(self, verbose: bool = True) -> pd.DataFrame:
        """
        Run ablation on delta (uncertainty_sensitivity) parameter.

        Tests: delta = [0, 0.5, 1.0, 1.5, 2.0, 2.5]
        delta=0 serves as baseline (no uncertainty-based spread widening)
        """
        if verbose:
            print("\n" + "=" * 60)
            print("DELTA ABLATION STUDY")
            print("=" * 60)
            print(f"Testing delta values: {self.config.delta_values}")
            print(f"Replications per config: {self.config.n_replications}")

        results = []

        for delta in self.config.delta_values:
            if verbose:
                print(f"\n--- Delta = {delta} ---")

            replication_results = []
            for rep in range(self.config.n_replications):
                seed = self.config.seed_base + rep
                result = self._run_single_simulation(
                    config_name=f"delta_{delta}",
                    delta=delta,
                    seed=seed,
                )
                replication_results.append(result)

            # Aggregate across replications
            agg = self._aggregate_results(replication_results)
            agg['delta'] = delta
            results.append(agg)

            if verbose:
                print(f"  Spread-Uncertainty Corr: {agg['spread_uncertainty_corr']:.4f} +/- {agg['spread_uncertainty_corr_std']:.4f}")
                print(f"  Return Volatility: {agg['return_volatility']:.6f}")
                print(f"  Mean Spread: {agg['mean_spread_bps']:.4f} bps")

        df = pd.DataFrame(results)
        df = df.set_index('delta')
        return df

    def run_component_weight_ablation(self, verbose: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Run ablation on epistemic/aleatoric component weights.

        Varies weights by +/- 20% and +/- 40% from baseline.
        """
        if verbose:
            print("\n" + "=" * 60)
            print("COMPONENT WEIGHT ABLATION STUDY")
            print("=" * 60)

        # Baseline weights from UncertaintyDecomposer defaults
        baseline_weights = {
            'w_regulatory': 0.35,
            'w_data_missing': 0.25,
            'w_mc_variance': 0.40,
            'w_vix': 0.35,
            'w_peg': 0.25,
            'w_tvl': 0.15,
            'w_entropy': 0.25,
        }

        results = {'epistemic': [], 'aleatoric': []}

        # Epistemic weight variations
        if verbose:
            print("\n--- Epistemic Weight Variations ---")

        for variation in self.config.weight_variations:
            # Apply variation to epistemic weights
            modified_weights = baseline_weights.copy()
            modified_weights['w_mc_variance'] = baseline_weights['w_mc_variance'] * (1 + variation)

            # Renormalize epistemic weights
            epi_total = modified_weights['w_regulatory'] + modified_weights['w_data_missing'] + modified_weights['w_mc_variance']
            modified_weights['w_regulatory'] /= epi_total
            modified_weights['w_data_missing'] /= epi_total
            modified_weights['w_mc_variance'] /= epi_total

            replication_results = []
            for rep in range(self.config.n_replications):
                seed = self.config.seed_base + rep
                result = self._run_single_simulation(
                    config_name=f"epi_var_{int(variation*100)}",
                    custom_decomposer_weights=modified_weights,
                    seed=seed,
                )
                replication_results.append(result)

            agg = self._aggregate_results(replication_results)
            agg['variation'] = variation
            agg['variation_pct'] = int(variation * 100)
            results['epistemic'].append(agg)

            if verbose:
                print(f"  Variation {int(variation*100):+d}%: Spread-Epi Corr = {agg['spread_epistemic_corr']:.4f}")

        # Aleatoric weight variations
        if verbose:
            print("\n--- Aleatoric Weight Variations ---")

        for variation in self.config.weight_variations:
            # Apply variation to aleatoric weights
            modified_weights = baseline_weights.copy()
            modified_weights['w_entropy'] = baseline_weights['w_entropy'] * (1 + variation)

            # Renormalize aleatoric weights
            ale_total = modified_weights['w_vix'] + modified_weights['w_peg'] + modified_weights['w_tvl'] + modified_weights['w_entropy']
            modified_weights['w_vix'] /= ale_total
            modified_weights['w_peg'] /= ale_total
            modified_weights['w_tvl'] /= ale_total
            modified_weights['w_entropy'] /= ale_total

            replication_results = []
            for rep in range(self.config.n_replications):
                seed = self.config.seed_base + rep
                result = self._run_single_simulation(
                    config_name=f"ale_var_{int(variation*100)}",
                    custom_decomposer_weights=modified_weights,
                    seed=seed,
                )
                replication_results.append(result)

            agg = self._aggregate_results(replication_results)
            agg['variation'] = variation
            agg['variation_pct'] = int(variation * 100)
            results['aleatoric'].append(agg)

            if verbose:
                print(f"  Variation {int(variation*100):+d}%: Spread-Ale Corr = {agg['spread_aleatoric_corr']:.4f}")

        return {
            'epistemic': pd.DataFrame(results['epistemic']),
            'aleatoric': pd.DataFrame(results['aleatoric']),
        }

    def run_blending_weight_ablation(self, verbose: bool = True) -> pd.DataFrame:
        """
        Run ablation on macro/micro blending weights.

        Varies macro_weight from 0.0 (micro-only) to 0.75 (macro-heavy).
        """
        if verbose:
            print("\n" + "=" * 60)
            print("BLENDING WEIGHT ABLATION STUDY")
            print("=" * 60)
            print(f"Testing macro_weight values: {self.config.macro_weights}")

        results = []

        for macro_weight in self.config.macro_weights:
            if verbose:
                print(f"\n--- Macro Weight = {macro_weight:.2f} ---")

            replication_results = []
            for rep in range(self.config.n_replications):
                seed = self.config.seed_base + rep
                result = self._run_single_simulation(
                    config_name=f"macro_{int(macro_weight*100)}",
                    macro_weight=macro_weight,
                    seed=seed,
                )
                replication_results.append(result)

            agg = self._aggregate_results(replication_results)
            agg['macro_weight'] = macro_weight
            agg['micro_weight'] = 1.0 - macro_weight
            results.append(agg)

            if verbose:
                print(f"  Spread-Uncertainty Corr: {agg['spread_uncertainty_corr']:.4f}")
                print(f"  Mean Spread: {agg['mean_spread_bps']:.4f} bps")

        df = pd.DataFrame(results)
        df = df.set_index('macro_weight')
        return df

    def _run_single_simulation(
        self,
        config_name: str,
        delta: float = 1.5,
        custom_decomposer_weights: Dict[str, float] = None,
        macro_weight: float = 0.3,
        seed: int = 42,
    ) -> AblationResult:
        """Run a single simulation with given configuration."""
        start_time = datetime.now()

        np.random.seed(seed)

        # Create custom decomposer if weights specified
        if custom_decomposer_weights:
            decomposer = UncertaintyDecomposer(
                weight_regulatory_opacity=custom_decomposer_weights.get('w_regulatory', 0.35),
                weight_data_missing=custom_decomposer_weights.get('w_data_missing', 0.25),
                weight_mc_variance=custom_decomposer_weights.get('w_mc_variance', 0.40),
                weight_vix=custom_decomposer_weights.get('w_vix', 0.35),
                weight_peg=custom_decomposer_weights.get('w_peg', 0.25),
                weight_tvl=custom_decomposer_weights.get('w_tvl', 0.15),
                weight_entropy=custom_decomposer_weights.get('w_entropy', 0.25),
            )
        else:
            decomposer = UncertaintyDecomposer()

        # Create sentiment generator with macro/micro blending
        sentiment_gen = SyntheticSentimentGenerator(
            decomposer=decomposer,
            macro_weight=macro_weight,
        )

        # Create market model
        model = CryptoMarketModel(
            initial_price=self.config.initial_price,
            seed=seed,
        )

        # Add market makers with specified delta
        for i in range(self.config.n_market_makers):
            mm = MarketMakerAgent(
                f"mm_{i}",
                model,
                base_spread_bps=8.0 + i * 2,
                uncertainty_sensitivity=delta,  # Key parameter
                quote_size=0.5,
            )
            model.add_agent(mm)

        # Add informed traders
        for i in range(self.config.n_informed):
            informed = InformedTraderAgent(
                f"informed_{i}",
                model,
                sentiment_threshold=0.2 + i * 0.1,
                trade_size=0.3,
            )
            model.add_agent(informed)

        # Add noise traders
        for i in range(self.config.n_noise):
            noise = NoiseTraderAgent(
                f"noise_{i}",
                model,
                trade_probability=0.2,
            )
            model.add_agent(noise)

        # Run simulation
        history = model.run_simulation(
            self.config.n_steps,
            sentiment_generator=sentiment_gen,
        )

        # Convert to DataFrame
        df = pd.DataFrame(history)

        # Compute derived columns
        df['log_return'] = np.log(df['mid_price'] / df['mid_price'].shift(1))
        df['total_uncertainty'] = df['epistemic_uncertainty'] + df['aleatoric_uncertainty']
        df = df.dropna()

        # Compute metrics
        result = self._compute_metrics(df, config_name, delta, seed, start_time)

        return result

    def _compute_metrics(
        self,
        df: pd.DataFrame,
        config_name: str,
        delta: float,
        seed: int,
        start_time: datetime,
    ) -> AblationResult:
        """Compute all metrics from simulation DataFrame."""

        # Filter valid spread values
        spread_valid = df['spread_bps'].dropna()

        # Correlations
        spread_unc_corr = df['spread_bps'].corr(df['total_uncertainty']) if len(df) > 10 else 0.0
        spread_epi_corr = df['spread_bps'].corr(df['epistemic_uncertainty']) if len(df) > 10 else 0.0
        spread_ale_corr = df['spread_bps'].corr(df['aleatoric_uncertainty']) if len(df) > 10 else 0.0

        # Return statistics
        returns = df['log_return'].dropna()
        if len(returns) > 10:
            return_vol = returns.std()
            return_mean = returns.mean()
            return_skew = stats.skew(returns)
            return_kurt = stats.kurtosis(returns)
        else:
            return_vol = return_mean = return_skew = return_kurt = 0.0

        # Regime persistence
        regime_persistence = self._compute_regime_persistence(df)

        # Mean regime duration
        mean_duration = self._compute_mean_regime_duration(df)

        # Spread statistics
        mean_spread = spread_valid.mean() if len(spread_valid) > 0 else 0.0
        std_spread = spread_valid.std() if len(spread_valid) > 0 else 0.0
        median_spread = spread_valid.median() if len(spread_valid) > 0 else 0.0

        # Stylized facts
        acf_lag10 = self._compute_acf(returns.abs(), lag=10) if len(returns) > 20 else 0.0
        has_vol_clustering = acf_lag10 > 0.1

        # Jarque-Bera test
        if len(returns) > 10:
            jb_stat, jb_pval = stats.jarque_bera(returns)
        else:
            jb_stat, jb_pval = 0.0, 1.0

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

        return AblationResult(
            config_name=config_name,
            config_params={'delta': delta, 'seed': seed},
            spread_uncertainty_corr=spread_unc_corr,
            spread_epistemic_corr=spread_epi_corr,
            spread_aleatoric_corr=spread_ale_corr,
            return_volatility=return_vol,
            return_mean=return_mean,
            return_skewness=return_skew,
            return_kurtosis=return_kurt,
            regime_persistence_bullish=regime_persistence.get('bullish', 0.0),
            regime_persistence_neutral=regime_persistence.get('neutral', 0.0),
            regime_persistence_bearish=regime_persistence.get('bearish', 0.0),
            mean_regime_duration=mean_duration,
            mean_spread_bps=mean_spread,
            std_spread_bps=std_spread,
            median_spread_bps=median_spread,
            has_volatility_clustering=has_vol_clustering,
            acf_abs_returns_lag10=acf_lag10,
            jarque_bera_stat=jb_stat,
            jarque_bera_pval=jb_pval,
            n_observations=len(df),
            simulation_time_ms=elapsed_ms,
        )

    def _compute_regime_persistence(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute regime persistence (diagonal of transition matrix)."""
        regimes = ['bullish', 'neutral', 'bearish']
        persistence = {}

        for regime in regimes:
            mask = df['regime'] == regime
            if mask.sum() < 2:
                persistence[regime] = 0.0
                continue

            # Compute P(regime_t+1 = regime | regime_t = regime)
            regime_series = df['regime']
            transitions = sum(
                (regime_series.iloc[i] == regime) and (regime_series.iloc[i+1] == regime)
                for i in range(len(regime_series) - 1)
                if regime_series.iloc[i] == regime
            )
            total = sum(regime_series.iloc[i] == regime for i in range(len(regime_series) - 1))

            persistence[regime] = transitions / total if total > 0 else 0.0

        return persistence

    def _compute_mean_regime_duration(self, df: pd.DataFrame) -> float:
        """Compute mean regime duration in steps."""
        regime_changes = df['regime'] != df['regime'].shift(1)
        regime_changes.iloc[0] = True

        episode_starts = df.index[regime_changes].tolist()
        episode_starts.append(len(df))

        durations = [episode_starts[i+1] - episode_starts[i] for i in range(len(episode_starts) - 1)]

        return np.mean(durations) if durations else 0.0

    def _compute_acf(self, series: pd.Series, lag: int) -> float:
        """Compute autocorrelation at specified lag."""
        if len(series) <= lag:
            return 0.0
        return series.autocorr(lag=lag)

    def _aggregate_results(self, results: List[AblationResult]) -> Dict[str, Any]:
        """Aggregate results across replications."""
        # Extract metrics into arrays
        metrics = {
            'spread_uncertainty_corr': [],
            'spread_epistemic_corr': [],
            'spread_aleatoric_corr': [],
            'return_volatility': [],
            'return_mean': [],
            'return_skewness': [],
            'return_kurtosis': [],
            'regime_persistence_bullish': [],
            'regime_persistence_neutral': [],
            'regime_persistence_bearish': [],
            'mean_regime_duration': [],
            'mean_spread_bps': [],
            'std_spread_bps': [],
            'acf_abs_returns_lag10': [],
            'jarque_bera_stat': [],
        }

        for r in results:
            for key in metrics:
                metrics[key].append(getattr(r, key))

        # Compute mean and std
        agg = {}
        for key, values in metrics.items():
            agg[key] = np.mean(values)
            agg[f'{key}_std'] = np.std(values)

        # Count stylized facts
        agg['pct_volatility_clustering'] = sum(r.has_volatility_clustering for r in results) / len(results)
        agg['n_replications'] = len(results)

        return agg


# =============================================================================
# Results Export
# =============================================================================

class AblationExporter:
    """Export ablation results to various formats."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_delta_table_latex(
        self,
        df: pd.DataFrame,
        filename: str = "ablation_delta_results.tex",
    ) -> str:
        """Export delta ablation results as LaTeX table."""

        # Select and rename columns for paper
        table_df = df[[
            'spread_uncertainty_corr',
            'spread_epistemic_corr',
            'spread_aleatoric_corr',
            'return_volatility',
            'mean_spread_bps',
            'mean_regime_duration',
        ]].copy()

        table_df.columns = [
            r'$\rho(\text{Spread}, U)$',
            r'$\rho(\text{Spread}, \epsilon)$',
            r'$\rho(\text{Spread}, \alpha)$',
            r'$\sigma_r$',
            r'$\bar{s}$ (bps)',
            r'$\bar{D}_{\text{regime}}$',
        ]

        table_df.index.name = r'$\delta$'

        latex = table_df.to_latex(
            float_format='%.4f',
            escape=False,
            caption=(
                r'Ablation study: Effect of uncertainty sensitivity parameter $\delta$ on market dynamics. '
                r'$\delta=0$ represents baseline (no uncertainty-based spread widening). '
                r'$\rho(\cdot)$ denotes Pearson correlation, $\sigma_r$ return volatility, '
                r'$\bar{s}$ mean spread, $\bar{D}$ mean regime duration.'
            ),
            label='tab:ablation_delta',
            column_format='l' + 'r' * len(table_df.columns),
        )

        # Write to file
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            f.write(latex)

        return latex

    def export_weight_sensitivity_latex(
        self,
        epi_df: pd.DataFrame,
        ale_df: pd.DataFrame,
        filename: str = "ablation_weight_sensitivity.tex",
    ) -> str:
        """Export component weight sensitivity as LaTeX table."""

        # Combine epistemic and aleatoric results
        epi_df = epi_df.copy()
        ale_df = ale_df.copy()

        epi_df['component'] = 'Epistemic'
        ale_df['component'] = 'Aleatoric'

        combined = pd.concat([epi_df, ale_df], ignore_index=True)

        # Pivot for nice table
        pivot = combined.pivot(
            index='variation_pct',
            columns='component',
            values=['spread_epistemic_corr', 'spread_aleatoric_corr', 'return_volatility']
        )

        # Flatten column names
        pivot.columns = [f'{val}_{comp}' for val, comp in pivot.columns]
        pivot.index.name = 'Variation (%)'

        latex = pivot.to_latex(
            float_format='%.4f',
            escape=False,
            caption=(
                r'Sensitivity analysis: Effect of $\pm 20\%$ and $\pm 40\%$ variations in '
                r'epistemic and aleatoric component weights on spread correlations and volatility.'
            ),
            label='tab:weight_sensitivity',
        )

        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            f.write(latex)

        return latex

    def export_blending_table_latex(
        self,
        df: pd.DataFrame,
        filename: str = "ablation_blending_results.tex",
    ) -> str:
        """Export blending weight ablation results as LaTeX table."""

        table_df = df[[
            'spread_uncertainty_corr',
            'return_volatility',
            'mean_spread_bps',
            'regime_persistence_neutral',
        ]].copy()

        table_df.columns = [
            r'$\rho(\text{Spread}, U)$',
            r'$\sigma_r$',
            r'$\bar{s}$ (bps)',
            r'Regime Persist.',
        ]

        # Add micro weight column
        table_df.insert(0, r'$w_{\text{micro}}$', 1.0 - table_df.index)
        table_df.index.name = r'$w_{\text{macro}}$'

        latex = table_df.to_latex(
            float_format='%.4f',
            escape=False,
            caption=(
                r'Ablation study: Effect of macro/micro blending weights on market dynamics. '
                r'$w_{\text{macro}}=0$ corresponds to CryptoBERT-only signals.'
            ),
            label='tab:ablation_blending',
        )

        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            f.write(latex)

        return latex

    def export_json_summary(
        self,
        delta_results: pd.DataFrame,
        weight_results: Dict[str, pd.DataFrame],
        blending_results: pd.DataFrame,
        filename: str = "ablation_summary.json",
    ) -> dict:
        """Export all results as JSON for programmatic use."""

        summary = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'description': 'Ablation and sensitivity analysis results',
            },
            'delta_ablation': {
                'description': 'Effect of uncertainty_sensitivity (delta) parameter',
                'results': delta_results.reset_index().to_dict(orient='records'),
                'key_finding': self._summarize_delta_finding(delta_results),
            },
            'component_weight_sensitivity': {
                'description': 'Effect of epistemic/aleatoric weight variations',
                'epistemic': weight_results['epistemic'].to_dict(orient='records'),
                'aleatoric': weight_results['aleatoric'].to_dict(orient='records'),
            },
            'blending_ablation': {
                'description': 'Effect of macro/micro blending weights',
                'results': blending_results.reset_index().to_dict(orient='records'),
            },
        }

        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        return summary

    def _summarize_delta_finding(self, df: pd.DataFrame) -> str:
        """Generate key finding summary for delta ablation."""
        baseline_corr = df.loc[0.0, 'spread_uncertainty_corr'] if 0.0 in df.index else df.iloc[0]['spread_uncertainty_corr']
        best_delta = df['spread_uncertainty_corr'].idxmax()
        best_corr = df.loc[best_delta, 'spread_uncertainty_corr']

        return (
            f"Baseline (delta=0): spread-uncertainty correlation = {baseline_corr:.4f}. "
            f"Optimal delta={best_delta}: correlation = {best_corr:.4f}."
        )

    def export_combined_latex(
        self,
        delta_df: pd.DataFrame,
        weight_results: Dict[str, pd.DataFrame],
        blending_df: pd.DataFrame,
        filename: str = "ablation_results.tex",
    ) -> str:
        """
        Export combined ablation results as a single LaTeX file for the paper.

        This is the main output file for addressing reviewer concerns.
        """
        latex_content = r"""%% Ablation and Sensitivity Analysis Results
%% Auto-generated by ablation_analysis.py
%% For: Sentiment-Microstructure ABM Paper

\begin{table}[htbp]
\centering
\caption{Ablation Study: Uncertainty Sensitivity Parameter ($\delta$)}
\label{tab:ablation_delta}
\small
\begin{tabular}{@{}lcccccc@{}}
\toprule
$\delta$ & $\rho(\text{Spread}, U)$ & $\rho(\text{Spread}, \epsilon)$ & $\rho(\text{Spread}, \alpha)$ & $\sigma_r$ & $\bar{s}$ (bps) & $\bar{D}$ \\
\midrule
"""
        # Add delta rows
        for delta in delta_df.index:
            row = delta_df.loc[delta]
            latex_content += (
                f"{delta:.1f} & {row['spread_uncertainty_corr']:.4f} & "
                f"{row['spread_epistemic_corr']:.4f} & {row['spread_aleatoric_corr']:.4f} & "
                f"{row['return_volatility']:.6f} & {row['mean_spread_bps']:.2f} & "
                f"{row['mean_regime_duration']:.1f} \\\\\n"
            )

        latex_content += r"""\bottomrule
\end{tabular}

\vspace{0.5em}
\raggedright\footnotesize
\textit{Notes:} $\delta=0$ represents baseline (no uncertainty-based spread widening).
$\rho(\cdot)$ denotes Pearson correlation, $U$ total uncertainty, $\epsilon$ epistemic,
$\alpha$ aleatoric. $\sigma_r$ is return volatility, $\bar{s}$ mean spread,
$\bar{D}$ mean regime duration (steps). Results averaged over """ + str(delta_df['n_replications'].iloc[0] if 'n_replications' in delta_df.columns else 5) + r""" replications.
\end{table}

\begin{table}[htbp]
\centering
\caption{Sensitivity Analysis: Uncertainty Component Weight Variations}
\label{tab:weight_sensitivity}
\small
\begin{tabular}{@{}lcccc@{}}
\toprule
Variation & \multicolumn{2}{c}{Epistemic Weights} & \multicolumn{2}{c}{Aleatoric Weights} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
(\%) & $\rho(\text{Spread}, \epsilon)$ & $\sigma_r$ & $\rho(\text{Spread}, \alpha)$ & $\sigma_r$ \\
\midrule
"""
        # Add weight sensitivity rows
        epi_df = weight_results['epistemic']
        ale_df = weight_results['aleatoric']

        for idx, epi_row in epi_df.iterrows():
            var_pct = int(epi_row['variation_pct'])
            ale_row = ale_df[ale_df['variation_pct'] == var_pct].iloc[0]
            latex_content += (
                f"{var_pct:+d} & {epi_row['spread_epistemic_corr']:.4f} & "
                f"{epi_row['return_volatility']:.6f} & "
                f"{ale_row['spread_aleatoric_corr']:.4f} & {ale_row['return_volatility']:.6f} \\\\\n"
            )

        latex_content += r"""\bottomrule
\end{tabular}

\vspace{0.5em}
\raggedright\footnotesize
\textit{Notes:} Weight variations of $\pm 20\%$ and $\pm 40\%$ from baseline values.
Epistemic weights affect regulatory opacity and MC dropout variance contributions.
Aleatoric weights affect VIX, peg stability, and entropy contributions.
\end{table}

\begin{table}[htbp]
\centering
\caption{Ablation Study: Macro/Micro Sentiment Blending Weights}
\label{tab:ablation_blending}
\small
\begin{tabular}{@{}ccccc@{}}
\toprule
$w_{\text{macro}}$ & $w_{\text{micro}}$ & $\rho(\text{Spread}, U)$ & $\sigma_r$ & $\bar{s}$ (bps) \\
\midrule
"""
        # Add blending rows
        for macro_w in blending_df.index:
            row = blending_df.loc[macro_w]
            micro_w = 1.0 - macro_w
            latex_content += (
                f"{macro_w:.2f} & {micro_w:.2f} & {row['spread_uncertainty_corr']:.4f} & "
                f"{row['return_volatility']:.6f} & {row['mean_spread_bps']:.2f} \\\\\n"
            )

        latex_content += r"""\bottomrule
\end{tabular}

\vspace{0.5em}
\raggedright\footnotesize
\textit{Notes:} $w_{\text{macro}}=0$ corresponds to CryptoBERT-only (retail) signals.
Higher macro weights incorporate smoother institutional sentiment from ASRI.
$w_{\text{macro}}=0.30$ represents the default configuration used in main results.
\end{table}
"""
        # Write to file
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            f.write(latex_content)

        return latex_content

    def plot_delta_sensitivity(
        self,
        df: pd.DataFrame,
        filename: str = "ablation_delta_sensitivity.pdf",
    ) -> None:
        """Generate delta sensitivity plot."""

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Plot 1: Correlation vs Delta
        ax1 = axes[0]
        ax1.plot(df.index, df['spread_uncertainty_corr'], 'b-o', linewidth=2, markersize=8)
        ax1.fill_between(
            df.index,
            df['spread_uncertainty_corr'] - df['spread_uncertainty_corr_std'],
            df['spread_uncertainty_corr'] + df['spread_uncertainty_corr_std'],
            alpha=0.3,
        )
        ax1.set_xlabel(r'$\delta$ (Uncertainty Sensitivity)')
        ax1.set_ylabel(r'$\rho$(Spread, Uncertainty)')
        ax1.set_title('(a) Spread-Uncertainty Correlation')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Volatility vs Delta
        ax2 = axes[1]
        ax2.plot(df.index, df['return_volatility'] * 100, 'r-s', linewidth=2, markersize=8)
        ax2.fill_between(
            df.index,
            (df['return_volatility'] - df['return_volatility_std']) * 100,
            (df['return_volatility'] + df['return_volatility_std']) * 100,
            alpha=0.3, color='red',
        )
        ax2.set_xlabel(r'$\delta$ (Uncertainty Sensitivity)')
        ax2.set_ylabel('Return Volatility (%)')
        ax2.set_title('(b) Return Volatility')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Spread vs Delta
        ax3 = axes[2]
        ax3.plot(df.index, df['mean_spread_bps'], 'g-^', linewidth=2, markersize=8)
        ax3.fill_between(
            df.index,
            df['mean_spread_bps'] - df['std_spread_bps'],
            df['mean_spread_bps'] + df['std_spread_bps'],
            alpha=0.3, color='green',
        )
        ax3.set_xlabel(r'$\delta$ (Uncertainty Sensitivity)')
        ax3.set_ylabel('Mean Spread (bps)')
        ax3.set_title('(c) Mean Bid-Ask Spread')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_weight_heatmap(
        self,
        epi_df: pd.DataFrame,
        ale_df: pd.DataFrame,
        filename: str = "ablation_weight_heatmap.pdf",
    ) -> None:
        """Generate sensitivity heatmap for weight variations."""

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Epistemic heatmap
        epi_pivot = epi_df.pivot_table(
            index='variation_pct',
            values=['spread_epistemic_corr', 'return_volatility'],
            aggfunc='mean',
        )

        sns.heatmap(
            epi_pivot,
            ax=axes[0],
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            center=0,
        )
        axes[0].set_title('(a) Epistemic Weight Sensitivity')
        axes[0].set_ylabel('Weight Variation (%)')

        # Aleatoric heatmap
        ale_pivot = ale_df.pivot_table(
            index='variation_pct',
            values=['spread_aleatoric_corr', 'return_volatility'],
            aggfunc='mean',
        )

        sns.heatmap(
            ale_pivot,
            ax=axes[1],
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            center=0,
        )
        axes[1].set_title('(b) Aleatoric Weight Sensitivity')
        axes[1].set_ylabel('Weight Variation (%)')

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, bbox_inches='tight', dpi=300)
        plt.close()


# =============================================================================
# Main Entry Point
# =============================================================================

def run_full_ablation_study(
    output_dir: Path = None,
    n_steps: int = 1000,
    n_replications: int = 5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run complete ablation study and generate all outputs.

    Args:
        output_dir: Directory for outputs (defaults to analysis/results/)
        n_steps: Simulation steps per run
        n_replications: Number of replications for statistical robustness
        verbose: Print progress

    Returns:
        Dictionary with all results DataFrames
    """
    if output_dir is None:
        output_dir = project_root / 'analysis' / 'results'

    output_dir = Path(output_dir)

    # Create config
    config = AblationConfig(
        n_steps=n_steps,
        n_replications=n_replications,
    )

    # Initialize runner
    runner = AblationRunner(config)

    if verbose:
        print("\n" + "=" * 70)
        print("ABLATION AND SENSITIVITY ANALYSIS")
        print("=" * 70)
        print(f"Output directory: {output_dir}")
        print(f"Simulation steps: {config.n_steps}")
        print(f"Replications: {config.n_replications}")
        print(f"Timestamp: {datetime.now().isoformat()}")

    # Run ablations
    delta_results = runner.run_delta_ablation(verbose=verbose)
    weight_results = runner.run_component_weight_ablation(verbose=verbose)
    blending_results = runner.run_blending_weight_ablation(verbose=verbose)

    # Export results
    exporter = AblationExporter(output_dir)

    if verbose:
        print("\n" + "=" * 60)
        print("EXPORTING RESULTS")
        print("=" * 60)

    # Generate LaTeX tables
    exporter.export_delta_table_latex(delta_results)
    if verbose:
        print(f"  Saved: {output_dir / 'ablation_delta_results.tex'}")

    exporter.export_weight_sensitivity_latex(
        weight_results['epistemic'],
        weight_results['aleatoric'],
    )
    if verbose:
        print(f"  Saved: {output_dir / 'ablation_weight_sensitivity.tex'}")

    exporter.export_blending_table_latex(blending_results)
    if verbose:
        print(f"  Saved: {output_dir / 'ablation_blending_results.tex'}")

    # Generate plots
    exporter.plot_delta_sensitivity(delta_results)
    if verbose:
        print(f"  Saved: {output_dir / 'ablation_delta_sensitivity.pdf'}")

    exporter.plot_weight_heatmap(
        weight_results['epistemic'],
        weight_results['aleatoric'],
    )
    if verbose:
        print(f"  Saved: {output_dir / 'ablation_weight_heatmap.pdf'}")

    # Generate combined LaTeX table (main output for paper)
    exporter.export_combined_latex(
        delta_results,
        weight_results,
        blending_results,
    )
    if verbose:
        print(f"  Saved: {output_dir / 'ablation_results.tex'}")

    # Generate JSON summary
    summary = exporter.export_json_summary(
        delta_results,
        weight_results,
        blending_results,
    )
    if verbose:
        print(f"  Saved: {output_dir / 'ablation_summary.json'}")

    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("ABLATION STUDY COMPLETE")
        print("=" * 70)
        print("\nKey Findings:")
        print(f"  {summary['delta_ablation']['key_finding']}")

        # Delta effect summary
        print("\nDelta Parameter Effect:")
        print(delta_results[['spread_uncertainty_corr', 'return_volatility', 'mean_spread_bps']].to_string())

    return {
        'delta': delta_results,
        'weights': weight_results,
        'blending': blending_results,
        'summary': summary,
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run ablation study for Sentiment-Microstructure ABM')
    parser.add_argument('--steps', type=int, default=1000, help='Simulation steps per run')
    parser.add_argument('--reps', type=int, default=5, help='Number of replications')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')

    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else None

    results = run_full_ablation_study(
        output_dir=output_dir,
        n_steps=args.steps,
        n_replications=args.reps,
        verbose=not args.quiet,
    )
