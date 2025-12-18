"""
Uncertainty Decomposition

Maps ASRI macro signals and CryptoBERT micro signals to
epistemic (reducible) and aleatoric (irreducible) uncertainty components.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from .models import MacroSignals


@dataclass
class UncertaintyComponents:
    """Detailed breakdown of uncertainty sources."""

    # Final outputs
    epistemic: float  # [0, 1]
    aleatoric: float  # [0, 1]

    # Macro contributions
    epistemic_regulatory: float = 0.0  # Regulatory opacity
    epistemic_data_missing: float = 0.0  # Missing data sources
    aleatoric_vix: float = 0.0  # Market fear
    aleatoric_peg: float = 0.0  # Stablecoin instability
    aleatoric_tvl: float = 0.0  # DeFi volatility

    # Micro contributions
    epistemic_mc_variance: float = 0.0  # MC Dropout variance
    aleatoric_entropy: float = 0.0  # Prediction entropy


class UncertaintyDecomposer:
    """
    Decomposes uncertainty into epistemic and aleatoric components.

    Epistemic Uncertainty (reducible with more knowledge):
    - Regulatory opacity: unclear regulations = unknown model behavior
    - Data availability: missing sources increase model uncertainty
    - MC Dropout variance: model's own confidence

    Aleatoric Uncertainty (irreducible market noise):
    - VIX level: market fear is inherently stochastic
    - Peg deviation: stablecoin instability is market noise
    - TVL volatility: DeFi flows are unpredictable
    - Prediction entropy: inherent class ambiguity
    """

    def __init__(
        self,
        # Weights for macro epistemic sources
        weight_regulatory_opacity: float = 0.35,
        weight_data_missing: float = 0.25,
        weight_mc_variance: float = 0.40,

        # Weights for aleatoric sources
        weight_vix: float = 0.35,
        weight_peg: float = 0.25,
        weight_tvl: float = 0.15,
        weight_entropy: float = 0.25,

        # Regime modulation
        crisis_epistemic_boost: float = 0.2,
        crisis_aleatoric_boost: float = 0.3,
    ):
        # Epistemic weights
        self.w_regulatory = weight_regulatory_opacity
        self.w_data_missing = weight_data_missing
        self.w_mc_variance = weight_mc_variance

        # Aleatoric weights
        self.w_vix = weight_vix
        self.w_peg = weight_peg
        self.w_tvl = weight_tvl
        self.w_entropy = weight_entropy

        # Crisis modulation
        self.crisis_epi_boost = crisis_epistemic_boost
        self.crisis_aleat_boost = crisis_aleatoric_boost

    def decompose(
        self,
        macro_signals: Optional[MacroSignals],
        micro_epistemic: float,  # From CryptoBERT MC Dropout variance
        micro_aleatoric: float,  # From CryptoBERT prediction entropy
        regime: str = 'neutral',
    ) -> Tuple[float, float]:
        """
        Decompose uncertainty into epistemic and aleatoric components.

        Args:
            macro_signals: ASRI macro data (can be None)
            micro_epistemic: MC Dropout variance from CryptoBERT [0, ~0.3 typical]
            micro_aleatoric: Shannon entropy from CryptoBERT [0, ~1.1 max]
            regime: Current market regime for modulation

        Returns:
            (epistemic_uncertainty, aleatoric_uncertainty) both in [0, 1]
        """
        # Compute macro epistemic
        if macro_signals is not None:
            epi_regulatory = self._epistemic_from_regulatory(macro_signals)
            epi_data_missing = self._epistemic_from_data_availability(macro_signals)
        else:
            epi_regulatory = 0.5  # High uncertainty when no macro data
            epi_data_missing = 1.0  # Maximum missing data penalty

        # Normalize micro epistemic (MC variance typically 0-0.3)
        epi_mc = np.clip(micro_epistemic / 0.3, 0, 1)

        # Combine epistemic sources
        epistemic = (
            self.w_regulatory * epi_regulatory +
            self.w_data_missing * epi_data_missing +
            self.w_mc_variance * epi_mc
        )

        # Compute macro aleatoric
        if macro_signals is not None:
            aleat_vix = self._aleatoric_from_vix(macro_signals)
            aleat_peg = self._aleatoric_from_peg(macro_signals)
            aleat_tvl = self._aleatoric_from_tvl(macro_signals)
        else:
            aleat_vix = 0.3  # Default moderate volatility
            aleat_peg = 0.1
            aleat_tvl = 0.2

        # Normalize micro aleatoric (entropy max ~1.1 for 3 classes)
        aleat_entropy = np.clip(micro_aleatoric / 1.1, 0, 1)

        # Combine aleatoric sources
        aleatoric = (
            self.w_vix * aleat_vix +
            self.w_peg * aleat_peg +
            self.w_tvl * aleat_tvl +
            self.w_entropy * aleat_entropy
        )

        # Regime modulation
        if regime == 'crisis':
            epistemic = min(1.0, epistemic + self.crisis_epi_boost)
            aleatoric = min(1.0, aleatoric + self.crisis_aleat_boost)
        elif regime == 'regulatory':
            epistemic = min(1.0, epistemic + 0.15)  # Regulatory events increase epistemic

        return np.clip(epistemic, 0, 1), np.clip(aleatoric, 0, 1)

    def decompose_detailed(
        self,
        macro_signals: Optional[MacroSignals],
        micro_epistemic: float,
        micro_aleatoric: float,
        regime: str = 'neutral',
    ) -> UncertaintyComponents:
        """
        Return detailed breakdown of uncertainty components.

        Useful for research analysis of which factors drive uncertainty.
        """
        # Compute all components
        if macro_signals is not None:
            epi_regulatory = self._epistemic_from_regulatory(macro_signals)
            epi_data_missing = self._epistemic_from_data_availability(macro_signals)
            aleat_vix = self._aleatoric_from_vix(macro_signals)
            aleat_peg = self._aleatoric_from_peg(macro_signals)
            aleat_tvl = self._aleatoric_from_tvl(macro_signals)
        else:
            epi_regulatory = 0.5
            epi_data_missing = 1.0
            aleat_vix = 0.3
            aleat_peg = 0.1
            aleat_tvl = 0.2

        epi_mc = np.clip(micro_epistemic / 0.3, 0, 1)
        aleat_entropy = np.clip(micro_aleatoric / 1.1, 0, 1)

        # Compute totals
        epistemic = (
            self.w_regulatory * epi_regulatory +
            self.w_data_missing * epi_data_missing +
            self.w_mc_variance * epi_mc
        )

        aleatoric = (
            self.w_vix * aleat_vix +
            self.w_peg * aleat_peg +
            self.w_tvl * aleat_tvl +
            self.w_entropy * aleat_entropy
        )

        # Apply regime modulation
        if regime == 'crisis':
            epistemic = min(1.0, epistemic + self.crisis_epi_boost)
            aleatoric = min(1.0, aleatoric + self.crisis_aleat_boost)

        return UncertaintyComponents(
            epistemic=np.clip(epistemic, 0, 1),
            aleatoric=np.clip(aleatoric, 0, 1),
            epistemic_regulatory=epi_regulatory,
            epistemic_data_missing=epi_data_missing,
            epistemic_mc_variance=epi_mc,
            aleatoric_vix=aleat_vix,
            aleatoric_peg=aleat_peg,
            aleatoric_tvl=aleat_tvl,
            aleatoric_entropy=aleat_entropy,
        )

    def _epistemic_from_regulatory(self, signals: MacroSignals) -> float:
        """
        Higher regulatory risk score = more regulatory opacity = higher epistemic.

        The idea: when regulators are active/negative, the model doesn't know
        how markets will react (structural uncertainty).
        """
        if signals.arbitrage_opacity is not None:
            # Direct ASRI sub-index
            return signals.arbitrage_opacity / 100.0
        elif signals.regulatory_score_raw is not None:
            # Use news-derived regulatory risk as proxy
            # High risk (80+) = high opacity, low risk = clear environment
            return np.clip((signals.regulatory_score_raw - 30) / 50, 0, 1)
        return 0.3  # Default moderate

    def _epistemic_from_data_availability(self, signals: MacroSignals) -> float:
        """
        Missing data sources increase epistemic uncertainty.

        If we can't observe key metrics, our model is less informed.
        """
        return 1.0 - signals.data_completeness

    def _aleatoric_from_vix(self, signals: MacroSignals) -> float:
        """
        VIX measures market fear - inherently stochastic.

        VIX 10-15: Low volatility environment
        VIX 15-25: Normal
        VIX 25-35: Elevated
        VIX 35+: High fear
        """
        if signals.vix_normalized is not None:
            return signals.vix_normalized
        elif signals.vix_level is not None:
            return np.clip((signals.vix_level - 10) / 40, 0, 1)
        return 0.25  # Default moderate

    def _aleatoric_from_peg(self, signals: MacroSignals) -> float:
        """
        Stablecoin peg deviation indicates market stress.

        0.001 (0.1%) deviation: Normal
        0.01 (1%) deviation: Stressed
        0.05 (5%) deviation: Crisis
        """
        if signals.peg_stability is not None:
            # peg_stability is the max deviation from 1.0
            return np.clip(signals.peg_stability * 20, 0, 1)  # 5% deviation = 1.0
        return 0.1  # Default low

    def _aleatoric_from_tvl(self, signals: MacroSignals) -> float:
        """
        TVL stress indicates DeFi liquidity conditions.

        High TVL drawdown from max = unstable liquidity environment.
        """
        if signals.tvl_stress is not None:
            return signals.tvl_stress
        return 0.2  # Default moderate
