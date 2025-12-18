"""
Signal Composer

The main orchestrator that blends macro (ASRI) and micro (CryptoBERT)
sentiment signals into a unified SentimentTick for ABM consumption.
"""

import numpy as np
from typing import Optional, Tuple, Generator, Callable
from datetime import datetime
import logging

from .models import SentimentTick, MacroSignals
from .macro_sentiment import MacroSentimentBlender
from .uncertainty_decomposer import UncertaintyDecomposer
from .regime_detector import RegimeDetector
from .divergence_tracker import DivergenceTracker

logger = logging.getLogger(__name__)


class SignalComposer:
    """
    Composes final sentiment tick from multiple signal sources.

    This is the main entry point for the signal processing pipeline.
    It takes macro signals (ASRI) and micro signals (CryptoBERT),
    blends them with adaptive weighting, decomposes uncertainty,
    detects regime, and tracks divergence.

    Usage:
        composer = SignalComposer()

        # Fetch macro data
        macro = await asri_adapter.fetch_all()

        # Get micro sentiment from CryptoBERT
        micro = (sentiment, epistemic, aleatoric)

        # Compose unified tick
        tick = composer.compose(macro, micro)

        # Use in ABM
        model.set_sentiment_tick(tick)
    """

    # Default weights
    DEFAULT_MACRO_WEIGHT = 0.30  # ASRI contribution
    DEFAULT_MICRO_WEIGHT = 0.70  # CryptoBERT contribution

    # Adaptive weight adjustments by regime
    REGIME_WEIGHTS = {
        'crisis': (0.60, 0.40),       # Trust institutions in crisis
        'regulatory': (0.70, 0.30),   # Heavy macro during regulatory events
        'bearish': (0.45, 0.55),      # Slight macro boost in bearish
        'bullish': (0.25, 0.75),      # Trust retail momentum in bullish
        'neutral': (0.30, 0.70),      # Default balanced
    }

    def __init__(
        self,
        macro_weight: float = 0.30,
        micro_weight: float = 0.70,
        adaptive_weighting: bool = True,
        divergence_threshold: float = 0.4,
    ):
        """
        Initialize signal composer.

        Args:
            macro_weight: Base weight for ASRI signals [0, 1]
            micro_weight: Base weight for CryptoBERT signals [0, 1]
            adaptive_weighting: Adjust weights based on regime
            divergence_threshold: Threshold for flagging divergence events
        """
        # Normalize weights
        total = macro_weight + micro_weight
        self.base_macro_weight = macro_weight / total
        self.base_micro_weight = micro_weight / total

        self.adaptive_weighting = adaptive_weighting
        self.divergence_threshold = divergence_threshold

        # Sub-components
        self.macro_blender = MacroSentimentBlender()
        self.uncertainty_decomposer = UncertaintyDecomposer()
        self.regime_detector = RegimeDetector()
        self.divergence_tracker = DivergenceTracker(
            significant_threshold=divergence_threshold
        )

    def compose(
        self,
        macro_signals: Optional[MacroSignals],
        micro_sentiment: Tuple[float, float, float],
        price: Optional[float] = None,
    ) -> SentimentTick:
        """
        Compose unified sentiment tick from macro and micro signals.

        Args:
            macro_signals: ASRI data (can be None if unavailable)
            micro_sentiment: Tuple of (sentiment, epistemic, aleatoric) from CryptoBERT
            price: Current price for divergence tracking

        Returns:
            SentimentTick ready for ABM consumption
        """
        retail_sent, micro_epi, micro_aleat = micro_sentiment

        # Compute institutional sentiment from ASRI
        if macro_signals is not None:
            inst_result = self.macro_blender.compute_detailed(macro_signals)
            inst_sent = inst_result.sentiment
            macro_confidence = inst_result.confidence
        else:
            inst_sent = 0.0
            macro_confidence = 0.0

        # Detect regime for adaptive weighting
        regime = self.regime_detector.detect(
            macro_signals,
            retail_sent,
            inst_sent,
        )

        # Get adaptive weights
        if self.adaptive_weighting and macro_signals is not None:
            macro_w, micro_w = self._get_adaptive_weights(
                regime,
                macro_confidence,
            )
        else:
            macro_w = self.base_macro_weight if macro_signals else 0.0
            micro_w = 1.0 - macro_w

        # Blend sentiment
        blended_sentiment = macro_w * inst_sent + micro_w * retail_sent

        # Decompose uncertainty
        epistemic, aleatoric = self.uncertainty_decomposer.decompose(
            macro_signals,
            micro_epi,
            micro_aleat,
            regime,
        )

        # Get detailed uncertainty breakdown
        uncertainty_detail = self.uncertainty_decomposer.decompose_detailed(
            macro_signals, micro_epi, micro_aleat, regime
        )

        # Track divergence
        divergence = self.divergence_tracker.update(
            retail_sent,
            inst_sent,
            regime,
            macro_signals.asri_alert_level if macro_signals else 'unknown',
            price,
        )

        # Data completeness
        data_completeness = macro_signals.data_completeness if macro_signals else 0.0

        return SentimentTick(
            timestamp=datetime.utcnow(),
            sentiment=np.clip(blended_sentiment, -1, 1),
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            regime=regime,
            retail_sentiment=retail_sent,
            institutional_sentiment=inst_sent,
            divergence=divergence,
            macro_weight=macro_w,
            micro_weight=micro_w,
            asri_alert_level=macro_signals.asri_alert_level if macro_signals else 'unknown',
            data_completeness=data_completeness,
            epistemic_macro=uncertainty_detail.epistemic_regulatory + uncertainty_detail.epistemic_data_missing,
            epistemic_micro=uncertainty_detail.epistemic_mc_variance,
            aleatoric_macro=uncertainty_detail.aleatoric_vix + uncertainty_detail.aleatoric_peg,
            aleatoric_micro=uncertainty_detail.aleatoric_entropy,
        )

    def _get_adaptive_weights(
        self,
        regime: str,
        macro_confidence: float,
    ) -> Tuple[float, float]:
        """
        Get regime-adaptive weights.

        Args:
            regime: Current market regime
            macro_confidence: Confidence in macro signal [0, 1]

        Returns:
            (macro_weight, micro_weight) summing to 1.0
        """
        # Get base weights for regime
        base_macro, base_micro = self.REGIME_WEIGHTS.get(
            regime,
            (self.base_macro_weight, self.base_micro_weight)
        )

        # Modulate by macro confidence
        # Low confidence -> reduce macro weight
        effective_macro = base_macro * macro_confidence
        effective_micro = 1.0 - effective_macro

        return effective_macro, effective_micro

    def compose_micro_only(
        self,
        micro_sentiment: Tuple[float, float, float],
    ) -> SentimentTick:
        """
        Compose tick from CryptoBERT only (backward compatibility).

        Useful when ASRI data is unavailable.
        """
        return self.compose(None, micro_sentiment)

    def create_generator(
        self,
        macro_fetcher: Callable[[], MacroSignals],
        micro_analyzer: Callable[[str], Tuple[float, float, float]],
        texts: list,
        macro_refresh_interval: int = 50,
    ) -> Generator[SentimentTick, None, None]:
        """
        Create a generator that yields SentimentTicks.

        Useful for simulation loops.

        Args:
            macro_fetcher: Async function to fetch MacroSignals
            micro_analyzer: Function that takes text and returns (sent, epi, aleat)
            texts: List of texts to analyze
            macro_refresh_interval: Steps between macro data refreshes

        Yields:
            SentimentTick for each step
        """
        macro_signals = None
        text_idx = 0

        for step in range(len(texts)):
            # Refresh macro data periodically
            if step % macro_refresh_interval == 0:
                try:
                    macro_signals = macro_fetcher()
                except Exception as e:
                    logger.warning(f"Macro fetch failed: {e}")

            # Get micro sentiment
            text = texts[text_idx % len(texts)]
            micro = micro_analyzer(text)
            text_idx += 1

            # Compose tick
            tick = self.compose(macro_signals, micro)
            yield tick

    def get_divergence_stats(self) -> dict:
        """Get divergence tracking statistics."""
        return self.divergence_tracker.get_stats().__dict__

    def get_divergence_analysis(self) -> dict:
        """Analyze predictive power of divergence."""
        return self.divergence_tracker.analyze_predictive_power()

    def get_regime_distribution(self, window: int = 50) -> dict:
        """Get recent regime distribution."""
        return self.regime_detector.get_regime_distribution(window)

    def reset(self):
        """Reset all component states."""
        self.regime_detector.reset()
        self.divergence_tracker.reset()


# Convenience function for creating pre-configured composers
def create_default_composer() -> SignalComposer:
    """Create composer with default settings."""
    return SignalComposer(
        macro_weight=0.30,
        micro_weight=0.70,
        adaptive_weighting=True,
        divergence_threshold=0.4,
    )


def create_macro_heavy_composer() -> SignalComposer:
    """Create composer that weights ASRI more heavily."""
    return SignalComposer(
        macro_weight=0.50,
        micro_weight=0.50,
        adaptive_weighting=True,
        divergence_threshold=0.3,
    )


def create_micro_only_composer() -> SignalComposer:
    """Create composer that uses only CryptoBERT."""
    return SignalComposer(
        macro_weight=0.0,
        micro_weight=1.0,
        adaptive_weighting=False,
    )
