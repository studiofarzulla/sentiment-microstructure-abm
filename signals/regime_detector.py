"""
Regime Detector

Classifies market regime from combined macro and micro signals.
Used for adaptive weight adjustment and agent behavior modulation.
"""

import numpy as np
from typing import Optional, List
from dataclasses import dataclass
from collections import deque
from datetime import datetime

from .models import MacroSignals


@dataclass
class RegimeState:
    """Current regime classification with metadata."""

    regime: str  # 'bullish' | 'bearish' | 'neutral' | 'crisis' | 'regulatory'
    confidence: float  # [0, 1] confidence in classification
    timestamp: datetime

    # Contributing factors
    macro_signal: float  # Institutional sentiment
    micro_signal: float  # Retail sentiment
    divergence: float  # retail - institutional

    # Alert level from ASRI
    asri_alert: str

    # Regime persistence
    steps_in_regime: int = 1


class RegimeDetector:
    """
    Detects market regime from combined sentiment signals.

    Regimes:
    - bullish: Positive sentiment, low risk, upward momentum
    - bearish: Negative sentiment, elevated risk, downward momentum
    - neutral: Mixed signals, range-bound, normal volatility
    - crisis: High risk alerts, extreme uncertainty, potential contagion
    - regulatory: Significant regulatory news driving sentiment

    The detector uses hysteresis to prevent rapid regime switching.
    """

    # Regime thresholds
    BULLISH_THRESHOLD = 0.25
    BEARISH_THRESHOLD = -0.25
    CRISIS_ALERT_LEVELS = ('high', 'critical')
    REGULATORY_NEWS_THRESHOLD = -0.35  # Strongly negative regulatory news

    # Hysteresis settings
    MIN_STEPS_FOR_SWITCH = 3  # Minimum steps before regime can change
    SMOOTHING_WINDOW = 5  # Window for signal smoothing

    def __init__(
        self,
        bullish_threshold: float = 0.25,
        bearish_threshold: float = -0.25,
        crisis_alert_levels: tuple = ('high', 'critical'),
        min_steps_for_switch: int = 3,
        smoothing_window: int = 5,
    ):
        self.bullish_thresh = bullish_threshold
        self.bearish_thresh = bearish_threshold
        self.crisis_alerts = crisis_alert_levels
        self.min_steps = min_steps_for_switch
        self.smooth_window = smoothing_window

        # State tracking
        self._current_regime = 'neutral'
        self._steps_in_regime = 0
        self._signal_history: deque = deque(maxlen=smoothing_window)
        self._regime_history: List[RegimeState] = []

    def detect(
        self,
        macro_signals: Optional[MacroSignals],
        retail_sentiment: float,
        institutional_sentiment: Optional[float] = None,
    ) -> str:
        """
        Detect current market regime.

        Args:
            macro_signals: ASRI data (can be None)
            retail_sentiment: CryptoBERT sentiment [-1, 1]
            institutional_sentiment: Pre-computed institutional sentiment (optional)

        Returns:
            Regime string: 'bullish', 'bearish', 'neutral', 'crisis', or 'regulatory'
        """
        state = self.detect_detailed(macro_signals, retail_sentiment, institutional_sentiment)
        return state.regime

    def detect_detailed(
        self,
        macro_signals: Optional[MacroSignals],
        retail_sentiment: float,
        institutional_sentiment: Optional[float] = None,
    ) -> RegimeState:
        """
        Detect regime with full metadata.

        Returns RegimeState with confidence and contributing factors.
        """
        # Compute institutional sentiment if not provided
        if institutional_sentiment is None and macro_signals is not None:
            from .macro_sentiment import MacroSentimentBlender
            blender = MacroSentimentBlender()
            institutional_sentiment = blender.compute_institutional_sentiment(macro_signals)
        elif institutional_sentiment is None:
            institutional_sentiment = 0.0

        # Compute blended signal for regime detection
        blended = 0.5 * retail_sentiment + 0.5 * institutional_sentiment
        divergence = retail_sentiment - institutional_sentiment

        # Add to history for smoothing
        self._signal_history.append(blended)
        smoothed_signal = np.mean(self._signal_history)

        # Get ASRI alert level
        asri_alert = macro_signals.asri_alert_level if macro_signals else 'unknown'

        # Determine candidate regime
        candidate_regime, confidence = self._classify_regime(
            smoothed_signal,
            asri_alert,
            macro_signals,
            divergence,
        )

        # Apply hysteresis
        final_regime = self._apply_hysteresis(candidate_regime)

        # Update state
        if final_regime == self._current_regime:
            self._steps_in_regime += 1
        else:
            self._current_regime = final_regime
            self._steps_in_regime = 1

        state = RegimeState(
            regime=final_regime,
            confidence=confidence,
            timestamp=datetime.utcnow(),
            macro_signal=institutional_sentiment,
            micro_signal=retail_sentiment,
            divergence=divergence,
            asri_alert=asri_alert,
            steps_in_regime=self._steps_in_regime,
        )

        self._regime_history.append(state)

        return state

    def _classify_regime(
        self,
        smoothed_signal: float,
        asri_alert: str,
        macro_signals: Optional[MacroSignals],
        divergence: float,
    ) -> tuple[str, float]:
        """
        Classify regime based on signals.

        Returns (regime, confidence).
        """
        # Priority 1: Crisis detection (overrides other signals)
        if asri_alert in self.crisis_alerts:
            return 'crisis', 0.9

        # Priority 2: Regulatory event detection
        if macro_signals is not None:
            reg_sent = macro_signals.regulatory_sentiment
            if reg_sent is not None and reg_sent < self.REGULATORY_NEWS_THRESHOLD:
                # Strong negative regulatory news
                if macro_signals.regulatory_article_count >= 3:
                    return 'regulatory', 0.85

        # Priority 3: Standard regime classification
        if smoothed_signal > self.bullish_thresh:
            # How far above threshold determines confidence
            excess = smoothed_signal - self.bullish_thresh
            conf = min(0.5 + excess * 2, 0.95)
            return 'bullish', conf

        elif smoothed_signal < self.bearish_thresh:
            excess = self.bearish_thresh - smoothed_signal
            conf = min(0.5 + excess * 2, 0.95)
            return 'bearish', conf

        else:
            # Neutral - confidence higher when signal closer to zero
            conf = 0.5 + 0.3 * (1 - abs(smoothed_signal) / self.bullish_thresh)
            return 'neutral', conf

    def _apply_hysteresis(self, candidate: str) -> str:
        """
        Apply hysteresis to prevent rapid regime switching.

        Only switch if:
        1. We've been in current regime for MIN_STEPS, or
        2. New regime is 'crisis' (always switch immediately)
        """
        # Crisis always takes priority
        if candidate == 'crisis':
            return candidate

        # If already in candidate regime, stay there
        if candidate == self._current_regime:
            return candidate

        # Only switch if we've been stable long enough
        if self._steps_in_regime >= self.min_steps:
            return candidate

        # Otherwise stay in current regime
        return self._current_regime

    @property
    def current_regime(self) -> str:
        """Current regime classification."""
        return self._current_regime

    @property
    def regime_stability(self) -> float:
        """
        How stable is the current regime?

        Returns [0, 1] where 1 = very stable (many steps in same regime).
        """
        return min(1.0, self._steps_in_regime / 10)

    def get_regime_distribution(self, window: int = 50) -> dict:
        """
        Get distribution of regimes over recent history.

        Useful for understanding regime persistence.
        """
        recent = self._regime_history[-window:] if self._regime_history else []
        if not recent:
            return {'neutral': 1.0}

        counts = {}
        for state in recent:
            counts[state.regime] = counts.get(state.regime, 0) + 1

        total = len(recent)
        return {k: v / total for k, v in counts.items()}

    def reset(self):
        """Reset detector state."""
        self._current_regime = 'neutral'
        self._steps_in_regime = 0
        self._signal_history.clear()
        self._regime_history.clear()
