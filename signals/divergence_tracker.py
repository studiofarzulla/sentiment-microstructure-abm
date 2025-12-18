"""
Divergence Tracker

Tracks and analyzes divergence between retail (CryptoBERT) and
institutional (ASRI) sentiment signals.

Key research hypothesis: Large divergences predict volatility spikes.
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
import json


@dataclass
class DivergenceEvent:
    """Records a significant divergence event."""

    timestamp: datetime
    divergence: float  # retail - institutional
    retail_sentiment: float
    institutional_sentiment: float
    regime: str
    asri_alert_level: str

    # Filled retrospectively after forward window
    volatility_after: Optional[float] = None
    price_change_after: Optional[float] = None
    regime_after: Optional[str] = None
    steps_to_convergence: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'divergence': self.divergence,
            'retail_sentiment': self.retail_sentiment,
            'institutional_sentiment': self.institutional_sentiment,
            'regime': self.regime,
            'asri_alert_level': self.asri_alert_level,
            'volatility_after': self.volatility_after,
            'price_change_after': self.price_change_after,
            'regime_after': self.regime_after,
            'steps_to_convergence': self.steps_to_convergence,
        }


@dataclass
class DivergenceStats:
    """Summary statistics for divergence analysis."""

    mean_divergence: float
    std_divergence: float
    max_positive: float  # Max retail > institutional
    max_negative: float  # Max institutional > retail
    n_significant_events: int
    correlation_with_volatility: Optional[float] = None


class DivergenceTracker:
    """
    Tracks divergence between retail and institutional sentiment.

    Divergence = retail_sentiment - institutional_sentiment

    Positive divergence: Retail more bullish than institutions
    Negative divergence: Institutions more bullish than retail

    Research hypothesis: When retail and institutional sentiment diverge
    significantly, volatility tends to spike as the signals converge.
    """

    # Divergence thresholds
    SIGNIFICANT_THRESHOLD = 0.4  # Absolute divergence for event logging
    EXTREME_THRESHOLD = 0.6     # Extreme divergence

    def __init__(
        self,
        significant_threshold: float = 0.4,
        extreme_threshold: float = 0.6,
        history_window: int = 500,
        forward_window: int = 24,  # Steps to look forward for impact
    ):
        self.sig_threshold = significant_threshold
        self.extreme_threshold = extreme_threshold
        self.history_window = history_window
        self.forward_window = forward_window

        # Rolling history
        self._divergence_history: deque = deque(maxlen=history_window)
        self._price_history: deque = deque(maxlen=history_window)

        # Significant events
        self._events: List[DivergenceEvent] = []
        self._pending_events: List[Tuple[int, DivergenceEvent]] = []  # (step, event)

        # Current state
        self._current_step = 0
        self._ewma_divergence = 0.0
        self._ewma_alpha = 0.1

    def update(
        self,
        retail_sentiment: float,
        institutional_sentiment: float,
        regime: str,
        asri_alert_level: str = 'unknown',
        price: Optional[float] = None,
    ) -> float:
        """
        Update tracker with new sentiment readings.

        Args:
            retail_sentiment: CryptoBERT sentiment [-1, 1]
            institutional_sentiment: ASRI-derived sentiment [-1, 1]
            regime: Current market regime
            asri_alert_level: ASRI alert level
            price: Current price (for retrospective analysis)

        Returns:
            Current divergence value.
        """
        divergence = retail_sentiment - institutional_sentiment

        # Update histories
        self._divergence_history.append(divergence)
        if price is not None:
            self._price_history.append(price)

        # Update EWMA
        self._ewma_divergence = (
            self._ewma_alpha * divergence +
            (1 - self._ewma_alpha) * self._ewma_divergence
        )

        # Check for significant event
        if abs(divergence) >= self.sig_threshold:
            event = DivergenceEvent(
                timestamp=datetime.utcnow(),
                divergence=divergence,
                retail_sentiment=retail_sentiment,
                institutional_sentiment=institutional_sentiment,
                regime=regime,
                asri_alert_level=asri_alert_level,
            )
            self._events.append(event)
            self._pending_events.append((self._current_step, event))

        # Process pending events (fill forward metrics)
        self._process_pending_events()

        self._current_step += 1
        return divergence

    def _process_pending_events(self):
        """Fill in retrospective metrics for events past their forward window."""
        if not self._pending_events or len(self._price_history) < 2:
            return

        completed = []
        for i, (event_step, event) in enumerate(self._pending_events):
            steps_elapsed = self._current_step - event_step

            if steps_elapsed >= self.forward_window:
                # Calculate forward volatility
                if len(self._price_history) >= self.forward_window:
                    recent_prices = list(self._price_history)[-self.forward_window:]
                    returns = np.diff(np.log(recent_prices))
                    event.volatility_after = np.std(returns) if len(returns) > 1 else 0

                    # Price change
                    if recent_prices[0] > 0:
                        event.price_change_after = (
                            recent_prices[-1] / recent_prices[0] - 1
                        )

                # Steps to convergence (divergence returning to < threshold)
                if len(self._divergence_history) >= steps_elapsed:
                    recent_div = list(self._divergence_history)[-steps_elapsed:]
                    for j, d in enumerate(recent_div):
                        if abs(d) < self.sig_threshold:
                            event.steps_to_convergence = j + 1
                            break

                completed.append(i)

        # Remove completed events from pending
        for i in reversed(completed):
            self._pending_events.pop(i)

    @property
    def current_divergence(self) -> float:
        """Most recent divergence value."""
        if self._divergence_history:
            return self._divergence_history[-1]
        return 0.0

    @property
    def smoothed_divergence(self) -> float:
        """EWMA-smoothed divergence."""
        return self._ewma_divergence

    @property
    def is_significantly_diverged(self) -> bool:
        """Whether current divergence exceeds threshold."""
        return abs(self.current_divergence) >= self.sig_threshold

    @property
    def divergence_direction(self) -> str:
        """
        Direction of current divergence.

        Returns:
            'retail_bullish': Retail more bullish than institutional
            'institutional_bullish': Institutional more bullish
            'aligned': Signals roughly aligned
        """
        div = self.current_divergence
        if div > 0.1:
            return 'retail_bullish'
        elif div < -0.1:
            return 'institutional_bullish'
        return 'aligned'

    def get_stats(self) -> DivergenceStats:
        """Get summary statistics."""
        if not self._divergence_history:
            return DivergenceStats(
                mean_divergence=0,
                std_divergence=0,
                max_positive=0,
                max_negative=0,
                n_significant_events=0,
            )

        divs = list(self._divergence_history)

        return DivergenceStats(
            mean_divergence=np.mean(divs),
            std_divergence=np.std(divs),
            max_positive=max(divs),
            max_negative=min(divs),
            n_significant_events=len(self._events),
        )

    def get_events(self, n_recent: Optional[int] = None) -> List[DivergenceEvent]:
        """Get logged divergence events."""
        if n_recent:
            return self._events[-n_recent:]
        return self._events

    def export_events_json(self, filepath: str):
        """Export events to JSON file."""
        events_data = [e.to_dict() for e in self._events]
        with open(filepath, 'w') as f:
            json.dump(events_data, f, indent=2)

    def analyze_predictive_power(self) -> dict:
        """
        Analyze whether divergence predicts forward volatility.

        Returns correlation and basic stats.
        """
        # Only use events with forward metrics filled
        complete_events = [
            e for e in self._events
            if e.volatility_after is not None
        ]

        if len(complete_events) < 5:
            return {
                'n_events': len(complete_events),
                'correlation': None,
                'message': 'Insufficient data for analysis'
            }

        divergences = [abs(e.divergence) for e in complete_events]
        volatilities = [e.volatility_after for e in complete_events]

        correlation = np.corrcoef(divergences, volatilities)[0, 1]

        # Average volatility by divergence bucket
        high_div = [e.volatility_after for e in complete_events if abs(e.divergence) >= self.extreme_threshold]
        mod_div = [e.volatility_after for e in complete_events if self.sig_threshold <= abs(e.divergence) < self.extreme_threshold]

        return {
            'n_events': len(complete_events),
            'correlation': correlation,
            'avg_vol_extreme_divergence': np.mean(high_div) if high_div else None,
            'avg_vol_moderate_divergence': np.mean(mod_div) if mod_div else None,
            'avg_steps_to_convergence': np.mean([
                e.steps_to_convergence for e in complete_events
                if e.steps_to_convergence is not None
            ]) if complete_events else None,
        }

    def reset(self):
        """Reset tracker state."""
        self._divergence_history.clear()
        self._price_history.clear()
        self._events.clear()
        self._pending_events.clear()
        self._current_step = 0
        self._ewma_divergence = 0.0
