"""
Multi-Scale Sentiment Signal Processing

Integrates macro (ASRI) and micro (CryptoBERT) sentiment signals
with uncertainty decomposition for ABM consumption.
"""

from .models import SentimentTick, MacroSignals
from .asri_adapter import ASRIAdapter
from .macro_sentiment import MacroSentimentBlender
from .uncertainty_decomposer import UncertaintyDecomposer
from .regime_detector import RegimeDetector
from .signal_composer import SignalComposer
from .divergence_tracker import DivergenceTracker

__all__ = [
    'SentimentTick',
    'MacroSignals',
    'ASRIAdapter',
    'MacroSentimentBlender',
    'UncertaintyDecomposer',
    'RegimeDetector',
    'SignalComposer',
    'DivergenceTracker',
]
