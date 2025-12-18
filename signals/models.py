"""
Data models for multi-scale sentiment signals.

Defines the core dataclasses used throughout the signal processing pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List
import numpy as np


@dataclass
class MacroSignals:
    """Aggregated signals from ASRI data sources."""

    timestamp: datetime

    # News & Regulatory
    regulatory_sentiment: float  # [-1, 1] from NewsAggregator (inverted from 0-100)
    regulatory_score_raw: float  # Original 0-100 score
    article_count: int = 0
    regulatory_article_count: int = 0
    top_headlines: List[Dict] = field(default_factory=list)

    # DeFi Metrics
    total_tvl: Optional[float] = None  # Total DeFi TVL in USD
    tvl_change_24h: Optional[float] = None  # Percentage change
    tvl_stress: Optional[float] = None  # Deviation from historical max [0, 1]

    # Stablecoin Health
    peg_stability: Optional[float] = None  # Max peg deviation [0, 1]
    stablecoin_dominance: Optional[float] = None  # USDT+USDC market share

    # TradFi Linkage (from FRED)
    vix_level: Optional[float] = None  # Raw VIX
    vix_normalized: Optional[float] = None  # Normalized to typical range
    treasury_10y: Optional[float] = None  # 10Y Treasury yield
    yield_spread: Optional[float] = None  # 10Y-2Y spread

    # Crypto-Equity Correlation
    btc_spy_correlation: Optional[float] = None  # 30-day rolling correlation

    # ASRI Composite (if full calculation available)
    asri_composite: Optional[float] = None  # 0-100 aggregate score
    asri_alert_level: str = "unknown"  # low/moderate/elevated/high/critical

    # Sub-indices (if available)
    stablecoin_risk: Optional[float] = None
    defi_liquidity_risk: Optional[float] = None
    contagion_risk: Optional[float] = None
    arbitrage_opacity: Optional[float] = None

    @property
    def has_defi_data(self) -> bool:
        return self.total_tvl is not None

    @property
    def has_tradfi_data(self) -> bool:
        return self.vix_level is not None

    @property
    def data_completeness(self) -> float:
        """Fraction of optional fields that are populated."""
        optional_fields = [
            self.total_tvl, self.tvl_change_24h, self.peg_stability,
            self.vix_level, self.treasury_10y, self.btc_spy_correlation,
            self.asri_composite
        ]
        populated = sum(1 for f in optional_fields if f is not None)
        return populated / len(optional_fields)


@dataclass
class SentimentTick:
    """
    ABM-compatible sentiment signal with uncertainty decomposition.

    This is the unified output format consumed by the Mesa simulation.
    """

    timestamp: datetime

    # Core 4-tuple for ABM (required interface)
    sentiment: float  # [-1, 1] blended sentiment
    epistemic_uncertainty: float  # [0, 1] model/knowledge uncertainty
    aleatoric_uncertainty: float  # [0, 1] irreducible market noise
    regime: str  # 'bullish' | 'bearish' | 'neutral' | 'crisis' | 'regulatory'

    # Signal decomposition (research diagnostics)
    retail_sentiment: float = 0.0  # CryptoBERT raw signal
    institutional_sentiment: float = 0.0  # ASRI-derived signal
    divergence: float = 0.0  # retail - institutional

    # Source attribution
    macro_weight: float = 0.3  # Weight given to ASRI in blend
    micro_weight: float = 0.7  # Weight given to CryptoBERT

    # ASRI metadata
    asri_alert_level: str = "unknown"
    data_completeness: float = 0.0  # How much macro data was available

    # Uncertainty components (for detailed analysis)
    epistemic_macro: float = 0.0  # From ASRI sources
    epistemic_micro: float = 0.0  # From MC Dropout
    aleatoric_macro: float = 0.0  # From market volatility indicators
    aleatoric_micro: float = 0.0  # From prediction entropy

    @property
    def total_uncertainty(self) -> float:
        """Combined uncertainty for agents that don't differentiate."""
        return self.epistemic_uncertainty + self.aleatoric_uncertainty

    @property
    def is_high_divergence(self) -> bool:
        """Flag when retail and institutional signals disagree significantly."""
        return abs(self.divergence) > 0.4

    @property
    def is_crisis(self) -> bool:
        return self.regime == 'crisis'

    def to_tuple(self) -> tuple:
        """Return classic 4-tuple for backward compatibility."""
        return (
            self.sentiment,
            self.epistemic_uncertainty,
            self.aleatoric_uncertainty,
            self.regime
        )

    def to_dict(self) -> dict:
        """Serialize for CSV/JSON output."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'sentiment': self.sentiment,
            'epistemic_uncertainty': self.epistemic_uncertainty,
            'aleatoric_uncertainty': self.aleatoric_uncertainty,
            'regime': self.regime,
            'retail_sentiment': self.retail_sentiment,
            'institutional_sentiment': self.institutional_sentiment,
            'divergence': self.divergence,
            'macro_weight': self.macro_weight,
            'micro_weight': self.micro_weight,
            'asri_alert_level': self.asri_alert_level,
            'data_completeness': self.data_completeness,
            'total_uncertainty': self.total_uncertainty,
            'is_high_divergence': self.is_high_divergence,
        }


@dataclass
class DivergenceEvent:
    """Records a significant divergence between retail and institutional sentiment."""

    timestamp: datetime
    divergence: float  # retail - institutional
    retail_sentiment: float
    institutional_sentiment: float
    regime_before: str
    asri_alert_level: str

    # Forward-looking metrics (filled in retrospectively)
    volatility_24h_after: Optional[float] = None
    price_change_24h_after: Optional[float] = None
    regime_24h_after: Optional[str] = None
