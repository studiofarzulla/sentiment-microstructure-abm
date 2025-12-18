"""
Macro Sentiment Blender

Converts ASRI sub-indices and news sentiment into an institutional
sentiment signal comparable to CryptoBERT's retail sentiment.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from .models import MacroSignals


@dataclass
class MacroSentimentResult:
    """Result of macro sentiment computation."""

    sentiment: float  # [-1, 1] institutional sentiment
    confidence: float  # [0, 1] confidence in the signal

    # Component contributions
    regulatory_contribution: float
    defi_contribution: float
    tradfi_contribution: float

    # Raw inputs used
    regulatory_score: float
    vix_signal: float
    peg_signal: float


class MacroSentimentBlender:
    """
    Converts ASRI macro signals to a directional sentiment score.

    The core insight: ASRI measures *risk*, which is inversely related
    to bullish sentiment. High risk = bearish, low risk = bullish.

    Components:
    1. Regulatory sentiment: Direct news sentiment + opacity
    2. DeFi health: TVL stability + stablecoin peg
    3. TradFi linkage: VIX level + yield curve
    """

    # Component weights for blending
    WEIGHTS = {
        'regulatory': 0.40,  # News sentiment + regulatory risk
        'defi': 0.35,        # TVL + stablecoin health
        'tradfi': 0.25,      # VIX + yield curve
    }

    def __init__(
        self,
        regulatory_weight: float = 0.40,
        defi_weight: float = 0.35,
        tradfi_weight: float = 0.25,
    ):
        self.w_regulatory = regulatory_weight
        self.w_defi = defi_weight
        self.w_tradfi = tradfi_weight

        # Normalize weights
        total = self.w_regulatory + self.w_defi + self.w_tradfi
        self.w_regulatory /= total
        self.w_defi /= total
        self.w_tradfi /= total

    def compute_institutional_sentiment(
        self,
        signals: MacroSignals
    ) -> float:
        """
        Compute single institutional sentiment value from macro signals.

        Returns:
            float in [-1, 1] where:
            -1 = extremely bearish (high risk, negative news, market fear)
            +1 = extremely bullish (low risk, positive news, calm markets)
        """
        result = self.compute_detailed(signals)
        return result.sentiment

    def compute_detailed(self, signals: MacroSignals) -> MacroSentimentResult:
        """
        Compute institutional sentiment with full breakdown.

        Useful for understanding which factors drive the sentiment.
        """
        # 1. Regulatory component
        regulatory_sent, reg_conf = self._compute_regulatory_sentiment(signals)

        # 2. DeFi health component
        defi_sent, defi_conf = self._compute_defi_sentiment(signals)

        # 3. TradFi linkage component
        tradfi_sent, tradfi_conf = self._compute_tradfi_sentiment(signals)

        # Confidence-weighted blending
        # If a component has low confidence (missing data), reduce its weight
        effective_reg_w = self.w_regulatory * reg_conf
        effective_defi_w = self.w_defi * defi_conf
        effective_tradfi_w = self.w_tradfi * tradfi_conf

        total_weight = effective_reg_w + effective_defi_w + effective_tradfi_w

        if total_weight > 0:
            sentiment = (
                effective_reg_w * regulatory_sent +
                effective_defi_w * defi_sent +
                effective_tradfi_w * tradfi_sent
            ) / total_weight
        else:
            sentiment = 0.0  # No data available

        # Overall confidence
        overall_confidence = total_weight / (self.w_regulatory + self.w_defi + self.w_tradfi)

        return MacroSentimentResult(
            sentiment=np.clip(sentiment, -1, 1),
            confidence=overall_confidence,
            regulatory_contribution=regulatory_sent * effective_reg_w / max(total_weight, 0.01),
            defi_contribution=defi_sent * effective_defi_w / max(total_weight, 0.01),
            tradfi_contribution=tradfi_sent * effective_tradfi_w / max(total_weight, 0.01),
            regulatory_score=signals.regulatory_score_raw if signals.regulatory_score_raw else 50,
            vix_signal=tradfi_sent,
            peg_signal=defi_sent,
        )

    def _compute_regulatory_sentiment(
        self,
        signals: MacroSignals
    ) -> tuple[float, float]:
        """
        Compute sentiment from regulatory signals.

        Returns (sentiment, confidence).
        """
        if signals.regulatory_sentiment is not None:
            # Already converted to [-1, 1] in adapter
            sentiment = signals.regulatory_sentiment

            # Confidence based on article count
            conf = min(1.0, signals.article_count / 20) if signals.article_count else 0.5
            return sentiment, conf

        # Fallback to ASRI arbitrage_opacity if available
        if signals.arbitrage_opacity is not None:
            # High opacity = bearish (regulatory uncertainty)
            sentiment = 1.0 - 2.0 * (signals.arbitrage_opacity / 100.0)
            return sentiment, 0.8

        return 0.0, 0.3  # Neutral with low confidence

    def _compute_defi_sentiment(
        self,
        signals: MacroSignals
    ) -> tuple[float, float]:
        """
        Compute sentiment from DeFi health indicators.

        Returns (sentiment, confidence).
        """
        components = []
        confidences = []

        # TVL stress (if available)
        if signals.tvl_stress is not None:
            # Low stress = bullish, high stress = bearish
            tvl_sent = 1.0 - 2.0 * signals.tvl_stress
            components.append(tvl_sent)
            confidences.append(0.8)

        # Stablecoin peg stability
        if signals.peg_stability is not None:
            # Low deviation = bullish, high deviation = bearish
            # 0.01 (1%) deviation maps to -0.5 sentiment
            peg_sent = 1.0 - min(signals.peg_stability * 100, 2.0)
            components.append(peg_sent)
            confidences.append(0.9)

        # ASRI sub-indices if available
        if signals.stablecoin_risk is not None:
            sc_sent = 1.0 - 2.0 * (signals.stablecoin_risk / 100.0)
            components.append(sc_sent)
            confidences.append(0.9)

        if signals.defi_liquidity_risk is not None:
            defi_sent = 1.0 - 2.0 * (signals.defi_liquidity_risk / 100.0)
            components.append(defi_sent)
            confidences.append(0.9)

        if components:
            avg_conf = np.mean(confidences)
            sentiment = np.average(components, weights=confidences)
            return np.clip(sentiment, -1, 1), avg_conf

        return 0.0, 0.2  # Neutral with low confidence

    def _compute_tradfi_sentiment(
        self,
        signals: MacroSignals
    ) -> tuple[float, float]:
        """
        Compute sentiment from TradFi indicators.

        Returns (sentiment, confidence).
        """
        components = []
        confidences = []

        # VIX level
        if signals.vix_level is not None:
            # VIX 15 = neutral (0), VIX 10 = bullish (+0.5), VIX 30 = bearish (-0.5)
            vix_sent = 0.5 - (signals.vix_level - 15) / 30
            components.append(np.clip(vix_sent, -1, 1))
            confidences.append(0.9)

        # Yield spread (10Y-2Y)
        if signals.yield_spread is not None:
            # Positive spread = healthy economy = bullish
            # Negative spread = recession signal = bearish
            # Typical range: -1% to +2%
            spread_sent = np.clip(signals.yield_spread / 2, -1, 1)
            components.append(spread_sent)
            confidences.append(0.7)

        # Crypto-equity correlation
        if signals.btc_spy_correlation is not None:
            # High correlation = crypto follows tradfi = less independent
            # This is more of a risk factor than directional signal
            # High correlation + bearish tradfi = very bearish for crypto
            pass  # Could integrate more sophisticated logic here

        # ASRI contagion risk
        if signals.contagion_risk is not None:
            contagion_sent = 1.0 - 2.0 * (signals.contagion_risk / 100.0)
            components.append(contagion_sent)
            confidences.append(0.85)

        if components:
            avg_conf = np.mean(confidences)
            sentiment = np.average(components, weights=confidences)
            return np.clip(sentiment, -1, 1), avg_conf

        return 0.0, 0.2  # Neutral with low confidence

    def compute_from_asri_result(self, asri_normalized: float) -> float:
        """
        Directly convert ASRI normalized score to sentiment.

        ASRI 0-100 (higher = more risk) -> Sentiment -1 to +1 (higher = bullish)

        Args:
            asri_normalized: ASRI composite score 0-100

        Returns:
            Institutional sentiment -1 to +1
        """
        return 1.0 - 2.0 * (asri_normalized / 100.0)
