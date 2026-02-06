"""
Example: Integrating Dome API Prediction Markets with SignalComposer

This demonstrates how to add prediction market sentiment as a third
dimension to your existing macro (ASRI) + micro (Reddit) sentiment blend.

Prediction markets provide "informed sentiment" - aggregated beliefs
from participants with real money at stake, distinct from:
- Retail sentiment (Reddit/CryptoBERT)
- Institutional sentiment (ASRI/Fear & Greed)
"""

from datetime import datetime
from typing import Optional, Tuple
import logging

from dome_client import DomeAPIClient, get_prediction_market_sentiment
from signals.signal_composer import SignalComposer
from signals.models import MacroSignals, SentimentTick

logger = logging.getLogger(__name__)


class EnhancedSignalComposer:
    """
    Extended SignalComposer that includes prediction market sentiment.
    
    Blends three sentiment sources:
    1. Macro (ASRI) - institutional risk indicators
    2. Micro (CryptoBERT) - retail social media sentiment  
    3. Prediction Markets - informed market-implied sentiment
    """
    
    def __init__(
        self,
        prediction_weight: float = 0.20,  # Weight for prediction markets
        macro_weight: float = 0.30,
        micro_weight: float = 0.50,
        dome_api_key: Optional[str] = None,
    ):
        """
        Initialize enhanced composer.
        
        Args:
            prediction_weight: Weight for prediction market sentiment [0, 1]
            macro_weight: Weight for ASRI signals [0, 1]
            micro_weight: Weight for CryptoBERT signals [0, 1]
            dome_api_key: Dome API key (defaults to env var)
        """
        # Normalize weights
        total = prediction_weight + macro_weight + micro_weight
        self.prediction_weight = prediction_weight / total
        self.macro_weight = macro_weight / total
        self.micro_weight = micro_weight / total
        
        # Base composer for macro + micro blend
        self.base_composer = SignalComposer(
            macro_weight=self.macro_weight / (self.macro_weight + self.micro_weight),
            micro_weight=self.micro_weight / (self.macro_weight + self.micro_weight),
        )
        
        # Dome API client
        self.dome_client = DomeAPIClient(api_key=dome_api_key)
        
        logger.info(f"Enhanced SignalComposer initialized:")
        logger.info(f"  Prediction Markets: {self.prediction_weight:.1%}")
        logger.info(f"  Macro (ASRI): {self.macro_weight:.1%}")
        logger.info(f"  Micro (Reddit): {self.micro_weight:.1%}")
    
    def compose_enhanced(
        self,
        macro_signals: Optional[MacroSignals],
        micro_sentiment: Tuple[float, float, float],
        prediction_market_slugs: Optional[list] = None,
        price: Optional[float] = None,
    ) -> SentimentTick:
        """
        Compose sentiment tick with all three sources.
        
        Args:
            macro_signals: ASRI macro signals (can be None)
            micro_sentiment: (sentiment, epistemic, aleatoric) from CryptoBERT
            prediction_market_slugs: List of market slugs to aggregate
            price: Current price for divergence tracking
            
        Returns:
            Enhanced SentimentTick with prediction market component
        """
        # Get base tick (macro + micro)
        base_tick = self.base_composer.compose(macro_signals, micro_sentiment, price)
        
        # Get prediction market sentiment
        pred_result = self.dome_client.get_crypto_sentiment(
            market_slugs=prediction_market_slugs
        )
        
        if pred_result:
            pred_sentiment, pred_uncertainty = pred_result.to_sentiment_tick_component()
            
            # Blend with base sentiment
            blended_sentiment = (
                self.prediction_weight * pred_sentiment +
                (1 - self.prediction_weight) * base_tick.sentiment
            )
            
            # Combine uncertainties (max of epistemic components)
            combined_epistemic = max(
                base_tick.epistemic_uncertainty,
                pred_uncertainty
            )
            
            # Create enhanced tick
            enhanced_tick = SentimentTick(
                timestamp=datetime.utcnow(),
                sentiment=blended_sentiment,
                epistemic_uncertainty=combined_epistemic,
                aleatoric_uncertainty=base_tick.aleatoric_uncertainty,
                regime=base_tick.regime,
                retail_sentiment=base_tick.retail_sentiment,
                institutional_sentiment=base_tick.institutional_sentiment,
                divergence=base_tick.divergence,
                macro_weight=self.macro_weight,
                micro_weight=self.micro_weight,
                asri_alert_level=base_tick.asri_alert_level,
                data_completeness=base_tick.data_completeness,
            )
            
            logger.info(f"Enhanced sentiment: {blended_sentiment:.3f} "
                       f"(prediction: {pred_sentiment:.3f}, base: {base_tick.sentiment:.3f})")
            
            return enhanced_tick
        else:
            logger.warning("Prediction market data unavailable, using base tick")
            return base_tick


# ============================================================================
# Usage Example
# ============================================================================

def example_usage():
    """Example of using enhanced signal composer."""
    
    # Initialize composer
    composer = EnhancedSignalComposer(
        prediction_weight=0.20,
        macro_weight=0.30,
        micro_weight=0.50,
    )
    
    # Simulate macro signals (normally from ASRI adapter)
    macro_signals = None  # Would come from ASRIAdapter.fetch_all()
    
    # Simulate micro sentiment (normally from CryptoBERT)
    micro_sentiment = (0.3, 0.05, 0.15)  # (sentiment, epistemic, aleatoric)
    
    # Crypto prediction market slugs
    crypto_markets = [
        "will-bitcoin-price-be-above-50000-usd-on-december-31-2025",
        "will-bitcoin-price-be-above-60000-usd-on-december-31-2025",
    ]
    
    # Compose enhanced tick
    tick = composer.compose_enhanced(
        macro_signals=macro_signals,
        micro_sentiment=micro_sentiment,
        prediction_market_slugs=crypto_markets,
        price=45000.0,  # Current BTC price
    )
    
    print(f"\n{'='*60}")
    print("ENHANCED SENTIMENT TICK")
    print(f"{'='*60}")
    print(f"Final Sentiment: {tick.sentiment:.3f}")
    print(f"Epistemic Uncertainty: {tick.epistemic_uncertainty:.3f}")
    print(f"Aleatoric Uncertainty: {tick.aleatoric_uncertainty:.3f}")
    print(f"Regime: {tick.regime}")
    print(f"\nComponents:")
    print(f"  Retail (Reddit): {tick.retail_sentiment:.3f}")
    print(f"  Institutional (ASRI): {tick.institutional_sentiment:.3f}")
    print(f"  Prediction Markets: (blended in)")
    print(f"  Divergence: {tick.divergence:.3f}")


if __name__ == '__main__':
    example_usage()
