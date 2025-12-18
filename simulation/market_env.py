"""
Mesa Market Environment for Agent-Based Simulation

Coordinates agent interactions, order book matching, and sentiment signals.
Uses Mesa's scheduling infrastructure with custom market mechanics.

Author: Murad Farzulla
Date: December 2025
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import logging

from mesa import Model
from mesa.time import RandomActivation

from simulation.order_book import OrderBook, Order, Fill, Side, OrderType, create_order

# Import SentimentTick for enhanced signal support
try:
    from signals.models import SentimentTick
    SENTIMENT_TICK_AVAILABLE = True
except ImportError:
    SENTIMENT_TICK_AVAILABLE = False
    SentimentTick = None

logger = logging.getLogger(__name__)


@dataclass
class MarketState:
    """Current state of the market for agent decision-making."""
    timestamp: datetime
    step: int
    mid_price: Optional[float]
    best_bid: Optional[float]
    best_ask: Optional[float]
    spread: Optional[float]
    spread_bps: Optional[float]
    imbalance: float
    sentiment: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    total_uncertainty: float
    regime: str
    recent_fills: List[Fill] = field(default_factory=list)
    total_volume: float = 0.0
    trade_count: int = 0

    # Extended fields from SentimentTick (for multi-scale analysis)
    retail_sentiment: float = 0.0
    institutional_sentiment: float = 0.0
    divergence: float = 0.0
    asri_alert_level: str = "unknown"
    is_high_divergence: bool = False
    macro_weight: float = 0.0
    micro_weight: float = 1.0


@dataclass
class AgentPosition:
    """Track an agent's position and PnL."""
    agent_id: str
    cash: float = 100000.0  # Starting cash
    position: float = 0.0  # BTC position
    realized_pnl: float = 0.0
    trades: int = 0

    @property
    def unrealized_pnl(self) -> Callable[[float], float]:
        """Returns function to compute unrealized PnL given current price."""
        def compute(current_price: float) -> float:
            return self.position * current_price
        return compute

    def update_from_fill(self, fill: Fill, is_buyer: bool) -> None:
        """Update position from a fill."""
        if is_buyer:
            self.cash -= fill.price * fill.size
            self.position += fill.size
        else:
            self.cash += fill.price * fill.size
            self.position -= fill.size
        self.trades += 1


class BaseMarketAgent:
    """Base class for market agents in Mesa environment."""

    def __init__(self, unique_id: str, model: 'CryptoMarketModel'):
        self.unique_id = unique_id
        self.model = model
        self.position = AgentPosition(agent_id=unique_id)

    def step(self) -> List[Order]:
        """
        Agent's action for this timestep.

        Returns list of orders to submit to the order book.
        Override in subclasses.
        """
        raise NotImplementedError

    def on_fill(self, fill: Fill, is_buyer: bool) -> None:
        """Called when one of agent's orders is filled."""
        self.position.update_from_fill(fill, is_buyer)

    def get_state(self) -> dict:
        """Get agent's current state for logging/analysis."""
        return {
            "agent_id": self.unique_id,
            "cash": self.position.cash,
            "position": self.position.position,
            "realized_pnl": self.position.realized_pnl,
            "trades": self.position.trades,
        }


class MarketMakerAgent(BaseMarketAgent):
    """
    Market maker that provides two-sided liquidity.

    Quotes are adjusted based on:
    - Inventory risk (Avellaneda-Stoikov)
    - Sentiment signal
    - Uncertainty premium
    """

    def __init__(
        self,
        unique_id: str,
        model: 'CryptoMarketModel',
        base_spread_bps: float = 10.0,
        inventory_aversion: float = 0.001,
        sentiment_sensitivity: float = 0.5,
        uncertainty_sensitivity: float = 1.5,
        quote_size: float = 0.1,
    ):
        super().__init__(unique_id, model)
        self.base_spread_bps = base_spread_bps
        self.inventory_aversion = inventory_aversion
        self.sentiment_sensitivity = sentiment_sensitivity
        self.uncertainty_sensitivity = uncertainty_sensitivity
        self.quote_size = quote_size

    def step(self) -> List[Order]:
        """Generate two-sided quotes."""
        state = self.model.get_market_state()

        if state.mid_price is None:
            return []

        mid = state.mid_price

        # Base spread
        half_spread = (self.base_spread_bps / 10000) * mid / 2

        # Inventory skew (mean reversion)
        inventory_skew = -self.inventory_aversion * self.position.position * mid

        # Sentiment adjustment
        sentiment_adj = state.sentiment * self.sentiment_sensitivity * half_spread

        # Uncertainty premium
        uncertainty_premium = state.total_uncertainty * self.uncertainty_sensitivity * half_spread

        # Final quotes
        bid_price = mid - half_spread + inventory_skew + sentiment_adj - uncertainty_premium
        ask_price = mid + half_spread + inventory_skew + sentiment_adj + uncertainty_premium

        orders = [
            create_order(self.unique_id, "buy", self.quote_size, bid_price),
            create_order(self.unique_id, "sell", self.quote_size, ask_price),
        ]

        logger.debug(
            f"MM {self.unique_id}: bid={bid_price:.2f} ask={ask_price:.2f} "
            f"inv={self.position.position:.2f}"
        )

        return orders


class InformedTraderAgent(BaseMarketAgent):
    """
    Trades on sentiment signal when confident.

    Only trades when:
    - Sentiment exceeds threshold
    - Epistemic uncertainty is low (model confident)
    """

    def __init__(
        self,
        unique_id: str,
        model: 'CryptoMarketModel',
        sentiment_threshold: float = 0.3,
        uncertainty_threshold: float = 0.1,
        trade_size: float = 0.5,
        position_limit: float = 5.0,
    ):
        super().__init__(unique_id, model)
        self.sentiment_threshold = sentiment_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.trade_size = trade_size
        self.position_limit = position_limit

    def step(self) -> List[Order]:
        """Trade on sentiment signal if confident."""
        state = self.model.get_market_state()

        if state.mid_price is None:
            return []

        # Check if we have a confident signal
        if state.epistemic_uncertainty > self.uncertainty_threshold:
            return []  # Model not confident enough

        orders = []

        # Bullish signal
        if state.sentiment > self.sentiment_threshold:
            if self.position.position < self.position_limit:
                orders.append(
                    create_order(self.unique_id, "buy", self.trade_size, order_type="market")
                )
                logger.debug(f"Informed {self.unique_id}: BUY signal (sent={state.sentiment:.2f})")

        # Bearish signal
        elif state.sentiment < -self.sentiment_threshold:
            if self.position.position > -self.position_limit:
                orders.append(
                    create_order(self.unique_id, "sell", self.trade_size, order_type="market")
                )
                logger.debug(f"Informed {self.unique_id}: SELL signal (sent={state.sentiment:.2f})")

        return orders


class NoiseTraderAgent(BaseMarketAgent):
    """
    Random trader that generates liquidity demand.

    Submits market orders at random intervals with slight sentiment bias.
    """

    def __init__(
        self,
        unique_id: str,
        model: 'CryptoMarketModel',
        trade_probability: float = 0.3,
        sentiment_bias: float = 0.1,
        min_size: float = 0.01,
        max_size: float = 0.2,
    ):
        super().__init__(unique_id, model)
        self.trade_probability = trade_probability
        self.sentiment_bias = sentiment_bias
        self.min_size = min_size
        self.max_size = max_size

    def step(self) -> List[Order]:
        """Maybe submit a random order."""
        if np.random.random() > self.trade_probability:
            return []

        state = self.model.get_market_state()

        # Direction biased by sentiment
        buy_prob = 0.5 + state.sentiment * self.sentiment_bias
        is_buy = np.random.random() < buy_prob

        # Random size
        size = np.random.uniform(self.min_size, self.max_size)

        side = "buy" if is_buy else "sell"
        order = create_order(self.unique_id, side, size, order_type="market")

        logger.debug(f"Noise {self.unique_id}: {side} {size:.3f}")

        return [order]


class CryptoMarketModel(Model):
    """
    Mesa model coordinating agents and order book.

    Manages:
    - Order book state
    - Sentiment signal (from external source or synthetic)
    - Agent scheduling
    - Fill notification
    """

    def __init__(
        self,
        symbol: str = "BTC/USD",
        initial_price: float = 100.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.symbol = symbol
        self.initial_price = initial_price

        # Random state
        if seed is not None:
            np.random.seed(seed)

        # Order book
        self.order_book = OrderBook(symbol)

        # Agent scheduler
        self.schedule = RandomActivation(self)

        # Agent registry
        self._agents: Dict[str, BaseMarketAgent] = {}

        # Market state
        self._current_step = 0
        self._sentiment = 0.0
        self._epistemic_uncertainty = 0.05
        self._aleatoric_uncertainty = 0.2
        self._regime = "neutral"
        self._recent_fills: List[Fill] = []

        # Extended sentiment state (from SentimentTick)
        self._retail_sentiment = 0.0
        self._institutional_sentiment = 0.0
        self._divergence = 0.0
        self._asri_alert_level = "unknown"
        self._is_high_divergence = False
        self._macro_weight = 0.0
        self._micro_weight = 1.0

        # History for analysis
        self.history: List[dict] = []

        # Track last known price for reseeding
        self._last_known_price = initial_price

        # Initialize with some liquidity around initial price
        self._seed_liquidity()

    def _seed_liquidity(self) -> None:
        """Add initial liquidity to the book."""
        # Use last known price for reseeding (handles mid-run depletion)
        ref_price = self._last_known_price

        for i in range(5):
            bid_price = ref_price * (1 - 0.001 * (i + 1))
            ask_price = ref_price * (1 + 0.001 * (i + 1))

            self.order_book.submit_order(
                create_order("__seed__", "buy", 1.0, bid_price)
            )
            self.order_book.submit_order(
                create_order("__seed__", "sell", 1.0, ask_price)
            )

    def add_agent(self, agent: BaseMarketAgent) -> None:
        """Register an agent with the model."""
        self._agents[agent.unique_id] = agent
        self.schedule.add(agent)

    def set_sentiment(
        self,
        sentiment: float,
        epistemic: float,
        aleatoric: float,
        regime: str = "neutral"
    ) -> None:
        """Update the current sentiment state (called externally or from data feed)."""
        self._sentiment = sentiment
        self._epistemic_uncertainty = epistemic
        self._aleatoric_uncertainty = aleatoric
        self._regime = regime

    def set_sentiment_tick(self, tick: 'SentimentTick') -> None:
        """
        Update sentiment from a SentimentTick (multi-scale signal).

        This method accepts the full SentimentTick from the SignalComposer,
        extracting both the core 4-tuple and extended fields for analysis.
        """
        # Core sentiment state
        self._sentiment = tick.sentiment
        self._epistemic_uncertainty = tick.epistemic_uncertainty
        self._aleatoric_uncertainty = tick.aleatoric_uncertainty
        self._regime = tick.regime

        # Extended state for multi-scale analysis
        self._retail_sentiment = tick.retail_sentiment
        self._institutional_sentiment = tick.institutional_sentiment
        self._divergence = tick.divergence
        self._asri_alert_level = tick.asri_alert_level
        self._is_high_divergence = tick.is_high_divergence
        self._macro_weight = tick.macro_weight
        self._micro_weight = tick.micro_weight

    def get_market_state(self) -> MarketState:
        """Get current market state for agent decision-making."""
        return MarketState(
            timestamp=datetime.utcnow(),
            step=self._current_step,
            mid_price=self.order_book.mid_price,
            best_bid=self.order_book.best_bid,
            best_ask=self.order_book.best_ask,
            spread=self.order_book.spread,
            spread_bps=self.order_book.spread_bps,
            imbalance=self.order_book.imbalance,
            sentiment=self._sentiment,
            epistemic_uncertainty=self._epistemic_uncertainty,
            aleatoric_uncertainty=self._aleatoric_uncertainty,
            total_uncertainty=self._epistemic_uncertainty + self._aleatoric_uncertainty,
            regime=self._regime,
            recent_fills=self._recent_fills.copy(),
            total_volume=self.order_book.total_volume,
            trade_count=self.order_book.trade_count,
            # Extended fields
            retail_sentiment=self._retail_sentiment,
            institutional_sentiment=self._institutional_sentiment,
            divergence=self._divergence,
            asri_alert_level=self._asri_alert_level,
            is_high_divergence=self._is_high_divergence,
            macro_weight=self._macro_weight,
            micro_weight=self._micro_weight,
        )

    def step(self) -> None:
        """Advance simulation by one step."""
        self._current_step += 1
        self._recent_fills = []

        # Collect orders from all agents
        all_orders: List[Order] = []

        for agent in self._agents.values():
            orders = agent.step()
            all_orders.extend(orders)

        # Shuffle order arrival (random priority)
        np.random.shuffle(all_orders)

        # Submit orders to book and process fills
        for order in all_orders:
            fills = self.order_book.submit_order(order)
            self._recent_fills.extend(fills)

            # Notify agents of their fills
            for fill in fills:
                if fill.buyer_agent_id in self._agents:
                    self._agents[fill.buyer_agent_id].on_fill(fill, is_buyer=True)
                if fill.seller_agent_id in self._agents:
                    self._agents[fill.seller_agent_id].on_fill(fill, is_buyer=False)

        # Record history
        self._record_step()

    def _record_step(self) -> None:
        """Record current state for analysis."""
        state = self.get_market_state()

        # Update last known price for reseeding
        if state.mid_price is not None:
            self._last_known_price = state.mid_price

        record = {
            "step": self._current_step,
            "timestamp": state.timestamp.isoformat(),
            "mid_price": state.mid_price,
            "best_bid": state.best_bid,
            "best_ask": state.best_ask,
            "spread_bps": state.spread_bps,
            "imbalance": state.imbalance,
            "sentiment": state.sentiment,
            "epistemic_uncertainty": state.epistemic_uncertainty,
            "aleatoric_uncertainty": state.aleatoric_uncertainty,
            "regime": state.regime,
            "fills_this_step": len(self._recent_fills),
            "total_volume": state.total_volume,
            "trade_count": state.trade_count,
        }

        # Agent positions
        for agent_id, agent in self._agents.items():
            record[f"{agent_id}_position"] = agent.position.position
            record[f"{agent_id}_cash"] = agent.position.cash

        self.history.append(record)

    def run_simulation(
        self,
        n_steps: int,
        sentiment_generator: Optional[Callable[[int], tuple]] = None,
    ) -> List[dict]:
        """
        Run simulation for n_steps.

        Args:
            n_steps: Number of timesteps to simulate
            sentiment_generator: Optional function (step) -> (sentiment, epistemic, aleatoric, regime)
                                If None, uses current sentiment values
        """
        for step in range(n_steps):
            # Update sentiment if generator provided
            if sentiment_generator is not None:
                sent, epi, aleat, regime = sentiment_generator(step)
                self.set_sentiment(sent, epi, aleat, regime)

            self.step()

            if step % 100 == 0:
                mid = self.order_book.mid_price
                spread = self.order_book.spread_bps
                mid_str = f"{mid:.2f}" if mid is not None else "N/A"
                spread_str = f"{spread:.2f}" if spread is not None else "N/A"
                logger.info(
                    f"Step {step}/{n_steps}: mid={mid_str}, "
                    f"spread={spread_str}bps, "
                    f"trades={self.order_book.trade_count}"
                )

                # Reseed liquidity if book is depleted
                if mid is None:
                    logger.warning("Order book depleted, reseeding liquidity...")
                    self._seed_liquidity()

        return self.history


def create_default_market(
    n_market_makers: int = 2,
    n_informed: int = 3,
    n_noise: int = 10,
    initial_price: float = 100.0,
    seed: int = 42,
) -> CryptoMarketModel:
    """Create a market with default agent configuration."""

    model = CryptoMarketModel(initial_price=initial_price, seed=seed)

    # Add market makers
    for i in range(n_market_makers):
        mm = MarketMakerAgent(
            f"mm_{i}",
            model,
            base_spread_bps=8.0 + i * 2,  # Slight variation
            quote_size=0.5,
        )
        model.add_agent(mm)

    # Add informed traders
    for i in range(n_informed):
        informed = InformedTraderAgent(
            f"informed_{i}",
            model,
            sentiment_threshold=0.2 + i * 0.1,  # Different thresholds
            trade_size=0.3,
        )
        model.add_agent(informed)

    # Add noise traders
    for i in range(n_noise):
        noise = NoiseTraderAgent(
            f"noise_{i}",
            model,
            trade_probability=0.2,
        )
        model.add_agent(noise)

    return model


if __name__ == "__main__":
    import pandas as pd

    logging.basicConfig(level=logging.INFO)

    # Create market
    model = create_default_market(seed=42)

    # Simple sentiment generator (synthetic)
    def sentiment_gen(step):
        # Regime switching
        if step < 200:
            regime = "neutral"
            base_sent = 0.0
        elif step < 400:
            regime = "bullish"
            base_sent = 0.4
        elif step < 600:
            regime = "bearish"
            base_sent = -0.4
        else:
            regime = "neutral"
            base_sent = 0.0

        # Add noise
        sentiment = np.clip(base_sent + np.random.normal(0, 0.1), -1, 1)
        epistemic = 0.03 + abs(sentiment) * 0.02
        aleatoric = 0.15 + (1 - abs(sentiment)) * 0.1

        return sentiment, epistemic, aleatoric, regime

    # Run
    print("Running simulation...")
    history = model.run_simulation(1000, sentiment_generator=sentiment_gen)

    # Results
    df = pd.DataFrame(history)
    print(f"\nSimulation complete: {len(df)} steps")
    print(f"Total trades: {df['trade_count'].iloc[-1]}")
    print(f"Total volume: {df['total_volume'].iloc[-1]:.2f}")
    print(f"\nFinal price: {df['mid_price'].iloc[-1]:.2f}")
    print(f"Price range: {df['mid_price'].min():.2f} - {df['mid_price'].max():.2f}")
    print(f"\nMean spread: {df['spread_bps'].mean():.2f} bps")

    # Save
    df.to_csv("/tmp/mesa_simulation_test.csv", index=False)
    print(f"\nSaved to /tmp/mesa_simulation_test.csv")
