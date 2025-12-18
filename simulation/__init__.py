"""
Simulation module for agent-based market modeling.

Components:
- OrderBook: FIFO price-time priority matching engine
- CryptoMarketModel: Mesa-based market environment
- Agent types: MarketMaker, InformedTrader, NoiseTrader
"""

from simulation.order_book import (
    OrderBook,
    Order,
    Fill,
    Side,
    OrderType,
    PriceLevel,
    create_order,
)

from simulation.market_env import (
    CryptoMarketModel,
    MarketState,
    AgentPosition,
    BaseMarketAgent,
    MarketMakerAgent,
    InformedTraderAgent,
    NoiseTraderAgent,
    create_default_market,
)

__all__ = [
    # Order book
    "OrderBook",
    "Order",
    "Fill",
    "Side",
    "OrderType",
    "PriceLevel",
    "create_order",
    # Market environment
    "CryptoMarketModel",
    "MarketState",
    "AgentPosition",
    "BaseMarketAgent",
    "MarketMakerAgent",
    "InformedTraderAgent",
    "NoiseTraderAgent",
    "create_default_market",
]
