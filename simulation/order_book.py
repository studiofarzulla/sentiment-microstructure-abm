"""
Limit Order Book with FIFO Price-Time Priority Matching

A simple but functional order book implementation for agent-based simulation.
Supports limit orders, market orders, and order cancellation.

Author: Murad Farzulla
Date: December 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import deque
import uuid
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"


@dataclass
class Order:
    """Represents a single order in the book."""
    order_id: str
    agent_id: str
    side: Side
    price: float  # For market orders, this is None/ignored
    size: float
    order_type: OrderType = OrderType.LIMIT
    timestamp: datetime = field(default_factory=datetime.utcnow)
    filled_size: float = 0.0

    @property
    def remaining_size(self) -> float:
        return self.size - self.filled_size

    @property
    def is_filled(self) -> bool:
        return self.remaining_size <= 1e-10

    def __hash__(self):
        return hash(self.order_id)


@dataclass
class Fill:
    """Represents a trade execution."""
    fill_id: str
    buyer_order_id: str
    seller_order_id: str
    buyer_agent_id: str
    seller_agent_id: str
    price: float
    size: float
    timestamp: datetime
    aggressor_side: Side  # Which side initiated the trade


@dataclass
class PriceLevel:
    """A single price level with FIFO queue of orders."""
    price: float
    orders: deque = field(default_factory=deque)

    @property
    def total_size(self) -> float:
        return sum(o.remaining_size for o in self.orders)

    @property
    def order_count(self) -> int:
        return len(self.orders)

    def add_order(self, order: Order) -> None:
        self.orders.append(order)

    def remove_order(self, order_id: str) -> Optional[Order]:
        for i, order in enumerate(self.orders):
            if order.order_id == order_id:
                del self.orders[i]
                return order
        return None


class OrderBook:
    """
    Limit Order Book with price-time priority matching.

    Maintains separate bid and ask sides with sorted price levels.
    Supports limit orders, market orders, and cancellations.
    """

    def __init__(self, symbol: str = "BTC/USD"):
        self.symbol = symbol

        # Bids: highest price first (descending)
        # Asks: lowest price first (ascending)
        self._bids: Dict[float, PriceLevel] = {}
        self._asks: Dict[float, PriceLevel] = {}

        # Order lookup for O(1) cancellation
        self._orders: Dict[str, Order] = {}

        # Track all fills
        self._fills: List[Fill] = []

        # Statistics
        self.total_volume = 0.0
        self.trade_count = 0

    @property
    def best_bid(self) -> Optional[float]:
        """Highest bid price."""
        if not self._bids:
            return None
        return max(self._bids.keys())

    @property
    def best_ask(self) -> Optional[float]:
        """Lowest ask price."""
        if not self._asks:
            return None
        return min(self._asks.keys())

    @property
    def mid_price(self) -> Optional[float]:
        """Midpoint between best bid and ask."""
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2

    @property
    def spread(self) -> Optional[float]:
        """Absolute spread."""
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return ba - bb

    @property
    def spread_bps(self) -> Optional[float]:
        """Spread in basis points."""
        mid = self.mid_price
        spread = self.spread
        if mid is None or spread is None or mid == 0:
            return None
        return (spread / mid) * 10000

    @property
    def bid_volume(self) -> float:
        """Total volume on bid side."""
        return sum(level.total_size for level in self._bids.values())

    @property
    def ask_volume(self) -> float:
        """Total volume on ask side."""
        return sum(level.total_size for level in self._asks.values())

    @property
    def imbalance(self) -> float:
        """Order book imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)."""
        bv, av = self.bid_volume, self.ask_volume
        total = bv + av
        if total == 0:
            return 0.0
        return (bv - av) / total

    def submit_order(self, order: Order) -> List[Fill]:
        """
        Submit an order to the book.

        Returns list of fills if order matches against resting liquidity.
        Remaining unfilled portion (for limit orders) rests in the book.
        """
        if order.order_type == OrderType.MARKET:
            return self._execute_market_order(order)
        else:
            return self._execute_limit_order(order)

    def _execute_market_order(self, order: Order) -> List[Fill]:
        """Execute a market order against resting liquidity."""
        fills = []

        if order.side == Side.BUY:
            # Match against asks (lowest first)
            fills = self._match_against_side(order, self._asks, ascending=True)
        else:
            # Match against bids (highest first)
            fills = self._match_against_side(order, self._bids, ascending=False)

        return fills

    def _execute_limit_order(self, order: Order) -> List[Fill]:
        """Execute a limit order - match if possible, rest remainder."""
        fills = []

        if order.side == Side.BUY:
            # Check if we can match against asks
            if self.best_ask is not None and order.price >= self.best_ask:
                fills = self._match_against_side(
                    order, self._asks, ascending=True, limit_price=order.price
                )
        else:
            # Check if we can match against bids
            if self.best_bid is not None and order.price <= self.best_bid:
                fills = self._match_against_side(
                    order, self._bids, ascending=False, limit_price=order.price
                )

        # Rest any unfilled portion
        if not order.is_filled:
            self._add_to_book(order)

        return fills

    def _match_against_side(
        self,
        aggressor: Order,
        book_side: Dict[float, PriceLevel],
        ascending: bool,
        limit_price: Optional[float] = None
    ) -> List[Fill]:
        """
        Match an aggressor order against a side of the book.

        Args:
            aggressor: The incoming order
            book_side: Either self._bids or self._asks
            ascending: True for asks (low to high), False for bids (high to low)
            limit_price: Optional price limit for limit orders
        """
        fills = []
        prices_to_remove = []

        # Sort prices
        sorted_prices = sorted(book_side.keys(), reverse=not ascending)

        for price in sorted_prices:
            # Check limit price constraint
            if limit_price is not None:
                if ascending and price > limit_price:
                    break
                if not ascending and price < limit_price:
                    break

            level = book_side[price]
            orders_to_remove = []

            for resting_order in level.orders:
                if aggressor.is_filled:
                    break

                # Calculate fill size
                fill_size = min(aggressor.remaining_size, resting_order.remaining_size)

                # Create fill
                fill = Fill(
                    fill_id=str(uuid.uuid4()),
                    buyer_order_id=aggressor.order_id if aggressor.side == Side.BUY else resting_order.order_id,
                    seller_order_id=resting_order.order_id if aggressor.side == Side.BUY else aggressor.order_id,
                    buyer_agent_id=aggressor.agent_id if aggressor.side == Side.BUY else resting_order.agent_id,
                    seller_agent_id=resting_order.agent_id if aggressor.side == Side.BUY else aggressor.agent_id,
                    price=price,
                    size=fill_size,
                    timestamp=datetime.utcnow(),
                    aggressor_side=aggressor.side
                )
                fills.append(fill)
                self._fills.append(fill)

                # Update order states
                aggressor.filled_size += fill_size
                resting_order.filled_size += fill_size

                # Track statistics
                self.total_volume += fill_size
                self.trade_count += 1

                # Mark filled resting orders for removal
                if resting_order.is_filled:
                    orders_to_remove.append(resting_order.order_id)
                    if resting_order.order_id in self._orders:
                        del self._orders[resting_order.order_id]

                logger.debug(
                    f"Fill: {fill_size:.4f} @ {price:.2f} | "
                    f"Buyer: {fill.buyer_agent_id}, Seller: {fill.seller_agent_id}"
                )

            # Remove filled orders from level
            for order_id in orders_to_remove:
                level.remove_order(order_id)

            # Mark empty levels for removal
            if level.order_count == 0:
                prices_to_remove.append(price)

            if aggressor.is_filled:
                break

        # Clean up empty price levels
        for price in prices_to_remove:
            del book_side[price]

        return fills

    def _add_to_book(self, order: Order) -> None:
        """Add an order to the appropriate side of the book."""
        if order.side == Side.BUY:
            book_side = self._bids
        else:
            book_side = self._asks

        # Create price level if needed
        if order.price not in book_side:
            book_side[order.price] = PriceLevel(price=order.price)

        book_side[order.price].add_order(order)
        self._orders[order.order_id] = order

        logger.debug(
            f"Resting order: {order.side.value} {order.remaining_size:.4f} @ {order.price:.2f} "
            f"from {order.agent_id}"
        )

    def cancel_order(self, order_id: str) -> Optional[Order]:
        """Cancel an order by ID. Returns the cancelled order or None."""
        if order_id not in self._orders:
            return None

        order = self._orders[order_id]

        # Find and remove from book
        if order.side == Side.BUY:
            book_side = self._bids
        else:
            book_side = self._asks

        if order.price in book_side:
            book_side[order.price].remove_order(order_id)
            if book_side[order.price].order_count == 0:
                del book_side[order.price]

        del self._orders[order_id]
        return order

    def get_depth(self, levels: int = 10) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Get order book depth.

        Returns:
            (bids, asks) where each is list of (price, size) tuples
        """
        # Bids: highest first
        bid_prices = sorted(self._bids.keys(), reverse=True)[:levels]
        bids = [(p, self._bids[p].total_size) for p in bid_prices]

        # Asks: lowest first
        ask_prices = sorted(self._asks.keys())[:levels]
        asks = [(p, self._asks[p].total_size) for p in ask_prices]

        return bids, asks

    def get_snapshot(self) -> dict:
        """Get a snapshot of the current order book state."""
        bids, asks = self.get_depth(10)

        return {
            "symbol": self.symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "mid_price": self.mid_price,
            "spread": self.spread,
            "spread_bps": self.spread_bps,
            "bid_volume": self.bid_volume,
            "ask_volume": self.ask_volume,
            "imbalance": self.imbalance,
            "bids": bids,
            "asks": asks,
            "total_volume": self.total_volume,
            "trade_count": self.trade_count,
        }

    def reset(self) -> None:
        """Clear the order book."""
        self._bids.clear()
        self._asks.clear()
        self._orders.clear()
        self._fills.clear()
        self.total_volume = 0.0
        self.trade_count = 0

    def __repr__(self) -> str:
        return (
            f"OrderBook({self.symbol}, "
            f"bid={self.best_bid}, ask={self.best_ask}, "
            f"spread={self.spread_bps:.2f}bps)" if self.spread_bps else
            f"OrderBook({self.symbol}, empty)"
        )


# Convenience factory function
def create_order(
    agent_id: str,
    side: str,
    size: float,
    price: Optional[float] = None,
    order_type: str = "limit"
) -> Order:
    """Create an order with sensible defaults."""
    return Order(
        order_id=str(uuid.uuid4()),
        agent_id=agent_id,
        side=Side.BUY if side.lower() == "buy" else Side.SELL,
        price=price if price else 0.0,
        size=size,
        order_type=OrderType.LIMIT if order_type.lower() == "limit" else OrderType.MARKET
    )


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.DEBUG)

    book = OrderBook("BTC/USD")

    # Add some initial liquidity
    book.submit_order(create_order("mm1", "buy", 1.0, 99.0))
    book.submit_order(create_order("mm1", "buy", 2.0, 98.0))
    book.submit_order(create_order("mm1", "sell", 1.0, 101.0))
    book.submit_order(create_order("mm1", "sell", 2.0, 102.0))

    print(f"\nInitial book: {book}")
    print(f"Snapshot: {book.get_snapshot()}")

    # Incoming market buy
    fills = book.submit_order(create_order("trader1", "buy", 0.5, order_type="market"))
    print(f"\nMarket buy 0.5 BTC: {len(fills)} fills")
    for f in fills:
        print(f"  Fill: {f.size} @ {f.price}")

    print(f"\nAfter trade: {book}")
    print(f"Trade count: {book.trade_count}, Volume: {book.total_volume}")
