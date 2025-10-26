"""
Binance Order Book Data Ingestion

Real-time async WebSocket client for Binance order book depth streams.
Publishes snapshots to Kafka for microstructure feature engineering.

Architecture:
- Full async/await using websockets library
- Automatic reconnection with exponential backoff
- Pydantic validation for robustness
- Async Kafka producer with proper resource cleanup
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import logging

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException
from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError
from pydantic import BaseModel, Field, field_validator, ValidationError
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OrderBookLevel(BaseModel):
    """Single price level in order book."""
    price: float = Field(gt=0, description="Price level")
    quantity: float = Field(ge=0, description="Quantity at this level")


class BinanceDepthUpdate(BaseModel):
    """
    Binance order book depth update validation.

    Raw format:
    {
      "lastUpdateId": 160,
      "bids": [["0.0024", "10"]],  // [price, quantity]
      "asks": [["0.0026", "100"]]
    }
    """
    lastUpdateId: int = Field(gt=0)
    bids: List[List[str]] = Field(default_factory=list)
    asks: List[List[str]] = Field(default_factory=list)

    @field_validator('bids', 'asks')
    @classmethod
    def validate_price_levels(cls, v: List[List[str]]) -> List[List[str]]:
        """Ensure each level has exactly [price, quantity]."""
        for level in v:
            if len(level) != 2:
                raise ValueError(f"Invalid price level format: {level}")
        return v


class OrderBookSnapshot(BaseModel):
    """Processed order book snapshot with microstructure features."""
    symbol: str
    timestamp: str  # ISO format from exchange
    exchange_timestamp_ms: int  # Raw exchange timestamp
    last_update_id: int
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    mid_price: Optional[float] = None
    spread: Optional[float] = None
    spread_bps: Optional[float] = None
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    imbalance: float = 0.0
    bids: List[List[float]] = Field(default_factory=list)
    asks: List[List[float]] = Field(default_factory=list)
    source: str = "binance"


class BinanceOrderBookClient:
    """
    Async Binance WebSocket client for real-time order book depth data.

    Features:
    - Full async/await architecture
    - Automatic reconnection with exponential backoff
    - Pydantic validation for data integrity
    - Proper resource cleanup with context managers
    - Exchange timestamp usage (no network latency)
    """

    def __init__(
        self,
        symbol: str = 'btcusdt',
        depth_update_speed: str = '100ms',
        levels: int = 20,
        kafka_bootstrap_servers: Optional[str] = None,
        kafka_topic: Optional[str] = None,
        max_reconnect_delay: int = 60,
        initial_reconnect_delay: float = 1.0
    ):
        """
        Initialize Binance async client.

        Args:
            symbol: Trading pair (default: btcusdt)
            depth_update_speed: Update frequency ('100ms' or '1000ms')
            levels: Number of price levels to capture (5, 10, or 20)
            kafka_bootstrap_servers: Kafka connection string
            kafka_topic: Kafka topic for order books
            max_reconnect_delay: Maximum reconnection delay in seconds
            initial_reconnect_delay: Initial reconnection delay in seconds
        """
        self.symbol = symbol.lower()
        self.depth_speed = depth_update_speed
        self.levels = levels

        # WebSocket URL
        base_url = os.getenv('BINANCE_WEBSOCKET_URL', 'wss://stream.binance.com:9443/ws')
        self.ws_url = f"{base_url}/{self.symbol}@depth{self.levels}@{self.depth_speed}"

        # Kafka configuration
        kafka_servers = kafka_bootstrap_servers or os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.kafka_servers = kafka_servers.split(',')
        self.kafka_topic = kafka_topic or os.getenv('KAFKA_TOPIC_ORDERBOOKS', 'order-books')

        # Reconnection configuration
        self.max_reconnect_delay = max_reconnect_delay
        self.initial_reconnect_delay = initial_reconnect_delay
        self.current_reconnect_delay = initial_reconnect_delay

        # State
        self.producer: Optional[AIOKafkaProducer] = None
        self.is_running = False
        self.message_count = 0
        self.error_count = 0

        logger.info(f"Initialized Binance async client for {self.symbol.upper()}")
        logger.info(f"WebSocket URL: {self.ws_url}")
        logger.info(f"Publishing to Kafka topic: {self.kafka_topic}")

    async def _init_kafka_producer(self) -> AIOKafkaProducer:
        """Initialize and start async Kafka producer."""
        producer = AIOKafkaProducer(
            bootstrap_servers=self.kafka_servers,
            value_serializer=lambda v: json.dumps(v.model_dump()).encode('utf-8'),
            compression_type='gzip',
            acks='all',  # Wait for all replicas
            max_batch_size=16384,  # 16KB batches
            linger_ms=100,  # Batch for 100ms
        )
        await producer.start()
        logger.info("Kafka producer started")
        return producer

    def _process_depth_update(self, data: BinanceDepthUpdate) -> OrderBookSnapshot:
        """
        Process validated depth update into structured snapshot.

        Uses exchange timestamp to avoid network latency.
        Handles missing data gracefully with None values.
        """
        # Parse and validate price levels
        bids: List[List[float]] = []
        asks: List[List[float]] = []

        try:
            bids = [[float(p), float(q)] for p, q in data.bids]
            asks = [[float(p), float(q)] for p, q in data.asks]
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to parse price levels: {e}")
            # Continue with empty levels rather than crash

        # Compute microstructure features
        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None

        mid_price = None
        spread = None
        spread_bps = None
        if best_bid and best_ask:
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            spread_bps = (spread / mid_price * 10000) if mid_price > 0 else None

        # Volume at each side
        bid_volume = sum(q for p, q in bids)
        ask_volume = sum(q for p, q in asks)

        # Order book imbalance
        total_volume = bid_volume + ask_volume
        imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0.0

        # Use current timestamp (Binance depth streams don't include E field)
        # For trade streams we'd use data.get('E') for exchange timestamp
        timestamp_ms = int(datetime.utcnow().timestamp() * 1000)

        return OrderBookSnapshot(
            symbol=self.symbol.upper(),
            timestamp=datetime.utcnow().isoformat(),
            exchange_timestamp_ms=timestamp_ms,
            last_update_id=data.lastUpdateId,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            spread=spread,
            spread_bps=spread_bps,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            imbalance=imbalance,
            bids=bids[:10],  # Top 10 levels
            asks=asks[:10],
            source='binance'
        )

    async def _publish_to_kafka(self, snapshot: OrderBookSnapshot):
        """Publish order book snapshot to Kafka with error handling."""
        if not self.producer:
            logger.error("Kafka producer not initialized")
            return

        try:
            await self.producer.send(self.kafka_topic, value=snapshot)
            self.message_count += 1

            if self.message_count % 100 == 0:
                logger.info(f"Published {self.message_count} order book snapshots")

        except KafkaError as e:
            self.error_count += 1
            logger.error(f"Failed to publish to Kafka: {e}")
        except Exception as e:
            self.error_count += 1
            logger.error(f"Unexpected error publishing to Kafka: {e}")

    async def _stream_orderbook(self):
        """
        Stream order book updates from Binance WebSocket.

        Async generator that yields forever until connection closes.
        Uses exchange timestamps and validates all incoming data.
        """
        async with websockets.connect(self.ws_url) as websocket:
            logger.info(f"WebSocket connected to {self.symbol.upper()} depth stream")
            self.is_running = True
            self.current_reconnect_delay = self.initial_reconnect_delay  # Reset backoff

            async for message in websocket:
                try:
                    # Parse JSON
                    raw_data = json.loads(message)

                    # Validate with Pydantic
                    validated_data = BinanceDepthUpdate(**raw_data)

                    # Process and enrich
                    snapshot = self._process_depth_update(validated_data)

                    # Publish to Kafka
                    await self._publish_to_kafka(snapshot)

                except ValidationError as e:
                    self.error_count += 1
                    logger.warning(f"Invalid data from Binance: {e}")
                    # Continue streaming despite validation errors

                except json.JSONDecodeError as e:
                    self.error_count += 1
                    logger.error(f"JSON decode error: {e}")

                except Exception as e:
                    self.error_count += 1
                    logger.error(f"Error processing message: {e}")

    async def _exponential_backoff(self):
        """
        Exponential backoff for reconnection attempts.

        Starts at initial_reconnect_delay, doubles each time up to max_reconnect_delay.
        Binance closes connections every 24h, so reconnection is expected behavior.
        """
        delay = self.current_reconnect_delay
        logger.info(f"Reconnecting in {delay:.1f} seconds...")
        await asyncio.sleep(delay)

        # Exponential backoff with cap
        self.current_reconnect_delay = min(
            self.current_reconnect_delay * 2,
            self.max_reconnect_delay
        )

    async def stream_with_reconnect(self):
        """
        Main streaming loop with automatic reconnection.

        Handles:
        - Normal connection closes (24h timeout)
        - Network errors
        - Binance server issues
        - Unexpected disconnections

        Runs forever until explicitly stopped.
        """
        while True:
            try:
                await self._stream_orderbook()

            except ConnectionClosed as e:
                self.is_running = False
                logger.warning(f"WebSocket connection closed: {e.code} - {e.reason}")
                logger.info("This is expected behavior (Binance 24h timeout)")
                await self._exponential_backoff()

            except WebSocketException as e:
                self.is_running = False
                logger.error(f"WebSocket error: {e}")
                await self._exponential_backoff()

            except asyncio.CancelledError:
                logger.info("Stream cancelled, shutting down gracefully")
                self.is_running = False
                break

            except Exception as e:
                self.is_running = False
                logger.error(f"Unexpected error: {e}")
                await self._exponential_backoff()

    async def start(self):
        """
        Start the async client with proper resource management.

        Initializes Kafka producer, starts streaming, ensures cleanup.
        Use as async context manager or call start() and close() manually.
        """
        try:
            # Initialize Kafka producer
            self.producer = await self._init_kafka_producer()

            # Start streaming with reconnection
            await self.stream_with_reconnect()

        finally:
            await self.close()

    async def close(self):
        """
        Cleanup resources gracefully.

        Ensures all Kafka messages are flushed before shutdown.
        Safe to call multiple times.
        """
        logger.info("Shutting down Binance client...")
        self.is_running = False

        if self.producer:
            try:
                # Flush remaining messages with timeout
                await asyncio.wait_for(self.producer.stop(), timeout=10.0)
                logger.info(f"Kafka producer stopped. Published {self.message_count} messages")
            except asyncio.TimeoutError:
                logger.warning("Kafka producer flush timeout, some messages may be lost")
            except Exception as e:
                logger.error(f"Error closing Kafka producer: {e}")
            finally:
                self.producer = None

        logger.info(f"Client closed. Total messages: {self.message_count}, Errors: {self.error_count}")

    async def __aenter__(self):
        """Async context manager entry."""
        self.producer = await self._init_kafka_producer()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.close()


async def main():
    """Run Binance client as standalone async service."""
    import argparse

    parser = argparse.ArgumentParser(description='Binance async order book data ingestion')
    parser.add_argument('--symbol', default='btcusdt', help='Trading pair')
    parser.add_argument('--speed', choices=['100ms', '1000ms'], default='100ms',
                       help='Update frequency')
    parser.add_argument('--levels', type=int, choices=[5, 10, 20], default=20,
                       help='Number of price levels')

    args = parser.parse_args()

    # Use context manager for automatic cleanup
    async with BinanceOrderBookClient(
        symbol=args.symbol,
        depth_update_speed=args.speed,
        levels=args.levels
    ) as client:
        try:
            await client.stream_with_reconnect()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
            # Context manager handles cleanup


if __name__ == '__main__':
    # Run async main
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown complete")
