"""
Kafka → Mesa Bridge

Real-time streaming interface that connects Kafka data streams
to the Mesa ABM simulation.

This module provides:
1. KafkaDataConsumer - Async consumer for order book + sentiment topics
2. LiveSimulationRunner - Runs Mesa model driven by live Kafka data
3. DataAligner - Aligns order book and sentiment by timestamp

Architecture:
    Binance WS → Kafka (order-books) ─┐
                                      ├─→ KafkaDataConsumer → Mesa Model
    Reddit API → Kafka (sentiment) ───┘

Usage:
    # Start live simulation with Kafka streams
    runner = LiveSimulationRunner(
        kafka_servers='localhost:9092',
        orderbook_topic='order-books',
        sentiment_topic='reddit-sentiment',
    )
    await runner.run()

Author: Murad Farzulla  
Date: January 2026
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from collections import deque
import numpy as np

# Async Kafka imports (optional - graceful fallback)
try:
    from aiokafka import AIOKafkaConsumer
    from aiokafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    AIOKafkaConsumer = None
    KafkaError = Exception

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class OrderBookUpdate:
    """Order book update from Kafka."""
    timestamp: datetime
    symbol: str
    mid_price: float
    best_bid: float
    best_ask: float
    spread_bps: float
    imbalance: float
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    
    @classmethod
    def from_kafka_message(cls, data: dict) -> 'OrderBookUpdate':
        """Parse from Kafka message format (matches binance_client output)."""
        return cls(
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.utcnow().isoformat())),
            symbol=data.get('symbol', 'BTCUSDT'),
            mid_price=float(data.get('mid_price', 0)),
            best_bid=float(data.get('best_bid', 0)),
            best_ask=float(data.get('best_ask', 0)),
            spread_bps=float(data.get('spread_bps', 0)),
            imbalance=float(data.get('imbalance', 0)),
            bid_volume=float(data.get('bid_volume', 0)),
            ask_volume=float(data.get('ask_volume', 0)),
        )


@dataclass
class SentimentUpdate:
    """Sentiment update from Kafka (post-CryptoBERT analysis)."""
    timestamp: datetime
    sentiment: float  # [-1, 1]
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    regime: str = "neutral"
    source: str = "reddit"
    text_snippet: str = ""
    
    @classmethod
    def from_kafka_message(cls, data: dict) -> 'SentimentUpdate':
        """Parse from Kafka message format."""
        # Handle both raw reddit posts and analyzed sentiment
        if 'sentiment' in data:
            # Already analyzed
            return cls(
                timestamp=datetime.fromisoformat(data.get('timestamp', datetime.utcnow().isoformat())),
                sentiment=float(data.get('sentiment', 0)),
                epistemic_uncertainty=float(data.get('epistemic_uncertainty', 0.1)),
                aleatoric_uncertainty=float(data.get('aleatoric_uncertainty', 0.2)),
                regime=data.get('regime', 'neutral'),
                source=data.get('source', 'reddit'),
                text_snippet=str(data.get('text', ''))[:100],
            )
        else:
            # Raw post - needs analysis (return placeholder)
            return cls(
                timestamp=datetime.fromisoformat(data.get('timestamp', datetime.utcnow().isoformat())),
                sentiment=0.0,  # Will be filled by analyzer
                epistemic_uncertainty=0.1,
                aleatoric_uncertainty=0.2,
                regime='neutral',
                source='reddit',
                text_snippet=str(data.get('title', data.get('text', '')))[:100],
            )


@dataclass
class AlignedTick:
    """Combined order book + sentiment tick for simulation."""
    timestamp: datetime
    # Order book
    mid_price: float
    best_bid: float
    best_ask: float
    spread_bps: float
    imbalance: float
    # Sentiment
    sentiment: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    regime: str
    # Metadata
    orderbook_age_ms: float = 0.0
    sentiment_age_ms: float = 0.0


# ============================================================================
# Data Aligner
# ============================================================================

class DataAligner:
    """
    Aligns order book and sentiment streams by timestamp.
    
    Order books arrive ~10x/second, sentiment ~1/minute.
    This class maintains the latest of each and produces aligned ticks.
    """
    
    def __init__(
        self,
        max_sentiment_age_sec: float = 300,  # 5 minutes max
        sentiment_decay_rate: float = 0.99,  # Decay toward neutral
    ):
        self.max_sentiment_age_sec = max_sentiment_age_sec
        self.sentiment_decay_rate = sentiment_decay_rate
        
        # Latest state
        self._latest_orderbook: Optional[OrderBookUpdate] = None
        self._latest_sentiment: Optional[SentimentUpdate] = None
        
        # Decayed sentiment state
        self._current_sentiment: float = 0.0
        self._current_epistemic: float = 0.1
        self._current_aleatoric: float = 0.2
        self._current_regime: str = "neutral"
        
        # Statistics
        self.orderbook_count = 0
        self.sentiment_count = 0
        
    def update_orderbook(self, update: OrderBookUpdate):
        """Process new order book update."""
        self._latest_orderbook = update
        self.orderbook_count += 1
        
    def update_sentiment(self, update: SentimentUpdate):
        """Process new sentiment update."""
        self._latest_sentiment = update
        self._current_sentiment = update.sentiment
        self._current_epistemic = update.epistemic_uncertainty
        self._current_aleatoric = update.aleatoric_uncertainty
        self._current_regime = update.regime
        self.sentiment_count += 1
        
    def get_aligned_tick(self) -> Optional[AlignedTick]:
        """
        Get current aligned tick.
        
        Returns None if no order book data available.
        Sentiment decays toward neutral if stale.
        """
        if self._latest_orderbook is None:
            return None
            
        now = datetime.utcnow()
        ob = self._latest_orderbook
        
        # Calculate ages
        ob_age_ms = (now - ob.timestamp).total_seconds() * 1000
        
        if self._latest_sentiment is not None:
            sent_age_sec = (now - self._latest_sentiment.timestamp).total_seconds()
            sent_age_ms = sent_age_sec * 1000
            
            # Decay sentiment if stale
            if sent_age_sec > 60:  # Start decay after 1 minute
                decay_factor = self.sentiment_decay_rate ** (sent_age_sec / 60)
                self._current_sentiment *= decay_factor
                
                # Update regime based on decayed sentiment
                if abs(self._current_sentiment) < 0.1:
                    self._current_regime = 'neutral'
        else:
            sent_age_ms = self.max_sentiment_age_sec * 1000
            
        return AlignedTick(
            timestamp=now,
            mid_price=ob.mid_price,
            best_bid=ob.best_bid,
            best_ask=ob.best_ask,
            spread_bps=ob.spread_bps,
            imbalance=ob.imbalance,
            sentiment=self._current_sentiment,
            epistemic_uncertainty=self._current_epistemic,
            aleatoric_uncertainty=self._current_aleatoric,
            regime=self._current_regime,
            orderbook_age_ms=ob_age_ms,
            sentiment_age_ms=sent_age_ms,
        )
    
    def get_stats(self) -> dict:
        """Get alignment statistics."""
        return {
            'orderbook_count': self.orderbook_count,
            'sentiment_count': self.sentiment_count,
            'current_sentiment': self._current_sentiment,
            'current_regime': self._current_regime,
            'has_orderbook': self._latest_orderbook is not None,
            'has_sentiment': self._latest_sentiment is not None,
        }


# ============================================================================
# Kafka Consumer
# ============================================================================

class KafkaDataConsumer:
    """
    Async Kafka consumer for market data streams.
    
    Consumes from:
    - order-books: Real-time order book snapshots from Binance
    - reddit-sentiment: Analyzed sentiment from Reddit posts
    
    Provides aligned ticks to the simulation.
    """
    
    def __init__(
        self,
        kafka_servers: str = 'localhost:9092',
        orderbook_topic: str = 'order-books',
        sentiment_topic: str = 'reddit-sentiment',
        group_id: str = 'abm-simulation',
    ):
        if not KAFKA_AVAILABLE:
            raise ImportError(
                "aiokafka not installed. Run: pip install aiokafka"
            )
            
        self.kafka_servers = kafka_servers.split(',')
        self.orderbook_topic = orderbook_topic
        self.sentiment_topic = sentiment_topic
        self.group_id = group_id
        
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.aligner = DataAligner()
        
        self._running = False
        self._tick_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        
    async def start(self):
        """Start Kafka consumer."""
        logger.info(f"Starting Kafka consumer...")
        logger.info(f"  Servers: {self.kafka_servers}")
        logger.info(f"  Topics: {self.orderbook_topic}, {self.sentiment_topic}")
        
        self.consumer = AIOKafkaConsumer(
            self.orderbook_topic,
            self.sentiment_topic,
            bootstrap_servers=self.kafka_servers,
            group_id=self.group_id,
            auto_offset_reset='latest',  # Start from newest messages
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        )
        
        await self.consumer.start()
        self._running = True
        logger.info("Kafka consumer started")
        
    async def stop(self):
        """Stop Kafka consumer."""
        self._running = False
        if self.consumer:
            await self.consumer.stop()
            logger.info("Kafka consumer stopped")
            
    async def consume_loop(self):
        """
        Main consumption loop.
        
        Processes messages and updates the aligner.
        Call get_tick() to retrieve aligned ticks.
        """
        if not self.consumer:
            raise RuntimeError("Consumer not started. Call start() first.")
            
        logger.info("Starting consumption loop...")
        
        async for msg in self.consumer:
            if not self._running:
                break
                
            try:
                topic = msg.topic
                data = msg.value
                
                if topic == self.orderbook_topic:
                    update = OrderBookUpdate.from_kafka_message(data)
                    self.aligner.update_orderbook(update)
                    
                    # Generate aligned tick
                    tick = self.aligner.get_aligned_tick()
                    if tick:
                        try:
                            self._tick_queue.put_nowait(tick)
                        except asyncio.QueueFull:
                            # Drop oldest tick if queue full
                            try:
                                self._tick_queue.get_nowait()
                                self._tick_queue.put_nowait(tick)
                            except:
                                pass
                                
                elif topic == self.sentiment_topic:
                    update = SentimentUpdate.from_kafka_message(data)
                    self.aligner.update_sentiment(update)
                    logger.debug(f"Sentiment update: {update.sentiment:.3f} ({update.regime})")
                    
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                continue
                
        logger.info("Consumption loop ended")
        
    async def get_tick(self, timeout: float = 1.0) -> Optional[AlignedTick]:
        """
        Get next aligned tick.
        
        Args:
            timeout: Max seconds to wait
            
        Returns:
            AlignedTick or None if timeout
        """
        try:
            return await asyncio.wait_for(
                self._tick_queue.get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None
            
    def get_current_tick(self) -> Optional[AlignedTick]:
        """Get current aligned tick without waiting (sync version)."""
        return self.aligner.get_aligned_tick()


# ============================================================================
# Live Simulation Runner
# ============================================================================

class LiveSimulationRunner:
    """
    Runs Mesa simulation driven by live Kafka streams.
    
    This is the main entry point for real-time simulation.
    
    Usage:
        runner = LiveSimulationRunner()
        await runner.run(duration_minutes=60)
    """
    
    def __init__(
        self,
        kafka_servers: str = 'localhost:9092',
        orderbook_topic: str = 'order-books',
        sentiment_topic: str = 'reddit-sentiment',
        step_interval_ms: float = 500,
        n_market_makers: int = 2,
        n_informed: int = 5,
        n_noise: int = 10,
    ):
        self.kafka_servers = kafka_servers
        self.orderbook_topic = orderbook_topic
        self.sentiment_topic = sentiment_topic
        self.step_interval_ms = step_interval_ms
        
        # Agent configuration
        self.n_market_makers = n_market_makers
        self.n_informed = n_informed
        self.n_noise = n_noise
        
        # Components (initialized in run())
        self.consumer: Optional[KafkaDataConsumer] = None
        self.model = None
        
        # State
        self._running = False
        self.step_count = 0
        self.history: List[dict] = []
        
    async def run(
        self,
        duration_minutes: Optional[float] = None,
        max_steps: Optional[int] = None,
        on_step_callback: Optional[Callable[[int, AlignedTick], None]] = None,
    ):
        """
        Run live simulation.
        
        Args:
            duration_minutes: Run for this many minutes (None = forever)
            max_steps: Stop after this many steps (None = no limit)
            on_step_callback: Called after each step with (step, tick)
        """
        # Import here to avoid circular imports
        from simulation.market_env import create_default_market
        
        logger.info("=" * 60)
        logger.info("LIVE SIMULATION STARTING")
        logger.info("=" * 60)
        
        # Initialize Kafka consumer
        self.consumer = KafkaDataConsumer(
            kafka_servers=self.kafka_servers,
            orderbook_topic=self.orderbook_topic,
            sentiment_topic=self.sentiment_topic,
        )
        
        try:
            await self.consumer.start()
            
            # Start consumption in background
            consume_task = asyncio.create_task(self.consumer.consume_loop())
            
            # Wait for first tick to get initial price
            logger.info("Waiting for initial market data...")
            initial_tick = None
            for _ in range(30):  # Wait up to 30 seconds
                initial_tick = await self.consumer.get_tick(timeout=1.0)
                if initial_tick:
                    break
                    
            if not initial_tick:
                raise RuntimeError("No market data received within 30 seconds")
                
            logger.info(f"Initial price: ${initial_tick.mid_price:.2f}")
            
            # Create model
            self.model = create_default_market(
                n_market_makers=self.n_market_makers,
                n_informed=self.n_informed,
                n_noise=self.n_noise,
                initial_price=initial_tick.mid_price,
            )
            
            # Run simulation loop
            self._running = True
            start_time = datetime.utcnow()
            step_interval = timedelta(milliseconds=self.step_interval_ms)
            next_step_time = start_time
            
            logger.info("Simulation loop starting...")
            
            while self._running:
                # Check stopping conditions
                if duration_minutes and (datetime.utcnow() - start_time).total_seconds() > duration_minutes * 60:
                    logger.info(f"Duration limit reached ({duration_minutes} minutes)")
                    break
                    
                if max_steps and self.step_count >= max_steps:
                    logger.info(f"Step limit reached ({max_steps} steps)")
                    break
                
                # Wait for next step time
                now = datetime.utcnow()
                if now < next_step_time:
                    await asyncio.sleep((next_step_time - now).total_seconds())
                next_step_time = datetime.utcnow() + step_interval
                
                # Get current tick
                tick = self.consumer.get_current_tick()
                if tick is None:
                    continue
                    
                # Update model sentiment
                self.model.set_sentiment(
                    tick.sentiment,
                    tick.epistemic_uncertainty,
                    tick.aleatoric_uncertainty,
                    tick.regime,
                )
                
                # Anchor price to real market
                if tick.mid_price > 0:
                    self.model._anchor_price_to_real(tick.mid_price, strength=0.1)
                
                # Step simulation
                self.model.step()
                self.step_count += 1
                
                # Record history
                record = {
                    'step': self.step_count,
                    'timestamp': tick.timestamp.isoformat(),
                    'real_price': tick.mid_price,
                    'sim_price': self.model.order_book.mid_price,
                    'spread_bps': self.model.order_book.spread_bps,
                    'sentiment': tick.sentiment,
                    'regime': tick.regime,
                    'trades': self.model.order_book.trade_count,
                }
                self.history.append(record)
                
                # Callback
                if on_step_callback:
                    on_step_callback(self.step_count, tick)
                    
                # Progress logging
                if self.step_count % 100 == 0:
                    stats = self.consumer.aligner.get_stats()
                    logger.info(
                        f"Step {self.step_count}: "
                        f"real=${tick.mid_price:.2f}, "
                        f"sim=${self.model.order_book.mid_price:.2f}, "
                        f"sent={tick.sentiment:+.2f}, "
                        f"trades={self.model.order_book.trade_count}"
                    )
                    
            logger.info("=" * 60)
            logger.info("SIMULATION COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Total steps: {self.step_count}")
            logger.info(f"Total trades: {self.model.order_book.trade_count}")
            
        finally:
            self._running = False
            if self.consumer:
                await self.consumer.stop()
                
        return self.history
    
    def stop(self):
        """Signal the simulation to stop."""
        self._running = False


# ============================================================================
# Standalone Mock Consumer (for testing without Kafka)
# ============================================================================

class MockKafkaConsumer:
    """
    Mock consumer that generates synthetic data.
    
    Useful for testing the simulation without running Kafka.
    """
    
    def __init__(
        self,
        initial_price: float = 42000.0,
        volatility: float = 0.0005,
        sentiment_interval_sec: float = 30,
    ):
        self.price = initial_price
        self.volatility = volatility
        self.sentiment_interval_sec = sentiment_interval_sec
        
        self.aligner = DataAligner()
        self._running = False
        self._last_sentiment_time = datetime.utcnow()
        self._sentiment = 0.0
        
    async def start(self):
        self._running = True
        logger.info("Mock consumer started")
        
    async def stop(self):
        self._running = False
        logger.info("Mock consumer stopped")
        
    async def consume_loop(self):
        """Generate synthetic data."""
        while self._running:
            now = datetime.utcnow()
            
            # Generate order book update
            ret = np.random.normal(0, self.volatility)
            self.price *= (1 + ret)
            
            spread_bps = 5 + abs(ret) * 1000 + np.random.uniform(0, 3)
            spread = self.price * spread_bps / 10000
            
            ob = OrderBookUpdate(
                timestamp=now,
                symbol='BTCUSDT',
                mid_price=self.price,
                best_bid=self.price - spread/2,
                best_ask=self.price + spread/2,
                spread_bps=spread_bps,
                imbalance=np.random.uniform(-0.3, 0.3),
            )
            self.aligner.update_orderbook(ob)
            
            # Generate sentiment update periodically
            if (now - self._last_sentiment_time).total_seconds() > self.sentiment_interval_sec:
                # Random walk sentiment
                self._sentiment = np.clip(
                    self._sentiment + np.random.normal(0, 0.1),
                    -1, 1
                )
                
                regime = 'bullish' if self._sentiment > 0.2 else ('bearish' if self._sentiment < -0.2 else 'neutral')
                
                sent = SentimentUpdate(
                    timestamp=now,
                    sentiment=self._sentiment,
                    epistemic_uncertainty=0.05 + abs(self._sentiment) * 0.02,
                    aleatoric_uncertainty=0.15 + (1 - abs(self._sentiment)) * 0.1,
                    regime=regime,
                )
                self.aligner.update_sentiment(sent)
                self._last_sentiment_time = now
                
            await asyncio.sleep(0.1)  # 10 updates/second
            
    def get_current_tick(self) -> Optional[AlignedTick]:
        return self.aligner.get_aligned_tick()


# ============================================================================
# CLI Demo
# ============================================================================

async def run_demo_with_mock():
    """Run a demo simulation with mock data (no Kafka needed)."""
    from simulation.market_env import create_default_market
    
    logger.info("=" * 60)
    logger.info("MOCK LIVE SIMULATION DEMO")
    logger.info("=" * 60)
    
    # Create mock consumer
    consumer = MockKafkaConsumer(initial_price=42000)
    await consumer.start()
    
    # Start mock data generation
    consume_task = asyncio.create_task(consumer.consume_loop())
    
    # Wait for initial data
    await asyncio.sleep(0.5)
    
    initial_tick = consumer.get_current_tick()
    if not initial_tick:
        raise RuntimeError("No initial tick")
        
    logger.info(f"Initial price: ${initial_tick.mid_price:.2f}")
    
    # Create model
    model = create_default_market(
        n_market_makers=2,
        n_informed=3,
        n_noise=10,
        initial_price=initial_tick.mid_price,
    )
    
    # Run simulation
    history = []
    n_steps = 500
    step_interval = 0.1  # 100ms steps
    
    logger.info(f"Running {n_steps} steps...")
    
    for step in range(n_steps):
        tick = consumer.get_current_tick()
        if tick:
            model.set_sentiment(
                tick.sentiment,
                tick.epistemic_uncertainty,
                tick.aleatoric_uncertainty,
                tick.regime,
            )
            model._anchor_price_to_real(tick.mid_price, strength=0.05)
            
        model.step()
        
        if tick:
            history.append({
                'step': step,
                'real_price': tick.mid_price,
                'sim_price': model.order_book.mid_price,
                'sentiment': tick.sentiment,
                'regime': tick.regime,
            })
            
        if step % 100 == 0:
            logger.info(f"Step {step}: price=${tick.mid_price:.2f}, trades={model.order_book.trade_count}")
            
        await asyncio.sleep(step_interval)
        
    # Stop
    consumer._running = False
    await consumer.stop()
    
    logger.info("\n" + "=" * 60)
    logger.info("DEMO COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Steps: {n_steps}")
    logger.info(f"Trades: {model.order_book.trade_count}")
    
    if history:
        import pandas as pd
        df = pd.DataFrame(history)
        logger.info(f"Price drift: ${df['real_price'].iloc[0]:.2f} -> ${df['real_price'].iloc[-1]:.2f}")
        
        # Save results
        output_path = '/tmp/mock_live_simulation.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to: {output_path}")
        
    return history


if __name__ == '__main__':
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Kafka-Mesa Bridge Demo')
    parser.add_argument('--mock', action='store_true', help='Run with mock data (no Kafka)')
    parser.add_argument('--kafka', type=str, default='localhost:9092', help='Kafka servers')
    parser.add_argument('--duration', type=float, default=5, help='Duration in minutes')
    
    args = parser.parse_args()
    
    if args.mock:
        asyncio.run(run_demo_with_mock())
    else:
        if not KAFKA_AVAILABLE:
            logger.error("aiokafka not installed. Use --mock for mock data demo.")
            exit(1)
            
        runner = LiveSimulationRunner(kafka_servers=args.kafka)
        asyncio.run(runner.run(duration_minutes=args.duration))
