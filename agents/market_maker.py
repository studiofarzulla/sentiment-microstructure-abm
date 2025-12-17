"""
Sentiment-Aware Market Maker Agent

Strategies:
1. Inventory Skew: Adjust quotes to revert to target inventory.
2. Sentiment Alpha: Shift mid-price based on Reddit sentiment velocity.
3. Microstructure: Adjust spread based on order book imbalance.
"""

import asyncio
import json
import os
import math
from typing import Dict, List, Optional
from datetime import datetime

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from pydantic import BaseModel

from .base import BaseAgent

# Import data models (assuming they are importable, otherwise we redefine for now)
# from data_ingestion.binance_client import OrderBookSnapshot

class MarketState(BaseModel):
    """Internal state of the market maker."""
    mid_price: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread: float = 0.0
    imbalance: float = 0.0
    sentiment_score: float = 0.0  # -1.0 to 1.0
    sentiment_velocity: float = 0.0
    inventory: float = 0.0
    cash: float = 100000.0  # USDT
    
class Quote(BaseModel):
    """Output quote."""
    timestamp: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    inventory_skew: float
    sentiment_skew: float

class SentimentAwareMarketMaker(BaseAgent):
    """
    Market Maker that adjusts quotes based on order book microstructure
    and real-time social sentiment.
    """
    
    def __init__(self, name: str, config: Dict = None):
        super().__init__(name, config)
        
        # Configuration
        self.symbol = config.get('symbol', 'BTCUSDT')
        self.kafka_servers = config.get('kafka_servers', 'localhost:9092')
        self.orderbook_topic = config.get('topic_orderbooks', 'order-books')
        self.sentiment_topic = config.get('topic_sentiment', 'reddit-sentiment')
        self.quote_topic = config.get('topic_quotes', 'mm-quotes')
        
        # Strategy Parameters
        self.target_inventory = config.get('target_inventory', 0.0)
        self.inventory_risk_aversion = config.get('gamma', 0.1)
        self.sentiment_sensitivity = config.get('kappa', 50.0)  # Dollar adjustment per sentiment unit
        self.min_spread_bps = config.get('min_spread_bps', 2.0)
        
        # State
        self.state = MarketState()
        
        # Kafka
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.producer: Optional[AIOKafkaProducer] = None

    async def _init_kafka(self):
        """Initialize Kafka consumer and producer."""
        self.consumer = AIOKafkaConsumer(
            self.orderbook_topic,
            self.sentiment_topic,
            bootstrap_servers=self.kafka_servers,
            group_id=f"{self.name}-group",
            auto_offset_reset='latest'
        )
        
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.kafka_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        await self.consumer.start()
        await self.producer.start()
        self.logger.info("Kafka connection established")

    async def _process_orderbook(self, data: Dict):
        """Update internal state from order book snapshot."""
        # data matches OrderBookSnapshot schema
        self.state.mid_price = data.get('mid_price', 0.0)
        self.state.best_bid = data.get('best_bid', 0.0)
        self.state.best_ask = data.get('best_ask', 0.0)
        self.state.spread = data.get('spread', 0.0)
        self.state.imbalance = data.get('imbalance', 0.0)
        
        await self._recalculate_quotes()

    async def _process_sentiment(self, data: Dict):
        """Update internal state from sentiment analysis."""
        # Expecting {'score': 0.5, 'velocity': 0.1}
        new_score = data.get('score', 0.0)
        self.state.sentiment_velocity = new_score - self.state.sentiment_score
        self.state.sentiment_score = new_score
        
        self.logger.info(f"Sentiment update: {new_score:.3f} (Vel: {self.state.sentiment_velocity:.3f})")
        # Trigger requote on significant sentiment shift
        if abs(self.state.sentiment_velocity) > 0.1:
            await self._recalculate_quotes()

    async def _recalculate_quotes(self):
        """
        Core Pricing Logic:
        Ref Price = Mid Price + Inventory Skew + Sentiment Skew
        """
        if self.state.mid_price == 0:
            return

        # 1. Inventory Skew (Avellaneda-Stoikov approx)
        # Shift price against inventory to encourage mean reversion
        inventory_diff = self.state.inventory - self.target_inventory
        inventory_skew = -1 * self.inventory_risk_aversion * inventory_diff
        
        # 2. Sentiment Skew
        # Bullish -> Shift prices UP (buy higher, sell higher) -> Accumulate inventory? 
        # Actually, if sentiment is high, we expect price rise. 
        # We should skew bid UP to capture volume, skew ask UP to sell higher.
        sentiment_skew = self.state.sentiment_score * self.state.sentiment_sensitivity
        
        # 3. Reservation Price
        reservation_price = self.state.mid_price + inventory_skew + sentiment_skew
        
        # 4. Spread Calculation
        # Widen spread if volatility (implied by spread) is high
        half_spread = max(self.state.spread / 2, self.state.mid_price * (self.min_spread_bps / 10000))
        
        my_bid = reservation_price - half_spread
        my_ask = reservation_price + half_spread
        
        # Quote Sizing (Simple constant for now)
        quote_size = 0.1  # BTC
        
        # Publish Quote
        quote = {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': self.symbol,
            'bid_price': round(my_bid, 2),
            'ask_price': round(my_ask, 2),
            'bid_size': quote_size,
            'ask_size': quote_size,
            'ref_price': round(reservation_price, 2),
            'inv_skew': round(inventory_skew, 4),
            'sent_skew': round(sentiment_skew, 4)
        }
        
        if self.producer:
            await self.producer.send(self.quote_topic, quote)
            
        # self.logger.debug(f"Quote: {quote['bid_price']} / {quote['ask_price']} (Skew: {inventory_skew:.2f}/{sentiment_skew:.2f})")

    async def run(self):
        """Main agent loop."""
        await self._init_kafka()
        self.logger.info("Market Maker Agent Running...")
        
        try:
            async for msg in self.consumer:
                if self.is_running is False:
                    break
                    
                topic = msg.topic
                try:
                    payload = json.loads(msg.value)
                    
                    if topic == self.orderbook_topic:
                        await self._process_orderbook(payload)
                    elif topic == self.sentiment_topic:
                        await self._process_sentiment(payload)
                        
                except json.JSONDecodeError:
                    self.logger.error("Failed to decode message")
                    
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Stop Kafka connections."""
        if self.consumer:
            await self.consumer.stop()
        if self.producer:
            await self.producer.stop()
        await super().cleanup()

if __name__ == "__main__":
    # Standalone execution
    config = {
        'symbol': 'BTCUSDT',
        'kafka_servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
        'topic_orderbooks': os.getenv('KAFKA_TOPIC_ORDERBOOKS', 'order-books'),
        'topic_sentiment': 'reddit-sentiment'
    }
    
    agent = SentimentAwareMarketMaker("MM-Alpha", config)
    try:
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        pass
