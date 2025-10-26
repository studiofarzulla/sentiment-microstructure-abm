"""
Configuration Management for Sentiment-Microstructure ABM

Centralized, type-safe configuration using Pydantic BaseSettings.
Validates all environment variables on startup with clear error messages.

Usage:
    from config.settings import settings

    # Access validated config
    producer = KafkaProducer(
        bootstrap_servers=settings.kafka.bootstrap_servers_list
    )
"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import List, Optional
import os


class KafkaConfig(BaseSettings):
    """Kafka broker and topic configuration."""

    bootstrap_servers: str = Field(
        ...,
        validation_alias='KAFKA_BOOTSTRAP_SERVERS',
        description='Comma-separated list of Kafka brokers (e.g., localhost:9092,broker2:9092)'
    )

    topic_orderbooks: str = Field(
        default='order-books',
        validation_alias='KAFKA_TOPIC_ORDERBOOKS',
        description='Topic for Binance order book snapshots'
    )

    topic_reddit: str = Field(
        default='reddit-posts',
        validation_alias='KAFKA_TOPIC_REDDIT',
        description='Topic for Reddit posts and comments'
    )

    topic_sentiment: str = Field(
        default='sentiment-ticks',
        validation_alias='KAFKA_TOPIC_SENTIMENT',
        description='Topic for sentiment analysis results'
    )

    compression_type: str = Field(
        default='gzip',
        validation_alias='KAFKA_COMPRESSION_TYPE',
        description='Compression algorithm (gzip, snappy, lz4, zstd)'
    )

    @field_validator('bootstrap_servers')
    @classmethod
    def validate_bootstrap_servers(cls, v):
        """Ensure bootstrap servers is not empty."""
        if not v or not v.strip():
            raise ValueError(
                "KAFKA_BOOTSTRAP_SERVERS is required. "
                "Example: KAFKA_BOOTSTRAP_SERVERS=localhost:9092"
            )

        # Check format (basic validation)
        servers = [s.strip() for s in v.split(',')]
        for server in servers:
            if ':' not in server:
                raise ValueError(
                    f"Invalid Kafka server format: '{server}'. "
                    "Expected format: host:port"
                )

        return v

    @field_validator('compression_type')
    @classmethod
    def validate_compression(cls, v):
        """Validate compression type."""
        valid_types = ['gzip', 'snappy', 'lz4', 'zstd', 'none']
        if v not in valid_types:
            raise ValueError(
                f"Invalid compression type: '{v}'. "
                f"Must be one of: {', '.join(valid_types)}"
            )
        return v

    @property
    def bootstrap_servers_list(self) -> List[str]:
        """Get bootstrap servers as list (commonly needed)."""
        return [s.strip() for s in self.bootstrap_servers.split(',')]

    model_config = {
        'env_file': '.env',
        'case_sensitive': False,
        'extra': 'ignore'
    }


class RedditConfig(BaseSettings):
    """Reddit API credentials and target subreddits."""

    client_id: str = Field(
        ...,
        validation_alias='REDDIT_CLIENT_ID',
        description='Reddit API client ID (get from https://www.reddit.com/prefs/apps)'
    )

    client_secret: str = Field(
        ...,
        validation_alias='REDDIT_CLIENT_SECRET',
        description='Reddit API client secret'
    )

    user_agent: str = Field(
        default='SentimentMicrostructureABM/1.0',
        validation_alias='REDDIT_USER_AGENT',
        description='Reddit API user agent string'
    )

    subreddits: str = Field(
        default='CryptoCurrency,Bitcoin,ethereum,CryptoMarkets,bitcoinmarkets,ethtrader,CryptoTechnology',
        validation_alias='REDDIT_SUBREDDITS',
        description='Comma-separated list of subreddits to monitor'
    )

    @field_validator('client_id')
    @classmethod
    def validate_client_id(cls, v):
        """Ensure client_id is not placeholder."""
        if not v or v in ['your_client_id_here', 'YOUR_CLIENT_ID']:
            raise ValueError(
                "REDDIT_CLIENT_ID is required. "
                "Get credentials from https://www.reddit.com/prefs/apps"
            )
        return v

    @field_validator('client_secret')
    @classmethod
    def validate_client_secret(cls, v):
        """Ensure client_secret is not placeholder."""
        if not v or v in ['your_client_secret_here', 'YOUR_CLIENT_SECRET']:
            raise ValueError(
                "REDDIT_CLIENT_SECRET is required. "
                "Get credentials from https://www.reddit.com/prefs/apps"
            )
        return v

    @property
    def subreddits_list(self) -> List[str]:
        """Get subreddits as list."""
        return [s.strip() for s in self.subreddits.split(',')]

    model_config = {
        'env_file': '.env',
        'case_sensitive': False,
        'extra': 'ignore'
    }


class BinanceConfig(BaseSettings):
    """Binance WebSocket configuration."""

    websocket_url: str = Field(
        default='wss://stream.binance.com:9443/ws',
        validation_alias='BINANCE_WEBSOCKET_URL',
        description='Binance WebSocket base URL'
    )

    symbols: str = Field(
        default='btcusdt',
        validation_alias='BINANCE_SYMBOLS',
        description='Comma-separated list of trading pairs to stream'
    )

    depth_update_speed: str = Field(
        default='100ms',
        validation_alias='BINANCE_DEPTH_UPDATE_SPEED',
        description='Order book update frequency (100ms or 1000ms)'
    )

    depth_levels: int = Field(
        default=20,
        validation_alias='BINANCE_DEPTH_LEVELS',
        description='Number of price levels to capture (5, 10, or 20)'
    )

    @field_validator('websocket_url')
    @classmethod
    def validate_websocket_url(cls, v):
        """Ensure WebSocket URL is valid."""
        if not v.startswith('wss://') and not v.startswith('ws://'):
            raise ValueError(
                f"Invalid WebSocket URL: '{v}'. "
                "Must start with wss:// or ws://"
            )
        return v

    @field_validator('depth_update_speed')
    @classmethod
    def validate_depth_speed(cls, v):
        """Validate update speed."""
        valid_speeds = ['100ms', '1000ms']
        if v not in valid_speeds:
            raise ValueError(
                f"Invalid depth_update_speed: '{v}'. "
                f"Must be one of: {', '.join(valid_speeds)}"
            )
        return v

    @field_validator('depth_levels')
    @classmethod
    def validate_depth_levels(cls, v):
        """Validate depth levels."""
        valid_levels = [5, 10, 20]
        if v not in valid_levels:
            raise ValueError(
                f"Invalid depth_levels: {v}. "
                f"Must be one of: {', '.join(map(str, valid_levels))}"
            )
        return v

    @property
    def symbols_list(self) -> List[str]:
        """Get symbols as list."""
        return [s.strip().lower() for s in self.symbols.split(',')]

    model_config = {
        'env_file': '.env',
        'case_sensitive': False,
        'extra': 'ignore'
    }


class SentimentConfig(BaseSettings):
    """Sentiment analysis model configuration."""

    model_name: str = Field(
        default='distilroberta-base',
        validation_alias='SENTIMENT_MODEL_NAME',
        description='HuggingFace model identifier'
    )

    model_path: Optional[str] = Field(
        default=None,
        validation_alias='SENTIMENT_MODEL_PATH',
        description='Path to fine-tuned model (if using local model)'
    )

    model_cache: str = Field(
        default='models/cache',
        validation_alias='SENTIMENT_MODEL_CACHE',
        description='Directory for model cache'
    )

    batch_size: int = Field(
        default=8,
        validation_alias='SENTIMENT_BATCH_SIZE',
        description='Batch size for inference'
    )

    mc_samples: int = Field(
        default=20,
        validation_alias='SENTIMENT_MC_SAMPLES',
        description='Number of Monte Carlo dropout samples for uncertainty'
    )

    ewma_alpha: float = Field(
        default=0.3,
        validation_alias='SENTIMENT_EWMA_ALPHA',
        description='EWMA smoothing factor (0-1, higher = more responsive)'
    )

    device: Optional[str] = Field(
        default=None,
        validation_alias='SENTIMENT_DEVICE',
        description='PyTorch device (cuda, cpu, or None for auto-detect)'
    )

    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v):
        """Ensure batch size is positive."""
        if v < 1:
            raise ValueError(f"batch_size must be >= 1, got {v}")
        if v > 128:
            raise ValueError(f"batch_size too large ({v}), recommended <= 128")
        return v

    @field_validator('mc_samples')
    @classmethod
    def validate_mc_samples(cls, v):
        """Ensure MC samples is reasonable."""
        if v < 1:
            raise ValueError(f"mc_samples must be >= 1, got {v}")
        if v > 100:
            raise ValueError(f"mc_samples too large ({v}), recommended <= 100")
        return v

    @field_validator('ewma_alpha')
    @classmethod
    def validate_ewma_alpha(cls, v):
        """Ensure alpha is in valid range."""
        if not 0 <= v <= 1:
            raise ValueError(f"ewma_alpha must be in [0, 1], got {v}")
        return v

    @field_validator('device')
    @classmethod
    def validate_device(cls, v):
        """Validate device string."""
        if v is not None and v not in ['cuda', 'cpu', 'mps']:
            raise ValueError(
                f"Invalid device: '{v}'. "
                "Must be 'cuda', 'cpu', 'mps', or None (auto-detect)"
            )
        return v

    model_config = {
        'env_file': '.env',
        'case_sensitive': False,
        'extra': 'ignore'
    }


class TimescaleConfig(BaseSettings):
    """TimescaleDB connection configuration."""

    host: str = Field(
        default='localhost',
        validation_alias='TIMESCALE_HOST',
        description='Database host'
    )

    port: int = Field(
        default=5432,
        validation_alias='TIMESCALE_PORT',
        description='Database port'
    )

    database: str = Field(
        default='market_sim',
        validation_alias='TIMESCALE_DB',
        description='Database name'
    )

    user: str = Field(
        default='postgres',
        validation_alias='TIMESCALE_USER',
        description='Database user'
    )

    password: str = Field(
        default='postgres',
        validation_alias='TIMESCALE_PASSWORD',
        description='Database password'
    )

    @field_validator('port')
    @classmethod
    def validate_port(cls, v):
        """Validate port range."""
        if not 1 <= v <= 65535:
            raise ValueError(f"Invalid port: {v}. Must be in range 1-65535")
        return v

    @property
    def connection_string(self) -> str:
        """Get PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    model_config = {
        'env_file': '.env',
        'case_sensitive': False,
        'extra': 'ignore'
    }


class SimulationConfig(BaseSettings):
    """Agent-based simulation parameters."""

    tick_interval_ms: int = Field(
        default=500,
        validation_alias='SIM_TICK_INTERVAL_MS',
        description='Simulation tick interval in milliseconds'
    )

    n_market_makers: int = Field(
        default=10,
        validation_alias='SIM_N_MARKET_MAKERS',
        description='Number of market maker agents'
    )

    n_informed_traders: int = Field(
        default=20,
        validation_alias='SIM_N_INFORMED_TRADERS',
        description='Number of informed trader agents'
    )

    n_noise_traders: int = Field(
        default=50,
        validation_alias='SIM_N_NOISE_TRADERS',
        description='Number of noise trader agents'
    )

    n_arbitrageurs: int = Field(
        default=5,
        validation_alias='SIM_N_ARBITRAGEURS',
        description='Number of arbitrageur agents'
    )

    @field_validator('tick_interval_ms', 'n_market_makers', 'n_informed_traders',
                     'n_noise_traders', 'n_arbitrageurs')
    @classmethod
    def validate_positive(cls, v):
        """Ensure values are positive."""
        if v < 0:
            raise ValueError(f"Value must be non-negative, got {v}")
        return v

    @property
    def total_agents(self) -> int:
        """Total number of agents in simulation."""
        return (
            self.n_market_makers +
            self.n_informed_traders +
            self.n_noise_traders +
            self.n_arbitrageurs
        )

    model_config = {
        'env_file': '.env',
        'case_sensitive': False,
        'extra': 'ignore'
    }


class DFMConfig(BaseSettings):
    """Dynamic Factor Model parameters."""

    window_size_minutes: int = Field(
        default=60,
        validation_alias='DFM_WINDOW_SIZE_MINUTES',
        description='Rolling window size for factor estimation'
    )

    update_interval_minutes: int = Field(
        default=5,
        validation_alias='DFM_UPDATE_INTERVAL_MINUTES',
        description='How often to update factor model'
    )

    n_factors: int = Field(
        default=3,
        validation_alias='DFM_N_FACTORS',
        description='Number of latent factors to extract'
    )

    @field_validator('window_size_minutes', 'update_interval_minutes')
    @classmethod
    def validate_positive_time(cls, v):
        """Ensure time values are positive."""
        if v <= 0:
            raise ValueError(f"Time interval must be > 0, got {v}")
        return v

    @field_validator('n_factors')
    @classmethod
    def validate_n_factors(cls, v):
        """Ensure number of factors is reasonable."""
        if v < 1:
            raise ValueError(f"n_factors must be >= 1, got {v}")
        if v > 10:
            raise ValueError(f"n_factors too large ({v}), recommended <= 10")
        return v

    model_config = {
        'env_file': '.env',
        'case_sensitive': False,
        'extra': 'ignore'
    }


class DashboardConfig(BaseSettings):
    """Dashboard server configuration."""

    host: str = Field(
        default='0.0.0.0',
        validation_alias='DASH_HOST',
        description='Dashboard host (0.0.0.0 for all interfaces)'
    )

    port: int = Field(
        default=8050,
        validation_alias='DASH_PORT',
        description='Dashboard port'
    )

    debug: bool = Field(
        default=False,
        validation_alias='DASH_DEBUG',
        description='Enable debug mode'
    )

    @field_validator('port')
    @classmethod
    def validate_port(cls, v):
        """Validate port range."""
        if not 1024 <= v <= 65535:
            raise ValueError(
                f"Invalid port: {v}. "
                "Must be in range 1024-65535 (non-privileged ports)"
            )
        return v

    model_config = {
        'env_file': '.env',
        'case_sensitive': False,
        'extra': 'ignore'
    }


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    level: str = Field(
        default='INFO',
        validation_alias='LOG_LEVEL',
        description='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)'
    )

    file: Optional[str] = Field(
        default=None,
        validation_alias='LOG_FILE',
        description='Log file path (None = stdout only)'
    )

    format: str = Field(
        default='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        validation_alias='LOG_FORMAT',
        description='Log message format'
    )

    @field_validator('level')
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(
                f"Invalid log level: '{v}'. "
                f"Must be one of: {', '.join(valid_levels)}"
            )
        return v_upper

    model_config = {
        'env_file': '.env',
        'case_sensitive': False,
        'extra': 'ignore'
    }


class Settings(BaseSettings):
    """
    Global settings aggregator.

    Loads and validates all configuration from environment variables.
    Raises clear errors on startup if required values are missing.

    Usage:
        from config.settings import settings

        # Access nested configs
        kafka_servers = settings.kafka.bootstrap_servers_list
        reddit_client = praw.Reddit(
            client_id=settings.reddit.client_id,
            client_secret=settings.reddit.client_secret,
            user_agent=settings.reddit.user_agent
        )
    """

    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    reddit: RedditConfig = Field(default_factory=RedditConfig)
    binance: BinanceConfig = Field(default_factory=BinanceConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    timescale: TimescaleConfig = Field(default_factory=TimescaleConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    dfm: DFMConfig = Field(default_factory=DFMConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = {
        'env_file': '.env',
        'case_sensitive': False,
        'extra': 'ignore'
    }

    def __repr__(self) -> str:
        """Display configuration (with secrets masked)."""
        lines = [
            "=== Configuration ===",
            "",
            "Kafka:",
            f"  Bootstrap servers: {self.kafka.bootstrap_servers}",
            f"  Topics: {self.kafka.topic_orderbooks}, {self.kafka.topic_reddit}, {self.kafka.topic_sentiment}",
            "",
            "Reddit:",
            f"  Client ID: {self.reddit.client_id[:8]}...",
            f"  Subreddits: {len(self.reddit.subreddits_list)} monitored",
            "",
            "Binance:",
            f"  WebSocket: {self.binance.websocket_url}",
            f"  Symbols: {', '.join(self.binance.symbols_list)}",
            f"  Update speed: {self.binance.depth_update_speed}",
            "",
            "Sentiment:",
            f"  Model: {self.sentiment.model_name}",
            f"  MC samples: {self.sentiment.mc_samples}",
            f"  Device: {self.sentiment.device or 'auto'}",
            "",
            "TimescaleDB:",
            f"  Host: {self.timescale.host}:{self.timescale.port}",
            f"  Database: {self.timescale.database}",
            "",
            "Simulation:",
            f"  Total agents: {self.simulation.total_agents}",
            f"  Tick interval: {self.simulation.tick_interval_ms}ms",
            "",
            "Dashboard:",
            f"  Address: http://{self.dashboard.host}:{self.dashboard.port}",
            "",
            "Logging:",
            f"  Level: {self.logging.level}",
            f"  File: {self.logging.file or 'stdout'}",
        ]
        return "\n".join(lines)


# Global settings instance
# Import this in your modules:
#   from config.settings import settings
try:
    settings = Settings()
except Exception as e:
    print(f"\n{'='*60}")
    print("CONFIGURATION ERROR")
    print(f"{'='*60}")
    print(f"\n{e}\n")
    print("Please check your .env file or environment variables.")
    print("See config/README.md for setup instructions.")
    print(f"{'='*60}\n")
    raise


if __name__ == '__main__':
    # Test configuration loading
    print(settings)
