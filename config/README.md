# Configuration Management

Type-safe, validated configuration system using Pydantic BaseSettings. All environment variables are validated on startup with clear error messages.

## Quick Start

1. **Copy the example configuration:**
   ```bash
   cp config/.env.example .env
   ```

2. **Edit `.env` with your actual values:**
   ```bash
   nano .env  # or your favorite editor
   ```

3. **Test configuration loading:**
   ```bash
   python -m config.settings
   ```

If configuration is valid, you'll see a summary. If invalid, you'll get clear error messages explaining what's wrong.

## Usage in Code

```python
from config.settings import settings

# Access validated configuration
kafka_producer = KafkaProducer(
    bootstrap_servers=settings.kafka.bootstrap_servers_list,
    compression_type=settings.kafka.compression_type
)

reddit_client = praw.Reddit(
    client_id=settings.reddit.client_id,
    client_secret=settings.reddit.client_secret,
    user_agent=settings.reddit.user_agent
)

# Use convenience properties
subreddits = settings.reddit.subreddits_list  # List[str]
total_agents = settings.simulation.total_agents  # Computed property
```

## Configuration Sections

### Kafka Configuration

**Required:**
- `KAFKA_BOOTSTRAP_SERVERS` - Comma-separated broker addresses

**Optional:**
- `KAFKA_TOPIC_ORDERBOOKS` - Topic for order book data (default: `order-books`)
- `KAFKA_TOPIC_REDDIT` - Topic for Reddit posts (default: `reddit-posts`)
- `KAFKA_TOPIC_SENTIMENT` - Topic for sentiment results (default: `sentiment-ticks`)
- `KAFKA_COMPRESSION_TYPE` - Compression algorithm (default: `gzip`)

**Valid compression types:** `gzip`, `snappy`, `lz4`, `zstd`, `none`

**Examples:**
```bash
# Single broker
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Multiple brokers (high availability)
KAFKA_BOOTSTRAP_SERVERS=broker1:9092,broker2:9092,broker3:9092
```

**Validation:**
- Servers list must not be empty
- Each server must be in `host:port` format
- Compression type must be from valid list

### Reddit API Configuration

**Required:**
- `REDDIT_CLIENT_ID` - API client ID
- `REDDIT_CLIENT_SECRET` - API client secret

**Optional:**
- `REDDIT_USER_AGENT` - User agent string (default: `SentimentMicrostructureABM/1.0`)
- `REDDIT_SUBREDDITS` - Comma-separated subreddit names (defaults to crypto-related subs)

**How to get credentials:**
1. Go to https://www.reddit.com/prefs/apps
2. Click "create another app" at the bottom
3. Fill in:
   - **Name:** SentimentMicrostructureABM
   - **Type:** script
   - **Redirect URI:** http://localhost:8080
4. Copy the client ID (under the app name) and secret

**Example:**
```bash
REDDIT_CLIENT_ID=abcdef123456789
REDDIT_CLIENT_SECRET=xyz789secretabc123
REDDIT_USER_AGENT=SentimentMicrostructureABM/1.0 by /u/yourusername
REDDIT_SUBREDDITS=CryptoCurrency,Bitcoin,ethereum
```

**Validation:**
- Client ID/secret cannot be placeholders (`your_client_id_here`, etc.)
- Subreddits parsed into list automatically via `settings.reddit.subreddits_list`

### Binance WebSocket Configuration

**Optional (all have sensible defaults):**
- `BINANCE_WEBSOCKET_URL` - WebSocket base URL (default: `wss://stream.binance.com:9443/ws`)
- `BINANCE_SYMBOLS` - Trading pairs to stream (default: `btcusdt`)
- `BINANCE_DEPTH_UPDATE_SPEED` - Update frequency (default: `100ms`)
- `BINANCE_DEPTH_LEVELS` - Number of price levels (default: `20`)

**Valid update speeds:** `100ms` (10 updates/sec), `1000ms` (1 update/sec)

**Valid depth levels:** `5`, `10`, `20`

**Example:**
```bash
BINANCE_WEBSOCKET_URL=wss://stream.binance.com:9443/ws
BINANCE_SYMBOLS=btcusdt,ethusdt,bnbusdt
BINANCE_DEPTH_UPDATE_SPEED=100ms
BINANCE_DEPTH_LEVELS=20
```

**Validation:**
- WebSocket URL must start with `wss://` or `ws://`
- Update speed must be `100ms` or `1000ms`
- Depth levels must be `5`, `10`, or `20`
- Symbols parsed into list automatically via `settings.binance.symbols_list`

### Sentiment Analysis Configuration

**Optional (all have defaults):**
- `SENTIMENT_MODEL_NAME` - HuggingFace model ID (default: `distilroberta-base`)
- `SENTIMENT_MODEL_PATH` - Path to fine-tuned model (optional)
- `SENTIMENT_MODEL_CACHE` - Model cache directory (default: `models/cache`)
- `SENTIMENT_BATCH_SIZE` - Inference batch size (default: `8`)
- `SENTIMENT_MC_SAMPLES` - Monte Carlo samples for uncertainty (default: `20`)
- `SENTIMENT_EWMA_ALPHA` - EWMA smoothing factor (default: `0.3`)
- `SENTIMENT_DEVICE` - PyTorch device (default: auto-detect)

**Example:**
```bash
SENTIMENT_MODEL_NAME=distilroberta-base
SENTIMENT_MODEL_PATH=models/sentiment-crypto-finetuned
SENTIMENT_BATCH_SIZE=16
SENTIMENT_MC_SAMPLES=30
SENTIMENT_EWMA_ALPHA=0.3
SENTIMENT_DEVICE=cuda
```

**Validation:**
- Batch size must be >= 1 and <= 128
- MC samples must be >= 1 and <= 100 (higher values = slower but better uncertainty estimates)
- EWMA alpha must be in range [0, 1]
- Device must be `cuda`, `cpu`, `mps`, or empty (auto-detect)

**Tuning guidelines:**
- **Batch size:** Higher = faster inference but more memory. Start with 8, increase if you have GPU headroom.
- **MC samples:** Higher = better uncertainty estimates but slower. 20 is good baseline, 50 for research quality.
- **EWMA alpha:** Higher = more responsive to new data. 0.3 balances responsiveness and smoothing.

### TimescaleDB Configuration

**Optional (all have defaults for local development):**
- `TIMESCALE_HOST` - Database host (default: `localhost`)
- `TIMESCALE_PORT` - Database port (default: `5432`)
- `TIMESCALE_DB` - Database name (default: `market_sim`)
- `TIMESCALE_USER` - Database user (default: `postgres`)
- `TIMESCALE_PASSWORD` - Database password (default: `postgres`)

**Example:**
```bash
TIMESCALE_HOST=timescale.example.com
TIMESCALE_PORT=5432
TIMESCALE_DB=market_sim_prod
TIMESCALE_USER=market_user
TIMESCALE_PASSWORD=secure_password_here
```

**Usage:**
```python
# Get connection string
conn_str = settings.timescale.connection_string
# postgresql://user:pass@host:port/database
```

**Validation:**
- Port must be in range 1-65535

### Simulation Parameters

**Optional (all have defaults):**
- `SIM_TICK_INTERVAL_MS` - Tick interval in milliseconds (default: `500`)
- `SIM_N_MARKET_MAKERS` - Number of market maker agents (default: `10`)
- `SIM_N_INFORMED_TRADERS` - Number of informed traders (default: `20`)
- `SIM_N_NOISE_TRADERS` - Number of noise traders (default: `50`)
- `SIM_N_ARBITRAGEURS` - Number of arbitrageurs (default: `5`)

**Example:**
```bash
SIM_TICK_INTERVAL_MS=250
SIM_N_MARKET_MAKERS=15
SIM_N_INFORMED_TRADERS=30
SIM_N_NOISE_TRADERS=100
SIM_N_ARBITRAGEURS=10
```

**Usage:**
```python
total = settings.simulation.total_agents  # Computed property
# total = 15 + 30 + 100 + 10 = 155 agents
```

**Validation:**
- All values must be non-negative

### Dynamic Factor Model (DFM) Parameters

**Optional (all have defaults):**
- `DFM_WINDOW_SIZE_MINUTES` - Rolling window size (default: `60`)
- `DFM_UPDATE_INTERVAL_MINUTES` - Update frequency (default: `5`)
- `DFM_N_FACTORS` - Number of latent factors (default: `3`)

**Example:**
```bash
DFM_WINDOW_SIZE_MINUTES=120
DFM_UPDATE_INTERVAL_MINUTES=10
DFM_N_FACTORS=5
```

**Validation:**
- Window size and update interval must be > 0
- Number of factors must be >= 1 and <= 10

### Dashboard Configuration

**Optional (all have defaults):**
- `DASH_HOST` - Server host (default: `0.0.0.0`)
- `DASH_PORT` - Server port (default: `8050`)
- `DASH_DEBUG` - Enable debug mode (default: `false`)

**Example:**
```bash
DASH_HOST=0.0.0.0
DASH_PORT=8050
DASH_DEBUG=false
```

**Validation:**
- Port must be in range 1024-65535 (non-privileged ports)

### Logging Configuration

**Optional (all have defaults):**
- `LOG_LEVEL` - Logging level (default: `INFO`)
- `LOG_FILE` - Log file path (default: stdout only)
- `LOG_FORMAT` - Log message format (default: Python logging format)

**Valid log levels:** `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

**Example:**
```bash
LOG_LEVEL=DEBUG
LOG_FILE=logs/market_sim.log
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

**Validation:**
- Log level must be one of the valid levels (case-insensitive)

## Error Messages

The configuration system provides clear error messages when values are missing or invalid:

### Missing Required Value
```
CONFIGURATION ERROR
================================================================

1 validation error for RedditConfig
client_id
  REDDIT_CLIENT_ID is required. Get credentials from https://www.reddit.com/prefs/apps
  (type=value_error)

Please check your .env file or environment variables.
See config/README.md for setup instructions.
================================================================
```

### Invalid Format
```
CONFIGURATION ERROR
================================================================

1 validation error for KafkaConfig
bootstrap_servers
  Invalid Kafka server format: 'localhost'. Expected format: host:port
  (type=value_error)

Please check your .env file or environment variables.
See config/README.md for setup instructions.
================================================================
```

### Out of Range Value
```
CONFIGURATION ERROR
================================================================

1 validation error for SentimentConfig
ewma_alpha
  ewma_alpha must be in [0, 1], got 1.5
  (type=value_error)

Please check your .env file or environment variables.
See config/README.md for setup instructions.
================================================================
```

## Convenience Properties

The configuration objects provide convenience properties for common operations:

```python
# Get comma-separated values as lists
kafka_servers = settings.kafka.bootstrap_servers_list  # List[str]
subreddits = settings.reddit.subreddits_list  # List[str]
symbols = settings.binance.symbols_list  # List[str]

# Get computed values
total_agents = settings.simulation.total_agents  # int
db_connection = settings.timescale.connection_string  # str
```

## Testing Configuration

Test your configuration without running the full application:

```bash
# Load and validate configuration
python -m config.settings

# Should output configuration summary if valid
# Or clear error messages if invalid
```

## Environment Variable Precedence

Configuration is loaded in the following order (later overrides earlier):

1. Default values in code
2. `.env` file in project root
3. System environment variables

This allows you to:
- Use `.env` for local development
- Override with environment variables in production (Docker, K8s)
- Have sensible defaults for optional values

## Docker/Kubernetes Usage

In containerized environments, pass configuration via environment variables:

**Docker Compose:**
```yaml
services:
  reddit-client:
    image: sentiment-abm:latest
    environment:
      KAFKA_BOOTSTRAP_SERVERS: kafka:9092
      REDDIT_CLIENT_ID: ${REDDIT_CLIENT_ID}
      REDDIT_CLIENT_SECRET: ${REDDIT_CLIENT_SECRET}
```

**Kubernetes ConfigMap:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: abm-config
data:
  KAFKA_BOOTSTRAP_SERVERS: "kafka-service:9092"
  BINANCE_SYMBOLS: "btcusdt,ethusdt"
  LOG_LEVEL: "INFO"
```

**Kubernetes Secret:**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: abm-secrets
type: Opaque
stringData:
  REDDIT_CLIENT_ID: "your_client_id"
  REDDIT_CLIENT_SECRET: "your_client_secret"
  TIMESCALE_PASSWORD: "secure_password"
```

## Troubleshooting

### Configuration not loading

**Problem:** `python -m config.settings` fails with import errors

**Solution:** Install pydantic first:
```bash
pip install pydantic python-dotenv
```

### Values not being read from .env

**Problem:** Environment variables showing defaults instead of .env values

**Solution:**
1. Ensure `.env` is in the project root (same directory as `config/`)
2. Check `.env` syntax (no spaces around `=`)
3. Verify no conflicting system environment variables

### Secrets showing in logs

**Problem:** Worried about secrets in configuration output

**Solution:** The `__repr__` method masks sensitive values:
```python
print(settings)
# Client ID: abcdef12... (truncated)
```

Never log `settings.reddit.client_secret` directly in production.

## Best Practices

1. **Never commit `.env`** - Add to `.gitignore` (already done)
2. **Use example file** - Keep `.env.example` updated with all variables
3. **Validate early** - Import `settings` at module level to fail fast
4. **Use type hints** - IDEs will autocomplete configuration fields
5. **Document changes** - Update this README when adding new config options

## Adding New Configuration

To add a new configuration section:

1. **Create new config class in `settings.py`:**
   ```python
   class NewFeatureConfig(BaseSettings):
       some_value: str = Field(
           default='default_value',
           env='NEW_FEATURE_VALUE',
           description='What this value does'
       )

       @validator('some_value')
       def validate_some_value(cls, v):
           # Add validation logic
           return v

       class Config:
           env_file = '.env'
           case_sensitive = False
   ```

2. **Add to `Settings` class:**
   ```python
   class Settings(BaseSettings):
       # ... existing configs ...
       new_feature: NewFeatureConfig = NewFeatureConfig()
   ```

3. **Update `.env.example`:**
   ```bash
   # New Feature Configuration
   NEW_FEATURE_VALUE=default_value
   ```

4. **Document in this README** (add new section above)

5. **Test:**
   ```bash
   python -m config.settings
   ```

## Migration from Old Config

If you have existing code using `os.getenv()`:

**Before:**
```python
import os
from dotenv import load_dotenv

load_dotenv()
kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092').split(',')
```

**After:**
```python
from config.settings import settings

kafka_servers = settings.kafka.bootstrap_servers_list
```

**Benefits:**
- Type safety (no more string parsing)
- Validation on startup
- Clear error messages
- IDE autocomplete
- Centralized configuration

## Support

If you encounter configuration issues:

1. Check error messages (they're designed to be helpful!)
2. Verify `.env` syntax
3. Test with `python -m config.settings`
4. Review this README for variable documentation
5. Check `.env.example` for correct format

For questions about specific configuration options, see the relevant section above or check the inline documentation in `config/settings.py`.
