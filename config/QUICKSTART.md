# Configuration Quickstart

Get up and running with the configuration system in 5 minutes.

## Step 1: Install Dependencies (30 seconds)

```bash
cd /home/kawaiikali/Documents/Resurrexi/coding-with-buddy/sentiment-microstructure-abm

# Install pydantic-settings if not already installed
pip install pydantic-settings==2.1.0 pydantic==2.5.0 python-dotenv==1.0.0

# Or install all requirements
pip install -r requirements.txt
```

## Step 2: Create Configuration File (1 minute)

```bash
# Copy the example configuration
cp config/.env.example .env

# Edit with your values
nano .env
```

**Minimum required values:**

```bash
# REQUIRED: Kafka broker address
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# REQUIRED: Reddit API credentials (get from https://www.reddit.com/prefs/apps)
REDDIT_CLIENT_ID=your_actual_client_id
REDDIT_CLIENT_SECRET=your_actual_client_secret
```

Everything else has sensible defaults.

## Step 3: Test Configuration (10 seconds)

```bash
# Validate your configuration
python -m config.settings
```

**Success looks like:**
```
=== Configuration ===

Kafka:
  Bootstrap servers: localhost:9092
  Topics: order-books, reddit-posts, sentiment-ticks

Reddit:
  Client ID: abcdef12...
  Subreddits: 7 monitored

Binance:
  WebSocket: wss://stream.binance.com:9443/ws
  Symbols: btcusdt
  Update speed: 100ms

...
```

**Failure looks like:**
```
============================================================
CONFIGURATION ERROR
============================================================

1 validation error for RedditConfig
client_id
  REDDIT_CLIENT_ID is required. Get credentials from https://www.reddit.com/prefs/apps
  (type=value_error)

Please check your .env file or environment variables.
See config/README.md for setup instructions.
============================================================
```

Fix the error and test again.

## Step 4: Use in Your Code (1 minute)

**Replace this:**
```python
import os
from dotenv import load_dotenv

load_dotenv()

kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
```

**With this:**
```python
from config.settings import settings

kafka_servers = settings.kafka.bootstrap_servers_list  # Already a list!
reddit_client_id = settings.reddit.client_id
reddit_client_secret = settings.reddit.client_secret
```

**That's it!** Your configuration is now:
- Type-safe
- Validated on startup
- Auto-completed by your IDE

## Common First-Time Issues

### Issue: "ModuleNotFoundError: No module named 'pydantic_settings'"

**Solution:**
```bash
pip install pydantic-settings==2.1.0
```

### Issue: "REDDIT_CLIENT_ID is required"

**Solution:** You need actual Reddit API credentials:

1. Go to https://www.reddit.com/prefs/apps
2. Click "create another app"
3. Select "script" as type
4. Copy the client ID and secret to your `.env` file

### Issue: "Invalid Kafka server format"

**Solution:** Kafka servers must be in `host:port` format:

```bash
# Wrong
KAFKA_BOOTSTRAP_SERVERS=localhost

# Right
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
```

### Issue: Configuration not found

**Solution:** Make sure `.env` is in the project root (same directory as `config/`):

```bash
# Check file location
ls -la .env

# Should be at:
# /home/kawaiikali/Documents/Resurrexi/coding-with-buddy/sentiment-microstructure-abm/.env
```

## Quick Reference

### Access Patterns

```python
from config.settings import settings

# Kafka
kafka_servers = settings.kafka.bootstrap_servers_list  # List[str]
kafka_topic = settings.kafka.topic_reddit  # str
compression = settings.kafka.compression_type  # str

# Reddit
client_id = settings.reddit.client_id  # str
subreddits = settings.reddit.subreddits_list  # List[str]

# Binance
ws_url = settings.binance.websocket_url  # str
symbols = settings.binance.symbols_list  # List[str]
update_speed = settings.binance.depth_update_speed  # str

# Sentiment
model = settings.sentiment.model_name  # str
samples = settings.sentiment.mc_samples  # int
alpha = settings.sentiment.ewma_alpha  # float

# Database
conn_str = settings.timescale.connection_string  # str
db_host = settings.timescale.host  # str
db_port = settings.timescale.port  # int

# Simulation
total_agents = settings.simulation.total_agents  # int (computed)
tick_ms = settings.simulation.tick_interval_ms  # int

# Logging
log_level = settings.logging.level  # str
log_file = settings.logging.file  # Optional[str]
```

### Environment Variables

```bash
# Kafka (required)
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Reddit (required)
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret

# All others optional with defaults
BINANCE_SYMBOLS=btcusdt,ethusdt
SENTIMENT_MC_SAMPLES=30
LOG_LEVEL=DEBUG
```

## Next Steps

Now that you have configuration working:

1. **Migrate existing code:** See [MIGRATION.md](MIGRATION.md)
2. **Understand validation:** See [README.md](zArchive/README.md)
3. **Explore architecture:** See [ARCHITECTURE.md](ARCHITECTURE.md)
4. **Review all options:** See [.env.example](.env.example)

## Getting Help

- **Error messages:** They're designed to be helpful! Read them carefully.
- **Documentation:** [README.md](zArchive/README.md) has complete docs for every config option
- **Examples:** [MIGRATION.md](MIGRATION.md) shows before/after patterns
- **Test config:** `python -m config.settings` validates without running the app

## Cheat Sheet

```bash
# Setup
cp config/.env.example .env
nano .env
python -m config.settings

# In code
from config.settings import settings
kafka_servers = settings.kafka.bootstrap_servers_list

# Debug
export LOG_LEVEL=DEBUG
python -m config.settings

# Docker
export KAFKA_BOOTSTRAP_SERVERS=kafka:9092
export REDDIT_CLIENT_ID=...
```

That's everything you need to get started! Configuration is now bulletproof.
