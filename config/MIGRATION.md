# Configuration Migration Guide

This guide shows how to refactor existing code to use the new centralized configuration system.

## Overview

**Before:** Scattered `os.getenv()` calls with manual parsing and validation
**After:** Type-safe, validated configuration via `settings` object

## Benefits

- **Fail fast:** Configuration errors caught on startup, not at runtime
- **Type safety:** IDE autocomplete, no string parsing bugs
- **Clear errors:** "KAFKA_BOOTSTRAP_SERVERS is required" instead of "NoneType has no split()"
- **Centralized:** One place to see all configuration options
- **Validated:** Invalid values rejected with helpful messages

## Migration Examples

### Example 1: Reddit Client

**Before:**
```python
import os
from dotenv import load_dotenv

load_dotenv()

self.reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent=os.getenv('REDDIT_USER_AGENT')
)

kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
self.kafka_topic = os.getenv('KAFKA_TOPIC_REDDIT', 'reddit-posts')

self.producer = KafkaProducer(
    bootstrap_servers=kafka_servers.split(','),
    compression_type='gzip'
)
```

**After:**
```python
from config.settings import settings

self.reddit = praw.Reddit(
    client_id=settings.reddit.client_id,
    client_secret=settings.reddit.client_secret,
    user_agent=settings.reddit.user_agent
)

self.producer = KafkaProducer(
    bootstrap_servers=settings.kafka.bootstrap_servers_list,
    compression_type=settings.kafka.compression_type
)

self.kafka_topic = settings.kafka.topic_reddit
```

**What changed:**
- No more `load_dotenv()` (settings object handles it)
- No more `.split(',')` (use `bootstrap_servers_list` property)
- No more defaults in code (centralized in settings)
- Configuration validated on import

### Example 2: Binance Client

**Before:**
```python
import os
from dotenv import load_dotenv

load_dotenv()

base_url = os.getenv('BINANCE_WEBSOCKET_URL', 'wss://stream.binance.com:9443/ws')
self.ws_url = f"{base_url}/{symbol}@depth{levels}@{speed}"

kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
self.kafka_topic = os.getenv('KAFKA_TOPIC_ORDERBOOKS', 'order-books')

self.producer = KafkaProducer(
    bootstrap_servers=kafka_servers.split(','),
    compression_type='gzip'
)
```

**After:**
```python
from config.settings import settings

self.ws_url = f"{settings.binance.websocket_url}/{symbol}@depth{levels}@{speed}"

self.producer = KafkaProducer(
    bootstrap_servers=settings.kafka.bootstrap_servers_list,
    compression_type=settings.kafka.compression_type
)

self.kafka_topic = settings.kafka.topic_orderbooks
```

**What changed:**
- Direct access to validated config values
- No string parsing or splitting
- Type-safe access (IDE knows these are strings/ints)

### Example 3: Sentiment Analyzer

**Before:**
```python
import os
from dotenv import load_dotenv

load_dotenv()

model_name = os.getenv('SENTIMENT_MODEL_NAME', 'distilroberta-base')
n_samples = int(os.getenv('SENTIMENT_MC_SAMPLES', '20'))
alpha = float(os.getenv('SENTIMENT_EWMA_ALPHA', '0.3'))
device = os.getenv('SENTIMENT_DEVICE', None)

self.tokenizer = AutoTokenizer.from_pretrained(model_name)
self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
self.n_samples = n_samples
self.ewma_alpha = alpha
self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
```

**After:**
```python
from config.settings import settings

self.tokenizer = AutoTokenizer.from_pretrained(settings.sentiment.model_name)
self.model = AutoModelForSequenceClassification.from_pretrained(settings.sentiment.model_name)
self.n_samples = settings.sentiment.mc_samples
self.ewma_alpha = settings.sentiment.ewma_alpha
self.device = settings.sentiment.device or ('cuda' if torch.cuda.is_available() else 'cpu')
```

**What changed:**
- No manual type conversion (`int()`, `float()`)
- No defaults scattered in code
- Values already validated (alpha in [0,1], samples > 0, etc.)

### Example 4: Initialization with Overrides

Sometimes you want to allow function arguments to override config:

**Before:**
```python
def __init__(self, kafka_bootstrap_servers: str = None, kafka_topic: str = None):
    kafka_servers = kafka_bootstrap_servers or os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    self.kafka_topic = kafka_topic or os.getenv('KAFKA_TOPIC_REDDIT', 'reddit-posts')
```

**After:**
```python
from config.settings import settings

def __init__(self, kafka_bootstrap_servers: str = None, kafka_topic: str = None):
    kafka_servers = kafka_bootstrap_servers or settings.kafka.bootstrap_servers
    self.kafka_topic = kafka_topic or settings.kafka.topic_reddit
```

**Pattern:**
- Use function argument if provided
- Fall back to validated settings
- No need for third fallback (settings has defaults)

## Complete Refactored Example: RedditClient

Here's a complete refactored version of `reddit_client.py`:

```python
"""
Reddit Sentiment Data Ingestion

Streams posts and comments from crypto-related subreddits in real-time.
Publishes to Kafka topic for sentiment analysis pipeline.
"""

import praw
import json
from kafka import KafkaProducer
from datetime import datetime
from typing import List, Dict, Optional
import logging

from config.settings import settings

# Setup logging with configured level
logging.basicConfig(
    level=settings.logging.level,
    format=settings.logging.format
)
logger = logging.getLogger(__name__)


class RedditClient:
    """
    Reddit API client for crypto sentiment data collection.

    Streams from multiple subreddits and publishes to Kafka.
    """

    def __init__(
        self,
        subreddits: Optional[List[str]] = None,
        kafka_bootstrap_servers: Optional[str] = None,
        kafka_topic: Optional[str] = None
    ):
        """
        Initialize Reddit client.

        Args:
            subreddits: List of subreddit names (default: from config)
            kafka_bootstrap_servers: Kafka connection string (default: from config)
            kafka_topic: Kafka topic for reddit posts (default: from config)
        """
        # Reddit API credentials (always from config - required values)
        self.reddit = praw.Reddit(
            client_id=settings.reddit.client_id,
            client_secret=settings.reddit.client_secret,
            user_agent=settings.reddit.user_agent
        )

        # Subreddits to monitor (allow override)
        self.subreddits = subreddits or settings.reddit.subreddits_list

        # Kafka configuration (allow overrides for testing)
        kafka_servers = kafka_bootstrap_servers or settings.kafka.bootstrap_servers
        self.kafka_topic = kafka_topic or settings.kafka.topic_reddit

        self.producer = KafkaProducer(
            bootstrap_servers=kafka_servers.split(',') if isinstance(kafka_servers, str) else kafka_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            compression_type=settings.kafka.compression_type
        )

        logger.info(f"Initialized Reddit client for subreddits: {self.subreddits}")
        logger.info(f"Publishing to Kafka topic: {self.kafka_topic}")

    # ... rest of implementation unchanged ...
```

**Key changes:**
1. Import `settings` at module level
2. Remove `load_dotenv()` call
3. Use `settings.reddit.*` for credentials
4. Use `settings.kafka.*` for Kafka config
5. Use `settings.logging.*` for logging setup
6. Still allow overrides via function arguments (useful for testing)

## Migration Checklist

For each module that uses configuration:

- [ ] Add `from config.settings import settings` at top
- [ ] Remove `from dotenv import load_dotenv` and `load_dotenv()` calls
- [ ] Replace `os.getenv()` calls with `settings.*.` access
- [ ] Remove manual string parsing (`.split()`, `int()`, `float()`)
- [ ] Remove default values (use settings defaults instead)
- [ ] Update type hints (no more `str` for integer configs)
- [ ] Remove manual validation (settings validates)
- [ ] Test with `python -m config.settings` first
- [ ] Test module functionality

## Testing Strategy

### 1. Test configuration loading first

```bash
# This will validate all config and show summary
python -m config.settings
```

If this fails, fix your `.env` before proceeding.

### 2. Test module imports

```python
# Test that module imports successfully (validates config)
python -c "from data_ingestion.reddit_client import RedditClient"
```

### 3. Test module functionality

```python
# Test with actual execution
python data_ingestion/reddit_client.py --mode submissions --limit 5
```

## Common Migration Issues

### Issue: Import fails with validation error

**Error:**
```
CONFIGURATION ERROR
REDDIT_CLIENT_ID is required. Get credentials from https://www.reddit.com/prefs/apps
```

**Solution:** Fix your `.env` file before importing modules.

### Issue: Can't override config in tests

**Before (doesn't work):**
```python
os.environ['KAFKA_BOOTSTRAP_SERVERS'] = 'test-kafka:9092'
from config.settings import settings  # Too late, already loaded
```

**After (works):**
```python
# Option 1: Pass overrides to constructors
client = RedditClient(kafka_bootstrap_servers='test-kafka:9092')

# Option 2: Mock the settings (for unit tests)
from unittest.mock import patch

with patch('config.settings.settings.kafka.bootstrap_servers', 'test-kafka:9092'):
    client = RedditClient()
```

### Issue: Default values changed behavior

If you had different defaults in different modules, you may see behavior changes.

**Solution:** Check the defaults in `config/settings.py` and adjust if needed, or pass explicit values.

### Issue: Type mismatches

**Before:** Everything was strings from `os.getenv()`
**After:** Correct types (int, float, bool, list)

If you were doing `if kafka_servers:` checks, you may need to adjust since values are now guaranteed to be present (or validation fails).

## Gradual Migration

You don't have to migrate everything at once:

1. **Add config system** (already done)
2. **Create `.env` file** with all values
3. **Test config loading**: `python -m config.settings`
4. **Migrate one module at a time**:
   - Start with simplest module
   - Test thoroughly
   - Commit
   - Move to next module
5. **Remove old patterns** once all modules migrated

## Getting Help

If you encounter issues:

1. Check error messages (they're descriptive!)
2. Verify `.env` syntax and values
3. Run `python -m config.settings` to test config
4. Check this migration guide for patterns
5. Review `config/README.md` for variable documentation

## Benefits Recap

After migration, you get:

- **No more cryptic runtime errors** like `'NoneType' object has no attribute 'split'`
- **Clear startup errors** with helpful messages
- **Type safety** and IDE autocomplete
- **Centralized configuration** - see all options in one place
- **Validation** - invalid values rejected immediately
- **Documentation** - each config field has description
- **Defaults** - sensible defaults for optional values
- **Testing** - easier to mock and override
- **Maintainability** - configuration changes in one file

Migration effort is small, benefits are significant!
