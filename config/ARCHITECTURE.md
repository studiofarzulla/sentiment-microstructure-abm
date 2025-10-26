# Configuration System Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Configuration Management System                    │
│                                                                       │
│  ┌────────────────┐      ┌──────────────┐      ┌─────────────────┐ │
│  │  Environment   │─────▶│   Pydantic   │─────▶│   Application   │ │
│  │   Variables    │      │  BaseSettings │      │     Modules     │ │
│  │                │      │              │      │                 │ │
│  │ • .env file    │      │ • Validation │      │ • reddit_client │ │
│  │ • System env   │      │ • Type conv. │      │ • binance_client│ │
│  │ • Defaults     │      │ • Parsing    │      │ • sentiment_*   │ │
│  └────────────────┘      └──────────────┘      └─────────────────┘ │
│                                                                       │
│  Flow: ENV → Validate → Type-safe config objects → Application      │
└─────────────────────────────────────────────────────────────────────┘
```

## Configuration Class Hierarchy

```
Settings (Global aggregator)
│
├── KafkaConfig
│   ├── bootstrap_servers: str (required)
│   ├── topic_orderbooks: str
│   ├── topic_reddit: str
│   ├── topic_sentiment: str
│   ├── compression_type: str (validated enum)
│   └── bootstrap_servers_list: List[str] (property)
│
├── RedditConfig
│   ├── client_id: str (required)
│   ├── client_secret: str (required)
│   ├── user_agent: str
│   ├── subreddits: str
│   └── subreddits_list: List[str] (property)
│
├── BinanceConfig
│   ├── websocket_url: str (validated format)
│   ├── symbols: str
│   ├── depth_update_speed: str (validated enum)
│   ├── depth_levels: int (validated enum)
│   └── symbols_list: List[str] (property)
│
├── SentimentConfig
│   ├── model_name: str
│   ├── model_path: Optional[str]
│   ├── model_cache: str
│   ├── batch_size: int (range validated)
│   ├── mc_samples: int (range validated)
│   ├── ewma_alpha: float (range validated)
│   └── device: Optional[str] (validated enum)
│
├── TimescaleConfig
│   ├── host: str
│   ├── port: int (range validated)
│   ├── database: str
│   ├── user: str
│   ├── password: str
│   └── connection_string: str (property)
│
├── SimulationConfig
│   ├── tick_interval_ms: int
│   ├── n_market_makers: int
│   ├── n_informed_traders: int
│   ├── n_noise_traders: int
│   ├── n_arbitrageurs: int
│   └── total_agents: int (property)
│
├── DFMConfig
│   ├── window_size_minutes: int
│   ├── update_interval_minutes: int
│   └── n_factors: int (range validated)
│
├── DashboardConfig
│   ├── host: str
│   ├── port: int (range validated)
│   └── debug: bool
│
└── LoggingConfig
    ├── level: str (validated enum)
    ├── file: Optional[str]
    └── format: str
```

## Data Flow

### 1. Configuration Loading (Startup)

```
Application Start
      │
      ▼
Import settings module
      │
      ▼
Settings() instantiation
      │
      ├──▶ Load .env file (if exists)
      │
      ├──▶ Read system environment variables
      │
      ├──▶ Apply defaults for optional values
      │
      ▼
For each config class:
      │
      ├──▶ Parse environment variables
      │
      ├──▶ Convert types (str → int/float/bool)
      │
      ├──▶ Run validators (@field_validator)
      │     │
      │     ├──▶ Format validation (URLs, host:port)
      │     ├──▶ Range validation (ports, alpha, samples)
      │     ├──▶ Enum validation (compression, log level)
      │     └──▶ Custom validation (no placeholders)
      │
      └──▶ Create config object
            │
            ├──▶ SUCCESS: settings object ready
            │
            └──▶ FAILURE: Clear error message + exit
```

### 2. Configuration Access (Runtime)

```
Module Import
      │
      ▼
from config.settings import settings
      │
      ▼
Access configuration
      │
      ├──▶ settings.kafka.bootstrap_servers_list
      │     └──▶ Returns: List[str] (already parsed)
      │
      ├──▶ settings.sentiment.mc_samples
      │     └──▶ Returns: int (already validated)
      │
      └──▶ settings.simulation.total_agents
            └──▶ Returns: int (computed property)
```

## Validation Architecture

### Three Layers of Validation

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Type Validation (Pydantic automatic)              │
│                                                              │
│  bootstrap_servers: str  ────▶ Must be string              │
│  port: int              ────▶ Must be integer              │
│  debug: bool            ────▶ Must be boolean              │
│  ewma_alpha: float      ────▶ Must be float                │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: Field Validation (@field_validator)               │
│                                                              │
│  Format:                                                     │
│    • URLs must start with wss:// or ws://                  │
│    • Servers must be host:port format                      │
│                                                              │
│  Range:                                                      │
│    • Ports in 1-65535                                       │
│    • Alpha in [0, 1]                                        │
│    • Batch size 1-128                                       │
│                                                              │
│  Enum:                                                       │
│    • Compression: gzip|snappy|lz4|zstd|none               │
│    • Log level: DEBUG|INFO|WARNING|ERROR|CRITICAL          │
│    • Depth speed: 100ms|1000ms                             │
│                                                              │
│  Custom:                                                     │
│    • No placeholder values (your_client_id_here)           │
│    • Non-empty required fields                             │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Business Logic (Application level)                │
│                                                              │
│  • Kafka brokers are reachable                             │
│  • Reddit credentials are valid                            │
│  • Model files exist at specified paths                    │
│  • Database connection succeeds                            │
│                                                              │
│  (These checks happen after config validation)             │
└─────────────────────────────────────────────────────────────┘
```

## Error Handling Flow

```
Configuration Error Detected
      │
      ▼
Catch Exception in settings.py
      │
      ▼
Print Formatted Error:
┌────────────────────────────────────┐
│ CONFIGURATION ERROR                │
│ ────────────────────────────────── │
│                                    │
│ 1 validation error for RedditConfig│
│ client_id                          │
│   REDDIT_CLIENT_ID is required.   │
│   Get credentials from             │
│   https://www.reddit.com/...       │
│                                    │
│ Please check your .env file or     │
│ environment variables.             │
│ See config/README.md for setup     │
│ instructions.                      │
└────────────────────────────────────┘
      │
      ▼
Re-raise Exception
      │
      ▼
Application Exits (fail fast)
```

## Integration Points

### Module Integration Pattern

```python
# Old pattern (scattered, error-prone)
import os
from dotenv import load_dotenv

load_dotenv()
kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092').split(',')
# Error: What if KAFKA_BOOTSTRAP_SERVERS is None?
# Error: What if it's not in host:port format?
# Error: Discovered at runtime, not startup

# New pattern (centralized, validated)
from config.settings import settings

kafka_servers = settings.kafka.bootstrap_servers_list
# ✓ Validated on startup
# ✓ Always a list
# ✓ Always in host:port format
# ✓ Type-safe with IDE autocomplete
```

### Testing Integration

```python
# Unit tests can mock settings
from unittest.mock import patch

def test_kafka_connection():
    with patch('config.settings.settings.kafka.bootstrap_servers', 'test:9092'):
        client = MyKafkaClient()
        assert client.servers == 'test:9092'

# Or pass overrides to constructors
def test_with_override():
    client = MyKafkaClient(kafka_bootstrap_servers='test:9092')
    assert client.servers == 'test:9092'
```

### Docker Integration

```dockerfile
# Dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

# Configuration via environment variables
ENV KAFKA_BOOTSTRAP_SERVERS=kafka:9092
ENV REDDIT_CLIENT_ID=${REDDIT_CLIENT_ID}
ENV REDDIT_CLIENT_SECRET=${REDDIT_CLIENT_SECRET}

CMD ["python", "main.py"]
```

### Kubernetes Integration

```yaml
# ConfigMap for non-sensitive config
apiVersion: v1
kind: ConfigMap
metadata:
  name: abm-config
data:
  KAFKA_BOOTSTRAP_SERVERS: "kafka-service:9092"
  BINANCE_SYMBOLS: "btcusdt,ethusdt"
  LOG_LEVEL: "INFO"

---

# Secret for sensitive config
apiVersion: v1
kind: Secret
metadata:
  name: abm-secrets
type: Opaque
stringData:
  REDDIT_CLIENT_ID: "..."
  REDDIT_CLIENT_SECRET: "..."
  TIMESCALE_PASSWORD: "..."

---

# Deployment using config
apiVersion: apps/v1
kind: Deployment
metadata:
  name: reddit-client
spec:
  template:
    spec:
      containers:
      - name: reddit-client
        image: abm:latest
        envFrom:
        - configMapRef:
            name: abm-config
        - secretRef:
            name: abm-secrets
```

## Performance Characteristics

### Startup Time

- **Config loading:** ~10-50ms (Pydantic parsing + validation)
- **Validation overhead:** Minimal (one-time on startup)
- **Memory footprint:** ~1KB per config object (~10KB total)

### Runtime Performance

- **Config access:** O(1) attribute lookup (no parsing at runtime)
- **Property evaluation:** Computed once per access (e.g., `total_agents`)
- **No file I/O:** All config loaded into memory on startup

### Scaling Considerations

- **Large deployments:** Environment variables scale better than files
- **Config hot-reload:** Not implemented (restart required for changes)
- **Distributed systems:** Each service loads own config independently

## Security Architecture

### Secret Management

```
┌─────────────────────────────────────────────────────────────┐
│ Secret Hierarchy (from least to most secure)               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ 1. Hardcoded in code          ❌ NEVER DO THIS             │
│                                                              │
│ 2. .env file in repo          ❌ NEVER DO THIS             │
│                                                              │
│ 3. .env file (gitignored)     ⚠️  OK for local dev        │
│                                                              │
│ 4. System env variables       ✓  Good for containers       │
│                                                              │
│ 5. K8s Secrets                ✓  Good for production       │
│                                                              │
│ 6. HashiCorp Vault            ✓✓ Best for enterprise       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Current Implementation

- **Secrets in .env:** Gitignored, used for local development
- **Validation:** Rejects obvious placeholders (`your_client_id_here`)
- **Logging:** `__repr__` masks secrets (shows `abcdef12...` not full value)
- **No persistence:** Secrets stay in memory, never written to logs/files

### Future Enhancements

- **Vault integration:** Load secrets from HashiCorp Vault
- **Secret rotation:** Support dynamic secret updates
- **Encryption at rest:** Encrypt .env files with key derivation
- **Audit logging:** Log config access for compliance

## Extension Points

### Adding New Configuration

```python
# 1. Create new config class
class NewFeatureConfig(BaseSettings):
    some_value: str = Field(
        default='default',
        validation_alias='NEW_FEATURE_VALUE',
        description='What this does'
    )

    @field_validator('some_value')
    @classmethod
    def validate_some_value(cls, v):
        # Custom validation logic
        return v

    model_config = {
        'env_file': '.env',
        'case_sensitive': False,
        'extra': 'ignore'
    }

# 2. Add to Settings class
class Settings(BaseSettings):
    # ... existing configs ...
    new_feature: NewFeatureConfig = Field(default_factory=NewFeatureConfig)
```

### Custom Validators

```python
# Validator with multiple fields
@field_validator('port')
@classmethod
def validate_port(cls, v, info):
    # Access other fields via info.data
    host = info.data.get('host')
    if host == 'localhost' and v < 1024:
        raise ValueError("Localhost requires port >= 1024")
    return v

# Cross-field validation
@model_validator(mode='after')
def validate_model(self):
    if self.min_value > self.max_value:
        raise ValueError("min_value must be <= max_value")
    return self
```

## Debugging Guide

### Configuration Not Loading

```bash
# 1. Test configuration loading
python -m config.settings

# 2. Check if .env exists
ls -la .env

# 3. Check environment variables
env | grep -E "KAFKA|REDDIT|BINANCE"

# 4. Enable debug logging
export LOG_LEVEL=DEBUG
python -m config.settings
```

### Validation Errors

```python
# Get detailed error information
try:
    settings = Settings()
except ValidationError as e:
    print(e.json())  # JSON format
    print(e.errors())  # List of errors

    # Each error has:
    # - loc: Field location (e.g., ('kafka', 'bootstrap_servers'))
    # - msg: Error message
    # - type: Error type
```

### IDE Integration

```python
# Enable type hints for better IDE support
from config.settings import settings

# Now IDE knows:
reveal_type(settings.kafka.bootstrap_servers)  # str
reveal_type(settings.kafka.bootstrap_servers_list)  # List[str]
reveal_type(settings.sentiment.mc_samples)  # int
reveal_type(settings.simulation.total_agents)  # int
```

## Design Decisions

### Why Pydantic BaseSettings?

**Alternatives considered:**
- python-decouple: Less type-safe, minimal validation
- dynaconf: More complex, harder to debug
- configparser: Old-school INI files, limited validation
- dataclasses + env vars: Manual parsing, no built-in validation

**Pydantic chosen for:**
- Type safety out of the box
- Comprehensive validation framework
- Excellent error messages
- IDE autocomplete support
- Active development and community

### Why validation_alias over env prefix?

```python
# Using validation_alias (chosen approach)
bootstrap_servers: str = Field(..., validation_alias='KAFKA_BOOTSTRAP_SERVERS')

# Alternative: env_prefix (not used)
class KafkaConfig(BaseSettings):
    model_config = {'env_prefix': 'KAFKA_'}
    bootstrap_servers: str  # Reads KAFKA_BOOTSTRAP_SERVERS
```

**Reason:** Explicit is better than implicit. validation_alias makes it crystal clear which environment variable maps to which field.

### Why properties for parsed values?

```python
@property
def bootstrap_servers_list(self) -> List[str]:
    return [s.strip() for s in self.bootstrap_servers.split(',')]
```

**Reason:**
- Keep raw value for debugging
- Parse on demand (lazy evaluation)
- Type-safe access without manual parsing
- Consistent API across all modules

### Why fail fast on startup?

Configuration errors could be handled gracefully (defaults, warnings), but we choose to fail fast:

**Rationale:**
- Catch errors before allocating resources
- Prevent partial deployments with wrong config
- Clear feedback in CI/CD pipelines
- No silent failures in production

## Comparison Matrix

| Feature | Old (os.getenv) | New (Pydantic) |
|---------|-----------------|----------------|
| Type safety | ❌ All strings | ✅ Typed fields |
| Validation | ❌ Manual | ✅ Automatic |
| Error messages | ❌ Cryptic | ✅ Clear |
| IDE support | ❌ No autocomplete | ✅ Full autocomplete |
| Testing | ⚠️ Mock os.environ | ✅ Mock settings |
| Documentation | ❌ Scattered | ✅ Centralized |
| Fail-fast | ❌ Runtime errors | ✅ Startup validation |
| Maintenance | ❌ Hard to change | ✅ Easy to extend |

## Conclusion

This configuration system provides:

- **Bulletproof validation:** Errors caught on startup, not runtime
- **Type safety:** No more string parsing bugs
- **Developer experience:** IDE autocomplete, clear documentation
- **Production ready:** Works with Docker, K8s, CI/CD
- **Maintainable:** Centralized, extensible, well-documented

The architecture is designed for both rapid prototyping (sensible defaults, easy local setup) and production deployment (validation, secrets management, environment-based config).
