-- TimescaleDB initialization script

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Order book snapshots
CREATE TABLE order_book_snapshots (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    best_bid DOUBLE PRECISION,
    best_ask DOUBLE PRECISION,
    mid_price DOUBLE PRECISION,
    spread DOUBLE PRECISION,
    bid_volume DOUBLE PRECISION,
    ask_volume DOUBLE PRECISION,
    imbalance DOUBLE PRECISION
);

SELECT create_hypertable('order_book_snapshots', 'timestamp');

-- Sentiment ticks
CREATE TABLE sentiment_ticks (
    timestamp TIMESTAMPTZ NOT NULL,
    source TEXT NOT NULL,  -- 'reddit', 'twitter', etc.
    content_id TEXT,
    sentiment_score DOUBLE PRECISION,  -- [-1, 1]
    sigma_epistemic DOUBLE PRECISION,
    sigma_aleatoric DOUBLE PRECISION,
    sentiment_ewma DOUBLE PRECISION
);

SELECT create_hypertable('sentiment_ticks', 'timestamp');

-- DFM factors
CREATE TABLE dfm_factors (
    timestamp TIMESTAMPTZ NOT NULL,
    factor_1 DOUBLE PRECISION,  -- market mood
    factor_2 DOUBLE PRECISION,  -- volatility regime
    factor_3 DOUBLE PRECISION,  -- liquidity stress
    explained_variance DOUBLE PRECISION
);

SELECT create_hypertable('dfm_factors', 'timestamp');

-- Simulated trades
CREATE TABLE simulated_trades (
    timestamp TIMESTAMPTZ NOT NULL,
    agent_id INTEGER,
    agent_type TEXT,  -- 'MM', 'IT', 'NT', 'AR'
    side TEXT,  -- 'buy', 'sell'
    price DOUBLE PRECISION,
    quantity DOUBLE PRECISION,
    order_type TEXT  -- 'limit', 'market'
);

SELECT create_hypertable('simulated_trades', 'timestamp');

-- Agent PnL tracking
CREATE TABLE agent_pnl (
    timestamp TIMESTAMPTZ NOT NULL,
    agent_id INTEGER,
    agent_type TEXT,
    cash DOUBLE PRECISION,
    inventory DOUBLE PRECISION,
    mark_to_market_pnl DOUBLE PRECISION
);

SELECT create_hypertable('agent_pnl', 'timestamp');

-- Indexes for common queries
CREATE INDEX idx_sentiment_source ON sentiment_ticks (source, timestamp DESC);
CREATE INDEX idx_trades_agent ON simulated_trades (agent_id, timestamp DESC);
CREATE INDEX idx_pnl_agent ON agent_pnl (agent_id, timestamp DESC);

-- Continuous aggregates for dashboard
CREATE MATERIALIZED VIEW order_book_1min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', timestamp) AS bucket,
    symbol,
    AVG(mid_price) as avg_price,
    AVG(spread) as avg_spread,
    AVG(imbalance) as avg_imbalance
FROM order_book_snapshots
GROUP BY bucket, symbol;

CREATE MATERIALIZED VIEW sentiment_1min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', timestamp) AS bucket,
    source,
    AVG(sentiment_score) as avg_sentiment,
    AVG(sigma_epistemic) as avg_uncertainty,
    COUNT(*) as num_posts
FROM sentiment_ticks
GROUP BY bucket, source;

-- Refresh policies
SELECT add_continuous_aggregate_policy('order_book_1min',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute');

SELECT add_continuous_aggregate_policy('sentiment_1min',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute');
