"""
Pytest configuration and shared fixtures for sentiment-microstructure-abm tests.

Provides mock objects for external services (Kafka, Reddit API, Binance WebSocket)
and common test utilities.
"""

import pytest
import json
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Dict, List
import numpy as np
import torch


# ============================================================================
# KAFKA MOCKS
# ============================================================================

@pytest.fixture
def mock_kafka_producer():
    """Mock KafkaProducer for testing without actual Kafka connection."""
    with patch('kafka.KafkaProducer') as mock_producer_class:
        mock_producer = MagicMock()

        # Mock send() to return a successful future
        mock_future = MagicMock()
        mock_future.get.return_value = MagicMock()
        mock_producer.send.return_value = mock_future

        # Mock flush() and close()
        mock_producer.flush.return_value = None
        mock_producer.close.return_value = None

        mock_producer_class.return_value = mock_producer

        yield mock_producer


# ============================================================================
# REDDIT API MOCKS
# ============================================================================

@pytest.fixture
def mock_reddit_submission():
    """Mock Reddit submission object."""
    submission = MagicMock()
    submission.id = "abc123"
    submission.title = "Bitcoin hitting new ATH! 🚀"
    submission.selftext = "This is the content of the post"
    submission.is_self = True
    submission.url = "https://reddit.com/r/CryptoCurrency/abc123"
    submission.author.name = "crypto_bull"
    submission.author = MagicMock()
    submission.author.__str__ = Mock(return_value="crypto_bull")
    submission.subreddit = MagicMock()
    submission.subreddit.__str__ = Mock(return_value="CryptoCurrency")
    submission.score = 420
    submission.num_comments = 69
    submission.created_utc = 1698360000.0
    return submission


@pytest.fixture
def mock_reddit_comment():
    """Mock Reddit comment object."""
    comment = MagicMock()
    comment.id = "def456"
    comment.body = "I totally agree, bullish!"
    comment.author = MagicMock()
    comment.author.__str__ = Mock(return_value="hodler123")
    comment.subreddit = MagicMock()
    comment.subreddit.__str__ = Mock(return_value="CryptoCurrency")
    comment.score = 42
    comment.parent_id = "t3_abc123"
    comment.created_utc = 1698360100.0
    return comment


@pytest.fixture
def mock_praw_reddit():
    """Mock PRAW Reddit client."""
    with patch('praw.Reddit') as mock_reddit_class:
        mock_reddit = MagicMock()
        mock_reddit_class.return_value = mock_reddit
        yield mock_reddit


@pytest.fixture
def mock_reddit_stream(mock_reddit_submission, mock_reddit_comment):
    """Mock Reddit subreddit stream."""
    mock_subreddit = MagicMock()

    # Mock submission stream
    mock_submission_stream = MagicMock()
    mock_submission_stream.submissions.return_value = iter([mock_reddit_submission])

    # Mock comment stream
    mock_comment_stream = MagicMock()
    mock_comment_stream.comments.return_value = iter([mock_reddit_comment])

    mock_subreddit.stream = MagicMock()
    mock_subreddit.stream.submissions = Mock(return_value=iter([mock_reddit_submission]))
    mock_subreddit.stream.comments = Mock(return_value=iter([mock_reddit_comment]))

    return mock_subreddit


# ============================================================================
# BINANCE WEBSOCKET MOCKS
# ============================================================================

@pytest.fixture
def mock_binance_depth_message():
    """Mock Binance order book depth update message."""
    return {
        "lastUpdateId": 160,
        "bids": [
            ["43250.50", "2.5"],
            ["43250.00", "1.8"],
            ["43249.50", "3.2"],
            ["43249.00", "0.9"],
            ["43248.50", "1.5"]
        ],
        "asks": [
            ["43251.00", "1.2"],
            ["43251.50", "2.1"],
            ["43252.00", "1.7"],
            ["43252.50", "2.8"],
            ["43253.00", "1.1"]
        ]
    }


@pytest.fixture
def mock_binance_malformed_messages():
    """Mock malformed Binance messages for error handling tests."""
    return [
        # Missing bids/asks
        {"lastUpdateId": 160},
        # Invalid float values
        {"lastUpdateId": 161, "bids": [["invalid", "2.5"]], "asks": [["43251.00", "1.2"]]},
        # Missing price/quantity
        {"lastUpdateId": 162, "bids": [["43250.50"]], "asks": [["43251.00", "1.2"]]},
        # Empty arrays
        {"lastUpdateId": 163, "bids": [], "asks": []},
        # Invalid JSON structure
        "not a dict",
        # Null values
        {"lastUpdateId": 164, "bids": None, "asks": None}
    ]


@pytest.fixture
def mock_websocket_app():
    """Mock websocket.WebSocketApp."""
    with patch('websocket.WebSocketApp') as mock_ws_class:
        mock_ws = MagicMock()
        mock_ws_class.return_value = mock_ws
        yield mock_ws


# ============================================================================
# SENTIMENT ANALYZER MOCKS
# ============================================================================

@pytest.fixture
def mock_transformers():
    """Mock transformers library for sentiment analyzer tests."""
    with patch('transformers.AutoTokenizer') as mock_tokenizer_class, \
         patch('transformers.AutoModelForSequenceClassification') as mock_model_class:

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[101, 2023, 2003, 1037, 3231, 102]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1]])
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = MagicMock()

        # Create mock logits that will produce reasonable sentiment predictions
        # Shape: (1, 3) for [negative, neutral, positive]
        mock_logits = torch.tensor([[0.1, 0.2, 0.7]])  # Positive bias

        mock_output = MagicMock()
        mock_output.logits = mock_logits
        mock_model.return_value = mock_output

        # Mock to() method for device placement
        mock_model.to.return_value = mock_model

        # Mock modules() for dropout detection
        mock_dropout = MagicMock(spec=torch.nn.Dropout)
        mock_model.modules.return_value = [mock_dropout]

        mock_model_class.from_pretrained.return_value = mock_model

        yield {
            'tokenizer': mock_tokenizer,
            'model': mock_model,
            'tokenizer_class': mock_tokenizer_class,
            'model_class': mock_model_class
        }


@pytest.fixture
def sample_texts():
    """Sample texts for sentiment analysis testing."""
    return [
        "Bitcoin is going to the moon! 🚀",
        "Crypto crash incoming, sell everything!",
        "Ethereum update looks interesting",
        "Not sure about this market...",
        "HODL strong, we're still early!",
        "This is a scam, don't buy!",
        "Market is consolidating, waiting for breakout",
        "Just bought the dip, let's see what happens"
    ]


# ============================================================================
# PERFORMANCE TESTING UTILITIES
# ============================================================================

@pytest.fixture
def performance_timer():
    """Timer utility for performance benchmarking."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.end_time = time.perf_counter()

        @property
        def elapsed(self):
            if self.start_time is None or self.end_time is None:
                return None
            return self.end_time - self.start_time

    return Timer


@pytest.fixture
def memory_profiler():
    """Memory profiling utility for leak detection."""
    import gc
    import sys

    class MemoryProfiler:
        def __init__(self):
            self.baseline = None
            self.current = None

        def start(self):
            gc.collect()
            self.baseline = self._get_memory_usage()

        def measure(self):
            gc.collect()
            self.current = self._get_memory_usage()

        def _get_memory_usage(self):
            """Get current memory usage in MB."""
            try:
                import psutil
                process = psutil.Process()
                return process.memory_info().rss / 1024 / 1024
            except ImportError:
                # Fallback to sys.getsizeof if psutil not available
                return sys.getsizeof(gc.get_objects()) / 1024 / 1024

        @property
        def delta(self):
            if self.baseline is None or self.current is None:
                return None
            return self.current - self.baseline

    return MemoryProfiler


# ============================================================================
# ENVIRONMENT MOCKING
# ============================================================================

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    env_vars = {
        'REDDIT_CLIENT_ID': 'test_client_id',
        'REDDIT_CLIENT_SECRET': 'test_client_secret',
        'REDDIT_USER_AGENT': 'test_user_agent',
        'KAFKA_BOOTSTRAP_SERVERS': 'localhost:9092',
        'KAFKA_TOPIC_REDDIT': 'test-reddit-posts',
        'KAFKA_TOPIC_ORDERBOOKS': 'test-order-books',
        'BINANCE_WEBSOCKET_URL': 'wss://stream.binance.com:9443/ws'
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "memory: marks tests that check for memory leaks"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their names."""
    for item in items:
        # Mark performance tests
        if "performance" in item.nodeid or "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.performance)

        # Mark memory tests
        if "memory" in item.nodeid or "leak" in item.nodeid:
            item.add_marker(pytest.mark.memory)

        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
