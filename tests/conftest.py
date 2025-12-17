import pytest
import os
from unittest.mock import MagicMock, patch, AsyncMock
import torch

@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch, tmp_path):
    hf_cache = tmp_path / "hf_cache"
    hf_cache.mkdir()
    
    env_vars = {
        'REDDIT_CLIENT_ID': 'test_client_id',
        'REDDIT_CLIENT_SECRET': 'test_client_secret',
        'REDDIT_USER_AGENT': 'test_user_agent',
        'KAFKA_BOOTSTRAP_SERVERS': 'localhost:9092',
        'KAFKA_TOPIC_REDDIT': 'test-reddit-posts',
        'KAFKA_TOPIC_ORDERBOOKS': 'test-order-books',
        'BINANCE_WEBSOCKET_URL': 'wss://stream.binance.com:9443/ws',
        'HF_HOME': str(hf_cache),
        'TRANSFORMERS_CACHE': str(hf_cache)
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars

@pytest.fixture
def mock_kafka_producer():
    with patch('data_ingestion.reddit_client.KafkaProducer') as mock_class:
        mock = MagicMock()
        mock.send.return_value = MagicMock()
        mock_class.return_value = mock
        yield mock

@pytest.fixture
def mock_praw_reddit():
    with patch('data_ingestion.reddit_client.praw.Reddit') as mock_class:
        mock = MagicMock()
        mock_class.return_value = mock
        yield mock

@pytest.fixture
def mock_reddit_submission():
    s = MagicMock()
    s.id = "abc123"
    s.title = "Test Post"
    s.selftext = "Content"
    s.is_self = True
    s.url = "http://url"
    s.author.__str__.return_value = "user"
    s.subreddit.__str__.return_value = "sub"
    s.score = 10
    s.num_comments = 5
    s.created_utc = 1000.0
    return s

@pytest.fixture
def mock_reddit_comment():
    c = MagicMock()
    c.id = "def456"
    c.body = "Comment"
    c.author.__str__.return_value = "user"
    c.subreddit.__str__.return_value = "sub"
    c.score = 5
    c.parent_id = "t3_abc123"
    c.created_utc = 1001.0
    return c

@pytest.fixture
def mock_transformers():
    with patch('transformers.AutoTokenizer') as mock_tok, \
         patch('transformers.AutoModelForSequenceClassification') as mock_mod:
        tok = MagicMock()
        tok.return_value = {'input_ids': torch.tensor([[1]]), 'attention_mask': torch.tensor([[1]])}
        mock_tok.from_pretrained.return_value = tok
        
        mod = MagicMock()
        mod.return_value.logits = torch.tensor([[0.1, 0.2, 0.7]])
        mod.to.return_value = mod
        mod.modules.return_value = [MagicMock(spec=torch.nn.Dropout)]
        mock_mod.from_pretrained.return_value = mod
        
        yield {'tokenizer': tok, 'model': mod}

@pytest.fixture
def mock_websocket_app():
    with patch('websocket.WebSocketApp') as mock_ws:
        yield mock_ws

@pytest.fixture
def mock_binance_depth_message():
    return {
        "lastUpdateId": 160,
        "bids": [["100.0", "1.0"]],
        "asks": [["101.0", "1.0"]]
    }

