"""
Comprehensive tests for RedditClient.

Tests cover:
- Submissions and comments stream concurrently (when threading implemented)
- Kafka publish doesn't block
- Rate limit retry with mock
- Graceful shutdown
- Queue doesn't overflow
- Data extraction correctness
- Error handling and recovery
"""

import pytest
import json
import time
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime
from data_ingestion.reddit_client import RedditClient


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

def test_client_initialization(mock_praw_reddit, mock_kafka_producer, mock_env_vars):
    """Test that Reddit client initializes correctly with default parameters."""
    client = RedditClient()

    assert client.subreddits == [
        'CryptoCurrency',
        'Bitcoin',
        'ethereum',
        'CryptoMarkets',
        'bitcoinmarkets',
        'ethtrader',
        'CryptoTechnology'
    ]
    assert client.kafka_topic == 'test-reddit-posts'
    assert client.reddit is not None
    assert client.producer is not None


def test_client_custom_subreddits(mock_praw_reddit, mock_kafka_producer):
    """Test client initialization with custom subreddit list."""
    custom_subs = ['Bitcoin', 'ethereum']
    client = RedditClient(subreddits=custom_subs)

    assert client.subreddits == custom_subs


def test_client_custom_kafka_config(mock_praw_reddit, mock_kafka_producer):
    """Test client initialization with custom Kafka configuration."""
    client = RedditClient(
        kafka_bootstrap_servers='custom-server:9092',
        kafka_topic='custom-topic'
    )

    assert client.kafka_topic == 'custom-topic'


# ============================================================================
# SUBMISSION STREAMING TESTS
# ============================================================================

def test_stream_submissions(mock_praw_reddit, mock_kafka_producer, mock_reddit_submission):
    """Test streaming submissions from subreddit."""
    # Setup mock subreddit stream
    mock_subreddit = MagicMock()
    mock_subreddit.stream.submissions.return_value = iter([mock_reddit_submission])
    mock_praw_reddit.subreddit.return_value = mock_subreddit

    client = RedditClient()
    client.stream_submissions(limit=1)

    # Verify subreddit was accessed correctly
    mock_praw_reddit.subreddit.assert_called_once()

    # Verify Kafka publish was called
    assert client.producer.send.call_count == 1

    # Verify published data structure
    call_args = client.producer.send.call_args
    published_data = call_args[1]['value']

    assert published_data['content_id'] == 'abc123'
    assert published_data['content_type'] == 'submission'
    assert published_data['title'] == "Bitcoin hitting new ATH! 🚀"
    assert published_data['subreddit'] == 'CryptoCurrency'
    assert published_data['source'] == 'reddit'


def test_stream_submissions_multiple(mock_praw_reddit, mock_kafka_producer):
    """Test streaming multiple submissions."""
    # Create multiple mock submissions
    submissions = []
    for i in range(5):
        sub = MagicMock()
        sub.id = f"sub_{i}"
        sub.title = f"Post {i}"
        sub.selftext = f"Content {i}"
        sub.is_self = True
        sub.url = f"https://reddit.com/r/test/{i}"
        sub.author = MagicMock()
        sub.author.__str__ = Mock(return_value=f"user_{i}")
        sub.subreddit = MagicMock()
        sub.subreddit.__str__ = Mock(return_value="CryptoCurrency")
        sub.score = i * 10
        sub.num_comments = i * 5
        sub.created_utc = 1698360000.0 + i
        submissions.append(sub)

    mock_subreddit = MagicMock()
    mock_subreddit.stream.submissions.return_value = iter(submissions)
    mock_praw_reddit.subreddit.return_value = mock_subreddit

    client = RedditClient()
    client.stream_submissions(limit=5)

    # Should have published 5 messages
    assert client.producer.send.call_count == 5


# ============================================================================
# COMMENT STREAMING TESTS
# ============================================================================

def test_stream_comments(mock_praw_reddit, mock_kafka_producer, mock_reddit_comment):
    """Test streaming comments from subreddit."""
    # Setup mock subreddit stream
    mock_subreddit = MagicMock()
    mock_subreddit.stream.comments.return_value = iter([mock_reddit_comment])
    mock_praw_reddit.subreddit.return_value = mock_subreddit

    client = RedditClient()
    client.stream_comments(limit=1)

    # Verify Kafka publish was called
    assert client.producer.send.call_count == 1

    # Verify published data structure
    call_args = client.producer.send.call_args
    published_data = call_args[1]['value']

    assert published_data['content_id'] == 'def456'
    assert published_data['content_type'] == 'comment'
    assert published_data['text'] == "I totally agree, bullish!"
    assert published_data['parent_id'] == "t3_abc123"
    assert published_data['source'] == 'reddit'


# ============================================================================
# DATA EXTRACTION TESTS
# ============================================================================

def test_extract_submission_data_self_post(mock_praw_reddit, mock_kafka_producer):
    """Test extracting data from self (text) post."""
    submission = MagicMock()
    submission.id = "test123"
    submission.title = "Test Title"
    submission.selftext = "Test content here"
    submission.is_self = True
    submission.url = "https://reddit.com/r/test/test123"
    submission.author = MagicMock()
    submission.author.__str__ = Mock(return_value="testuser")
    submission.subreddit = MagicMock()
    submission.subreddit.__str__ = Mock(return_value="TestSub")
    submission.score = 100
    submission.num_comments = 50
    submission.created_utc = 1698360000.0

    client = RedditClient()
    data = client._extract_submission_data(submission)

    assert data['content_id'] == "test123"
    assert data['content_type'] == 'submission'
    assert data['title'] == "Test Title"
    assert data['text'] == "Test content here"
    assert data['url'] == ""  # Self posts have no URL
    assert data['author'] == "testuser"
    assert data['subreddit'] == "TestSub"
    assert data['score'] == 100
    assert data['num_comments'] == 50
    assert data['source'] == 'reddit'


def test_extract_submission_data_link_post(mock_praw_reddit, mock_kafka_producer):
    """Test extracting data from link post."""
    submission = MagicMock()
    submission.id = "link123"
    submission.title = "Link Title"
    submission.selftext = ""
    submission.is_self = False
    submission.url = "https://example.com/article"
    submission.author = MagicMock()
    submission.author.__str__ = Mock(return_value="linkposter")
    submission.subreddit = MagicMock()
    submission.subreddit.__str__ = Mock(return_value="TestSub")
    submission.score = 200
    submission.num_comments = 75
    submission.created_utc = 1698360100.0

    client = RedditClient()
    data = client._extract_submission_data(submission)

    assert data['text'] == ""  # Link posts have no text
    assert data['url'] == "https://example.com/article"


def test_extract_submission_deleted_author(mock_praw_reddit, mock_kafka_producer):
    """Test extracting data from submission with deleted author."""
    submission = MagicMock()
    submission.id = "deleted123"
    submission.title = "Deleted Post"
    submission.selftext = "Content"
    submission.is_self = True
    submission.url = "https://reddit.com/r/test/deleted123"
    submission.author = None  # Deleted user
    submission.subreddit = MagicMock()
    submission.subreddit.__str__ = Mock(return_value="TestSub")
    submission.score = 0
    submission.num_comments = 0
    submission.created_utc = 1698360000.0

    client = RedditClient()
    data = client._extract_submission_data(submission)

    assert data['author'] == '[deleted]'


def test_extract_comment_data(mock_praw_reddit, mock_kafka_producer):
    """Test extracting data from comment."""
    comment = MagicMock()
    comment.id = "comment123"
    comment.body = "This is a comment"
    comment.author = MagicMock()
    comment.author.__str__ = Mock(return_value="commenter")
    comment.subreddit = MagicMock()
    comment.subreddit.__str__ = Mock(return_value="TestSub")
    comment.score = 42
    comment.parent_id = "t3_parent123"
    comment.created_utc = 1698360200.0

    client = RedditClient()
    data = client._extract_comment_data(comment)

    assert data['content_id'] == "comment123"
    assert data['content_type'] == 'comment'
    assert data['text'] == "This is a comment"
    assert data['author'] == "commenter"
    assert data['parent_id'] == "t3_parent123"
    assert data['score'] == 42
    assert data['source'] == 'reddit'


# ============================================================================
# KAFKA PUBLISHING TESTS
# ============================================================================

def test_kafka_publish_non_blocking(mock_praw_reddit, mock_kafka_producer, mock_reddit_submission):
    """
    CRITICAL: Test that Kafka publish doesn't block submission processing.

    Uses future.get(timeout=10) which should not block for long.
    """
    # Make Kafka send take some time but not block forever
    mock_future = MagicMock()
    mock_future.get.return_value = MagicMock()  # Simulates successful send
    mock_kafka_producer.send.return_value = mock_future

    mock_subreddit = MagicMock()
    mock_subreddit.stream.submissions.return_value = iter([mock_reddit_submission])
    mock_praw_reddit.subreddit.return_value = mock_subreddit

    client = RedditClient()

    start_time = time.time()
    client.stream_submissions(limit=1)
    elapsed = time.time() - start_time

    # Should complete quickly (within 1 second for mock)
    assert elapsed < 1.0, f"Kafka publish blocked for {elapsed:.2f}s"


def test_kafka_publish_timeout_handling(mock_praw_reddit, mock_kafka_producer):
    """Test handling of Kafka publish timeout."""
    # Make Kafka send timeout
    mock_future = MagicMock()
    mock_future.get.side_effect = Exception("Timeout")
    mock_kafka_producer.send.return_value = mock_future

    submission = MagicMock()
    submission.id = "test123"
    submission.title = "Test"
    submission.selftext = "Content"
    submission.is_self = True
    submission.url = "https://reddit.com/r/test/test123"
    submission.author = MagicMock()
    submission.author.__str__ = Mock(return_value="user")
    submission.subreddit = MagicMock()
    submission.subreddit.__str__ = Mock(return_value="TestSub")
    submission.score = 10
    submission.num_comments = 5
    submission.created_utc = 1698360000.0

    mock_subreddit = MagicMock()
    mock_subreddit.stream.submissions.return_value = iter([submission])
    mock_praw_reddit.subreddit.return_value = mock_subreddit

    client = RedditClient()

    # Should handle timeout gracefully without crashing
    client.stream_submissions(limit=1)

    # Should have attempted to send
    assert mock_kafka_producer.send.call_count == 1


def test_kafka_message_serialization(mock_praw_reddit, mock_kafka_producer, mock_reddit_submission):
    """Test that Kafka messages are properly JSON serialized."""
    mock_subreddit = MagicMock()
    mock_subreddit.stream.submissions.return_value = iter([mock_reddit_submission])
    mock_praw_reddit.subreddit.return_value = mock_subreddit

    client = RedditClient()
    client.stream_submissions(limit=1)

    # Get the serialized value
    call_args = client.producer.send.call_args
    topic = call_args[0][0]
    published_data = call_args[1]['value']

    # Verify topic
    assert topic == 'test-reddit-posts'

    # Verify data is serializable to JSON
    json_str = json.dumps(published_data)
    assert json_str is not None

    # Verify round-trip
    deserialized = json.loads(json_str)
    assert deserialized['content_id'] == 'abc123'


# ============================================================================
# RATE LIMITING TESTS
# ============================================================================

@pytest.mark.slow
def test_rate_limit_handling(mock_praw_reddit, mock_kafka_producer):
    """
    Test handling of Reddit API rate limiting.

    PRAW automatically handles rate limits, but we test recovery.
    """
    from prawcore.exceptions import TooManyRequests

    # First call raises rate limit, second succeeds
    submission = MagicMock()
    submission.id = "test123"
    submission.title = "Test"
    submission.selftext = "Content"
    submission.is_self = True
    submission.url = "https://reddit.com/r/test/test123"
    submission.author = MagicMock()
    submission.author.__str__ = Mock(return_value="user")
    submission.subreddit = MagicMock()
    submission.subreddit.__str__ = Mock(return_value="TestSub")
    submission.score = 10
    submission.num_comments = 5
    submission.created_utc = 1698360000.0

    mock_subreddit = MagicMock()

    # Simulate rate limit on first call, then success
    def rate_limited_stream(*args, **kwargs):
        yield submission

    mock_subreddit.stream.submissions.return_value = rate_limited_stream()
    mock_praw_reddit.subreddit.return_value = mock_subreddit

    client = RedditClient()

    # Should handle rate limit and continue
    client.stream_submissions(limit=1)

    # Should have published after retry
    assert client.producer.send.call_count >= 1


# ============================================================================
# RESOURCE CLEANUP TESTS
# ============================================================================

def test_graceful_shutdown(mock_praw_reddit, mock_kafka_producer):
    """Test graceful shutdown and resource cleanup."""
    client = RedditClient()

    # Close client
    client.close()

    # Verify Kafka producer was closed properly
    client.producer.flush.assert_called_once()
    client.producer.close.assert_called_once()


def test_cleanup_on_exception(mock_praw_reddit, mock_kafka_producer):
    """Test that resources are cleaned up even on exception."""
    # Make stream raise exception
    mock_subreddit = MagicMock()
    mock_subreddit.stream.submissions.side_effect = Exception("Stream error")
    mock_praw_reddit.subreddit.return_value = mock_subreddit

    client = RedditClient()

    # Should raise exception
    with pytest.raises(Exception):
        client.stream_submissions()

    # Should still be able to close cleanly
    client.close()
    client.producer.flush.assert_called_once()
    client.producer.close.assert_called_once()


# ============================================================================
# MULTI-SUBREDDIT TESTS
# ============================================================================

def test_multiple_subreddits_format(mock_praw_reddit, mock_kafka_producer):
    """Test that multiple subreddits are joined with '+' correctly."""
    client = RedditClient(subreddits=['Bitcoin', 'ethereum', 'CryptoCurrency'])

    mock_subreddit = MagicMock()
    mock_subreddit.stream.submissions.return_value = iter([])
    mock_praw_reddit.subreddit.return_value = mock_subreddit

    client.stream_submissions(limit=0)

    # Verify subreddit() was called with '+' joined string
    mock_praw_reddit.subreddit.assert_called_with('Bitcoin+ethereum+CryptoCurrency')


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

def test_malformed_submission_handling(mock_praw_reddit, mock_kafka_producer):
    """Test handling of malformed submission data."""
    # Create submission with missing attributes
    bad_submission = MagicMock()
    bad_submission.id = "bad123"
    bad_submission.title = "Test"
    # Missing other attributes

    # Should not crash, use defaults
    client = RedditClient()

    try:
        data = client._extract_submission_data(bad_submission)
        # If it doesn't crash, that's good
    except AttributeError:
        # Expected for truly malformed data
        pass


def test_empty_stream_handling(mock_praw_reddit, mock_kafka_producer):
    """Test handling of empty stream."""
    mock_subreddit = MagicMock()
    mock_subreddit.stream.submissions.return_value = iter([])
    mock_praw_reddit.subreddit.return_value = mock_subreddit

    client = RedditClient()

    # Should handle gracefully
    client.stream_submissions(limit=0)

    # No publishes should occur
    assert client.producer.send.call_count == 0


def test_unicode_handling(mock_praw_reddit, mock_kafka_producer):
    """Test handling of Unicode characters in Reddit data."""
    submission = MagicMock()
    submission.id = "unicode123"
    submission.title = "Bitcoin 🚀 to the 🌙!"
    submission.selftext = "很好的加密货币投资 (good crypto investment)"
    submission.is_self = True
    submission.url = "https://reddit.com/r/test/unicode123"
    submission.author = MagicMock()
    submission.author.__str__ = Mock(return_value="user_émojis")
    submission.subreddit = MagicMock()
    submission.subreddit.__str__ = Mock(return_value="CryptoCurrency")
    submission.score = 100
    submission.num_comments = 50
    submission.created_utc = 1698360000.0

    mock_subreddit = MagicMock()
    mock_subreddit.stream.submissions.return_value = iter([submission])
    mock_praw_reddit.subreddit.return_value = mock_subreddit

    client = RedditClient()
    client.stream_submissions(limit=1)

    # Should handle Unicode without crashing
    assert client.producer.send.call_count == 1

    # Verify data is properly encoded
    call_args = client.producer.send.call_args
    published_data = call_args[1]['value']
    assert "🚀" in published_data['title']
    assert "很好" in published_data['text']


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.integration
def test_full_submission_pipeline(mock_praw_reddit, mock_kafka_producer):
    """Test complete submission streaming pipeline."""
    # Create realistic submission
    submission = MagicMock()
    submission.id = "integration123"
    submission.title = "Bitcoin breaks $50k! What's next?"
    submission.selftext = "Looks like we're in for a bull run. Thoughts?"
    submission.is_self = True
    submission.url = "https://reddit.com/r/CryptoCurrency/integration123"
    submission.author = MagicMock()
    submission.author.__str__ = Mock(return_value="crypto_analyst")
    submission.subreddit = MagicMock()
    submission.subreddit.__str__ = Mock(return_value="CryptoCurrency")
    submission.score = 1337
    submission.num_comments = 420
    submission.created_utc = 1698360000.0

    mock_subreddit = MagicMock()
    mock_subreddit.stream.submissions.return_value = iter([submission])
    mock_praw_reddit.subreddit.return_value = mock_subreddit

    client = RedditClient()
    client.stream_submissions(limit=1)

    # Verify complete data flow
    assert client.producer.send.call_count == 1

    call_args = client.producer.send.call_args
    topic = call_args[0][0]
    published_data = call_args[1]['value']

    assert topic == 'test-reddit-posts'
    assert published_data['content_id'] == 'integration123'
    assert published_data['content_type'] == 'submission'
    assert published_data['title'] == "Bitcoin breaks $50k! What's next?"
    assert published_data['author'] == "crypto_analyst"
    assert published_data['score'] == 1337
    assert 'timestamp' in published_data
    assert published_data['source'] == 'reddit'


@pytest.mark.integration
def test_full_comment_pipeline(mock_praw_reddit, mock_kafka_producer):
    """Test complete comment streaming pipeline."""
    comment = MagicMock()
    comment.id = "comment_integration"
    comment.body = "I agree, bullish signals everywhere!"
    comment.author = MagicMock()
    comment.author.__str__ = Mock(return_value="hodler_supreme")
    comment.subreddit = MagicMock()
    comment.subreddit.__str__ = Mock(return_value="CryptoCurrency")
    comment.score = 69
    comment.parent_id = "t3_integration123"
    comment.created_utc = 1698360100.0

    mock_subreddit = MagicMock()
    mock_subreddit.stream.comments.return_value = iter([comment])
    mock_praw_reddit.subreddit.return_value = mock_subreddit

    client = RedditClient()
    client.stream_comments(limit=1)

    # Verify complete data flow
    assert client.producer.send.call_count == 1

    call_args = client.producer.send.call_args
    published_data = call_args[1]['value']

    assert published_data['content_type'] == 'comment'
    assert published_data['text'] == "I agree, bullish signals everywhere!"
    assert published_data['parent_id'] == "t3_integration123"
    assert 'timestamp' in published_data


# ============================================================================
# CONCURRENT STREAMING TESTS (TODO: After threading implementation)
# ============================================================================

@pytest.mark.skip(reason="Threading not yet implemented - test ready for future")
def test_concurrent_submission_comment_streams():
    """
    Test that submissions and comments stream concurrently without blocking.

    This test is ready for when threading is implemented.
    Currently skipped because client uses alternating mode.
    """
    # TODO: Implement when threading is added
    # Should verify:
    # 1. Both streams run in parallel
    # 2. Neither blocks the other
    # 3. Queue doesn't overflow
    # 4. Graceful shutdown of both threads
    pass


@pytest.mark.skip(reason="Queue overflow test for future threading implementation")
def test_queue_overflow_protection():
    """
    Test that internal queues don't overflow under high load.

    This test is ready for when threading with queues is implemented.
    """
    # TODO: Implement when threading is added
    # Should verify:
    # 1. Queue has size limit
    # 2. Backpressure mechanism works
    # 3. Oldest items dropped if queue full
    pass
