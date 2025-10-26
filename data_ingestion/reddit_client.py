"""
Reddit Sentiment Data Ingestion - Fixed Concurrent Version

Streams posts and comments from crypto-related subreddits in REAL-TIME.
Uses threading to capture submissions and comments SIMULTANEOUSLY.
Publishes to Kafka topic for sentiment analysis pipeline.

CRITICAL FIXES:
- Concurrent submission + comment streaming (no more data loss!)
- Non-blocking Kafka publishing with callback-based error handling
- Rate limit handling with exponential backoff
- Environment validation with clear error messages
- Graceful shutdown with proper thread cleanup
"""

import praw
import json
import time
from kafka import KafkaProducer
from kafka.errors import KafkaError
from datetime import datetime
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
import logging
import threading
from queue import Queue, Empty
import signal
import sys
from prawcore.exceptions import ResponseException, RequestException

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RedditClient:
    """
    Reddit API client for crypto sentiment data collection.

    Uses threading to stream submissions and comments concurrently.
    Non-blocking Kafka publishing via callback pattern.
    """

    def __init__(
        self,
        subreddits: List[str] = None,
        kafka_bootstrap_servers: str = None,
        kafka_topic: str = None
    ):
        """
        Initialize Reddit client with environment validation.

        Args:
            subreddits: List of subreddit names (default: crypto-related)
            kafka_bootstrap_servers: Kafka connection string
            kafka_topic: Kafka topic for reddit posts

        Raises:
            ValueError: If required environment variables are missing
        """
        # Validate environment variables
        self._validate_environment()

        # Reddit API credentials
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT')
        )

        # Subreddits to monitor
        self.subreddits = subreddits or [
            'CryptoCurrency',
            'Bitcoin',
            'ethereum',
            'CryptoMarkets',
            'bitcoinmarkets',
            'ethtrader',
            'CryptoTechnology'
        ]

        # Kafka producer configuration
        kafka_servers = kafka_bootstrap_servers or os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.kafka_topic = kafka_topic or os.getenv('KAFKA_TOPIC_REDDIT', 'reddit-posts')

        # Non-blocking Kafka producer with compression
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_servers.split(','),
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            compression_type='gzip',
            acks=1,  # Leader acknowledgment (balance between speed and reliability)
            retries=3,
            max_in_flight_requests_per_connection=5,
            linger_ms=10  # Small batching for efficiency
        )

        # Thread-safe queue for data flow
        self.data_queue = Queue(maxsize=1000)  # Backpressure if queue fills

        # Shutdown coordination
        self.shutdown_event = threading.Event()
        self.threads: List[threading.Thread] = []

        # Statistics tracking
        self.stats = {
            'submissions_processed': 0,
            'comments_processed': 0,
            'kafka_published': 0,
            'kafka_errors': 0,
            'rate_limit_hits': 0
        }
        self.stats_lock = threading.Lock()

        logger.info(f"Initialized Reddit client for subreddits: {self.subreddits}")
        logger.info(f"Publishing to Kafka topic: {self.kafka_topic} @ {kafka_servers}")

    def _validate_environment(self):
        """Validate required environment variables exist."""
        required_vars = [
            'REDDIT_CLIENT_ID',
            'REDDIT_CLIENT_SECRET',
            'REDDIT_USER_AGENT'
        ]

        missing = [var for var in required_vars if not os.getenv(var)]

        if missing:
            error_msg = (
                f"Missing required environment variables: {', '.join(missing)}\n"
                "Please create a .env file with:\n"
                "  REDDIT_CLIENT_ID=your_client_id\n"
                "  REDDIT_CLIENT_SECRET=your_client_secret\n"
                "  REDDIT_USER_AGENT=your_user_agent\n"
                "  KAFKA_BOOTSTRAP_SERVERS=localhost:9092 (optional)\n"
                "  KAFKA_TOPIC_REDDIT=reddit-posts (optional)"
            )
            raise ValueError(error_msg)

    def _submission_worker(self):
        """
        Worker thread: Stream submissions continuously.
        Pushes data to shared queue for Kafka publisher.
        """
        subreddit_str = '+'.join(self.subreddits)
        subreddit = self.reddit.subreddit(subreddit_str)

        logger.info(f"[SUBMISSION WORKER] Started streaming from r/{subreddit_str}")

        while not self.shutdown_event.is_set():
            try:
                for submission in subreddit.stream.submissions(skip_existing=True):
                    if self.shutdown_event.is_set():
                        break

                    # Extract data
                    post_data = self._extract_submission_data(submission)

                    # Push to queue (with timeout to allow shutdown checks)
                    try:
                        self.data_queue.put(post_data, timeout=1)

                        with self.stats_lock:
                            self.stats['submissions_processed'] += 1

                        logger.debug(f"[SUBMISSION] Queued: {submission.id} - {submission.title[:50]}")

                    except Exception as e:
                        logger.warning(f"[SUBMISSION] Queue full or error: {e}")

            except ResponseException as e:
                # Rate limit or API error
                if e.response.status_code == 429:
                    wait_time = self._handle_rate_limit()
                    logger.warning(f"[SUBMISSION] Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                else:
                    logger.error(f"[SUBMISSION] API error: {e}")
                    time.sleep(30)

            except RequestException as e:
                logger.error(f"[SUBMISSION] Network error: {e}")
                time.sleep(30)

            except Exception as e:
                logger.error(f"[SUBMISSION] Unexpected error: {e}", exc_info=True)
                time.sleep(30)

        logger.info("[SUBMISSION WORKER] Shutdown complete")

    def _comment_worker(self):
        """
        Worker thread: Stream comments continuously.
        Pushes data to shared queue for Kafka publisher.
        """
        subreddit_str = '+'.join(self.subreddits)
        subreddit = self.reddit.subreddit(subreddit_str)

        logger.info(f"[COMMENT WORKER] Started streaming from r/{subreddit_str}")

        while not self.shutdown_event.is_set():
            try:
                for comment in subreddit.stream.comments(skip_existing=True):
                    if self.shutdown_event.is_set():
                        break

                    # Extract data
                    comment_data = self._extract_comment_data(comment)

                    # Push to queue (with timeout to allow shutdown checks)
                    try:
                        self.data_queue.put(comment_data, timeout=1)

                        with self.stats_lock:
                            self.stats['comments_processed'] += 1

                        logger.debug(f"[COMMENT] Queued: {comment.id}")

                    except Exception as e:
                        logger.warning(f"[COMMENT] Queue full or error: {e}")

            except ResponseException as e:
                # Rate limit or API error
                if e.response.status_code == 429:
                    wait_time = self._handle_rate_limit()
                    logger.warning(f"[COMMENT] Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                else:
                    logger.error(f"[COMMENT] API error: {e}")
                    time.sleep(30)

            except RequestException as e:
                logger.error(f"[COMMENT] Network error: {e}")
                time.sleep(30)

            except Exception as e:
                logger.error(f"[COMMENT] Unexpected error: {e}", exc_info=True)
                time.sleep(30)

        logger.info("[COMMENT WORKER] Shutdown complete")

    def _kafka_publisher(self):
        """
        Worker thread: Publish data from queue to Kafka.
        Uses non-blocking send with callbacks for error handling.
        """
        logger.info("[KAFKA PUBLISHER] Started")

        while not self.shutdown_event.is_set():
            try:
                # Get data from queue (timeout allows shutdown checks)
                try:
                    data = self.data_queue.get(timeout=1)
                except Empty:
                    continue

                # Non-blocking Kafka send with callbacks
                future = self.producer.send(self.kafka_topic, value=data)

                # Add callbacks (executed in background)
                future.add_callback(self._kafka_success_callback, data)
                future.add_errback(self._kafka_error_callback, data)

            except Exception as e:
                logger.error(f"[KAFKA PUBLISHER] Unexpected error: {e}", exc_info=True)
                time.sleep(1)

        # Flush remaining messages on shutdown
        logger.info("[KAFKA PUBLISHER] Flushing remaining messages...")
        self.producer.flush(timeout=10)
        logger.info("[KAFKA PUBLISHER] Shutdown complete")

    def _kafka_success_callback(self, metadata, data: Dict):
        """Callback executed when Kafka publish succeeds."""
        with self.stats_lock:
            self.stats['kafka_published'] += 1

        logger.debug(
            f"[KAFKA SUCCESS] {data['content_type']} {data['content_id']} "
            f"-> partition {metadata.partition} offset {metadata.offset}"
        )

    def _kafka_error_callback(self, exc, data: Dict):
        """Callback executed when Kafka publish fails."""
        with self.stats_lock:
            self.stats['kafka_errors'] += 1

        logger.error(
            f"[KAFKA ERROR] Failed to publish {data['content_type']} {data['content_id']}: {exc}"
        )

    def _handle_rate_limit(self) -> int:
        """
        Handle Reddit API rate limiting with exponential backoff.

        Returns:
            Wait time in seconds
        """
        with self.stats_lock:
            self.stats['rate_limit_hits'] += 1
            hits = self.stats['rate_limit_hits']

        # Exponential backoff: 60s, 120s, 240s, max 600s (10 min)
        wait_time = min(60 * (2 ** (hits - 1)), 600)
        return wait_time

    def _extract_submission_data(self, submission) -> Dict:
        """Extract relevant fields from Reddit submission."""
        return {
            'content_id': submission.id,
            'content_type': 'submission',
            'subreddit': str(submission.subreddit),
            'title': submission.title,
            'text': submission.selftext if submission.is_self else '',
            'url': submission.url if not submission.is_self else '',
            'author': str(submission.author) if submission.author else '[deleted]',
            'score': submission.score,
            'num_comments': submission.num_comments,
            'created_utc': submission.created_utc,
            'timestamp': datetime.utcnow().isoformat(),
            'source': 'reddit'
        }

    def _extract_comment_data(self, comment) -> Dict:
        """Extract relevant fields from Reddit comment."""
        return {
            'content_id': comment.id,
            'content_type': 'comment',
            'subreddit': str(comment.subreddit),
            'text': comment.body,
            'author': str(comment.author) if comment.author else '[deleted]',
            'score': comment.score,
            'parent_id': comment.parent_id,
            'created_utc': comment.created_utc,
            'timestamp': datetime.utcnow().isoformat(),
            'source': 'reddit'
        }

    def start_streaming(self):
        """
        Start concurrent streaming threads.
        Launches 3 threads: submissions, comments, kafka publisher.
        """
        logger.info("Starting concurrent streaming threads...")

        # Create worker threads
        submission_thread = threading.Thread(
            target=self._submission_worker,
            name="SubmissionWorker",
            daemon=True
        )

        comment_thread = threading.Thread(
            target=self._comment_worker,
            name="CommentWorker",
            daemon=True
        )

        kafka_thread = threading.Thread(
            target=self._kafka_publisher,
            name="KafkaPublisher",
            daemon=True
        )

        # Start all threads
        submission_thread.start()
        comment_thread.start()
        kafka_thread.start()

        self.threads = [submission_thread, comment_thread, kafka_thread]

        logger.info("All threads started successfully")
        logger.info("Streaming submissions and comments concurrently...")

    def print_stats(self):
        """Print current statistics."""
        with self.stats_lock:
            stats_copy = self.stats.copy()

        logger.info("=" * 60)
        logger.info("STREAMING STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Submissions processed: {stats_copy['submissions_processed']}")
        logger.info(f"Comments processed:    {stats_copy['comments_processed']}")
        logger.info(f"Total items:           {stats_copy['submissions_processed'] + stats_copy['comments_processed']}")
        logger.info(f"Kafka published:       {stats_copy['kafka_published']}")
        logger.info(f"Kafka errors:          {stats_copy['kafka_errors']}")
        logger.info(f"Rate limit hits:       {stats_copy['rate_limit_hits']}")
        logger.info(f"Queue size:            {self.data_queue.qsize()}")
        logger.info("=" * 60)

    def wait_for_shutdown(self):
        """
        Wait for shutdown signal (Ctrl+C).
        Joins all threads gracefully.
        """
        # Setup signal handler for graceful shutdown
        def signal_handler(sig, frame):
            logger.info("\nReceived interrupt signal (Ctrl+C)")
            self.shutdown()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Stats printer thread (every 60 seconds)
        stats_interval = 60
        last_stats_time = time.time()

        try:
            while not self.shutdown_event.is_set():
                time.sleep(1)

                # Print stats periodically
                if time.time() - last_stats_time >= stats_interval:
                    self.print_stats()
                    last_stats_time = time.time()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt detected")
            self.shutdown()

    def shutdown(self):
        """
        Graceful shutdown of all threads.
        Ensures all data is flushed before exit.
        """
        if self.shutdown_event.is_set():
            return  # Already shutting down

        logger.info("Initiating graceful shutdown...")
        self.shutdown_event.set()

        # Wait for threads to finish
        for thread in self.threads:
            logger.info(f"Waiting for {thread.name} to finish...")
            thread.join(timeout=10)
            if thread.is_alive():
                logger.warning(f"{thread.name} did not finish in time")

        # Final stats
        self.print_stats()

        # Close producer
        logger.info("Closing Kafka producer...")
        self.producer.close(timeout=10)

        logger.info("Shutdown complete")


def main():
    """Run Reddit client as standalone service."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Reddit sentiment data ingestion (concurrent streaming)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stream submissions and comments concurrently (recommended)
  python reddit_client.py

  # Custom subreddit list
  python reddit_client.py --subreddits Bitcoin ethereum CryptoCurrency

  # Custom Kafka configuration
  python reddit_client.py --kafka-servers kafka1:9092,kafka2:9092 --topic my-reddit-topic

Environment Variables (.env file):
  REDDIT_CLIENT_ID=your_client_id           (required)
  REDDIT_CLIENT_SECRET=your_client_secret   (required)
  REDDIT_USER_AGENT=your_user_agent         (required)
  KAFKA_BOOTSTRAP_SERVERS=localhost:9092    (optional)
  KAFKA_TOPIC_REDDIT=reddit-posts           (optional)
        """
    )

    parser.add_argument(
        '--subreddits',
        nargs='+',
        default=None,
        help='Subreddits to monitor (default: crypto-related)'
    )

    parser.add_argument(
        '--kafka-servers',
        type=str,
        default=None,
        help='Kafka bootstrap servers (default: from .env or localhost:9092)'
    )

    parser.add_argument(
        '--topic',
        type=str,
        default=None,
        help='Kafka topic name (default: from .env or reddit-posts)'
    )

    parser.add_argument(
        '--stats-interval',
        type=int,
        default=60,
        help='Stats printing interval in seconds (default: 60)'
    )

    args = parser.parse_args()

    # Create client
    try:
        client = RedditClient(
            subreddits=args.subreddits,
            kafka_bootstrap_servers=args.kafka_servers,
            kafka_topic=args.topic
        )
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # Start streaming
    try:
        client.start_streaming()
        logger.info("Press Ctrl+C to stop streaming")
        client.wait_for_shutdown()

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        client.shutdown()
        sys.exit(1)


if __name__ == '__main__':
    main()
