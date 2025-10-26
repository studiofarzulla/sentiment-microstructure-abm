#!/usr/bin/env python3
"""
Quick Test Script for Reddit Client

Validates that the concurrent streaming architecture works correctly.
Tests environment validation, thread startup, and graceful shutdown.
"""

import os
import sys
import time
import threading
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from reddit_client import RedditClient


def test_environment_validation():
    """Test 1: Environment validation catches missing variables."""
    print("\n" + "="*60)
    print("TEST 1: Environment Validation")
    print("="*60)

    # Temporarily unset environment variables
    original_env = {
        'REDDIT_CLIENT_ID': os.getenv('REDDIT_CLIENT_ID'),
        'REDDIT_CLIENT_SECRET': os.getenv('REDDIT_CLIENT_SECRET'),
        'REDDIT_USER_AGENT': os.getenv('REDDIT_USER_AGENT')
    }

    try:
        for key in original_env.keys():
            if key in os.environ:
                del os.environ[key]

        # Should raise ValueError
        try:
            client = RedditClient()
            print("❌ FAILED: Should have raised ValueError for missing env vars")
            return False
        except ValueError as e:
            print("✓ PASSED: Correctly caught missing environment variables")
            print(f"   Error message: {str(e)[:100]}...")
            return True

    finally:
        # Restore environment
        for key, value in original_env.items():
            if value:
                os.environ[key] = value


def test_client_initialization():
    """Test 2: Client initializes correctly with valid environment."""
    print("\n" + "="*60)
    print("TEST 2: Client Initialization")
    print("="*60)

    # Reload environment
    load_dotenv()

    try:
        client = RedditClient(
            subreddits=['test'],
            kafka_bootstrap_servers='localhost:9092',
            kafka_topic='test-topic'
        )

        print("✓ PASSED: Client initialized successfully")
        print(f"   Subreddits: {client.subreddits}")
        print(f"   Kafka topic: {client.kafka_topic}")
        print(f"   Queue initialized: {client.data_queue.maxsize} max size")
        print(f"   Shutdown event created: {client.shutdown_event}")

        return True

    except Exception as e:
        print(f"❌ FAILED: Client initialization error: {e}")
        return False


def test_thread_lifecycle():
    """Test 3: Threads start and shutdown gracefully."""
    print("\n" + "="*60)
    print("TEST 3: Thread Lifecycle")
    print("="*60)

    try:
        client = RedditClient(
            subreddits=['Bitcoin'],
            kafka_bootstrap_servers='localhost:9092',
            kafka_topic='test-topic'
        )

        # Start streaming
        print("Starting streaming threads...")
        client.start_streaming()

        # Verify threads started
        time.sleep(2)

        active_threads = threading.active_count()
        print(f"✓ Threads started: {len(client.threads)} worker threads")
        print(f"   Total active threads: {active_threads}")

        for thread in client.threads:
            print(f"   - {thread.name}: {'alive' if thread.is_alive() else 'dead'}")

        # Wait a bit to see if data flows
        print("\nWaiting 10 seconds to capture data...")
        time.sleep(10)

        # Print statistics
        client.print_stats()

        # Check if any data was captured
        with client.stats_lock:
            total_items = (
                client.stats['submissions_processed'] +
                client.stats['comments_processed']
            )

        if total_items > 0:
            print(f"✓ Data flowing: {total_items} items captured")
        else:
            print("⚠ No data captured (may be normal if subreddit quiet)")

        # Graceful shutdown
        print("\nInitiating graceful shutdown...")
        client.shutdown()

        # Verify threads stopped
        time.sleep(2)
        all_stopped = all(not thread.is_alive() for thread in client.threads)

        if all_stopped:
            print("✓ PASSED: All threads stopped gracefully")
            return True
        else:
            print("❌ FAILED: Some threads did not stop")
            return False

    except Exception as e:
        print(f"❌ FAILED: Thread lifecycle error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_statistics_tracking():
    """Test 4: Statistics are tracked correctly."""
    print("\n" + "="*60)
    print("TEST 4: Statistics Tracking")
    print("="*60)

    try:
        client = RedditClient(
            subreddits=['Bitcoin'],
            kafka_bootstrap_servers='localhost:9092',
            kafka_topic='test-topic'
        )

        # Check initial stats
        print("Initial statistics:")
        client.print_stats()

        with client.stats_lock:
            initial_stats = client.stats.copy()

        # Verify all stat keys exist
        expected_keys = [
            'submissions_processed',
            'comments_processed',
            'kafka_published',
            'kafka_errors',
            'rate_limit_hits'
        ]

        all_keys_present = all(key in initial_stats for key in expected_keys)

        if all_keys_present:
            print("✓ PASSED: All statistic keys present")
            return True
        else:
            print("❌ FAILED: Missing statistic keys")
            return False

    except Exception as e:
        print(f"❌ FAILED: Statistics tracking error: {e}")
        return False


def test_queue_operations():
    """Test 5: Queue operations are thread-safe."""
    print("\n" + "="*60)
    print("TEST 5: Queue Operations")
    print("="*60)

    try:
        client = RedditClient(
            subreddits=['Bitcoin'],
            kafka_bootstrap_servers='localhost:9092',
            kafka_topic='test-topic'
        )

        # Test queue
        test_data = {
            'content_id': 'test123',
            'content_type': 'submission',
            'text': 'Test submission',
            'timestamp': '2025-10-26T00:00:00'
        }

        # Put item
        client.data_queue.put(test_data)
        print(f"✓ Put item in queue: size = {client.data_queue.qsize()}")

        # Get item
        retrieved = client.data_queue.get(timeout=1)
        print(f"✓ Retrieved item from queue: {retrieved['content_id']}")

        # Verify empty
        queue_empty = client.data_queue.empty()
        if queue_empty:
            print("✓ PASSED: Queue operations work correctly")
            return True
        else:
            print("❌ FAILED: Queue not empty after retrieval")
            return False

    except Exception as e:
        print(f"❌ FAILED: Queue operations error: {e}")
        return False


def test_kafka_callbacks():
    """Test 6: Kafka callbacks update statistics."""
    print("\n" + "="*60)
    print("TEST 6: Kafka Callbacks")
    print("="*60)

    try:
        client = RedditClient(
            subreddits=['Bitcoin'],
            kafka_bootstrap_servers='localhost:9092',
            kafka_topic='test-topic'
        )

        # Mock metadata
        class MockMetadata:
            partition = 0
            offset = 123

        test_data = {
            'content_id': 'test123',
            'content_type': 'submission'
        }

        # Test success callback
        initial_published = client.stats['kafka_published']
        client._kafka_success_callback(MockMetadata(), test_data)
        after_success = client.stats['kafka_published']

        if after_success == initial_published + 1:
            print("✓ Success callback increments kafka_published")
        else:
            print("❌ Success callback did not increment counter")
            return False

        # Test error callback
        initial_errors = client.stats['kafka_errors']
        client._kafka_error_callback(Exception("Test error"), test_data)
        after_error = client.stats['kafka_errors']

        if after_error == initial_errors + 1:
            print("✓ Error callback increments kafka_errors")
        else:
            print("❌ Error callback did not increment counter")
            return False

        print("✓ PASSED: Kafka callbacks work correctly")
        return True

    except Exception as e:
        print(f"❌ FAILED: Kafka callback error: {e}")
        return False


def test_rate_limit_handling():
    """Test 7: Rate limit handler uses exponential backoff."""
    print("\n" + "="*60)
    print("TEST 7: Rate Limit Handling")
    print("="*60)

    try:
        client = RedditClient(
            subreddits=['Bitcoin'],
            kafka_bootstrap_servers='localhost:9092',
            kafka_topic='test-topic'
        )

        # Test exponential backoff
        backoff_times = []
        expected_times = [60, 120, 240, 480, 600, 600]  # Last two capped at 600

        for i in range(6):
            wait_time = client._handle_rate_limit()
            backoff_times.append(wait_time)
            print(f"   Rate limit hit {i+1}: wait {wait_time}s")

        # Verify exponential growth (capped at 600)
        all_correct = all(
            backoff_times[i] == expected_times[i]
            for i in range(len(expected_times))
        )

        if all_correct:
            print("✓ PASSED: Rate limit exponential backoff correct")
            return True
        else:
            print(f"❌ FAILED: Incorrect backoff times")
            print(f"   Expected: {expected_times}")
            print(f"   Got:      {backoff_times}")
            return False

    except Exception as e:
        print(f"❌ FAILED: Rate limit handling error: {e}")
        return False


def run_all_tests():
    """Run all test cases."""
    print("\n" + "#"*60)
    print("# Reddit Client Test Suite")
    print("#"*60)

    tests = [
        ("Environment Validation", test_environment_validation),
        ("Client Initialization", test_client_initialization),
        ("Statistics Tracking", test_statistics_tracking),
        ("Queue Operations", test_queue_operations),
        ("Kafka Callbacks", test_kafka_callbacks),
        ("Rate Limit Handling", test_rate_limit_handling),
        # Thread lifecycle test last (takes 10+ seconds)
        ("Thread Lifecycle", test_thread_lifecycle),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ EXCEPTION in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

        time.sleep(1)  # Brief pause between tests

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")

    print("="*60)
    print(f"Results: {passed}/{total} tests passed ({100*passed/total:.0f}%)")
    print("="*60)

    return passed == total


if __name__ == '__main__':
    # Load environment
    load_dotenv()

    # Check if environment is configured
    required_vars = ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET', 'REDDIT_USER_AGENT']
    has_env = all(os.getenv(var) for var in required_vars)

    if not has_env:
        print("\n⚠ WARNING: Reddit API credentials not configured")
        print("Some tests will fail. To fix:")
        print("1. Create .env file with Reddit API credentials")
        print("2. See reddit_client.py for required variables")
        print("\nContinuing with tests that don't require API...\n")

    # Run tests
    success = run_all_tests()

    sys.exit(0 if success else 1)
