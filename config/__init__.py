"""
Configuration package for Sentiment-Microstructure ABM.

Provides centralized, type-safe configuration management.

Usage:
    from config.settings import settings

    # Access configuration
    kafka_servers = settings.kafka.bootstrap_servers_list
    reddit_creds = (settings.reddit.client_id, settings.reddit.client_secret)
"""

from .settings import settings, Settings

__all__ = ['settings', 'Settings']
