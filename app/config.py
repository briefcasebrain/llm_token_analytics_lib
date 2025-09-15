"""
Configuration settings for the Flask application.
"""

import os


class Config:
    """Base configuration class."""

    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')

    # Server settings
    # Note: 0.0.0.0 binding is intentional for server applications to accept external connections
    HOST = os.getenv('HOST', '0.0.0.0')  # nosec: intentional for server deployment
    PORT = int(os.getenv('PORT', 5000))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

    # Rate limiting settings
    RATELIMIT_STORAGE_URL = os.getenv('RATELIMIT_STORAGE_URL', 'memory://')

    # API settings
    API_VERSION = '1.0.0'
    API_TITLE = 'LLM Token Analytics API'


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
