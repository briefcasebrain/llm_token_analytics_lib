"""
Flask Application Factory
========================

Creates and configures the Flask application with all necessary components.
"""

from flask import Flask
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging

from .config import Config
from .storage import ResultsStorage
from .errors import register_error_handlers
from .routes import register_blueprints


def create_app(config=None):
    """Create and configure the Flask application."""

    # Initialize Flask app
    app = Flask(__name__)

    # Load configuration
    if config:
        app.config.update(config)
    else:
        app.config.from_object(Config)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize extensions
    CORS(app)

    # Rate limiting
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["100 per hour", "10 per minute"]
    )

    # Store limiter in app for blueprint access
    app.limiter = limiter

    # Initialize storage
    app.storage = ResultsStorage()

    # Register error handlers
    register_error_handlers(app)

    # Register blueprints
    register_blueprints(app)

    return app
