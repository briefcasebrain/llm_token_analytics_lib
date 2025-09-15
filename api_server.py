"""
REST API Server for LLM Token Analytics
========================================

Provides HTTP endpoints for running simulations and analysis.
"""

import os
import logging
from app import create_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the API server."""
    # Get configuration from environment
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    host = os.getenv('HOST', '0.0.0.0')

    # Create Flask app
    app = create_app()

    logger.info(f"Starting API server on {host}:{port}")

    app.run(
        host=host,
        port=port,
        debug=debug
    )


if __name__ == '__main__':
    main()

