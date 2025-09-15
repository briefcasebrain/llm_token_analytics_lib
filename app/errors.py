"""
Error handlers for the Flask application.
"""

from flask import jsonify


def register_error_handlers(app):
    """Register error handlers with the Flask app."""

    @app.errorhandler(404)
    def not_found(e):
        """404 error handler."""
        return jsonify({
            'error': 'Not found',
            'message': 'The requested resource was not found'
        }), 404

    @app.errorhandler(500)
    def internal_error(e):
        """500 error handler."""
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred'
        }), 500

    @app.errorhandler(429)
    def rate_limit_exceeded(e):
        """Rate limit error handler."""
        return jsonify({
            'error': 'Rate limit exceeded',
            'message': str(e.description)
        }), 429

    @app.errorhandler(400)
    def bad_request(e):
        """400 error handler."""
        return jsonify({
            'error': 'Bad request',
            'message': 'The request was malformed or invalid'
        }), 400
