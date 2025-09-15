"""
Data collection endpoints.
"""

from flask import Blueprint, request, jsonify, current_app
import logging
from functools import wraps

from llm_token_analytics import UnifiedCollector

collection_bp = Blueprint('collection', __name__)
logger = logging.getLogger(__name__)


def rate_limit(limit_string):
    """Decorator to apply rate limiting to routes."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if hasattr(current_app, 'limiter'):
                current_app.limiter.limit(limit_string).test()
            return f(*args, **kwargs)
        return decorated_function
    return decorator


@collection_bp.route('/collect', methods=['POST'])
@rate_limit("5 per hour")
def collect_data():
    """
    Collect usage data from LLM providers.

    Request body:
    {
        "providers": ["openai", "anthropic", "google"],
        "start_date": "2025-01-01",
        "end_date": "2025-01-31"
    }
    """
    try:

        data = request.json
        providers = data.get('providers', ['openai', 'anthropic', 'google'])
        start_date = data.get('start_date')
        end_date = data.get('end_date')

        logger.info(f"Collecting data from {providers}")

        # Initialize collector
        collector = UnifiedCollector(providers)

        # Collect data
        collected_data = collector.collect_all()

        if collected_data.empty:
            return jsonify({
                'error': 'No data collected',
                'message': 'Check API credentials and date range'
            }), 400

        # Get statistics
        stats = collector.get_summary_statistics(collected_data)

        # Store results
        collection_id = current_app.storage.store_collection_result(
            collected_data, stats
        )

        return jsonify({
            'success': True,
            'collection_id': collection_id,
            'records': len(collected_data),
            'statistics': stats
        })

    except Exception as e:
        logger.error(f"Collection error: {str(e)}")
        return jsonify({
            'error': 'Collection failed',
            'message': str(e)
        }), 500


@collection_bp.route('/results/<result_id>', methods=['GET'])
def get_results(result_id):
    """Get stored results by ID."""
    if not current_app.storage.result_exists(result_id):
        return jsonify({
            'error': 'Results not found'
        }), 404

    result = current_app.storage.get_result(result_id)

    # Format based on result type
    if result.get('type') == 'simulation':
        sim_results = result['results']
        response = {
            'id': result_id,
            'type': 'simulation',
            'timestamp': result['timestamp'],
            'results': {}
        }

        for mechanism, stats in sim_results.mechanism_results.items():
            response['results'][mechanism] = {
                'mean': stats['mean'],
                'median': stats['median'],
                'p95': stats['p95'],
                'cv': stats['cv']
            }
    else:  # Collection results
        response = {
            'id': result_id,
            'type': 'collection',
            'timestamp': result['timestamp'],
            'statistics': result.get('stats', {})
        }

    return jsonify(response)
