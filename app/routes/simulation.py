"""
Simulation endpoints.
"""

from flask import Blueprint, request, jsonify, current_app
import logging
import traceback
from functools import wraps

from llm_token_analytics import TokenSimulator, SimulationConfig

simulation_bp = Blueprint('simulation', __name__)
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


@simulation_bp.route('/simulate', methods=['POST'])
@rate_limit("10 per hour")
def run_simulation():
    """
    Run pricing simulation.

    Request body:
    {
        "n_simulations": 100000,
        "mechanisms": ["per_token", "bundle", "hybrid", "cached"],
        "data_source": "collection_id" or "synthetic",
        "collection_id": "collection_20250101_120000" (if data_source is collection_id)
    }
    """
    try:

        data = request.json
        n_simulations = data.get('n_simulations', 100000)
        mechanisms = data.get('mechanisms', ['per_token', 'bundle', 'hybrid', 'cached'])
        data_source = data.get('data_source', 'synthetic')

        logger.info(f"Running simulation with {n_simulations} iterations")

        # Configure simulation
        config = SimulationConfig(
            n_simulations=n_simulations,
            mechanisms=mechanisms,
            use_empirical_data=(data_source != 'synthetic')
        )

        # Load empirical data if specified
        if data_source != 'synthetic' and 'collection_id' in data:
            collection = current_app.storage.get_result(data['collection_id'])
            if collection:
                config.empirical_data_path = collection['file']
            else:
                return jsonify({
                    'error': 'Collection not found',
                    'message': f"Collection ID {data['collection_id']} not found"
                }), 404

        # Run simulation
        simulator = TokenSimulator(config)
        results = simulator.run(progress_bar=False)

        # Store results
        simulation_id = current_app.storage.store_simulation_result(results, config)

        # Prepare response
        response_data = {
            'success': True,
            'simulation_id': simulation_id,
            'results': {}
        }

        for mechanism, stats in results.mechanism_results.items():
            response_data['results'][mechanism] = {
                'mean': stats['mean'],
                'median': stats['median'],
                'std': stats['std'],
                'cv': stats['cv'],
                'p95': stats['p95'],
                'p99': stats['p99'],
                'tail_ratio': stats.get('tail_ratio', 0)
            }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Simulation error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Simulation failed',
            'message': str(e)
        }), 500
