"""
Comparison and optimization endpoints.
"""

from flask import Blueprint, request, jsonify, current_app
import logging

comparison_bp = Blueprint('comparison', __name__)
logger = logging.getLogger(__name__)


@comparison_bp.route('/compare', methods=['POST'])
def compare_mechanisms():
    """
    Compare pricing mechanisms.

    Request body:
    {
        "simulation_id": "simulation_20250101_120000"
    }
    """
    try:
        data = request.json
        simulation_id = data.get('simulation_id')

        if not simulation_id or not current_app.storage.result_exists(simulation_id):
            return jsonify({
                'error': 'Simulation not found'
            }), 404

        sim_results = current_app.storage.get_result(simulation_id)['results']

        # Create comparison
        from llm_token_analytics.analyzer import CostAnalyzer

        analyzer = CostAnalyzer(sim_results.mechanism_results)
        comparison_df = analyzer.compare_mechanisms()

        # Convert to JSON-friendly format
        comparison = comparison_df.to_dict('records')

        return jsonify({
            'success': True,
            'comparison': comparison
        })

    except Exception as e:
        logger.error(f"Comparison error: {str(e)}")
        return jsonify({
            'error': 'Comparison failed',
            'message': str(e)
        }), 500


@comparison_bp.route('/optimize', methods=['POST'])
def optimize_mechanism():
    """
    Find optimal pricing mechanism for user profile.

    Request body:
    {
        "simulation_id": "simulation_20250101_120000",
        "user_profile": {
            "risk_tolerance": "low",
            "usage_volume": 50000,
            "predictability_preference": 0.8,
            "budget_constraint": 100
        }
    }
    """
    try:
        data = request.json
        simulation_id = data.get('simulation_id')
        user_profile = data.get('user_profile', {})

        if not simulation_id or not current_app.storage.result_exists(simulation_id):
            return jsonify({
                'error': 'Simulation not found'
            }), 404

        sim_results = current_app.storage.get_result(simulation_id)['results']

        # Find optimal mechanism
        from llm_token_analytics.analyzer import CostAnalyzer

        analyzer = CostAnalyzer(sim_results.mechanism_results)
        optimal = analyzer.optimal_mechanism_selection(user_profile)

        # Get stats for optimal mechanism
        optimal_stats = sim_results.mechanism_results[optimal]

        return jsonify({
            'success': True,
            'recommended_mechanism': optimal,
            'expected_cost': optimal_stats['mean'],
            'cost_variance': optimal_stats['cv'],
            'p95_cost': optimal_stats['p95'],
            'user_profile': user_profile
        })

    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        return jsonify({
            'error': 'Optimization failed',
            'message': str(e)
        }), 500
