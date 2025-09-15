"""
Health check and API documentation endpoints.
"""

from flask import Blueprint, jsonify, current_app
from datetime import datetime

health_bp = Blueprint('health', __name__)


@health_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': current_app.config.get('API_VERSION', '1.0.0')
    })


@health_bp.route('/', methods=['GET'])
def index():
    """API documentation."""
    return jsonify({
        'name': current_app.config.get('API_TITLE', 'LLM Token Analytics API'),
        'version': current_app.config.get('API_VERSION', '1.0.0'),
        'endpoints': {
            'GET /health': 'Health check',
            'POST /collect': 'Collect usage data from providers',
            'POST /simulate': 'Run pricing simulation',
            'POST /analyze': 'Analyze token distributions',
            'GET /results/<id>': 'Get simulation results',
            'POST /compare': 'Compare pricing mechanisms',
            'POST /optimize': 'Find optimal mechanism for user profile',
            'GET /visualize/<simulation_id>': 'Generate visualization for simulation results'
        }
    })
