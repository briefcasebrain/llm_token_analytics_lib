"""
REST API Server for LLM Token Analytics
========================================

Provides HTTP endpoints for running simulations and analysis.
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import json
import tempfile
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import traceback

# Import library modules
from llm_token_analytics import (
    UnifiedCollector,
    TokenSimulator,
    SimulationConfig,
    TokenAnalyzer,
    SimulationVisualizer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour", "10 per minute"]
)

# Storage for results (in production, use database)
results_cache = {}


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


@app.route('/', methods=['GET'])
def index():
    """API documentation."""
    return jsonify({
        'name': 'LLM Token Analytics API',
        'version': '1.0.0',
        'endpoints': {
            'GET /health': 'Health check',
            'POST /collect': 'Collect usage data from providers',
            'POST /simulate': 'Run pricing simulation',
            'POST /analyze': 'Analyze token distributions',
            'GET /results/<id>': 'Get simulation results',
            'POST /compare': 'Compare pricing mechanisms',
            'POST /optimize': 'Find optimal mechanism for user profile'
        }
    })


# ============================================================================
# DATA COLLECTION
# ============================================================================

@app.route('/collect', methods=['POST'])
@limiter.limit("5 per hour")
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
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.parquet',
            delete=False
        )
        collected_data.to_parquet(temp_file.name)
        
        # Store reference
        collection_id = f"collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results_cache[collection_id] = {
            'file': temp_file.name,
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        }
        
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


# ============================================================================
# SIMULATION
# ============================================================================

@app.route('/simulate', methods=['POST'])
@limiter.limit("10 per hour")
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
            collection = results_cache.get(data['collection_id'])
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
        simulation_id = f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results_cache[simulation_id] = {
            'results': results,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }
        
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


# ============================================================================
# ANALYSIS
# ============================================================================

@app.route('/analyze', methods=['POST'])
def analyze_distributions():
    """
    Analyze token distributions.
    
    Request body:
    {
        "collection_id": "collection_20250101_120000" or
        "data": {
            "input_tokens": [...],
            "output_tokens": [...]
        }
    }
    """
    try:
        data = request.json
        
        # Load data
        if 'collection_id' in data:
            collection = results_cache.get(data['collection_id'])
            if not collection:
                return jsonify({
                    'error': 'Collection not found'
                }), 404
            
            df = pd.read_parquet(collection['file'])
        elif 'data' in data:
            df = pd.DataFrame(data['data'])
        else:
            return jsonify({
                'error': 'No data provided'
            }), 400
        
        # Analyze
        analyzer = TokenAnalyzer(df)
        
        # Distribution analysis
        dist_results = analyzer.analyze_distributions()
        
        # Correlation analysis
        corr_results = analyzer.analyze_correlations()
        
        # Prepare response
        response = {
            'success': True,
            'distributions': {},
            'correlations': {}
        }
        
        # Format distribution results
        for token_type, results in dist_results.items():
            response['distributions'][token_type] = {
                'best_fit': results['distribution']['type'],
                'parameters': results['distribution']['params'],
                'validation': {
                    'ks_pvalue': results['validation']['ks_pvalue'],
                    'is_valid': results['validation']['is_valid']
                },
                'statistics': results['statistics']
            }
        
        # Format correlation results
        response['correlations'] = {
            'pearson': corr_results['correlations']['linear']['pearson']['value'],
            'spearman': corr_results['correlations']['linear']['spearman']['value'],
            'kendall': corr_results['correlations']['linear']['kendall']['value'],
            'tail_correlation': corr_results['correlations']['tail_correlation']
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({
            'error': 'Analysis failed',
            'message': str(e)
        }), 500


# ============================================================================
# RESULTS RETRIEVAL
# ============================================================================

@app.route('/results/<result_id>', methods=['GET'])
def get_results(result_id):
    """Get stored results by ID."""
    if result_id not in results_cache:
        return jsonify({
            'error': 'Results not found'
        }), 404
    
    result = results_cache[result_id]
    
    # Format based on result type
    if 'results' in result:  # Simulation results
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


# ============================================================================
# COMPARISON
# ============================================================================

@app.route('/compare', methods=['POST'])
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
        
        if not simulation_id or simulation_id not in results_cache:
            return jsonify({
                'error': 'Simulation not found'
            }), 404
        
        sim_results = results_cache[simulation_id]['results']
        
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


# ============================================================================
# OPTIMIZATION
# ============================================================================

@app.route('/optimize', methods=['POST'])
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
        
        if not simulation_id or simulation_id not in results_cache:
            return jsonify({
                'error': 'Simulation not found'
            }), 404
        
        sim_results = results_cache[simulation_id]['results']
        
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


# ============================================================================
# VISUALIZATION
# ============================================================================

@app.route('/visualize/<simulation_id>', methods=['GET'])
def visualize_results(simulation_id):
    """Generate visualization for simulation results."""
    try:
        if simulation_id not in results_cache:
            return jsonify({
                'error': 'Simulation not found'
            }), 404
        
        sim_results = results_cache[simulation_id]['results']
        
        # Create visualization
        visualizer = SimulationVisualizer(sim_results)
        fig = visualizer.create_comparison_plot()
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.html',
            delete=False
        )
        fig.write_html(temp_file.name)
        
        return send_file(
            temp_file.name,
            mimetype='text/html',
            as_attachment=True,
            download_name=f'simulation_{simulation_id}.html'
        )
        
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}")
        return jsonify({
            'error': 'Visualization failed',
            'message': str(e)
        }), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

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


# ============================================================================
# MAIN
# ============================================================================

def create_app(config=None):
    """Create and configure the Flask app."""
    if config:
        app.config.update(config)
    
    return app


if __name__ == '__main__':
    # Get configuration from environment
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    host = os.getenv('HOST', '0.0.0.0')
    
    logger.info(f"Starting API server on {host}:{port}")
    
    app.run(
        host=host,
        port=port,
        debug=debug
    )
