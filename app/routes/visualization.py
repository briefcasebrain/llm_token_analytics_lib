"""
Visualization endpoints.
"""

from flask import Blueprint, send_file, jsonify, current_app
import tempfile
import logging

from llm_token_analytics import SimulationVisualizer

visualization_bp = Blueprint('visualization', __name__)
logger = logging.getLogger(__name__)


@visualization_bp.route('/visualize/<simulation_id>', methods=['GET'])
def visualize_results(simulation_id):
    """Generate visualization for simulation results."""
    try:
        if not current_app.storage.result_exists(simulation_id):
            return jsonify({
                'error': 'Simulation not found'
            }), 404

        sim_results = current_app.storage.get_result(simulation_id)['results']

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
