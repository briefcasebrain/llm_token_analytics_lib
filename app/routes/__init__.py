"""
Route blueprints registration.
"""

from .health import health_bp
from .collection import collection_bp
from .simulation import simulation_bp
from .analysis import analysis_bp
from .comparison import comparison_bp
from .visualization import visualization_bp


def register_blueprints(app):
    """Register all blueprints with the Flask app."""
    app.register_blueprint(health_bp)
    app.register_blueprint(collection_bp)
    app.register_blueprint(simulation_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(comparison_bp)
    app.register_blueprint(visualization_bp)
