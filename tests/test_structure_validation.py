"""
Structure validation tests - verify the refactored API structure is correct
"""

import os
import ast


class TestAPIStructure:
    """Test the overall API structure and module organization."""

    def test_app_directory_structure(self):
        """Test that all required files and directories exist."""
        expected_files = [
            'app/__init__.py',
            'app/config.py',
            'app/storage.py',
            'app/errors.py',
            'app/routes/__init__.py',
            'app/routes/health.py',
            'app/routes/collection.py',
            'app/routes/simulation.py',
            'app/routes/analysis.py',
            'app/routes/comparison.py',
            'app/routes/visualization.py'
        ]

        for file_path in expected_files:
            assert os.path.exists(file_path), f"Missing file: {file_path}"

    def test_route_files_have_blueprints(self):
        """Test that each route file defines a blueprint."""
        route_files = [
            'app/routes/health.py',
            'app/routes/collection.py',
            'app/routes/simulation.py',
            'app/routes/analysis.py',
            'app/routes/comparison.py',
            'app/routes/visualization.py'
        ]

        expected_blueprints = [
            'health_bp',
            'collection_bp',
            'simulation_bp',
            'analysis_bp',
            'comparison_bp',
            'visualization_bp'
        ]

        for route_file, expected_bp in zip(route_files, expected_blueprints):
            with open(route_file, 'r') as f:
                content = f.read()
                assert expected_bp in content, f"Blueprint {expected_bp} not found in {route_file}"

    def test_config_classes_exist(self):
        """Test that all configuration classes are defined."""
        config_path = 'app/config.py'

        with open(config_path, 'r') as f:
            content = f.read()
            tree = ast.parse(content)

        class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        expected_classes = ['Config', 'DevelopmentConfig', 'ProductionConfig', 'TestingConfig']
        for expected_class in expected_classes:
            assert expected_class in class_names, f"Missing class: {expected_class}"

    def test_storage_class_methods(self):
        """Test that ResultsStorage has all required methods."""
        storage_path = 'app/storage.py'

        with open(storage_path, 'r') as f:
            content = f.read()
            tree = ast.parse(content)

        # Find the ResultsStorage class
        storage_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'ResultsStorage':
                storage_class = node
                break

        assert storage_class is not None, "ResultsStorage class not found"

        # Check for required methods
        method_names = [node.name for node in storage_class.body if isinstance(node, ast.FunctionDef)]

        expected_methods = [
            '__init__',
            'store_collection_result',
            'store_simulation_result',
            'get_result',
            'get_collection_data',
            'result_exists',
            'list_results',
            'clear_cache'
        ]

        for expected_method in expected_methods:
            assert expected_method in method_names, f"Missing method: {expected_method}"

    def test_error_handlers_defined(self):
        """Test that error handlers are defined."""
        errors_path = 'app/errors.py'

        with open(errors_path, 'r') as f:
            content = f.read()
            tree = ast.parse(content)

        function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

        expected_functions = [
            'register_error_handlers',
            'not_found',
            'internal_error',
            'rate_limit_exceeded',
            'bad_request'
        ]

        for expected_function in expected_functions:
            assert expected_function in function_names, f"Missing function: {expected_function}"

    def test_main_api_server_simplified(self):
        """Test that main api_server.py is simplified."""
        api_server_path = 'api_server.py'

        with open(api_server_path, 'r') as f:
            content = f.read()

        # Check that it imports from app
        assert 'from app import create_app' in content

        # Check that it's much shorter than before (should be under 50 lines)
        lines = content.strip().split('\n')
        non_empty_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
        assert len(non_empty_lines) < 50, f"api_server.py should be simplified, but has {len(non_empty_lines)} non-empty lines"

    def test_routes_init_registers_blueprints(self):
        """Test that routes/__init__.py imports and registers all blueprints."""
        routes_init_path = 'app/routes/__init__.py'

        with open(routes_init_path, 'r') as f:
            content = f.read()

        expected_imports = [
            'from .health import health_bp',
            'from .collection import collection_bp',
            'from .simulation import simulation_bp',
            'from .analysis import analysis_bp',
            'from .comparison import comparison_bp',
            'from .visualization import visualization_bp'
        ]

        for expected_import in expected_imports:
            assert expected_import in content, f"Missing import: {expected_import}"

        # Check that register_blueprints function exists
        assert 'def register_blueprints(app):' in content

    def test_endpoints_defined_in_routes(self):
        """Test that expected endpoints are defined in route files."""
        endpoint_mapping = {
            'app/routes/health.py': ['/health', '/'],
            'app/routes/collection.py': ['/collect', '/results/<result_id>'],
            'app/routes/simulation.py': ['/simulate'],
            'app/routes/analysis.py': ['/analyze'],
            'app/routes/comparison.py': ['/compare', '/optimize'],
            'app/routes/visualization.py': ['/visualize/<simulation_id>']
        }

        for file_path, expected_endpoints in endpoint_mapping.items():
            with open(file_path, 'r') as f:
                content = f.read()

            for endpoint in expected_endpoints:
                # Check for route decorator with this endpoint
                route_pattern = f"@{os.path.basename(file_path)[:-3]}_bp.route('{endpoint}'"
                assert route_pattern in content, f"Endpoint {endpoint} not found in {file_path}"

    def test_imports_are_clean(self):
        """Test that imports are properly organized and don't have circular dependencies."""
        files_to_check = [
            'app/__init__.py',
            'app/config.py',
            'app/storage.py',
            'app/errors.py'
        ]

        for file_path in files_to_check:
            with open(file_path, 'r') as f:
                content = f.read()
                tree = ast.parse(content)

            # Check that there are no relative imports to parent modules
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if hasattr(node, 'module') and node.module:
                        # Should not import from parent app module in these files
                        if file_path != 'app/__init__.py':
                            assert not node.module.startswith('app.'), f"Circular import detected in {file_path}: {node.module}"

    def test_line_count_reduction(self):
        """Test that the refactoring actually reduced complexity by measuring line counts."""
        # Get line count of main api_server.py
        with open('api_server.py', 'r') as f:
            main_lines = len([line for line in f if line.strip() and not line.strip().startswith('#')])

        # Should be under 40 non-comment/blank lines
        assert main_lines < 40, f"Main api_server.py should be simplified to under 40 lines, but has {main_lines}"

        # Check that we have proper separation of concerns
        app_files = [
            'app/__init__.py',
            'app/config.py',
            'app/storage.py',
            'app/errors.py'
        ]

        route_files = [
            'app/routes/health.py',
            'app/routes/collection.py',
            'app/routes/simulation.py',
            'app/routes/analysis.py',
            'app/routes/comparison.py',
            'app/routes/visualization.py'
        ]

        # Each file should be reasonably sized (not too big)
        for file_path in app_files + route_files:
            with open(file_path, 'r') as f:
                lines = len([line for line in f if line.strip() and not line.strip().startswith('#')])
            assert lines < 200, f"{file_path} is too large ({lines} lines). Consider further refactoring."
