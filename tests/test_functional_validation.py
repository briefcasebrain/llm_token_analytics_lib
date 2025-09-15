"""
Functional validation tests - test that the refactored API maintains functionality
"""

import sys
from unittest.mock import patch, MagicMock


class TestFunctionalValidation:
    """Test that refactored code maintains original functionality."""

    def test_health_endpoint_functionality(self):
        """Test health endpoint returns correct structure."""
        # Mock Flask dependencies
        with patch.dict(sys.modules, {
            'flask': MagicMock(),
            'flask_cors': MagicMock(),
            'flask_limiter': MagicMock(),
            'flask_limiter.util': MagicMock()
        }):
            from app.routes.health import health_check

            with patch('app.routes.health.current_app') as mock_app, \
                 patch('app.routes.health.datetime') as mock_datetime:

                # Setup mocks
                mock_app.config.get.return_value = '1.0.0'
                mock_datetime.now.return_value.isoformat.return_value = '2025-01-01T00:00:00'

                # Call endpoint
                response = health_check()
                data = response[0]

                # Validate response structure
                assert isinstance(data, dict)
                assert 'status' in data
                assert 'timestamp' in data
                assert 'version' in data
                assert data['status'] == 'healthy'
                assert data['timestamp'] == '2025-01-01T00:00:00'
                assert data['version'] == '1.0.0'

    def test_storage_functionality_preserved(self):
        """Test that storage functionality is preserved after refactoring."""
        # Import storage without Flask dependencies
        import importlib.util
        spec = importlib.util.spec_from_file_location("storage", "app/storage.py")
        storage_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(storage_module)

        storage = storage_module.ResultsStorage()

        # Test basic functionality
        assert len(storage._cache) == 0

        # Test storing and retrieving data
        test_data = {'results': {'mean': 10.5}}
        test_config = {'n_simulations': 1000}

        sim_id = storage.store_simulation_result(test_data, test_config)
        assert storage.result_exists(sim_id)

        retrieved = storage.get_result(sim_id)
        assert retrieved['results'] == test_data
        assert retrieved['config'] == test_config
        assert retrieved['type'] == 'simulation'

    def test_config_functionality_preserved(self):
        """Test that configuration functionality is preserved."""
        import importlib.util
        import os
        from unittest.mock import patch

        spec = importlib.util.spec_from_file_location("config", "app/config.py")
        config_module = importlib.util.module_from_spec(spec)

        # Test environment variable handling
        with patch.dict(os.environ, {
            'SECRET_KEY': 'test-secret-123',
            'DEBUG': 'true',
            'PORT': '8080'
        }):
            spec.loader.exec_module(config_module)
            config = config_module.Config()

            assert config.SECRET_KEY == 'test-secret-123'
            assert config.DEBUG is True
            assert config.PORT == 8080

    def test_collection_endpoint_logic(self):
        """Test collection endpoint logic is preserved."""
        with patch.dict(sys.modules, {
            'flask': MagicMock(),
            'flask_cors': MagicMock(),
            'flask_limiter': MagicMock(),
            'flask_limiter.util': MagicMock(),
            'llm_token_analytics': MagicMock()
        }):
            from app.routes.collection import collect_data
            import pandas as pd

            # Mock the request and dependencies
            with patch('app.routes.collection.request') as mock_request, \
                 patch('app.routes.collection.current_app') as mock_app, \
                 patch('app.routes.collection.UnifiedCollector') as mock_collector_class:

                # Setup mocks
                mock_request.json = {'providers': ['openai']}
                mock_app.storage.store_collection_result.return_value = 'collection_123'

                sample_df = pd.DataFrame({'input_tokens': [100, 200]})
                sample_stats = {'total_requests': 2}

                mock_collector = MagicMock()
                mock_collector.collect_all.return_value = sample_df
                mock_collector.get_summary_statistics.return_value = sample_stats
                mock_collector_class.return_value = mock_collector

                # Call endpoint
                response = collect_data()
                data = response[0]

                # Verify functionality
                assert data['success'] is True
                assert data['collection_id'] == 'collection_123'
                assert data['records'] == 2
                assert data['statistics'] == sample_stats

    def test_error_handling_preserved(self):
        """Test that error handling is properly preserved."""
        with patch.dict(sys.modules, {
            'flask': MagicMock(),
            'flask_cors': MagicMock(),
            'flask_limiter': MagicMock(),
            'flask_limiter.util': MagicMock()
        }):
            # Test that error handlers can be imported and called
            from app.errors import register_error_handlers

            mock_app = MagicMock()

            # Should not raise any errors
            register_error_handlers(mock_app)

            # Verify error handlers were registered
            assert mock_app.errorhandler.call_count >= 4  # 404, 500, 429, 400

    def test_blueprints_registration_functionality(self):
        """Test that blueprint registration works correctly."""
        with patch.dict(sys.modules, {
            'flask': MagicMock(),
            'flask_cors': MagicMock(),
            'flask_limiter': MagicMock(),
            'flask_limiter.util': MagicMock(),
            'llm_token_analytics': MagicMock()
        }):
            from app.routes import register_blueprints

            mock_app = MagicMock()

            # Should not raise any errors
            register_blueprints(mock_app)

            # Verify blueprints were registered
            assert mock_app.register_blueprint.call_count == 6  # 6 blueprints

    def test_api_server_main_functionality(self):
        """Test that main API server functionality is preserved."""
        with patch.dict(sys.modules, {
            'flask': MagicMock(),
            'flask_cors': MagicMock(),
            'flask_limiter': MagicMock(),
            'flask_limiter.util': MagicMock(),
            'llm_token_analytics': MagicMock()
        }):
            # Import and test main function
            import importlib.util
            import os

            spec = importlib.util.spec_from_file_location("api_server", "api_server.py")
            api_module = importlib.util.module_from_spec(spec)

            with patch('app.create_app') as mock_create_app, \
                 patch.dict(os.environ, {'PORT': '5000', 'HOST': '127.0.0.1', 'DEBUG': 'false'}):

                mock_app = MagicMock()
                mock_create_app.return_value = mock_app

                spec.loader.exec_module(api_module)

                # Call main function
                api_module.main()

                # Verify app was created and run
                mock_create_app.assert_called_once()
                mock_app.run.assert_called_once_with(
                    host='127.0.0.1',
                    port=5000,
                    debug=False
                )

    def test_rate_limiting_functionality(self):
        """Test that rate limiting functionality is preserved."""
        with patch.dict(sys.modules, {
            'flask': MagicMock(),
            'flask_cors': MagicMock(),
            'flask_limiter': MagicMock(),
            'flask_limiter.util': MagicMock(),
            'llm_token_analytics': MagicMock()
        }):
            from app.routes.collection import rate_limit

            # Test rate limiting decorator
            @rate_limit("5 per hour")
            def test_function():
                return "success"

            with patch('app.routes.collection.current_app') as mock_app:
                mock_limiter = MagicMock()
                mock_app.limiter = mock_limiter

                # Call decorated function
                result = test_function()

                # Verify rate limiting was applied
                assert result == "success"
                mock_limiter.limit.assert_called_once_with("5 per hour")

    def test_module_imports_work(self):
        """Test that all modules can be imported without errors."""
        modules_to_test = [
            'app.config',
            'app.storage',
            'app.errors'
        ]

        for module_name in modules_to_test:
            if '.' in module_name:
                # Import module components individually to avoid Flask dependencies
                module_path = module_name.replace('.', '/') + '.py'
                import importlib.util
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)

                # Should not raise ImportError
                spec.loader.exec_module(module)
                assert module is not None

    def test_endpoint_route_definitions(self):
        """Test that all expected routes are properly defined."""
        expected_routes = {
            'health': ['/', '/health'],
            'collection': ['/collect', '/results/<result_id>'],
            'simulation': ['/simulate'],
            'analysis': ['/analyze'],
            'comparison': ['/compare', '/optimize'],
            'visualization': ['/visualize/<simulation_id>']
        }

        for blueprint_name, routes in expected_routes.items():
            file_path = f'app/routes/{blueprint_name}.py'

            with open(file_path, 'r') as f:
                content = f.read()

            for route in routes:
                # Check that route is defined
                assert f"'{route}'" in content, f"Route {route} not found in {blueprint_name}.py"
