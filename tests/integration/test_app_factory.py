"""
Integration tests for Flask app factory
"""

import pytest
from unittest.mock import patch, MagicMock
import sys


class TestAppFactory:
    """Test cases for Flask app factory."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self, mock_llm_modules):
        """Setup mocks for each test."""
        # Mock Flask and related modules
        self.flask_mock = MagicMock()
        self.cors_mock = MagicMock()
        self.limiter_mock = MagicMock()

        with patch.dict(sys.modules, {
            'flask': self.flask_mock,
            'flask_cors': self.cors_mock,
            'flask_limiter': self.limiter_mock,
            'flask_limiter.util': MagicMock()
        }):
            # Mock Flask class and its instance
            self.mock_app = MagicMock()
            self.flask_mock.Flask.return_value = self.mock_app

            # Mock CORS
            self.cors_mock.CORS = MagicMock()

            # Mock Limiter
            self.limiter_mock.Limiter = MagicMock()
            self.limiter_mock.util.get_remote_address = MagicMock()

            yield

    def test_create_app_basic(self):
        """Test basic app creation without custom config."""
        with patch.dict(sys.modules, {
            'flask': self.flask_mock,
            'flask_cors': self.cors_mock,
            'flask_limiter': self.limiter_mock,
            'flask_limiter.util': MagicMock()
        }):
            from app import create_app

            app = create_app()

            # Verify Flask app was created
            self.flask_mock.Flask.assert_called_once()

            # Verify CORS was initialized
            self.cors_mock.CORS.assert_called_once_with(self.mock_app)

            # Verify Limiter was initialized
            self.limiter_mock.Limiter.assert_called_once()

            # Verify app has required attributes
            assert hasattr(self.mock_app, 'limiter')
            assert hasattr(self.mock_app, 'storage')

    def test_create_app_with_custom_config(self, test_config):
        """Test app creation with custom configuration."""
        with patch.dict(sys.modules, {
            'flask': self.flask_mock,
            'flask_cors': self.cors_mock,
            'flask_limiter': self.limiter_mock,
            'flask_limiter.util': MagicMock()
        }):
            from app import create_app

            app = create_app(config=test_config)

            # Verify config was applied
            self.mock_app.config.update.assert_called_once_with(test_config)

    def test_create_app_extensions_initialization(self):
        """Test that all extensions are properly initialized."""
        with patch.dict(sys.modules, {
            'flask': self.flask_mock,
            'flask_cors': self.cors_mock,
            'flask_limiter': self.limiter_mock,
            'flask_limiter.util': MagicMock()
        }):
            from app import create_app

            app = create_app()

            # Verify CORS initialization
            self.cors_mock.CORS.assert_called_once_with(self.mock_app)

            # Verify Limiter initialization
            limiter_call = self.limiter_mock.Limiter.call_args
            assert limiter_call[1]['app'] == self.mock_app
            assert 'key_func' in limiter_call[1]
            assert 'default_limits' in limiter_call[1]

    def test_create_app_storage_initialization(self):
        """Test that storage is properly initialized."""
        with patch.dict(sys.modules, {
            'flask': self.flask_mock,
            'flask_cors': self.cors_mock,
            'flask_limiter': self.limiter_mock,
            'flask_limiter.util': MagicMock()
        }):
            with patch('app.ResultsStorage') as mock_storage_class:
                mock_storage_instance = MagicMock()
                mock_storage_class.return_value = mock_storage_instance

                from app import create_app

                app = create_app()

                # Verify storage was created and assigned
                mock_storage_class.assert_called_once()
                assert self.mock_app.storage == mock_storage_instance

    def test_create_app_error_handlers_registration(self):
        """Test that error handlers are registered."""
        with patch.dict(sys.modules, {
            'flask': self.flask_mock,
            'flask_cors': self.cors_mock,
            'flask_limiter': self.limiter_mock,
            'flask_limiter.util': MagicMock()
        }):
            with patch('app.register_error_handlers') as mock_error_handlers:
                from app import create_app

                app = create_app()

                # Verify error handlers were registered
                mock_error_handlers.assert_called_once_with(self.mock_app)

    def test_create_app_blueprints_registration(self):
        """Test that blueprints are registered."""
        with patch.dict(sys.modules, {
            'flask': self.flask_mock,
            'flask_cors': self.cors_mock,
            'flask_limiter': self.limiter_mock,
            'flask_limiter.util': MagicMock()
        }):
            with patch('app.register_blueprints') as mock_blueprints:
                from app import create_app

                app = create_app()

                # Verify blueprints were registered
                mock_blueprints.assert_called_once_with(self.mock_app)

    def test_create_app_rate_limiting_configuration(self):
        """Test rate limiting configuration."""
        with patch.dict(sys.modules, {
            'flask': self.flask_mock,
            'flask_cors': self.cors_mock,
            'flask_limiter': self.limiter_mock,
            'flask_limiter.util': MagicMock()
        }):
            from app import create_app

            app = create_app()

            # Verify Limiter was called with correct parameters
            limiter_call = self.limiter_mock.Limiter.call_args
            assert limiter_call[1]['app'] == self.mock_app
            assert limiter_call[1]['default_limits'] == ["100 per hour", "10 per minute"]

    def test_create_app_config_from_object(self):
        """Test that config is loaded from Config object when no custom config provided."""
        with patch.dict(sys.modules, {
            'flask': self.flask_mock,
            'flask_cors': self.cors_mock,
            'flask_limiter': self.limiter_mock,
            'flask_limiter.util': MagicMock()
        }):
            with patch('app.Config') as mock_config_class:
                from app import create_app

                app = create_app()

                # Verify config was loaded from Config class
                self.mock_app.config.from_object.assert_called_once_with(mock_config_class)

    def test_create_app_logging_configuration(self):
        """Test that logging is properly configured."""
        with patch.dict(sys.modules, {
            'flask': self.flask_mock,
            'flask_cors': self.cors_mock,
            'flask_limiter': self.limiter_mock,
            'flask_limiter.util': MagicMock()
        }):
            with patch('logging.basicConfig') as mock_logging:
                from app import create_app

                app = create_app()

                # Verify logging was configured
                mock_logging.assert_called_once()
                call_args = mock_logging.call_args[1]
                assert call_args['level'] == logging.INFO
                assert 'format' in call_args

    def test_create_app_returns_flask_instance(self):
        """Test that create_app returns the Flask app instance."""
        with patch.dict(sys.modules, {
            'flask': self.flask_mock,
            'flask_cors': self.cors_mock,
            'flask_limiter': self.limiter_mock,
            'flask_limiter.util': MagicMock()
        }):
            from app import create_app

            app = create_app()

            # Verify the returned app is the Flask instance
            assert app == self.mock_app


# Need to import logging for the test
import logging
