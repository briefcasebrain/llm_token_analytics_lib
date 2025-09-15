"""
Isolated unit tests for the config module (without Flask dependencies)
"""

import os
from unittest.mock import patch
import sys

# Mock Flask modules before importing our config
sys.modules['flask'] = None
sys.modules['flask_cors'] = None
sys.modules['flask_limiter'] = None
sys.modules['flask_limiter.util'] = None


def test_config_environment_override():
    """Test that environment variables override defaults."""
    # Import here to avoid import errors
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", "app/config.py")
    config_module = importlib.util.module_from_spec(spec)

    with patch.dict(os.environ, {
        'SECRET_KEY': 'test-secret',
        'HOST': '127.0.0.1',
        'PORT': '8080',
        'DEBUG': 'true'
    }):
        spec.loader.exec_module(config_module)
        config = config_module.Config()

        assert config.SECRET_KEY == 'test-secret'
        assert config.HOST == '127.0.0.1'
        assert config.PORT == 8080
        assert config.DEBUG is True


def test_config_debug_string_conversion():
    """Test DEBUG environment variable string conversion."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", "app/config.py")
    config_module = importlib.util.module_from_spec(spec)

    # Test 'true' (case insensitive)
    with patch.dict(os.environ, {'DEBUG': 'True'}):
        spec.loader.exec_module(config_module)
        config = config_module.Config()
        assert config.DEBUG is True

    # Test 'false' and other values
    with patch.dict(os.environ, {'DEBUG': 'false'}):
        spec.loader.exec_module(config_module)
        config = config_module.Config()
        assert config.DEBUG is False


def test_config_port_integer_conversion():
    """Test PORT environment variable integer conversion."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", "app/config.py")
    config_module = importlib.util.module_from_spec(spec)

    with patch.dict(os.environ, {'PORT': '3000'}):
        spec.loader.exec_module(config_module)
        config = config_module.Config()
        assert config.PORT == 3000
        assert isinstance(config.PORT, int)


def test_development_config():
    """Test DevelopmentConfig class."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", "app/config.py")
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    dev_config = config_module.DevelopmentConfig()
    assert dev_config.DEBUG is True
    # Should inherit base values
    assert hasattr(dev_config, 'SECRET_KEY')
    assert hasattr(dev_config, 'HOST')
    assert hasattr(dev_config, 'PORT')


def test_production_config():
    """Test ProductionConfig class."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", "app/config.py")
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    prod_config = config_module.ProductionConfig()
    assert prod_config.DEBUG is False
    # Should inherit base values
    assert hasattr(prod_config, 'SECRET_KEY')
    assert hasattr(prod_config, 'HOST')
    assert hasattr(prod_config, 'PORT')


def test_testing_config():
    """Test TestingConfig class."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", "app/config.py")
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    test_config = config_module.TestingConfig()
    assert test_config.TESTING is True
    assert test_config.DEBUG is True
    # Should inherit base values
    assert hasattr(test_config, 'SECRET_KEY')
    assert hasattr(test_config, 'HOST')
    assert hasattr(test_config, 'PORT')
