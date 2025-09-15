"""
Test configuration and fixtures
"""

import pytest
import pandas as pd
import tempfile
import os
from unittest.mock import MagicMock


@pytest.fixture
def mock_llm_modules():
    """Mock the LLM analytics modules to avoid import errors."""
    import sys
    from unittest.mock import MagicMock

    # Mock the main module and its components
    mock_module = MagicMock()
    mock_module.UnifiedCollector = MagicMock()
    mock_module.TokenSimulator = MagicMock()
    mock_module.SimulationConfig = MagicMock()
    mock_module.TokenAnalyzer = MagicMock()
    mock_module.SimulationVisualizer = MagicMock()

    sys.modules['llm_token_analytics'] = mock_module
    sys.modules['llm_token_analytics.analyzer'] = MagicMock()

    return mock_module


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'input_tokens': [100, 200, 150, 300, 250],
        'output_tokens': [50, 100, 75, 150, 125],
        'provider': ['openai', 'anthropic', 'google', 'openai', 'anthropic'],
        'timestamp': pd.date_range('2025-01-01', periods=5, freq='D')
    })


@pytest.fixture
def sample_stats():
    """Create sample statistics for testing."""
    return {
        'total_requests': 100,
        'total_input_tokens': 10000,
        'total_output_tokens': 5000,
        'avg_input_tokens': 100,
        'avg_output_tokens': 50,
        'providers': ['openai', 'anthropic', 'google']
    }


@pytest.fixture
def mock_simulation_results():
    """Create mock simulation results."""
    mock_results = MagicMock()
    mock_results.mechanism_results = {
        'per_token': {
            'mean': 10.5,
            'median': 9.8,
            'std': 2.3,
            'cv': 0.22,
            'p95': 14.2,
            'p99': 16.8,
            'tail_ratio': 0.15
        },
        'bundle': {
            'mean': 12.1,
            'median': 11.5,
            'std': 1.8,
            'cv': 0.15,
            'p95': 15.0,
            'p99': 16.2,
            'tail_ratio': 0.08
        }
    }
    return mock_results


@pytest.fixture
def temp_parquet_file(sample_dataframe):
    """Create a temporary parquet file for testing."""
    temp_file = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False)
    sample_dataframe.to_parquet(temp_file.name)

    yield temp_file.name

    # Cleanup
    try:
        os.unlink(temp_file.name)
    except OSError:
        pass


@pytest.fixture
def test_config():
    """Test configuration settings."""
    return {
        'TESTING': True,
        'SECRET_KEY': 'test-secret-key',
        'DEBUG': True,
        'API_VERSION': '1.0.0',
        'API_TITLE': 'Test LLM Token Analytics API'
    }
