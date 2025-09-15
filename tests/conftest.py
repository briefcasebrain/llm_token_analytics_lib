"""
Pytest Configuration and Shared Fixtures
========================================
Shared fixtures and configuration for the test suite.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import warnings
import logging

from llm_token_analytics import (
    SimulationConfig,
    CollectorConfig,
    TokenSimulator
)


# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Session-level fixtures
# ============================================================================

@pytest.fixture(scope="session")
def temp_directory():
    """Create a temporary directory for the test session."""
    temp_dir = tempfile.mkdtemp(prefix="llm_analytics_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def configure_test_environment():
    """Configure the test environment."""
    # Suppress warnings during tests
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Set numpy random seed for reproducible tests
    np.random.seed(42)

    # Configure matplotlib for headless testing
    try:
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        pass

    yield

    # Cleanup after all tests
    logger.info("Test session completed")


# ============================================================================
# Configuration fixtures
# ============================================================================

@pytest.fixture
def basic_simulation_config():
    """Basic simulation configuration for testing."""
    return SimulationConfig(
        n_simulations=1000,  # Small number for fast tests
        confidence_level=0.95,
        seed=42,
        mechanisms=['per_token', 'bundle'],
        use_empirical_data=False  # Use synthetic data
    )


@pytest.fixture
def extended_simulation_config():
    """Extended simulation configuration with all mechanisms."""
    return SimulationConfig(
        n_simulations=5000,
        confidence_level=0.99,
        seed=123,
        mechanisms=['per_token', 'bundle', 'hybrid', 'cached', 'outcome', 'dynamic'],
        use_empirical_data=False
    )


@pytest.fixture
def mock_collector_config():
    """Mock collector configuration for testing."""
    return CollectorConfig(
        api_key="test_api_key_123",
        provider="openai",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),
        org_id="test_org",
        output_format="parquet",
        cache_dir="./test_cache"
    )


# ============================================================================
# Data fixtures
# ============================================================================

@pytest.fixture
def sample_token_data():
    """Sample token usage data for testing."""
    np.random.seed(42)  # For reproducible data
    n_samples = 1000

    # Generate correlated token data
    correlation = 0.35
    mean = [0, 0]
    cov = [[1, correlation], [correlation, 1]]
    normal_samples = np.random.multivariate_normal(mean, cov, n_samples)

    input_tokens = np.exp(4.0 + 0.4 * normal_samples[:, 0])
    output_tokens = np.exp(4.3 + 0.6 * normal_samples[:, 1])

    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='min'),
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': input_tokens + output_tokens,
        'provider': np.random.choice(['openai', 'anthropic', 'google'], n_samples),
        'model': np.random.choice(['gpt-4', 'claude-3', 'gemini-pro'], n_samples),
        'cost': input_tokens * 0.03/1000 + output_tokens * 0.06/1000
    })


@pytest.fixture
def problematic_token_data():
    """Token data with various issues for testing data cleaning."""
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10),
        'input_tokens': [100, -50, 0, np.inf, np.nan, 1e10, 150, 200, 0.001, 300],
        'output_tokens': [50, 25, -10, 75, np.inf, np.nan, 100, 125, 1e9, 150],
        'provider': ['openai'] * 10,
        'model': ['gpt-4'] * 10
    })


@pytest.fixture
def large_token_dataset():
    """Large dataset for performance testing."""
    np.random.seed(42)
    n_samples = 50000

    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='30s'),
        'input_tokens': np.random.lognormal(4.0, 0.5, n_samples),
        'output_tokens': np.random.lognormal(4.3, 0.6, n_samples),
        'provider': np.random.choice(['openai', 'anthropic'], n_samples),
        'model': np.random.choice(['gpt-4', 'claude-3'], n_samples)
    })


@pytest.fixture
def mock_simulation_results():
    """Mock simulation results for testing."""
    from llm_token_analytics import SimulationResults

    mechanism_results = {
        'per_token': {
            'mean': 0.05,
            'median': 0.048,
            'std': 0.012,
            'cv': 0.24,
            'p95': 0.072,
            'p99': 0.089,
            'tail_ratio': 1.5,
            'skewness': 0.4,
            'kurtosis': 3.2,
            'percentiles': {i: 0.05 * (i/100) for i in range(1, 100)}
        },
        'bundle': {
            'mean': 0.042,
            'median': 0.041,
            'std': 0.015,
            'cv': 0.357,
            'p95': 0.068,
            'p99': 0.082,
            'tail_ratio': 1.66,
            'skewness': 0.6,
            'kurtosis': 3.5,
            'percentiles': {i: 0.042 * (i/100) for i in range(1, 100)}
        }
    }

    comparison = pd.DataFrame({
        'mechanism': ['per_token', 'bundle'],
        'mean': [0.05, 0.042],
        'cv': [0.24, 0.357]
    })

    metadata = {
        'timestamp': datetime.now().isoformat(),
        'n_simulations': 10000,
        'mechanisms': ['per_token', 'bundle']
    }

    return SimulationResults(
        mechanism_results=mechanism_results,
        comparison=comparison,
        metadata=metadata
    )


# ============================================================================
# Mock fixtures
# ============================================================================

@pytest.fixture
def mock_api_response():
    """Mock API response for testing collectors."""
    return {
        "data": [
            {
                "timestamp": "2024-01-15T10:00:00Z",
                "model": "gpt-4",
                "prompt_tokens": 1500,
                "completion_tokens": 800,
                "total_tokens": 2300,
                "cost": 0.092
            },
            {
                "timestamp": "2024-01-15T11:00:00Z",
                "model": "gpt-3.5-turbo",
                "prompt_tokens": 500,
                "completion_tokens": 300,
                "total_tokens": 800,
                "cost": 0.024
            }
        ]
    }


@pytest.fixture
def mock_requests_session():
    """Mock requests session for testing HTTP operations."""
    session_mock = Mock()
    response_mock = Mock()
    response_mock.json.return_value = {"data": []}
    response_mock.raise_for_status.return_value = None
    session_mock.get.return_value = response_mock
    return session_mock


# ============================================================================
# Parametrized fixtures
# ============================================================================

@pytest.fixture(params=['per_token', 'bundle', 'hybrid', 'cached'])
def pricing_mechanism_name(request):
    """Parametrized fixture for different pricing mechanisms."""
    return request.param


@pytest.fixture(params=['openai', 'anthropic', 'google'])
def provider_name(request):
    """Parametrized fixture for different providers."""
    return request.param


@pytest.fixture(params=[100, 1000, 5000])
def simulation_size(request):
    """Parametrized fixture for different simulation sizes."""
    return request.param


# ============================================================================
# Helper fixtures
# ============================================================================

@pytest.fixture
def numpy_random_state():
    """Fixture that provides a controlled random state."""
    return np.random.RandomState(42)


@pytest.fixture
def temporary_file():
    """Fixture that provides a temporary file."""
    import tempfile
    fd, path = tempfile.mkstemp(suffix='.tmp')
    yield path
    try:
        os.close(fd)
        os.unlink(path)
    except (OSError, FileNotFoundError):
        pass


@pytest.fixture
def clean_environment(monkeypatch):
    """Fixture that provides a clean environment."""
    # Clear relevant environment variables
    env_vars_to_clear = [
        'OPENAI_API_KEY',
        'ANTHROPIC_API_KEY',
        'GOOGLE_API_KEY',
        'GOOGLE_CLOUD_PROJECT'
    ]

    for var in env_vars_to_clear:
        monkeypatch.delenv(var, raising=False)

    yield monkeypatch


# ============================================================================
# Performance testing fixtures
# ============================================================================

@pytest.fixture
def performance_timer():
    """Fixture for timing test operations."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def __enter__(self):
            self.start_time = time.time()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.end_time = time.time()

        @property
        def elapsed(self):
            if self.end_time and self.start_time:
                return self.end_time - self.start_time
            return None

    return Timer


@pytest.fixture
def memory_monitor():
    """Fixture for monitoring memory usage during tests."""
    import psutil
    import os

    class MemoryMonitor:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.initial_memory = None
            self.peak_memory = None

        def start(self):
            self.initial_memory = self.process.memory_info().rss

        def update_peak(self):
            current_memory = self.process.memory_info().rss
            if self.peak_memory is None or current_memory > self.peak_memory:
                self.peak_memory = current_memory

        def get_usage_mb(self):
            if self.initial_memory and self.peak_memory:
                return (self.peak_memory - self.initial_memory) / 1024 / 1024
            return None

    return MemoryMonitor


# ============================================================================
# Test markers and configurations
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests for component interactions")
    config.addinivalue_line("markers", "performance: Performance and benchmark tests")
    config.addinivalue_line("markers", "mock: Tests using mocked dependencies")
    config.addinivalue_line("markers", "slow: Slow-running tests (> 5 seconds)")
    config.addinivalue_line("markers", "network: Tests requiring network access")
    config.addinivalue_line("markers", "api: Tests that make actual API calls")
    config.addinivalue_line("markers", "edge_case: Edge case and boundary condition tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on test file names
        if "test_performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)

        if "test_mocks" in item.nodeid:
            item.add_marker(pytest.mark.mock)

        if "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        if "test_schemas" in item.nodeid:
            item.add_marker(pytest.mark.schema)
            item.add_marker(pytest.mark.unit)

        if "test_contracts" in item.nodeid:
            item.add_marker(pytest.mark.contract)
            item.add_marker(pytest.mark.unit)

        if "edge_case" in item.name.lower():
            item.add_marker(pytest.mark.edge_case)


# ============================================================================
# Cleanup fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Automatic cleanup after each test."""
    yield
    # Clean up any temporary files or resources
    import gc
    gc.collect()