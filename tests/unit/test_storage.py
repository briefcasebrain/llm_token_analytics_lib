"""
Isolated unit tests for the storage module (without Flask dependencies)
"""

import pytest
import pandas as pd
import tempfile
import os
from unittest.mock import patch
import sys
import importlib.util

# Mock Flask modules before importing our storage
sys.modules['flask'] = None
sys.modules['flask_cors'] = None
sys.modules['flask_limiter'] = None
sys.modules['flask_limiter.util'] = None


@pytest.fixture
def storage_class():
    """Load storage class dynamically."""
    spec = importlib.util.spec_from_file_location("storage", "app/storage.py")
    storage_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(storage_module)
    return storage_module.ResultsStorage


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


def test_storage_init(storage_class):
    """Test storage initialization."""
    storage = storage_class()
    assert hasattr(storage, '_cache')
    assert isinstance(storage._cache, dict)
    assert len(storage._cache) == 0


def test_store_collection_result(storage_class, sample_dataframe, sample_stats):
    """Test storing collection results."""
    storage = storage_class()

    with patch('tempfile.NamedTemporaryFile') as mock_temp:
        mock_file = mock_temp.return_value
        mock_file.name = '/tmp/test_file.parquet'

        with patch.object(sample_dataframe, 'to_parquet') as mock_to_parquet:
            collection_id = storage.store_collection_result(sample_dataframe, sample_stats)

            # Verify collection_id format
            assert collection_id.startswith('collection_')
            assert len(collection_id) > 11  # collection_ + timestamp

            # Verify data was stored
            assert collection_id in storage._cache
            stored_data = storage._cache[collection_id]
            assert stored_data['file'] == '/tmp/test_file.parquet'
            assert stored_data['stats'] == sample_stats
            assert stored_data['type'] == 'collection'
            assert 'timestamp' in stored_data

            # Verify to_parquet was called
            mock_to_parquet.assert_called_once_with('/tmp/test_file.parquet')


def test_store_simulation_result(storage_class):
    """Test storing simulation results."""
    storage = storage_class()
    mock_results = {'mechanism_results': {'per_token': {'mean': 10.5}}}
    mock_config = {'n_simulations': 1000}

    simulation_id = storage.store_simulation_result(mock_results, mock_config)

    # Verify simulation_id format
    assert simulation_id.startswith('simulation_')
    assert len(simulation_id) > 11  # simulation_ + timestamp

    # Verify data was stored
    assert simulation_id in storage._cache
    stored_data = storage._cache[simulation_id]
    assert stored_data['results'] == mock_results
    assert stored_data['config'] == mock_config
    assert stored_data['type'] == 'simulation'
    assert 'timestamp' in stored_data


def test_get_result_existing(storage_class, sample_stats):
    """Test getting an existing result."""
    storage = storage_class()

    # Store some data first
    storage._cache['test_id'] = {
        'stats': sample_stats,
        'type': 'collection',
        'timestamp': '2025-01-01T00:00:00'
    }

    # Retrieve the data
    result = storage.get_result('test_id')
    assert result is not None
    assert result['stats'] == sample_stats
    assert result['type'] == 'collection'


def test_get_result_nonexistent(storage_class):
    """Test getting a non-existent result."""
    storage = storage_class()
    result = storage.get_result('nonexistent_id')
    assert result is None


def test_get_collection_data_existing(storage_class, sample_dataframe):
    """Test getting collection data that exists."""
    storage = storage_class()

    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as temp_file:
        # Create actual parquet file
        sample_dataframe.to_parquet(temp_file.name)

        # Store reference in cache
        storage._cache['test_collection'] = {
            'file': temp_file.name,
            'type': 'collection'
        }

        # Test retrieval
        result_df = storage.get_collection_data('test_collection')
        assert result_df is not None
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(sample_dataframe)

        # Cleanup
        os.unlink(temp_file.name)


def test_get_collection_data_nonexistent(storage_class):
    """Test getting collection data that doesn't exist."""
    storage = storage_class()
    result = storage.get_collection_data('nonexistent_collection')
    assert result is None


def test_result_exists(storage_class):
    """Test checking if result exists."""
    storage = storage_class()

    # Initially no results
    assert not storage.result_exists('test_id')

    # Add a result
    storage._cache['test_id'] = {'data': 'test'}

    # Now it should exist
    assert storage.result_exists('test_id')


def test_list_results(storage_class, sample_stats):
    """Test listing all results."""
    storage = storage_class()

    # Add multiple results
    storage._cache['collection_1'] = {
        'type': 'collection',
        'timestamp': '2025-01-01T00:00:00',
        'stats': sample_stats
    }
    storage._cache['simulation_1'] = {
        'type': 'simulation',
        'timestamp': '2025-01-02T00:00:00',
        'results': {'test': 'data'}
    }

    # Test listing
    results_summary = storage.list_results()

    assert len(results_summary) == 2
    assert 'collection_1' in results_summary
    assert 'simulation_1' in results_summary

    # Check that only type and timestamp are included
    for result_id, summary in results_summary.items():
        assert 'type' in summary
        assert 'timestamp' in summary
        assert len(summary) == 2  # Only type and timestamp


def test_clear_cache(storage_class):
    """Test clearing the cache."""
    storage = storage_class()

    # Add some data
    storage._cache['test_1'] = {'data': 'test1'}
    storage._cache['test_2'] = {'data': 'test2'}

    assert len(storage._cache) == 2

    # Clear cache
    storage.clear_cache()

    assert len(storage._cache) == 0
    assert storage._cache == {}


def test_get_collection_data_missing_file_key(storage_class):
    """Test getting collection data when file key is missing."""
    storage = storage_class()

    # Store collection without file key
    storage._cache['test_collection'] = {
        'type': 'collection',
        'stats': {'test': 'data'}
    }

    result = storage.get_collection_data('test_collection')
    assert result is None
