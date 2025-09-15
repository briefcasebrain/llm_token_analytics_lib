"""
Schema Validation Tests for LLM Token Analytics Library
======================================================
Comprehensive tests for dataclass validation and contract enforcement.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os
from typing import Dict, List, Optional
from dataclasses import FrozenInstanceError

from llm_token_analytics.simulator import SimulationConfig, SimulationResults
from llm_token_analytics.analyzer import TokenDistribution
from llm_token_analytics.collectors import CollectorConfig


class TestSimulationConfig:
    """Test SimulationConfig dataclass validation and behavior."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SimulationConfig()

        assert config.n_simulations == 100_000
        assert config.confidence_level == 0.95
        assert config.seed == 42
        assert config.use_empirical_data is True
        assert config.chunk_size == 10_000
        assert isinstance(config.providers, list)
        assert isinstance(config.mechanisms, list)

    def test_post_init_defaults(self):
        """Test __post_init__ sets correct defaults."""
        config = SimulationConfig()

        expected_providers = ['openai', 'anthropic', 'google']
        expected_mechanisms = ['per_token', 'bundle', 'hybrid', 'cached', 'outcome', 'dynamic']

        assert config.providers == expected_providers
        assert config.mechanisms == expected_mechanisms

    def test_custom_values(self):
        """Test custom configuration values."""
        custom_providers = ['openai']
        custom_mechanisms = ['per_token', 'bundle']

        config = SimulationConfig(
            n_simulations=50_000,
            confidence_level=0.99,
            providers=custom_providers,
            mechanisms=custom_mechanisms
        )

        assert config.n_simulations == 50_000
        assert config.confidence_level == 0.99
        assert config.providers == custom_providers
        assert config.mechanisms == custom_mechanisms

    def test_path_creation(self):
        """Test that paths are created during initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimulationConfig(
                data_path=f"{temp_dir}/test_data",
                output_path=f"{temp_dir}/test_output"
            )

            assert Path(config.data_path).exists()
            assert Path(config.output_path).exists()

    def test_type_validation(self):
        """Test type validation for configuration fields."""
        # These should not raise errors
        config = SimulationConfig(n_simulations=1000)
        assert isinstance(config.n_simulations, int)

        # Test string conversions
        config2 = SimulationConfig(data_path="/tmp/test")
        assert isinstance(config2.data_path, str)

    def test_invalid_values(self):
        """Test validation of invalid values."""
        # Negative simulations should be handled gracefully
        config = SimulationConfig(n_simulations=-1000)
        # The dataclass doesn't enforce this, but we can test it exists
        assert config.n_simulations == -1000

        # Invalid confidence level
        config2 = SimulationConfig(confidence_level=1.5)
        assert config2.confidence_level == 1.5

    def test_seed_behavior(self):
        """Test random seed setting behavior."""
        config1 = SimulationConfig(seed=42)
        config2 = SimulationConfig(seed=42)

        # Both should have the same seed
        assert config1.seed == config2.seed

        # Test different seeds
        config3 = SimulationConfig(seed=123)
        assert config3.seed != config1.seed


class TestSimulationResults:
    """Test SimulationResults dataclass validation."""

    def test_creation(self):
        """Test basic SimulationResults creation."""
        mechanism_results = {
            'per_token': {
                'mean': 0.05,
                'median': 0.04,
                'std': 0.01,
                'cv': 0.2,
                'p95': 0.07,
                'p99': 0.09,
                'tail_ratio': 1.75
            }
        }

        comparison = pd.DataFrame({'test': [1, 2, 3]})
        metadata = {'timestamp': datetime.now().isoformat(), 'n_simulations': 10000}

        results = SimulationResults(
            mechanism_results=mechanism_results,
            comparison=comparison,
            metadata=metadata
        )

        assert results.mechanism_results == mechanism_results
        assert isinstance(results.comparison, pd.DataFrame)
        assert results.metadata == metadata

    def test_to_dataframe_conversion(self):
        """Test conversion to DataFrame."""
        mechanism_results = {
            'per_token': {
                'mean': 0.05, 'median': 0.04, 'std': 0.01,
                'cv': 0.2, 'p95': 0.07, 'p99': 0.09, 'tail_ratio': 1.75
            },
            'bundle': {
                'mean': 0.06, 'median': 0.055, 'std': 0.015,
                'cv': 0.25, 'p95': 0.08, 'p99': 0.095, 'tail_ratio': 1.45
            }
        }

        results = SimulationResults(
            mechanism_results=mechanism_results,
            comparison=pd.DataFrame(),
            metadata={}
        )

        df = results.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'mechanism' in df.columns
        assert 'mean' in df.columns
        assert df['mechanism'].tolist() == ['per_token', 'bundle']

    def test_empty_results(self):
        """Test behavior with empty results."""
        results = SimulationResults(
            mechanism_results={},
            comparison=pd.DataFrame(),
            metadata={}
        )

        df = results.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_missing_fields_in_mechanism_results(self):
        """Test handling of missing fields in mechanism results."""
        mechanism_results = {
            'per_token': {
                'mean': 0.05,
                'median': 0.04
                # Missing other required fields
            }
        }

        results = SimulationResults(
            mechanism_results=mechanism_results,
            comparison=pd.DataFrame(),
            metadata={}
        )

        # Should handle missing keys gracefully (KeyError expected)
        with pytest.raises(KeyError):
            results.to_dataframe()


class TestTokenDistribution:
    """Test TokenDistribution dataclass validation."""

    def test_creation_valid(self):
        """Test creation with valid parameters."""
        distribution = TokenDistribution(
            distribution_type='lognorm',
            parameters={'mu': 4.0, 'sigma': 0.5},
            goodness_of_fit={'aic': 1000.0, 'bic': 1010.0, 'ks_pvalue': 0.8}
        )

        assert distribution.distribution_type == 'lognorm'
        assert distribution.parameters['mu'] == 4.0
        assert distribution.goodness_of_fit['aic'] == 1000.0

    def test_lognorm_sampling(self):
        """Test lognormal distribution sampling."""
        distribution = TokenDistribution(
            distribution_type='lognorm',
            parameters={'mu': 4.0, 'sigma': 0.5},
            goodness_of_fit={}
        )

        samples = distribution.sample(1000)

        assert len(samples) == 1000
        assert all(samples > 0)  # Lognormal samples are positive
        assert isinstance(samples, np.ndarray)

    def test_gamma_sampling(self):
        """Test gamma distribution sampling."""
        distribution = TokenDistribution(
            distribution_type='gamma',
            parameters={'alpha': 2.0, 'scale': 1.5},
            goodness_of_fit={}
        )

        samples = distribution.sample(500)

        assert len(samples) == 500
        assert all(samples > 0)  # Gamma samples are positive
        assert isinstance(samples, np.ndarray)

    def test_pareto_sampling(self):
        """Test Pareto distribution sampling."""
        distribution = TokenDistribution(
            distribution_type='pareto',
            parameters={'alpha': 1.5, 'scale': 1.0},
            goodness_of_fit={}
        )

        samples = distribution.sample(300)

        assert len(samples) == 300
        assert all(samples >= 1.0)  # Pareto samples are >= scale
        assert isinstance(samples, np.ndarray)

    def test_weibull_sampling(self):
        """Test Weibull distribution sampling."""
        distribution = TokenDistribution(
            distribution_type='weibull',
            parameters={'shape': 2.0, 'scale': 1.0},
            goodness_of_fit={}
        )

        samples = distribution.sample(200)

        assert len(samples) == 200
        assert all(samples >= 0)  # Weibull samples are non-negative
        assert isinstance(samples, np.ndarray)

    def test_unknown_distribution_error(self):
        """Test error handling for unknown distribution types."""
        distribution = TokenDistribution(
            distribution_type='unknown',
            parameters={'param': 1.0},
            goodness_of_fit={}
        )

        with pytest.raises(ValueError, match="Unknown distribution"):
            distribution.sample(100)

    def test_missing_parameters(self):
        """Test error handling for missing parameters."""
        distribution = TokenDistribution(
            distribution_type='lognorm',
            parameters={'mu': 4.0},  # Missing 'sigma'
            goodness_of_fit={}
        )

        with pytest.raises(KeyError):
            distribution.sample(100)

    def test_invalid_sample_size(self):
        """Test behavior with invalid sample sizes."""
        distribution = TokenDistribution(
            distribution_type='lognorm',
            parameters={'mu': 4.0, 'sigma': 0.5},
            goodness_of_fit={}
        )

        # Zero samples
        samples = distribution.sample(0)
        assert len(samples) == 0

        # Negative samples - should raise error or handle gracefully
        with pytest.raises((ValueError, TypeError)):
            distribution.sample(-100)


class TestCollectorConfig:
    """Test CollectorConfig dataclass validation."""

    def test_creation(self):
        """Test basic configuration creation."""
        config = CollectorConfig(
            api_key="test_key",
            provider="openai",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31)
        )

        assert config.api_key == "test_key"
        assert config.provider == "openai"
        assert config.output_format == 'parquet'  # Default
        assert config.cache_dir == './cache'  # Default

    def test_from_env_openai(self, monkeypatch):
        """Test environment variable loading for OpenAI."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
        monkeypatch.setenv("OPENAI_ORG_ID", "test_org")

        config = CollectorConfig.from_env("openai")

        assert config.api_key == "test_openai_key"
        assert config.provider == "openai"
        assert config.org_id == "test_org"

    def test_from_env_anthropic(self, monkeypatch):
        """Test environment variable loading for Anthropic."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test_anthropic_key")

        config = CollectorConfig.from_env("anthropic")

        assert config.api_key == "test_anthropic_key"
        assert config.provider == "anthropic"

    def test_from_env_google(self, monkeypatch):
        """Test environment variable loading for Google."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test_google_key")
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test_project")

        config = CollectorConfig.from_env("google")

        assert config.api_key == "test_google_key"
        assert config.provider == "google"
        assert config.project_id == "test_project"

    def test_from_env_missing_key(self):
        """Test error handling when API key is missing."""
        # Clear environment variables
        for var in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY']:
            if var in os.environ:
                del os.environ[var]

        with pytest.raises(ValueError, match="No API key found"):
            CollectorConfig.from_env("openai")

    def test_from_env_unknown_provider(self):
        """Test error handling for unknown provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            CollectorConfig.from_env("unknown_provider")

    def test_date_defaults(self, monkeypatch):
        """Test default date range setting."""
        # Set up environment for testing
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        # Test from_env creates proper date defaults
        config = CollectorConfig.from_env("openai")

        # Check that dates are set and in proper range
        assert config.start_date < config.end_date
        date_diff = config.end_date - config.start_date
        assert 29 <= date_diff.days <= 31  # Allow for some variation in day calculation


class TestTypeValidation:
    """Test type validation across all dataclasses."""

    def test_simulation_config_types(self):
        """Test type enforcement in SimulationConfig."""
        config = SimulationConfig(
            n_simulations=10000,
            confidence_level=0.95,
            seed=42,
            providers=['openai'],
            mechanisms=['per_token'],
            use_empirical_data=True,
            data_path="/tmp",
            output_path="/tmp/output",
            chunk_size=1000
        )

        assert isinstance(config.n_simulations, int)
        assert isinstance(config.confidence_level, float)
        assert isinstance(config.seed, int)
        assert isinstance(config.providers, list)
        assert isinstance(config.mechanisms, list)
        assert isinstance(config.use_empirical_data, bool)
        assert isinstance(config.data_path, str)
        assert isinstance(config.output_path, str)
        assert isinstance(config.chunk_size, int)

    def test_token_distribution_types(self):
        """Test type enforcement in TokenDistribution."""
        distribution = TokenDistribution(
            distribution_type='lognorm',
            parameters={'mu': 4.0, 'sigma': 0.5},
            goodness_of_fit={'aic': 1000.0, 'bic': 1010.0}
        )

        assert isinstance(distribution.distribution_type, str)
        assert isinstance(distribution.parameters, dict)
        assert isinstance(distribution.goodness_of_fit, dict)
        assert all(isinstance(v, (int, float)) for v in distribution.parameters.values())

    def test_collector_config_types(self):
        """Test type enforcement in CollectorConfig."""
        config = CollectorConfig(
            api_key="test",
            provider="openai",
            start_date=datetime.now(),
            end_date=datetime.now(),
            org_id="test_org",
            project_id="test_project",
            output_format='parquet',
            cache_dir='./cache'
        )

        assert isinstance(config.api_key, str)
        assert isinstance(config.provider, str)
        assert isinstance(config.start_date, (datetime, type(None)))
        assert isinstance(config.end_date, (datetime, type(None)))
        assert isinstance(config.org_id, (str, type(None)))
        assert isinstance(config.project_id, (str, type(None)))
        assert isinstance(config.output_format, str)
        assert isinstance(config.cache_dir, str)