"""
Integration Tests for LLM Token Analytics Library
=================================================
End-to-end tests for complete workflows and component interactions.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os
from pathlib import Path
import pickle
import json

from llm_token_analytics import (
    SimulationConfig,
    SimulationResults,
    TokenSimulator,
    CollectorConfig,
    UnifiedCollector,
    TokenAnalyzer,
    CostAnalyzer
)
from llm_token_analytics.collectors import DataProcessor
from llm_token_analytics.simulator import PerTokenPricing


class TestEndToEndSimulationWorkflow:
    """Test complete simulation workflow from configuration to results."""

    def test_basic_simulation_workflow(self):
        """Test complete simulation from config to analysis."""
        # Step 1: Create configuration
        config = SimulationConfig(
            n_simulations=1000,  # Small for fast testing
            mechanisms=['per_token', 'bundle', 'hybrid'],
            seed=42
        )

        # Step 2: Initialize and run simulator
        simulator = TokenSimulator(config)
        results = simulator.run()

        # Step 3: Validate results structure
        assert isinstance(results, SimulationResults)
        assert 'per_token' in results.mechanism_results
        assert 'bundle' in results.mechanism_results
        assert 'hybrid' in results.mechanism_results

        # Step 4: Analyze results
        analyzer = CostAnalyzer(results.mechanism_results)
        risk_metrics = analyzer.calculate_risk_metrics()

        assert isinstance(risk_metrics, pd.DataFrame)
        assert len(risk_metrics) == 3  # Three mechanisms

        # Step 5: Convert to DataFrame
        df = results.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert 'mechanism' in df.columns

    def test_simulation_with_custom_mechanisms(self):
        """Test simulation workflow with custom pricing mechanisms."""

        class CustomPricing(PerTokenPricing):
            def calculate(self, input_tokens: float, output_tokens: float) -> float:
                base_cost = super().calculate(input_tokens, output_tokens)
                volume_discount = 0.9 if (input_tokens + output_tokens) > 10000 else 1.0
                return base_cost * volume_discount

        # Create simulator with custom mechanism
        config = SimulationConfig(n_simulations=500, mechanisms=['per_token'])
        simulator = TokenSimulator(config)

        # Replace with custom mechanism
        custom_config = {"input_price": 30, "output_price": 60}
        simulator.pricing_mechanisms['custom'] = CustomPricing(custom_config)

        # Generate samples and test custom mechanism
        input_tokens, output_tokens = simulator.generate_token_samples(100)
        custom_costs = [
            simulator.pricing_mechanisms['custom'].calculate(inp, out)
            for inp, out in zip(input_tokens, output_tokens)
        ]

        assert len(custom_costs) == 100
        assert all(cost >= 0 for cost in custom_costs)

    def test_sensitivity_analysis_workflow(self):
        """Test complete sensitivity analysis workflow."""
        config = SimulationConfig(n_simulations=500)
        simulator = TokenSimulator(config)

        # Run sensitivity analysis
        param_ranges = {
            'cache_hit_rate': [0.3, 0.7, 0.9],
            'bundle_size': [50000, 100000, 200000]
        }

        sensitivity = simulator.sensitivity_analysis(param_ranges)

        # Validate structure
        assert 'cache_hit_rate' in sensitivity
        assert 'bundle_size' in sensitivity

        for param, results in sensitivity.items():
            for mechanism, data in results.items():
                assert 'values' in data
                assert 'means' in data
                assert 'mean_sensitivity' in data

    def test_reproducible_results(self):
        """Test that simulation results are reproducible with same seed."""
        config1 = SimulationConfig(n_simulations=1000, seed=42)
        config2 = SimulationConfig(n_simulations=1000, seed=42)

        simulator1 = TokenSimulator(config1)
        simulator2 = TokenSimulator(config2)

        results1 = simulator1.run()
        results2 = simulator2.run()

        # Results should be very similar (allowing for small numerical differences)
        for mechanism in results1.mechanism_results:
            if mechanism in results2.mechanism_results:
                mean1 = results1.mechanism_results[mechanism]['mean']
                mean2 = results2.mechanism_results[mechanism]['mean']
                assert abs(mean1 - mean2) < 0.1  # Allow for reasonable variation


class TestDataCollectionWorkflow:
    """Test data collection and processing workflows."""

    def test_data_processing_workflow(self):
        """Test complete data processing from raw to cleaned data."""
        # Step 1: Create synthetic raw data with issues
        raw_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='H'),
            'input_tokens': np.concatenate([
                np.random.lognormal(4.0, 0.5, 995),  # Normal data
                [-100, 0, np.inf, np.nan, 1e10]  # Problematic data (5 values)
            ]),
            'output_tokens': np.concatenate([
                np.random.lognormal(4.3, 0.6, 995),  # Normal data
                [-50, 0, np.inf, np.nan, 1e10]  # Problematic data (5 values)
            ]),
            'provider': np.random.choice(['openai', 'anthropic'], 1000),
            'model': np.random.choice(['gpt-4', 'claude-3'], 1000),
            'cost': np.random.uniform(0.01, 1.0, 1000)
        })

        # Step 2: Process data
        # DataProcessor methods are static
        clean_data = DataProcessor.clean_data(raw_data)

        # Step 3: Validate cleaning
        assert len(clean_data) < len(raw_data)  # Should remove problematic rows
        assert clean_data['input_tokens'].min() > 0
        assert clean_data['output_tokens'].min() > 0
        assert not clean_data.isnull().any().any()

        # Step 4: Calculate statistics
        stats = DataProcessor.calculate_statistics(clean_data)

        assert 'total_records' in stats
        assert 'token_stats' in stats
        assert stats['total_records'] == len(clean_data)

        # Step 5: Aggregate by period
        aggregated = DataProcessor.aggregate_by_period(clean_data, 'D')

        assert isinstance(aggregated, pd.DataFrame)
        assert 'input_tokens' in aggregated.columns

    def test_unified_collector_workflow(self):
        """Test unified collector with synthetic data."""
        # Mock the UnifiedCollector to avoid actual API calls
        collector = UnifiedCollector()

        # Since we don't have real API keys, this will use synthetic data
        # Test the structure even if no real collection occurs
        assert hasattr(collector, 'collectors')
        assert hasattr(collector, 'collect_all')


class TestAnalysisWorkflow:
    """Test statistical analysis workflows."""

    def test_distribution_analysis_workflow(self):
        """Test complete distribution analysis workflow."""
        # Step 1: Generate test data
        np.random.seed(42)
        correlation = 0.35
        n_samples = 2000

        mean = [0, 0]
        cov = [[1, correlation], [correlation, 1]]
        normal_samples = np.random.multivariate_normal(mean, cov, n_samples)

        input_tokens = np.exp(4.0 + 0.4 * normal_samples[:, 0])
        output_tokens = np.exp(4.3 + 0.6 * normal_samples[:, 1])

        # Step 2: Fit distributions
        analyzer = TokenAnalyzer()
        distributions = analyzer.fit_distributions(input_tokens, output_tokens)

        assert 'input' in distributions
        assert 'output' in distributions

        # Step 3: Analyze correlations
        correlations = analyzer.analyze_correlations(input_tokens, output_tokens)

        assert 'linear' in correlations
        assert 'tail_dependence' in correlations

        # Correlation should be close to what we generated
        pearson_corr = correlations['linear']['pearson']['value']
        assert abs(pearson_corr - correlation) < 0.1

        # Step 4: Full analysis
        mock_mechanism_results = {
            'per_token': {
                'mean': 0.05, 'std': 0.01, 'cv': 0.2,
                'p95': 0.07, 'tail_ratio': 1.4, 'skewness': 0.5, 'kurtosis': 3.0
            }
        }

        full_analysis = analyzer.full_analysis(
            input_tokens, output_tokens, mock_mechanism_results
        )

        assert 'distributions' in full_analysis
        assert 'correlations' in full_analysis
        assert 'risk_metrics' in full_analysis
        assert 'summary_statistics' in full_analysis

    def test_cost_analysis_workflow(self):
        """Test cost analysis and optimization workflow."""
        # Step 1: Create mock simulation results
        mechanism_results = {
            'per_token': {
                'mean': 0.05, 'std': 0.01, 'cv': 0.2, 'var_95': 0.02,
                'cvar_95': 0.08, 'tail_ratio': 1.4, 'skewness': 0.5, 'kurtosis': 3.0
            },
            'bundle': {
                'mean': 0.04, 'std': 0.015, 'cv': 0.375, 'var_95': 0.03,
                'cvar_95': 0.09, 'tail_ratio': 1.8, 'skewness': 0.8, 'kurtosis': 3.5
            },
            'hybrid': {
                'mean': 0.045, 'std': 0.008, 'cv': 0.178, 'var_95': 0.015,
                'cvar_95': 0.065, 'tail_ratio': 1.2, 'skewness': 0.3, 'kurtosis': 2.8
            }
        }

        # Step 2: Analyze costs
        analyzer = CostAnalyzer(mechanism_results)
        risk_metrics = analyzer.calculate_risk_metrics()

        assert isinstance(risk_metrics, pd.DataFrame)
        assert 'mechanism' in risk_metrics.columns
        assert len(risk_metrics) == 3

        # Step 3: Find optimal mechanism
        user_profiles = [
            {'risk_tolerance': 'low', 'usage_volume': 50000, 'budget_constraint': 200},
            {'risk_tolerance': 'medium', 'usage_volume': 100000, 'budget_constraint': None},
            {'risk_tolerance': 'high', 'usage_volume': 200000, 'budget_constraint': 500}
        ]

        for profile in user_profiles:
            optimal = analyzer.optimal_mechanism_selection(profile)
            assert optimal in mechanism_results.keys()


class TestConfigurationWorkflow:
    """Test configuration loading and validation workflows."""

    def test_configuration_file_workflow(self):
        """Test loading configuration from files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config file
            config_data = {
                'simulation': {
                    'n_simulations': 50000,
                    'confidence_level': 0.99,
                    'mechanisms': ['per_token', 'bundle']
                },
                'pricing': {
                    'input_price': 25,
                    'output_price': 50
                }
            }

            config_path = Path(temp_dir) / 'config.json'
            with open(config_path, 'w') as f:
                json.dump(config_data, f)

            # Load and use configuration
            with open(config_path) as f:
                loaded_config = json.load(f)

            sim_config = SimulationConfig(
                n_simulations=loaded_config['simulation']['n_simulations'],
                confidence_level=loaded_config['simulation']['confidence_level'],
                mechanisms=loaded_config['simulation']['mechanisms']
            )

            assert sim_config.n_simulations == 50000
            assert sim_config.confidence_level == 0.99
            assert sim_config.mechanisms == ['per_token', 'bundle']

    def test_environment_configuration_workflow(self, monkeypatch):
        """Test configuration from environment variables."""
        # Set environment variables
        monkeypatch.setenv("OPENAI_API_KEY", "test_key_123")
        monkeypatch.setenv("SIMULATION_ITERATIONS", "75000")

        # Create config from environment
        config = CollectorConfig.from_env("openai")
        assert config.api_key == "test_key_123"

        # Simulation config with env override
        iterations = int(os.getenv("SIMULATION_ITERATIONS", 100000))
        sim_config = SimulationConfig(n_simulations=iterations)
        assert sim_config.n_simulations == 75000


class TestPersistenceWorkflow:
    """Test saving and loading of results."""

    def test_results_persistence_workflow(self):
        """Test saving and loading simulation results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Generate results
            config = SimulationConfig(n_simulations=100, mechanisms=['per_token'])
            simulator = TokenSimulator(config)
            original_results = simulator.run()

            # Step 2: Save results
            results_path = Path(temp_dir) / 'results.pkl'
            with open(results_path, 'wb') as f:
                pickle.dump(original_results, f)

            # Step 3: Load results
            with open(results_path, 'rb') as f:
                loaded_results = pickle.load(f)

            # Step 4: Validate loaded results
            assert isinstance(loaded_results, SimulationResults)
            assert loaded_results.mechanism_results.keys() == original_results.mechanism_results.keys()

            # Compare key metrics
            for mechanism in original_results.mechanism_results:
                orig_mean = original_results.mechanism_results[mechanism]['mean']
                load_mean = loaded_results.mechanism_results[mechanism]['mean']
                assert abs(orig_mean - load_mean) < 1e-10

    def test_data_export_workflow(self):
        """Test exporting data in various formats."""
        # Create test data
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100),
            'input_tokens': np.random.lognormal(4.0, 0.5, 100),
            'output_tokens': np.random.lognormal(4.3, 0.6, 100),
            'provider': np.random.choice(['openai', 'anthropic'], 100)
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test CSV export
            csv_path = Path(temp_dir) / 'data.csv'
            data.to_csv(csv_path, index=False)
            loaded_csv = pd.read_csv(csv_path)
            assert len(loaded_csv) == len(data)

            # Test Parquet export
            parquet_path = Path(temp_dir) / 'data.parquet'
            data.to_parquet(parquet_path, index=False)
            loaded_parquet = pd.read_parquet(parquet_path)
            assert len(loaded_parquet) == len(data)


class TestErrorHandlingWorkflow:
    """Test error handling in complete workflows."""

    def test_graceful_degradation(self):
        """Test that workflows handle errors gracefully."""
        # Test with invalid configuration
        config = SimulationConfig(n_simulations=1)  # Minimal but valid
        simulator = TokenSimulator(config)

        # Should handle gracefully or raise appropriate error
        try:
            results = simulator.run()
            # If it doesn't error, results should be valid structure
            assert isinstance(results, SimulationResults)
        except (ValueError, RuntimeError):
            # Acceptable to raise error for invalid config
            pass

    def test_partial_failure_handling(self):
        """Test handling of partial failures in workflows."""
        # Test with missing mechanisms
        config = SimulationConfig(
            n_simulations=100,
            mechanisms=['per_token', 'nonexistent_mechanism']
        )

        simulator = TokenSimulator(config)
        results = simulator.run()

        # Should still have results for valid mechanisms
        assert 'per_token' in results.mechanism_results
        # nonexistent_mechanism should not be in results
        assert 'nonexistent_mechanism' not in results.mechanism_results


if __name__ == "__main__":
    pytest.main([__file__])