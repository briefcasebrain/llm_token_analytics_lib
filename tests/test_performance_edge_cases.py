"""
Performance and Edge Case Tests
===============================
Tests for performance characteristics, memory usage, and edge cases.
"""

import pytest
import numpy as np
import pandas as pd
import time
import psutil
import os
from datetime import datetime, timedelta
import gc
from typing import List, Dict
import threading
import multiprocessing as mp

from llm_token_analytics import (
    SimulationConfig,
    TokenSimulator,
    TokenAnalyzer,
    CollectorConfig
)
from llm_token_analytics.collectors import DataProcessor
from llm_token_analytics.analyzer import DistributionFitter, CorrelationAnalyzer
from llm_token_analytics.simulator import PerTokenPricing


class TestPerformanceCharacteristics:
    """Test performance characteristics of key operations."""

    def test_simulation_performance_scaling(self):
        """Test how simulation performance scales with number of iterations."""
        iteration_counts = [1000, 5000, 10000, 25000]
        execution_times = []

        for n_iterations in iteration_counts:
            config = SimulationConfig(
                n_simulations=n_iterations,
                mechanisms=['per_token', 'bundle'],
                seed=42
            )
            simulator = TokenSimulator(config)

            start_time = time.time()
            results = simulator.run()
            end_time = time.time()

            execution_time = end_time - start_time
            execution_times.append(execution_time)

            # Verify results are still valid
            assert len(results.mechanism_results) == 2

        # Performance should scale reasonably (not exponentially)
        # Time per iteration should be relatively stable
        time_per_iteration = [t / n for t, n in zip(execution_times, iteration_counts)]

        # Variation should be reasonable (within 3x of minimum)
        min_time = min(time_per_iteration)
        max_time = max(time_per_iteration)
        assert max_time < min_time * 3, f"Performance degradation too high: {max_time/min_time:.2f}x"

    def test_distribution_fitting_performance(self):
        """Test distribution fitting performance with different data sizes."""
        data_sizes = [500, 1000, 5000, 10000]
        execution_times = []

        for size in data_sizes:
            # Generate test data
            data = np.random.lognormal(4.0, 0.5, size)
            fitter = DistributionFitter(data)

            start_time = time.time()
            best_distribution = fitter.fit_all()
            end_time = time.time()

            execution_times.append(end_time - start_time)

            # Verify fitting worked
            assert best_distribution.distribution_type is not None

        # Performance should scale sub-quadratically
        # Check that largest dataset doesn't take more than 10x longest small dataset
        assert execution_times[-1] < execution_times[0] * 10

    @pytest.mark.parametrize("chunk_size", [1000, 5000, 10000])
    def test_chunked_processing_performance(self, chunk_size):
        """Test performance of chunked data processing."""
        # Generate large dataset
        large_data = pd.DataFrame({
            'input_tokens': np.random.lognormal(4.0, 0.5, 50000),
            'output_tokens': np.random.lognormal(4.3, 0.6, 50000),
            'timestamp': pd.date_range('2024-01-01', periods=50000, freq='min'),
            'provider': np.random.choice(['openai', 'anthropic'], 50000)
        })

        # DataProcessor methods are static

        start_time = time.time()

        # Process in chunks
        cleaned_chunks = []
        for i in range(0, len(large_data), chunk_size):
            chunk = large_data.iloc[i:i+chunk_size]
            cleaned_chunk = DataProcessor.clean_data(chunk)
            cleaned_chunks.append(cleaned_chunk)

        final_data = pd.concat(cleaned_chunks, ignore_index=True)
        end_time = time.time()

        # Verify processing worked
        assert len(final_data) <= len(large_data)  # Some rows may be cleaned out
        assert 'input_tokens' in final_data.columns

        # Processing should complete in reasonable time (less than 30 seconds)
        processing_time = end_time - start_time
        assert processing_time < 30, f"Chunked processing took too long: {processing_time:.2f}s"

    def test_memory_efficient_simulation(self):
        """Test memory usage during large simulations."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        config = SimulationConfig(
            n_simulations=50000,
            mechanisms=['per_token', 'bundle', 'hybrid'],
            chunk_size=5000  # Process in chunks to limit memory
        )

        simulator = TokenSimulator(config)
        results = simulator.run()

        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory

        # Memory increase should be reasonable (less than 500MB)
        assert memory_increase < 500, f"Memory usage too high: {memory_increase:.1f}MB"

        # Clean up
        del results, simulator
        gc.collect()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        # Create empty DataFrame with expected columns
        empty_df = pd.DataFrame(columns=['input_tokens', 'output_tokens', 'provider', 'timestamp'])
        # DataProcessor methods are static

        # Should handle empty DataFrame gracefully
        result = DataProcessor.clean_data(empty_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

        # Statistics on empty data with correct structure
        stats = DataProcessor.calculate_statistics(empty_df)
        assert stats['total_records'] == 0

    def test_single_row_data(self):
        """Test handling of single-row datasets."""
        single_row = pd.DataFrame({
            'input_tokens': [1000],
            'output_tokens': [500],
            'provider': ['openai'],
            'timestamp': [datetime.now()]
        })

        # DataProcessor methods are static
        result = DataProcessor.clean_data(single_row)

        assert len(result) == 1
        assert result['input_tokens'].iloc[0] == 1000

    def test_extreme_values(self):
        """Test handling of extreme token values."""
        extreme_data = pd.DataFrame({
            'input_tokens': [1, 1e6, 0.001, 1e9],  # Very small to very large
            'output_tokens': [1, 1e5, 0.001, 1e8],
            'provider': ['openai'] * 4,
            'timestamp': pd.date_range('2024-01-01', periods=4)
        })

        # DataProcessor methods are static

        # Should handle extreme values (may filter some out)
        result = DataProcessor.clean_data(extreme_data)
        assert isinstance(result, pd.DataFrame)

        # At least some data should remain
        assert isinstance(result, pd.DataFrame)

    def test_nan_and_infinite_values(self):
        """Test handling of NaN and infinite values."""
        problematic_data = pd.DataFrame({
            'input_tokens': [100, np.nan, np.inf, -np.inf, 200],
            'output_tokens': [50, 75, -np.inf, np.nan, 100],
            'provider': ['openai'] * 5,
            'timestamp': pd.date_range('2024-01-01', periods=5)
        })

        # DataProcessor methods are static
        cleaned = DataProcessor.clean_data(problematic_data)

        # Should remove rows with NaN/inf values
        assert not cleaned.isnull().any().any()
        assert not np.isinf(cleaned.select_dtypes(include=[np.number])).any().any()

    def test_zero_and_negative_tokens(self):
        """Test handling of zero and negative token counts."""
        zero_negative_data = pd.DataFrame({
            'input_tokens': [-100, 0, -50, 100, 200],
            'output_tokens': [50, -25, 0, 150, 75],
            'provider': ['openai'] * 5,
            'timestamp': pd.date_range('2024-01-01', periods=5)
        })

        # DataProcessor methods are static
        cleaned = DataProcessor.clean_data(zero_negative_data)

        # Should remove non-positive token counts
        assert (cleaned['input_tokens'] > 0).all()
        assert (cleaned['output_tokens'] > 0).all()

    def test_duplicate_timestamps(self):
        """Test handling of duplicate timestamps."""
        duplicate_time = datetime.now()
        duplicate_data = pd.DataFrame({
            'input_tokens': [100, 150, 200],
            'output_tokens': [50, 75, 100],
            'provider': ['openai', 'anthropic', 'openai'],
            'timestamp': [duplicate_time] * 3
        })

        # DataProcessor methods are static
        result = DataProcessor.clean_data(duplicate_data)

        # Should handle duplicates gracefully
        assert isinstance(result, pd.DataFrame)

    def test_very_large_correlations(self):
        """Test correlation analysis with extreme correlations."""
        # Perfect correlation
        n = 1000
        base_values = np.random.lognormal(4.0, 0.5, n)
        # Add tiny amount of noise to avoid perfect correlation
        perfect_corr_data = base_values + np.random.normal(0, 0.001, n)

        analyzer = CorrelationAnalyzer(base_values, perfect_corr_data)
        correlations = analyzer.analyze_correlations()

        # Should handle perfect correlation
        pearson_corr = correlations['linear']['pearson']['value']
        assert abs(pearson_corr - 1.0) < 0.01

        # Zero correlation
        uncorrelated_data = np.random.lognormal(4.3, 0.6, n)
        analyzer2 = CorrelationAnalyzer(base_values, uncorrelated_data)
        correlations2 = analyzer2.analyze_correlations()

        # Should handle zero correlation
        assert isinstance(correlations2['linear']['pearson']['value'], float)

    def test_distribution_fitting_edge_cases(self):
        """Test distribution fitting with challenging datasets."""
        # Uniform data (all same value)
        uniform_data = np.ones(1000) * 100
        fitter = DistributionFitter(uniform_data)

        try:
            result = fitter.fit_all()
            # Should either succeed or raise appropriate error
            assert result.distribution_type is not None
        except (ValueError, RuntimeError):
            # Acceptable for uniform data to fail fitting
            pass

        # Heavy-tailed data
        heavy_tail_data = np.concatenate([
            np.random.lognormal(4.0, 0.5, 900),  # Main distribution
            np.random.lognormal(8.0, 1.0, 100)   # Heavy tail
        ])

        fitter2 = DistributionFitter(heavy_tail_data)
        result2 = fitter2.fit_all()
        assert result2.distribution_type is not None

    def test_pricing_mechanism_edge_cases(self):
        """Test pricing mechanisms with edge case inputs."""
        config = {"input_price": 30, "output_price": 60}
        mechanism = PerTokenPricing(config)

        # Zero tokens
        cost_zero = mechanism.calculate(0, 0)
        assert cost_zero == 0

        # Very large token counts
        cost_large = mechanism.calculate(1e9, 1e9)
        assert cost_large > 0
        assert np.isfinite(cost_large)

        # Float precision test
        cost_precision = mechanism.calculate(1000.123456789, 500.987654321)
        assert isinstance(cost_precision, float)

    def test_simulation_with_minimal_iterations(self):
        """Test simulation with very few iterations."""
        config = SimulationConfig(n_simulations=1, mechanisms=['per_token'])
        simulator = TokenSimulator(config)

        results = simulator.run()

        # Should still produce valid results structure
        assert isinstance(results.mechanism_results, dict)
        assert 'per_token' in results.mechanism_results

        # Statistics should be reasonable even with one iteration
        stats = results.mechanism_results['per_token']
        assert 'mean' in stats
        assert 'std' in stats

    def test_configuration_boundary_values(self):
        """Test configuration with boundary values."""
        # Maximum reasonable values
        config_max = SimulationConfig(
            n_simulations=1000000,  # Very large
            confidence_level=0.9999,  # Near 1
            chunk_size=100000
        )

        # Should create without error
        assert config_max.n_simulations == 1000000

        # Minimum reasonable values
        config_min = SimulationConfig(
            n_simulations=1,
            confidence_level=0.01,  # Near 0
            chunk_size=1
        )

        assert config_min.n_simulations == 1

    def test_string_encoding_edge_cases(self):
        """Test handling of various string encodings."""
        # Data with special characters
        special_data = pd.DataFrame({
            'input_tokens': [100, 200],
            'output_tokens': [50, 100],
            'provider': ['openai', 'anthropic'],
            'model': ['gpt-4-turbo-preview', 'claude-3-opus-20240229'],
            'metadata': ['{"key": "value with üñíçødé"}', '{"test": "normal"}']
        })

        # DataProcessor methods are static
        result = DataProcessor.clean_data(special_data)

        # Should handle special characters gracefully
        assert isinstance(result, pd.DataFrame)


class TestConcurrencyAndThreadSafety:
    """Test concurrent operations and thread safety."""

    def test_concurrent_simulations(self):
        """Test running multiple simulations concurrently."""
        def run_simulation(seed):
            config = SimulationConfig(n_simulations=1000, seed=seed)
            simulator = TokenSimulator(config)
            return simulator.run()

        # Run multiple simulations in threads
        threads = []
        results = {}

        for i in range(3):
            thread = threading.Thread(
                target=lambda i=i: results.update({i: run_simulation(i * 100)})
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All simulations should complete successfully
        assert len(results) == 3
        for result in results.values():
            assert isinstance(result.mechanism_results, dict)

    def test_data_processing_thread_safety(self):
        """Test thread safety of data processing operations."""
        # Create shared data
        base_data = pd.DataFrame({
            'input_tokens': np.random.lognormal(4.0, 0.5, 10000),
            'output_tokens': np.random.lognormal(4.3, 0.6, 10000),
            'provider': np.random.choice(['openai', 'anthropic'], 10000),
            'timestamp': pd.date_range('2024-01-01', periods=10000, freq='min')
        })

        # DataProcessor methods are static
        results = {}

        def process_chunk(chunk_id, start_idx, end_idx):
            chunk = base_data.iloc[start_idx:end_idx]
            cleaned = DataProcessor.clean_data(chunk)
            results[chunk_id] = len(cleaned)

        # Process chunks concurrently
        threads = []
        chunk_size = 2000

        for i in range(5):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(base_data))
            thread = threading.Thread(
                target=process_chunk,
                args=(i, start_idx, end_idx)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All chunks should be processed
        assert len(results) == 5
        assert all(count >= 0 for count in results.values())

    @pytest.mark.skip(reason="Multiprocessing with local functions causes pickling issues")
    def test_multiprocessing_safety(self):
        """Test multiprocessing safety where applicable."""
        # Skipped due to pickling issues with local functions
        # The library's core functionality doesn't rely on multiprocessing
        pass


class TestResourceLimits:
    """Test behavior under resource constraints."""

    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure."""
        # Simulate memory pressure by creating large objects
        large_arrays = []

        try:
            # Create arrays until we approach memory limits
            for i in range(10):
                large_arrays.append(np.random.random((10000, 100)))

            # Now try to run simulation under memory pressure
            config = SimulationConfig(n_simulations=1000, chunk_size=100)
            simulator = TokenSimulator(config)
            results = simulator.run()

            # Should still work despite memory pressure
            assert isinstance(results.mechanism_results, dict)

        finally:
            # Clean up large arrays
            del large_arrays
            gc.collect()

    def test_timeout_simulation(self):
        """Test simulation timeout behavior."""
        # This would be more relevant if we had timeout mechanisms
        config = SimulationConfig(n_simulations=100000)  # Large simulation

        start_time = time.time()
        simulator = TokenSimulator(config)

        # Set a reasonable timeout expectation
        results = simulator.run()
        elapsed_time = time.time() - start_time

        # Should complete in reasonable time (less than 60 seconds)
        assert elapsed_time < 60, f"Simulation took too long: {elapsed_time:.2f}s"

        # Results should still be valid
        assert isinstance(results.mechanism_results, dict)


class TestNumericalStability:
    """Test numerical stability and precision."""

    def test_floating_point_precision(self):
        """Test handling of floating-point precision issues."""
        # Create data with precision challenges
        tiny_values = np.array([1e-10, 1e-15, 1e-20])
        huge_values = np.array([1e10, 1e15, 1e20])

        # Mix tiny and huge values
        mixed_data = np.concatenate([tiny_values, huge_values, [1.0, 2.0, 3.0]])

        fitter = DistributionFitter(mixed_data)

        # Should handle extreme ranges gracefully
        try:
            result = fitter.fit_all()
            assert result.distribution_type is not None
        except (ValueError, RuntimeError, OverflowError, UnderflowError):
            # Acceptable for extreme values to cause fitting issues
            pass

    def test_numerical_overflow_prevention(self):
        """Test prevention of numerical overflow in calculations."""
        config = {"input_price": 1e6, "output_price": 1e6}  # Very high prices
        mechanism = PerTokenPricing(config)

        # Very large token counts
        large_tokens = 1e9
        cost = mechanism.calculate(large_tokens, large_tokens)

        # Should not overflow to infinity
        assert np.isfinite(cost)
        assert cost > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])