"""
Contract Enforcement Tests
=========================
Tests to ensure abstract base classes and interface contracts are properly enforced.
"""

import pytest
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import inspect
from unittest.mock import Mock, patch

from llm_token_analytics.collectors import BaseCollector, CollectorConfig
from llm_token_analytics.simulator import PricingMechanism, TokenSimulator
from llm_token_analytics.analyzer import DistributionFitter, CorrelationAnalyzer


class TestBaseCollectorContract:
    """Test BaseCollector abstract base class contract enforcement."""

    def test_cannot_instantiate_base_collector(self):
        """Test that BaseCollector cannot be instantiated directly."""
        config = CollectorConfig(api_key="test", provider="test")

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseCollector(config)

    def test_concrete_collector_must_implement_abstract_methods(self):
        """Test that concrete collectors must implement all abstract methods."""

        class IncompleteCollector(BaseCollector):
            """Incomplete collector missing required methods."""

            def _get_headers(self) -> Dict[str, str]:
                return {"Authorization": "Bearer test"}

            # Missing collect() method

        config = CollectorConfig(api_key="test", provider="test")

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteCollector(config)

    def test_complete_collector_implementation(self):
        """Test that a complete collector implementation works."""

        class CompleteCollector(BaseCollector):
            """Complete collector with all required methods."""

            def _get_headers(self) -> Dict[str, str]:
                return {"Authorization": f"Bearer {self.config.api_key}"}

            def collect(self) -> pd.DataFrame:
                return pd.DataFrame({
                    'timestamp': [pd.Timestamp.now()],
                    'input_tokens': [100],
                    'output_tokens': [50],
                    'provider': [self.config.provider]
                })

        config = CollectorConfig(api_key="test", provider="test")
        collector = CompleteCollector(config)

        # Should be able to instantiate and call methods
        assert isinstance(collector, BaseCollector)
        headers = collector._get_headers()
        assert isinstance(headers, dict)
        assert "Authorization" in headers

        data = collector.collect()
        assert isinstance(data, pd.DataFrame)
        assert not data.empty

    def test_abstract_method_signatures(self):
        """Test that abstract methods have correct signatures."""
        # Get abstract methods from BaseCollector
        abstract_methods = BaseCollector.__abstractmethods__

        assert '_get_headers' in abstract_methods
        assert 'collect' in abstract_methods

        # Check method signatures
        get_headers_method = getattr(BaseCollector, '_get_headers')
        collect_method = getattr(BaseCollector, 'collect')

        # These should be abstract methods
        assert hasattr(get_headers_method, '__isabstractmethod__')
        assert hasattr(collect_method, '__isabstractmethod__')


class TestPricingMechanismContract:
    """Test PricingMechanism base class contract enforcement."""

    def test_cannot_instantiate_base_pricing_mechanism(self):
        """Test that PricingMechanism base class enforces implementation."""
        config = {"input_price": 0.03, "output_price": 0.06}

        # Should be able to create instance
        mechanism = PricingMechanism(config)

        # But calling calculate should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            mechanism.calculate(100, 50)

    def test_concrete_pricing_mechanism_implementation(self):
        """Test that concrete pricing mechanisms work correctly."""
        from llm_token_analytics.simulator import PerTokenPricing

        config = {"input_price": 30, "output_price": 60}  # per 1k tokens
        mechanism = PerTokenPricing(config)

        # Should be able to calculate
        cost = mechanism.calculate(1000, 500)  # 1k input, 0.5k output

        expected = (1000 * 30 / 1000) + (500 * 60 / 1000)  # $30 + $30 = $60
        assert abs(cost - 60.0) < 0.001

    def test_pricing_mechanism_inheritance(self):
        """Test that all pricing mechanisms inherit from base class."""
        from llm_token_analytics.simulator import (
            PerTokenPricing, BundlePricing, HybridPricing,
            CachedPricing, OutcomePricing, DynamicPricing
        )

        pricing_classes = [
            PerTokenPricing, BundlePricing, HybridPricing,
            CachedPricing, OutcomePricing, DynamicPricing
        ]

        for pricing_class in pricing_classes:
            assert issubclass(pricing_class, PricingMechanism)

    def test_pricing_mechanism_interface_compliance(self):
        """Test that pricing mechanisms comply with interface."""
        from llm_token_analytics.simulator import PerTokenPricing

        config = {"input_price": 30, "output_price": 60}
        mechanism = PerTokenPricing(config)

        # Check required attributes
        assert hasattr(mechanism, 'config')
        assert hasattr(mechanism, 'calculate')

        # Check method signature
        sig = inspect.signature(mechanism.calculate)
        params = list(sig.parameters.keys())
        assert 'input_tokens' in params
        assert 'output_tokens' in params

        # Check return type behavior
        result = mechanism.calculate(100.0, 50.0)
        assert isinstance(result, (int, float))
        assert result >= 0  # Costs should be non-negative

    def test_custom_pricing_mechanism(self):
        """Test implementing a custom pricing mechanism."""

        class CustomFixedPricing(PricingMechanism):
            """Custom pricing mechanism with fixed cost per request."""

            def calculate(self, input_tokens: float, output_tokens: float) -> float:
                fixed_cost = self.config.get('fixed_cost', 1.0)
                token_multiplier = self.config.get('token_multiplier', 0.001)
                return fixed_cost + (input_tokens + output_tokens) * token_multiplier

        config = {"fixed_cost": 5.0, "token_multiplier": 0.002}
        custom_mechanism = CustomFixedPricing(config)

        cost = custom_mechanism.calculate(1000, 500)
        expected = 5.0 + (1000 + 500) * 0.002  # 5.0 + 3.0 = 8.0
        assert abs(cost - 8.0) < 0.001


class TestMethodSignatureEnforcement:
    """Test method signature compliance across the library."""

    def test_collector_method_signatures(self):
        """Test that collector methods have consistent signatures."""
        from llm_token_analytics.collectors import OpenAICollector, AnthropicCollector

        collectors = [OpenAICollector, AnthropicCollector]

        for collector_class in collectors:
            # Check collect method signature
            collect_sig = inspect.signature(collector_class.collect)
            assert len(collect_sig.parameters) == 1  # Just self
            assert collect_sig.return_annotation in [pd.DataFrame, inspect.Parameter.empty]

            # Check _get_headers method signature
            headers_sig = inspect.signature(collector_class._get_headers)
            assert len(headers_sig.parameters) == 1  # Just self

    def test_simulator_method_signatures(self):
        """Test TokenSimulator method signatures."""
        from llm_token_analytics.simulator import TokenSimulator, SimulationConfig

        config = SimulationConfig(n_simulations=100)
        simulator = TokenSimulator(config)

        # Check run method - bound methods don't include 'self' in signature
        run_sig = inspect.signature(simulator.run)
        assert len(run_sig.parameters) == 0  # No parameters for bound method

        # Check generate_token_samples method
        samples_sig = inspect.signature(simulator.generate_token_samples)
        params = list(samples_sig.parameters.keys())
        assert 'n' in params

    def test_analyzer_method_signatures(self):
        """Test analyzer method signatures."""
        from llm_token_analytics.analyzer import TokenAnalyzer

        analyzer = TokenAnalyzer()

        # Check method signatures
        fit_sig = inspect.signature(analyzer.fit_distributions)
        params = list(fit_sig.parameters.keys())
        assert 'input_tokens' in params
        assert 'output_tokens' in params

        corr_sig = inspect.signature(analyzer.analyze_correlations)
        params = list(corr_sig.parameters.keys())
        assert 'input_tokens' in params
        assert 'output_tokens' in params


class TestInterfaceConsistency:
    """Test consistency of interfaces across similar classes."""

    def test_pricing_mechanism_consistency(self):
        """Test that all pricing mechanisms have consistent interfaces."""
        from llm_token_analytics.simulator import (
            PerTokenPricing, BundlePricing, HybridPricing, CachedPricing
        )

        mechanisms = [PerTokenPricing, BundlePricing, HybridPricing, CachedPricing]
        base_config = {"input_price": 30, "output_price": 60}

        for mechanism_class in mechanisms:
            # Add mechanism-specific config
            config = base_config.copy()
            if mechanism_class == BundlePricing:
                config.update({"bundle_size": 100_000, "bundle_price": 5.0})
            elif mechanism_class == HybridPricing:
                config.update({"seat_cost": 30, "included_tokens": 50_000})
            elif mechanism_class == CachedPricing:
                config.update({"cache_hit_rate": 0.7, "cache_discount": 0.8})

            mechanism = mechanism_class(config)

            # Test calculate method exists and works
            assert hasattr(mechanism, 'calculate')
            result = mechanism.calculate(1000, 500)
            assert isinstance(result, (int, float))
            assert result >= 0

            # Test consistent method signature - bound methods don't include 'self'
            sig = inspect.signature(mechanism.calculate)
            params = list(sig.parameters.keys())
            assert params == ['input_tokens', 'output_tokens']

    def test_collector_consistency(self):
        """Test that all collectors have consistent interfaces."""
        from llm_token_analytics.collectors import OpenAICollector, AnthropicCollector

        collectors = [OpenAICollector, AnthropicCollector]

        for collector_class in collectors:
            # Check that they all inherit from BaseCollector
            assert issubclass(collector_class, BaseCollector)

            # Check consistent constructor signature
            init_sig = inspect.signature(collector_class.__init__)
            params = list(init_sig.parameters.keys())
            assert 'config' in params

    def test_return_type_consistency(self):
        """Test that methods return consistent types."""
        from llm_token_analytics.simulator import SimulationConfig, TokenSimulator
        from llm_token_analytics.analyzer import DistributionFitter

        # Test SimulationResults consistency
        config = SimulationConfig(n_simulations=100)
        simulator = TokenSimulator(config)
        results = simulator.run()

        assert hasattr(results, 'mechanism_results')
        assert hasattr(results, 'comparison')
        assert hasattr(results, 'metadata')
        assert hasattr(results, 'to_dataframe')

        # Test DistributionFitter consistency
        data = np.random.lognormal(4.0, 0.5, 1000)
        fitter = DistributionFitter(data)
        distribution = fitter.fit_all()

        assert hasattr(distribution, 'distribution_type')
        assert hasattr(distribution, 'parameters')
        assert hasattr(distribution, 'goodness_of_fit')
        assert hasattr(distribution, 'sample')


class TestErrorHandlingContracts:
    """Test that error handling follows consistent patterns."""

    def test_configuration_validation(self):
        """Test that configuration validation is consistent."""
        from llm_token_analytics.collectors import CollectorConfig

        # Test invalid provider
        with pytest.raises(ValueError, match="Unknown provider"):
            CollectorConfig.from_env("invalid_provider")

    def test_data_validation_contracts(self):
        """Test that data validation follows consistent patterns."""
        from llm_token_analytics.analyzer import TokenDistribution

        # Test invalid distribution type
        distribution = TokenDistribution(
            distribution_type="invalid",
            parameters={"param": 1.0},
            goodness_of_fit={}
        )

        with pytest.raises(ValueError, match="Unknown distribution"):
            distribution.sample(100)

    def test_input_validation_patterns(self):
        """Test consistent input validation patterns."""
        from llm_token_analytics.simulator import PerTokenPricing

        mechanism = PerTokenPricing({})

        # Test with various input types - should handle gracefully
        result1 = mechanism.calculate(1000, 500)
        result2 = mechanism.calculate(1000.0, 500.0)

        assert isinstance(result1, (int, float))
        assert isinstance(result2, (int, float))

        # Test with negative values - behavior should be consistent
        result3 = mechanism.calculate(-100, 500)
        # The mechanism doesn't validate negative inputs, but should return a number
        assert isinstance(result3, (int, float))


class TestPolymorphism:
    """Test polymorphic behavior of interfaces."""

    def test_pricing_mechanism_polymorphism(self):
        """Test that pricing mechanisms can be used polymorphically."""
        from llm_token_analytics.simulator import PerTokenPricing, BundlePricing

        base_config = {"input_price": 30, "output_price": 60}
        bundle_config = {**base_config, "bundle_size": 100_000, "bundle_price": 5.0}

        mechanisms = [
            PerTokenPricing(base_config),
            BundlePricing(bundle_config)
        ]

        # Test polymorphic usage
        for mechanism in mechanisms:
            assert isinstance(mechanism, PricingMechanism)
            cost = mechanism.calculate(1000, 500)
            assert isinstance(cost, (int, float))
            assert cost >= 0

    def test_collector_polymorphism(self):
        """Test that collectors can be used polymorphically."""
        from llm_token_analytics.collectors import DataProcessor

        # Test with mock data that could come from any collector
        data1 = pd.DataFrame({
            'input_tokens': [100, 200, 300],
            'output_tokens': [50, 100, 150],
            'provider': ['openai', 'openai', 'openai']
        })

        data2 = pd.DataFrame({
            'input_tokens': [150, 250],
            'output_tokens': [75, 125],
            'provider': ['anthropic', 'anthropic']
        })

        processor = DataProcessor()

        # Should work with data from any provider
        clean1 = processor.clean_data(data1)
        clean2 = processor.clean_data(data2)

        assert isinstance(clean1, pd.DataFrame)
        assert isinstance(clean2, pd.DataFrame)