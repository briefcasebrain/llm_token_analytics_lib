"""
Monte Carlo Simulation Engine for LLM Token Pricing
====================================================
Production-grade simulation framework for token economics research.
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple
import logging
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for robust simulation runs."""
    n_simulations: int = 100_000
    confidence_level: float = 0.95
    seed: int = 42
    providers: List[str] = None
    mechanisms: List[str] = None
    use_empirical_data: bool = True
    data_path: str = "./data"
    output_path: str = "./results"
    chunk_size: int = 10_000

    def __post_init__(self):
        if self.providers is None:
            self.providers = ['openai', 'anthropic', 'google']
        if self.mechanisms is None:
            self.mechanisms = ['per_token', 'bundle', 'hybrid', 'cached', 'outcome', 'dynamic']

        Path(self.data_path).mkdir(parents=True, exist_ok=True)
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        np.random.seed(self.seed)


@dataclass
class SimulationResults:
    """Structured results from simulation runs."""
    mechanism_results: Dict[str, Dict]
    comparison: pd.DataFrame
    metadata: Dict

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for analysis."""
        df_data = []
        for mechanism, stats in self.mechanism_results.items():
            row = {
                'mechanism': mechanism,
                'mean': stats['mean'],
                'median': stats['median'],
                'std': stats['std'],
                'cv': stats['cv'],
                'p95': stats['p95'],
                'p99': stats['p99'],
                'tail_ratio': stats['tail_ratio']
            }
            df_data.append(row)
        return pd.DataFrame(df_data)


class PricingMechanism:
    """Base class for pricing mechanisms."""

    def __init__(self, config: Dict):
        self.config = config

    def calculate(self, input_tokens: float, output_tokens: float) -> float:
        raise NotImplementedError


class PerTokenPricing(PricingMechanism):
    """Standard per-token pricing."""

    def calculate(self, input_tokens: float, output_tokens: float) -> float:
        input_price = self.config.get('input_price', 0.03) / 1000
        output_price = self.config.get('output_price', 0.06) / 1000
        return input_tokens * input_price + output_tokens * output_price


class BundlePricing(PricingMechanism):
    """Bundle pricing with fixed allocation."""

    def calculate(self, input_tokens: float, output_tokens: float) -> float:
        total_tokens = input_tokens + output_tokens
        bundle_size = self.config.get('bundle_size', 100_000)
        bundle_price = self.config.get('bundle_price', 5.0)

        if total_tokens <= bundle_size:
            return bundle_price / bundle_size * total_tokens
        else:
            overage_rate = self.config.get('overage_rate', 0.08) / 1000
            return bundle_price + (total_tokens - bundle_size) * overage_rate


class HybridPricing(PricingMechanism):
    """Seat-based + usage pricing."""

    def calculate(self, input_tokens: float, output_tokens: float) -> float:
        seat_cost = self.config.get('seat_cost', 30.0)
        included_tokens = self.config.get('included_tokens', 50_000)
        total_tokens = input_tokens + output_tokens

        if total_tokens <= included_tokens:
            return seat_cost / included_tokens * total_tokens
        else:
            overage_rate = self.config.get('overage_rate', 0.05) / 1000
            return seat_cost + (total_tokens - included_tokens) * overage_rate


class CachedPricing(PricingMechanism):
    """Pricing with cache discounts."""

    def calculate(self, input_tokens: float, output_tokens: float) -> float:
        cache_hit_rate = self.config.get('cache_hit_rate', 0.7)
        cache_discount = self.config.get('cache_discount', 0.8)

        cached_input = input_tokens * cache_hit_rate
        uncached_input = input_tokens * (1 - cache_hit_rate)

        input_price = self.config.get('input_price', 0.03) / 1000
        output_price = self.config.get('output_price', 0.06) / 1000

        cached_cost = cached_input * input_price * (1 - cache_discount)
        uncached_cost = uncached_input * input_price
        output_cost = output_tokens * output_price

        return cached_cost + uncached_cost + output_cost


class OutcomePricing(PricingMechanism):
    """Outcome-based pricing with success metrics."""

    def calculate(self, input_tokens: float, output_tokens: float) -> float:
        base_rate = self.config.get('base_rate', 0.02) / 1000
        success_multiplier = self.config.get('success_multiplier', 2.0)
        success_rate = self.config.get('success_rate', 0.85)

        total_tokens = input_tokens + output_tokens
        base_cost = total_tokens * base_rate

        if np.random.random() < success_rate:
            return base_cost * success_multiplier
        else:
            return base_cost


class DynamicPricing(PricingMechanism):
    """Dynamic congestion-based pricing."""

    def calculate(self, input_tokens: float, output_tokens: float) -> float:
        base_price = self.config.get('base_price', 0.04) / 1000
        peak_multiplier = self.config.get('peak_multiplier', 1.5)
        congestion_threshold = self.config.get('congestion_threshold', 0.7)

        total_tokens = input_tokens + output_tokens

        congestion_level = np.random.beta(2, 5)

        if congestion_level > congestion_threshold:
            price_multiplier = 1 + (peak_multiplier - 1) * (
                (congestion_level - congestion_threshold) / (1 - congestion_threshold)
            )
        else:
            price_multiplier = 1.0

        return total_tokens * base_price * price_multiplier


class TokenSimulator:
    """Production-grade token pricing simulator."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.pricing_mechanisms = {}
        self._setup_pricing_mechanisms()

    def _setup_pricing_mechanisms(self):
        """Initialize pricing mechanisms."""
        base_config = {
            'input_price': 30,
            'output_price': 60,
        }

        self.pricing_mechanisms = {
            'per_token': PerTokenPricing(base_config),
            'bundle': BundlePricing({**base_config, 'bundle_size': 100_000, 'bundle_price': 5.0}),
            'hybrid': HybridPricing({**base_config, 'seat_cost': 30, 'included_tokens': 50_000}),
            'cached': CachedPricing({**base_config, 'cache_hit_rate': 0.7, 'cache_discount': 0.8}),
            'outcome': OutcomePricing({**base_config, 'success_rate': 0.85}),
            'dynamic': DynamicPricing({**base_config, 'peak_multiplier': 1.5})
        }

    def generate_token_samples(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate correlated token samples using copula."""
        correlation = 0.35

        mean = [0, 0]
        cov = [[1, correlation], [correlation, 1]]
        normal_samples = np.random.multivariate_normal(mean, cov, n)

        u = stats.norm.cdf(normal_samples[:, 0])
        v = stats.norm.cdf(normal_samples[:, 1])

        input_tokens = stats.lognorm.ppf(u, s=0.4, scale=np.exp(4.0))
        output_tokens = stats.lognorm.ppf(v, s=0.6, scale=np.exp(4.3))

        return input_tokens, output_tokens

    def run(self) -> SimulationResults:
        """Run full simulation with all mechanisms."""
        n = self.config.n_simulations
        logger.info(f"Running simulation with {n:,} iterations...")

        input_tokens, output_tokens = self.generate_token_samples(n)

        results = {}

        for mechanism_name in self.config.mechanisms:
            if mechanism_name in self.pricing_mechanisms:
                mechanism = self.pricing_mechanisms[mechanism_name]
                costs = np.array([
                    mechanism.calculate(inp, out)
                    for inp, out in zip(input_tokens, output_tokens)
                ])

                results[mechanism_name] = self._calculate_statistics(costs)

        comparison = self._compare_mechanisms(results)

        return SimulationResults(
            mechanism_results=results,
            comparison=comparison,
            metadata={
                'n_simulations': n,
                'mechanisms': list(results.keys()),
                'timestamp': pd.Timestamp.now().isoformat()
            }
        )

    def _calculate_statistics(self, costs: np.ndarray) -> Dict:
        """Calculate comprehensive statistics."""
        mean = np.mean(costs)
        median = np.median(costs)
        std = np.std(costs)

        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = np.percentile(costs, percentiles)

        ci_mean = self._bootstrap_ci(costs, np.mean)
        ci_median = self._bootstrap_ci(costs, np.median)
        ci_p95 = self._bootstrap_ci(costs, lambda x: np.percentile(x, 95))

        cv = std / mean if mean > 0 else np.inf
        var_95 = np.percentile(costs, 95) - median
        tail_ratio = np.percentile(costs, 95) / median if median > 0 else np.inf

        return {
            'mean': mean,
            'mean_ci': ci_mean,
            'median': median,
            'median_ci': ci_median,
            'std': std,
            'cv': cv,
            'percentiles': dict(zip(percentiles, percentile_values)),
            'p95': percentile_values[percentiles.index(95)],
            'p95_ci': ci_p95,
            'p99': percentile_values[percentiles.index(99)],
            'var_95': var_95,
            'tail_ratio': tail_ratio,
            'skewness': stats.skew(costs),
            'kurtosis': stats.kurtosis(costs)
        }

    def _bootstrap_ci(self, data: np.ndarray, statistic,
                      n_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        bootstrap_samples = []
        n = len(data)

        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_samples.append(statistic(sample))

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_samples, 100 * alpha / 2)
        upper = np.percentile(bootstrap_samples, 100 * (1 - alpha / 2))

        return (lower, upper)

    def _compare_mechanisms(self, results: Dict) -> pd.DataFrame:
        """Compare mechanisms across key metrics."""
        comparison = pd.DataFrame()

        for mechanism, stats in results.items():
            comparison[mechanism] = {
                'mean': stats['mean'],
                'median': stats['median'],
                'cv': stats['cv'],
                'p95': stats['p95'],
                'tail_ratio': stats['tail_ratio']
            }

        if 'per_token' in comparison.columns:
            baseline = comparison['per_token']
            for col in comparison.columns:
                if col != 'per_token':
                    comparison[f'{col}_vs_baseline'] = (
                        (comparison[col] - baseline) / baseline * 100
                    )

        return comparison

    def sensitivity_analysis(self, param_ranges: Dict) -> Dict:
        """Perform sensitivity analysis on key parameters."""
        logger.info("Running sensitivity analysis...")

        base_results = self.run()
        sensitivities = {}

        for param, values in param_ranges.items():
            param_results = []

            for value in tqdm(values, desc=f"Testing {param}"):
                if param == 'cache_hit_rate':
                    self.pricing_mechanisms['cached'].config['cache_hit_rate'] = value
                elif param == 'bundle_size':
                    self.pricing_mechanisms['bundle'].config['bundle_size'] = value

                results = self.run()
                param_results.append(results.mechanism_results)

                if param == 'cache_hit_rate':
                    self.pricing_mechanisms['cached'].config['cache_hit_rate'] = 0.7
                elif param == 'bundle_size':
                    self.pricing_mechanisms['bundle'].config['bundle_size'] = 100_000

            sensitivities[param] = self._analyze_sensitivity(
                param_results, values, base_results.mechanism_results
            )

        return sensitivities

    def _analyze_sensitivity(self, results: List[Dict], values: List[float],
                             base: Dict) -> Dict:
        """Analyze sensitivity results."""
        sensitivity = {}

        for mechanism in self.pricing_mechanisms.keys():
            if mechanism in results[0]:
                means = [r[mechanism]['mean'] for r in results]
                p95s = [r[mechanism]['p95'] for r in results]

                base_mean = base[mechanism]['mean'] if mechanism in base else means[0]
                mean_sensitivity = max(means) / base_mean - 1 if base_mean > 0 else 0

                sensitivity[mechanism] = {
                    'values': values,
                    'means': means,
                    'p95s': p95s,
                    'mean_sensitivity': mean_sensitivity,
                    'range': (min(means), max(means))
                }

        return sensitivity
