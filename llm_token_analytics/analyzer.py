"""
Statistical Analysis Tools for LLM Token Economics
===================================================
Comprehensive analysis framework for token usage patterns and pricing mechanisms.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde
from dataclasses import dataclass
from typing import Dict, Optional
import logging
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class TokenDistribution:
    """Fitted distribution parameters for token usage."""
    distribution_type: str
    parameters: Dict[str, float]
    goodness_of_fit: Dict[str, float]

    def sample(self, n: int) -> np.ndarray:
        """Generate samples from the fitted distribution."""
        if self.distribution_type == 'lognorm':
            return stats.lognorm.rvs(
                s=self.parameters['sigma'],
                scale=np.exp(self.parameters['mu']),
                size=n
            )
        elif self.distribution_type == 'gamma':
            return stats.gamma.rvs(
                a=self.parameters['alpha'],
                scale=self.parameters['scale'],
                size=n
            )
        elif self.distribution_type == 'pareto':
            return stats.pareto.rvs(
                b=self.parameters['alpha'],
                scale=self.parameters['scale'],
                size=n
            )
        elif self.distribution_type == 'weibull':
            return stats.weibull_min.rvs(
                c=self.parameters['shape'],
                scale=self.parameters['scale'],
                size=n
            )
        else:
            raise ValueError(f"Unknown distribution: {self.distribution_type}")


class DistributionFitter:
    """Fit and validate probability distributions to token data."""

    CANDIDATE_DISTRIBUTIONS = [
        'lognorm', 'gamma', 'weibull_min', 'pareto', 'burr12',
        'beta', 'expon', 'norm'
    ]

    def __init__(self, data: np.ndarray):
        self.data = data
        self.results = {}

    def fit_all(self) -> TokenDistribution:
        """Fit all candidate distributions and select the best."""
        logger.info("Fitting distributions to token data...")

        for dist_name in self.CANDIDATE_DISTRIBUTIONS:
            try:
                self.results[dist_name] = self._fit_single(dist_name)
            except Exception as e:
                logger.warning(f"Failed to fit {dist_name}: {e}")
                continue

        best_dist = min(self.results.items(), key=lambda x: x[1]['aic'])
        dist_name, dist_info = best_dist

        logger.info(f"Best distribution: {dist_name} (AIC={dist_info['aic']:.2f})")

        return TokenDistribution(
            distribution_type=dist_name,
            parameters=dist_info['params'],
            goodness_of_fit={
                'aic': dist_info['aic'],
                'bic': dist_info['bic'],
                'ks_pvalue': dist_info['ks_pvalue']
            }
        )

    def _fit_single(self, dist_name: str) -> Dict:
        """Fit a single distribution and calculate goodness of fit."""
        dist = getattr(stats, dist_name)

        params = dist.fit(self.data)

        log_likelihood = np.sum(dist.logpdf(self.data, *params))

        k = len(params)
        n = len(self.data)
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood

        ks_stat, ks_pval = stats.kstest(
            self.data,
            lambda x: dist.cdf(x, *params)
        )

        param_names = dist.shapes.split(', ') if dist.shapes else []
        param_names += ['loc', 'scale']
        param_dict = dict(zip(param_names, params))

        return {
            'params': param_dict,
            'aic': aic,
            'bic': bic,
            'ks_stat': ks_stat,
            'ks_pvalue': ks_pval
        }

    def validate_fit(self, distribution: TokenDistribution) -> Dict:
        """Validate the fitted distribution with visual and statistical tests."""
        fitted_samples = distribution.sample(len(self.data))

        ks_stat, ks_pval = stats.ks_2samp(self.data, fitted_samples)
        ad_result = stats.anderson_ksamp([self.data, fitted_samples])
        wasserstein = stats.wasserstein_distance(self.data, fitted_samples)

        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        empirical_percentiles = np.percentile(self.data, percentiles)
        fitted_percentiles = np.percentile(fitted_samples, percentiles)

        percentile_errors = np.abs(
            (fitted_percentiles - empirical_percentiles) / empirical_percentiles
        )

        return {
            'ks_test': {'statistic': ks_stat, 'p_value': ks_pval},
            'anderson_darling': {
                'statistic': ad_result.statistic,
                'p_value': ad_result.pvalue
            },
            'wasserstein_distance': wasserstein,
            'percentile_errors': dict(zip(percentiles, percentile_errors)),
            'mean_percentile_error': np.mean(percentile_errors),
            'is_valid': ks_pval > 0.05 and np.mean(percentile_errors) < 0.1
        }


class CorrelationAnalyzer:
    """Model correlation structure between input and output tokens."""

    def __init__(self, input_tokens: np.ndarray, output_tokens: np.ndarray):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def fit_copula(self, copula_type: str = 'gaussian') -> gaussian_kde:
        """Fit a copula to capture non-linear dependencies."""
        n = len(self.input_tokens)
        u = stats.rankdata(self.input_tokens) / (n + 1)
        v = stats.rankdata(self.output_tokens) / (n + 1)

        if copula_type == 'gaussian':
            copula = gaussian_kde(np.vstack([u, v]))
        elif copula_type == 'clayton':
            copula = self._fit_clayton_copula(u, v)
        elif copula_type == 'gumbel':
            copula = self._fit_gumbel_copula(u, v)
        else:
            raise ValueError(f"Unknown copula type: {copula_type}")

        return copula

    def _fit_clayton_copula(self, u: np.ndarray, v: np.ndarray) -> Dict:
        """Fit Clayton copula for lower tail dependence."""
        tau, _ = stats.kendalltau(u, v)
        theta = 2 * tau / (1 - tau) if tau < 1 else 10

        return {
            'type': 'clayton',
            'theta': theta,
            'tau': tau
        }

    def _fit_gumbel_copula(self, u: np.ndarray, v: np.ndarray) -> Dict:
        """Fit Gumbel copula for upper tail dependence."""
        tau, _ = stats.kendalltau(u, v)
        theta = 1 / (1 - tau) if tau < 1 else 10

        return {
            'type': 'gumbel',
            'theta': theta,
            'tau': tau
        }

    def analyze_correlations(self) -> Dict:
        """Comprehensive correlation analysis."""
        pearson_corr, pearson_pval = stats.pearsonr(
            self.input_tokens, self.output_tokens
        )
        spearman_corr, spearman_pval = stats.spearmanr(
            self.input_tokens, self.output_tokens
        )
        kendall_corr, kendall_pval = stats.kendalltau(
            self.input_tokens, self.output_tokens
        )

        threshold = np.percentile(self.input_tokens, 95)
        tail_mask = self.input_tokens > threshold
        if np.sum(tail_mask) > 10:
            tail_corr = np.corrcoef(
                self.input_tokens[tail_mask],
                self.output_tokens[tail_mask]
            )[0, 1]
        else:
            tail_corr = np.nan

        threshold_low = np.percentile(self.input_tokens, 5)
        lower_tail_mask = self.input_tokens < threshold_low
        if np.sum(lower_tail_mask) > 10:
            lower_tail_corr = np.corrcoef(
                self.input_tokens[lower_tail_mask],
                self.output_tokens[lower_tail_mask]
            )[0, 1]
        else:
            lower_tail_corr = np.nan

        return {
            'linear': {
                'pearson': {'value': pearson_corr, 'p_value': pearson_pval},
                'spearman': {'value': spearman_corr, 'p_value': spearman_pval},
                'kendall': {'value': kendall_corr, 'p_value': kendall_pval}
            },
            'tail_dependence': {
                'upper_tail': tail_corr,
                'lower_tail': lower_tail_corr
            },
            'copulas': {
                'gaussian': self.fit_copula('gaussian'),
                'clayton': self._fit_clayton_copula(
                    stats.rankdata(self.input_tokens) / (len(self.input_tokens) + 1),
                    stats.rankdata(self.output_tokens) / (len(self.output_tokens) + 1)
                ),
                'gumbel': self._fit_gumbel_copula(
                    stats.rankdata(self.input_tokens) / (len(self.input_tokens) + 1),
                    stats.rankdata(self.output_tokens) / (len(self.output_tokens) + 1)
                )
            }
        }


class CostAnalyzer:
    """Analyze costs and risk metrics for pricing mechanisms."""

    def __init__(self, mechanism_results: Dict[str, Dict]):
        self.results = mechanism_results

    def calculate_risk_metrics(self) -> pd.DataFrame:
        """Calculate comprehensive risk metrics for each mechanism."""
        risk_metrics = []

        for mechanism, stats in self.results.items():
            percentiles = stats.get('percentiles', {})

            var_95 = percentiles.get(95, 0) if percentiles else 0
            cvar_95 = self._calculate_cvar(stats, 0.95)

            sharpe_ratio = (stats['mean'] - 0) / stats['std'] if stats['std'] > 0 else 0

            max_drawdown = self._calculate_max_drawdown(stats)

            risk_metrics.append({
                'mechanism': mechanism,
                'mean': stats['mean'],
                'std': stats['std'],
                'cv': stats['cv'],
                'var_95': var_95,
                'cvar_95': cvar_95,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'tail_ratio': stats.get('tail_ratio', 0),
                'skewness': stats.get('skewness', 0),
                'kurtosis': stats.get('kurtosis', 0)
            })

        return pd.DataFrame(risk_metrics)

    def _calculate_cvar(self, stats: Dict, confidence: float) -> float:
        """Calculate Conditional Value at Risk (CVaR)."""
        percentile = int(confidence * 100)
        var = stats['percentiles'].get(percentile, 0) if 'percentiles' in stats else 0

        if percentile == 99:
            return stats['percentiles'].get(99, 0) if 'percentiles' in stats else 0
        elif percentile == 95:
            p95 = stats['percentiles'].get(95, 0) if 'percentiles' in stats else 0
            p99 = stats['percentiles'].get(99, 0) if 'percentiles' in stats else 0
            return (p95 + p99) / 2

        return var * 1.1

    def _calculate_max_drawdown(self, stats: Dict) -> float:
        """Estimate maximum drawdown from statistics."""
        return stats['std'] * 2.5 / stats['mean'] if stats['mean'] > 0 else 0

    def optimal_mechanism_selection(self, user_profile: Dict) -> str:
        """Select optimal mechanism based on user profile."""
        risk_tolerance = user_profile.get('risk_tolerance', 'medium')
        usage_volume = user_profile.get('usage_volume', 100_000)
        predictability_pref = user_profile.get('predictability_preference', 0.5)
        budget_constraint = user_profile.get('budget_constraint', None)

        risk_metrics = self.calculate_risk_metrics()

        scores = {}
        for _, row in risk_metrics.iterrows():
            mechanism = row['mechanism']
            score = 0

            if risk_tolerance == 'low':
                score -= row['cv'] * 2
                score -= row['tail_ratio']
                score += (1 - row['std'] / row['mean']) * predictability_pref
            elif risk_tolerance == 'high':
                score += row['sharpe_ratio']
                score -= row['mean'] / 100

            if budget_constraint:
                if row['mean'] * usage_volume / 1000 > budget_constraint:
                    score -= 1000

            scores[mechanism] = score

        return max(scores.items(), key=lambda x: x[1])[0]


class TokenAnalyzer:
    """Main interface for comprehensive token analytics."""

    def __init__(self, data: Optional[pd.DataFrame] = None):
        self.data = data
        self.distribution_fitter = None
        self.correlation_analyzer = None
        self.cost_analyzer = None

    def fit_distributions(self, input_tokens: np.ndarray,
                          output_tokens: np.ndarray) -> Dict:
        """Fit distributions to token data."""
        results = {}

        input_fitter = DistributionFitter(input_tokens)
        results['input'] = input_fitter.fit_all()

        output_fitter = DistributionFitter(output_tokens)
        results['output'] = output_fitter.fit_all()

        return results

    def analyze_correlations(self, input_tokens: np.ndarray,
                             output_tokens: np.ndarray) -> Dict:
        """Analyze correlation structure."""
        analyzer = CorrelationAnalyzer(input_tokens, output_tokens)
        return analyzer.analyze_correlations()

    def analyze_costs(self, mechanism_results: Dict) -> pd.DataFrame:
        """Analyze cost and risk metrics."""
        analyzer = CostAnalyzer(mechanism_results)
        return analyzer.calculate_risk_metrics()

    def full_analysis(self, input_tokens: np.ndarray,
                      output_tokens: np.ndarray,
                      mechanism_results: Dict) -> Dict:
        """Perform comprehensive analysis."""
        return {
            'distributions': self.fit_distributions(input_tokens, output_tokens),
            'correlations': self.analyze_correlations(input_tokens, output_tokens),
            'risk_metrics': self.analyze_costs(mechanism_results),
            'summary_statistics': {
                'input_tokens': {
                    'mean': np.mean(input_tokens),
                    'median': np.median(input_tokens),
                    'std': np.std(input_tokens),
                    'min': np.min(input_tokens),
                    'max': np.max(input_tokens)
                },
                'output_tokens': {
                    'mean': np.mean(output_tokens),
                    'median': np.median(output_tokens),
                    'std': np.std(output_tokens),
                    'min': np.min(output_tokens),
                    'max': np.max(output_tokens)
                }
            }
        }
