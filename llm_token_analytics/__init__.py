"""
LLM Token Analytics Library
===========================
A production-ready Python library for collecting, analyzing, and simulating
LLM token usage patterns across OpenAI, Anthropic, and Google AI platforms.
"""

__version__ = "1.0.0"
__author__ = "LLM Token Analytics Team"

from .collectors import (
    CollectorConfig,
    OpenAICollector,
    AnthropicCollector,
    GoogleAICollector,
    UnifiedCollector,
    DataProcessor
)

from .simulator import (
    SimulationConfig,
    SimulationResults,
    TokenSimulator,
    PricingMechanism,
    PerTokenPricing,
    BundlePricing,
    HybridPricing,
    CachedPricing,
    OutcomePricing,
    DynamicPricing
)

from .analyzer import (
    TokenDistribution,
    DistributionFitter,
    CorrelationAnalyzer,
    CostAnalyzer,
    TokenAnalyzer
)

from .visualizer import (
    SimulationVisualizer,
    plot_distributions,
    plot_cost_comparison,
    create_dashboard,
    create_report_plots
)

__all__ = [
    # Version
    "__version__",

    # Collectors
    "CollectorConfig",
    "OpenAICollector",
    "AnthropicCollector",
    "GoogleAICollector",
    "UnifiedCollector",
    "DataProcessor",

    # Simulator
    "SimulationConfig",
    "SimulationResults",
    "TokenSimulator",
    "PricingMechanism",
    "PerTokenPricing",
    "BundlePricing",
    "HybridPricing",
    "CachedPricing",
    "OutcomePricing",
    "DynamicPricing",

    # Analyzer
    "TokenDistribution",
    "DistributionFitter",
    "CorrelationAnalyzer",
    "CostAnalyzer",
    "TokenAnalyzer",

    # Visualizer
    "SimulationVisualizer",
    "plot_distributions",
    "plot_cost_comparison",
    "create_dashboard",
    "create_report_plots"
]
