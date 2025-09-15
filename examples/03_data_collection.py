#!/usr/bin/env python3
"""
Data Collection Example
========================

This example demonstrates how to collect real usage data from LLM providers
and use it for analysis. Note: This requires valid API keys.
"""

import os
import pandas as pd
from datetime import datetime, timedelta

# Import with error handling for optional dependencies
try:
    from llm_token_analytics import UnifiedCollector, TokenAnalyzer
    HAS_PROVIDERS = True
except ImportError:
    print("Provider libraries not installed. Install with:")
    print("   pip install llm-token-analytics[providers]")
    HAS_PROVIDERS = False


def check_api_keys():
    """Check if required API keys are available."""
    keys = {
        'OpenAI': os.getenv('OPENAI_API_KEY'),
        'Anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'Google Cloud': os.getenv('GOOGLE_CLOUD_PROJECT')
    }

    available = {}
    for provider, key in keys.items():
        available[provider] = key is not None
        status = "Available" if key else "Not configured"
        print(f"   {status}: {provider}")

    return available


def collect_sample_data():
    """Collect sample data from available providers."""
    if not HAS_PROVIDERS:
        return create_synthetic_data()

    print("Checking API credentials...")
    available_keys = check_api_keys()

    # Determine which providers to use
    providers = []
    if available_keys.get('OpenAI'):
        providers.append('openai')
    if available_keys.get('Anthropic'):
        providers.append('anthropic')
    if available_keys.get('Google Cloud'):
        providers.append('google')

    if not providers:
        print("\nNo API keys found. Using synthetic data instead.")
        return create_synthetic_data()

    print(f"\nCollecting data from: {', '.join(providers)}")

    try:
        # Initialize collector
        collector = UnifiedCollector(providers)

        # Collect data from the last 7 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        print(f"   Date range: {start_date.date()} to {end_date.date()}")

        # Collect data
        collected_data = collector.collect_all()

        if collected_data.empty:
            print("   No data collected. Using synthetic data instead.")
            return create_synthetic_data()

        print(f"   Collected {len(collected_data)} usage records")

        # Get summary statistics
        stats = collector.get_summary_statistics(collected_data)
        print("\nCollection Summary:")
        print(f"   Total requests: {stats.get('total_requests', 0):,}")
        print(f"   Total input tokens: {stats.get('total_input_tokens', 0):,}")
        print(f"   Total output tokens: {stats.get('total_output_tokens', 0):,}")
        print(f"   Average input tokens: {stats.get('avg_input_tokens', 0):.1f}")
        print(f"   Average output tokens: {stats.get('avg_output_tokens', 0):.1f}")

        return collected_data, stats

    except Exception as e:
        print(f"   Data collection failed: {e}")
        print("   Using synthetic data instead.")
        return create_synthetic_data()


def create_synthetic_data():
    """Create synthetic usage data for demonstration."""
    print("\nGenerating synthetic usage data...")

    import numpy as np

    # Generate realistic token usage patterns
    np.random.seed(42)  # For reproducible results

    n_records = 1000
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=7),
        end=datetime.now(),
        periods=n_records
    )

    # Simulate realistic token distributions
    # Input tokens: log-normal distribution (typical for user queries)
    input_tokens = np.random.lognormal(mean=5.5, sigma=0.8, size=n_records).astype(int)
    input_tokens = np.clip(input_tokens, 10, 8000)  # Reasonable bounds

    # Output tokens: correlated with input but with its own variance
    base_output = input_tokens * 0.6  # Base correlation
    noise = np.random.normal(0, input_tokens * 0.2, size=n_records)
    output_tokens = (base_output + noise).astype(int)
    output_tokens = np.clip(output_tokens, 5, 4000)  # Reasonable bounds

    # Provider distribution
    providers = np.random.choice(
        ['openai', 'anthropic', 'google'],
        size=n_records,
        p=[0.5, 0.3, 0.2]  # Market share approximation
    )

    # Model distribution
    models = []
    for provider in providers:
        if provider == 'openai':
            model = np.random.choice(['gpt-4', 'gpt-3.5-turbo'], p=[0.3, 0.7])
        elif provider == 'anthropic':
            model = np.random.choice(['claude-3-opus', 'claude-3-sonnet'], p=[0.4, 0.6])
        else:  # google
            model = np.random.choice(['gemini-pro', 'gemini-ultra'], p=[0.8, 0.2])
        models.append(model)

    synthetic_data = pd.DataFrame({
        'timestamp': dates,
        'provider': providers,
        'model': models,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': input_tokens + output_tokens,
    })

    # Calculate basic statistics
    stats = {
        'total_requests': len(synthetic_data),
        'total_input_tokens': synthetic_data['input_tokens'].sum(),
        'total_output_tokens': synthetic_data['output_tokens'].sum(),
        'avg_input_tokens': synthetic_data['input_tokens'].mean(),
        'avg_output_tokens': synthetic_data['output_tokens'].mean(),
        'providers': synthetic_data['provider'].unique().tolist()
    }

    print(f"   Generated {len(synthetic_data)} synthetic records")
    print(f"   Average input tokens: {stats['avg_input_tokens']:.1f}")
    print(f"   Average output tokens: {stats['avg_output_tokens']:.1f}")

    return synthetic_data, stats


def analyze_usage_patterns(data):
    """Analyze token usage patterns in the collected data."""
    print("\nAnalyzing Usage Patterns")
    print("-" * 40)

    if not HAS_PROVIDERS:
        print("   Analysis requires the full library installation")
        return

    try:
        # Initialize analyzer
        analyzer = TokenAnalyzer(data)

        # Distribution analysis
        print("1. Analyzing token distributions...")
        dist_results = analyzer.analyze_distributions()

        for token_type, results in dist_results.items():
            print(f"\n   {token_type.replace('_', ' ').title()} Distribution:")
            print(f"      Best fit: {results['distribution']['type']}")
            print(f"      Parameters: {results['distribution']['params']}")
            print(f"      Validation p-value: {results['validation']['ks_pvalue']:.4f}")
            print(f"      Valid fit: {'Yes' if results['validation']['is_valid'] else 'No'}")

        # Correlation analysis
        print("\n2. Analyzing correlations...")
        corr_results = analyzer.analyze_correlations()

        correlations = corr_results['correlations']['linear']
        print("   Input-Output Token Correlations:")
        print(f"      Pearson: {correlations['pearson']['value']:.3f}")
        print(f"      Spearman: {correlations['spearman']['value']:.3f}")
        print(f"      Kendall: {correlations['kendall']['value']:.3f}")

        # Usage by provider
        print("\n3. Usage by provider:")
        provider_stats = data.groupby('provider').agg({
            'input_tokens': ['count', 'mean', 'sum'],
            'output_tokens': ['mean', 'sum']
        }).round(1)

        for provider in data['provider'].unique():
            provider_data = data[data['provider'] == provider]
            print(f"   {provider.title()}:")
            print(f"      Requests: {len(provider_data):,}")
            print(f"      Avg input: {provider_data['input_tokens'].mean():.1f}")
            print(f"      Avg output: {provider_data['output_tokens'].mean():.1f}")
            print(f"      Total tokens: {provider_data['total_tokens'].sum():,}")

    except Exception as e:
        print(f"   Analysis failed: {e}")


def save_data(data, filename=None):
    """Save collected data for later use."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"usage_data_{timestamp}.parquet"

    try:
        data.to_parquet(filename, index=False)
        print(f"\nData saved as: {filename}")
        print(f"   Size: {os.path.getsize(filename) / 1024:.1f} KB")
        print(f"   Records: {len(data):,}")
        return filename
    except Exception as e:
        print(f"Failed to save data: {e}")
        return None


def main():
    """Demonstrate data collection and analysis."""

    print("LLM Token Usage Data Collection Example")
    print("=" * 50)

    # Collect data
    result = collect_sample_data()
    if isinstance(result, tuple):
        data, stats = result
    else:
        data = result
        stats = {}

    # Analyze the data
    analyze_usage_patterns(data)

    # Save the data
    saved_file = save_data(data)

    if saved_file:
        print("\nYou can now use this data for simulations:")
        print("   from llm_token_analytics import TokenSimulator, SimulationConfig")
        print(f"   config = SimulationConfig(empirical_data_path='{saved_file}')")

    print("\nData collection example completed!")


if __name__ == "__main__":
    main()
