"""
Complete Pipeline Example
=========================

Demonstrates the complete workflow from data collection through analysis
to simulation and visualization.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Add parent directory to path if running from examples folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_token_analytics import (
    UnifiedCollector,
    TokenSimulator,
    SimulationConfig,
    TokenAnalyzer,
    SimulationVisualizer
)


def run_complete_pipeline():
    """
    Complete example: Collect data, run simulation, analyze results.
    """
    print("=" * 60)
    print("COMPLETE PIPELINE EXAMPLE")
    print("=" * 60)

    # Step 1: Collect Data
    print("\n[Step 1] Collecting usage data...")

    # Option A: Collect from APIs (requires credentials)
    try:
        collector = UnifiedCollector(['openai', 'anthropic'])
        data = collector.collect_all()

        if not data.empty:
            print(f"Collected {len(data)} records from APIs")
            data.to_parquet("collected_usage.parquet")
    except Exception as e:
        print(f"API collection failed: {e}")
        print("Generating synthetic data instead...")

        # Option B: Generate synthetic data for demonstration
        np.random.seed(42)
        n_samples = 10000

        # Generate correlated token counts
        correlation = 0.35
        mean = [0, 0]
        cov = [[1, correlation], [correlation, 1]]
        normal_samples = np.random.multivariate_normal(mean, cov, n_samples)

        # Transform to lognormal
        data = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=n_samples, freq='5min'),
            'input_tokens': np.exp(4.0 + 0.4 * normal_samples[:, 0]),
            'output_tokens': np.exp(4.3 + 0.6 * normal_samples[:, 1]),
            'provider': np.random.choice(['openai', 'anthropic', 'google'], n_samples),
            'model': np.random.choice(['gpt-4', 'claude-3', 'gemini-pro'], n_samples)
        })

        # Calculate costs
        data['cost'] = (data['input_tokens'] * 0.03 +
                        data['output_tokens'] * 0.06) / 1000

        data.to_parquet("synthetic_usage.parquet")
        print(f"Generated {len(data)} synthetic records")

    # Step 2: Analyze Data
    print("\n[Step 2] Analyzing token distributions...")

    analyzer = TokenAnalyzer(data)

    # Analyze distributions
    dist_results = analyzer.analyze_distributions()

    print("\nDistribution Analysis:")
    for token_type, results in dist_results.items():
        dist = results['distribution']
        stats = results['statistics']
        print(f"\n{token_type.upper()} tokens:")
        print(f"  Best fit: {dist['type']}")
        print(f"  Mean: {stats['mean']:.1f}")
        print(f"  Median: {stats['median']:.1f}")
        print(f"  P95: {stats['p95']:.1f}")
        print(f"  CV: {stats['cv']:.2f}")

    # Analyze correlations
    corr_results = analyzer.analyze_correlations()
    correlations = corr_results['correlations']

    print("\nCorrelation Analysis:")
    print(f"  Pearson: {correlations['linear']['pearson']['value']:.3f}")
    print(f"  Spearman: {correlations['linear']['spearman']['value']:.3f}")
    print(f"  Tail correlation: {correlations['tail_correlation']['upper_tail']:.3f}")

    # Step 3: Run Simulation
    print("\n[Step 3] Running pricing simulation...")

    config = SimulationConfig(
        n_simulations=50_000,  # Reduced for example
        mechanisms=['per_token', 'bundle', 'hybrid', 'cached'],
        use_empirical_data=True,
        empirical_data_path="synthetic_usage.parquet"
    )

    simulator = TokenSimulator(config)
    results = simulator.run()

    print("\nSimulation Results:")
    print("-" * 60)
    print(f"{'Mechanism':<15} {'Mean':<10} {'Median':<10} {'P95':<10} {'CV':<10}")
    print("-" * 60)

    for mechanism, stats in results.mechanism_results.items():
        print(f"{mechanism:<15} "
              f"${stats['mean']:<9.4f} "
              f"${stats['median']:<9.4f} "
              f"${stats['p95']:<9.4f} "
              f"{stats['cv']:<9.2f}")

    # Step 4: Generate Visualizations
    print("\n[Step 4] Creating visualizations...")

    visualizer = SimulationVisualizer(results)
    fig = visualizer.create_comparison_plot()
    fig.write_html("simulation_results.html")
    print("Saved visualization to simulation_results.html")

    # Step 5: Generate Report
    print("\n[Step 5] Generating report...")

    report = {
        'summary': {
            'data_points': len(data),
            'simulation_iterations': config.n_simulations,
            'mechanisms_tested': config.mechanisms
        },
        'distributions': dist_results,
        'correlations': corr_results,
        'simulation_results': results.mechanism_results
    }

    with open('analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print("Report saved to analysis_report.json")

    return results


if __name__ == "__main__":
    run_complete_pipeline()
