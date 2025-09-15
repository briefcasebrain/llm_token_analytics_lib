"""
Sensitivity Analysis Example
============================

Demonstrates how to perform sensitivity analysis on key parameters
to understand their impact on pricing mechanisms.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path if running from examples folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_token_analytics import (
    TokenSimulator,
    SimulationConfig,
    SimulationVisualizer
)


def run_sensitivity_analysis():
    """
    Example of running sensitivity analysis on key parameters.
    """
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS EXAMPLE")
    print("=" * 60)

    # Configure simulation
    config = SimulationConfig(
        n_simulations=10_000,  # Smaller for faster sensitivity analysis
        mechanisms=['per_token', 'bundle', 'hybrid', 'cached']
    )

    simulator = TokenSimulator(config)

    # Define parameters to test
    sensitivity_params = {
        'correlation': np.linspace(0, 0.8, 5),
        'output_variance': np.linspace(0.3, 0.9, 5),
        'cache_hit_rate': np.linspace(0.3, 0.9, 5)
    }

    print("\nRunning sensitivity analysis...")
    print(f"Testing {len(sensitivity_params)} parameters")

    # Run sensitivity analysis
    sensitivity_results = simulator.sensitivity_analysis(sensitivity_params)

    # Display results
    print("\nSensitivity Results:")
    print("-" * 60)

    for param, results in sensitivity_results.items():
        print(f"\n{param.upper()}:")

        for mechanism, data in results.items():
            mean_range = data['mean_range']
            sensitivity = data['mean_sensitivity']

            print(f"  {mechanism}:")
            print(f"    Mean range: ${mean_range[0]:.4f} - ${mean_range[1]:.4f}")
            print(f"    Sensitivity: {sensitivity:.1%}")

    # Create sensitivity plot
    from llm_token_analytics.visualizer import SimulationVisualizer

    # Mock results object for visualization
    class MockResults:
        def __init__(self, sensitivity):
            self.sensitivity_results = sensitivity

    mock_results = MockResults(sensitivity_results)
    visualizer = SimulationVisualizer(mock_results)

    # Note: Would need to add sensitivity plot method to visualizer
    # For now, we'll print the key findings

    print("\nKey Findings:")
    print("-" * 40)

    # Find most sensitive mechanism
    max_sensitivity = 0
    most_sensitive = None

    for param, results in sensitivity_results.items():
        for mechanism, data in results.items():
            if data['mean_sensitivity'] > max_sensitivity:
                max_sensitivity = data['mean_sensitivity']
                most_sensitive = (mechanism, param)

    if most_sensitive:
        print(f"Most sensitive: {most_sensitive[0]} to {most_sensitive[1]} "
              f"({max_sensitivity:.1%} change)")

    # Create detailed sensitivity report
    sensitivity_df = []
    for param, results in sensitivity_results.items():
        for mechanism, data in results.items():
            sensitivity_df.append({
                'parameter': param,
                'mechanism': mechanism,
                'min_mean': data['mean_range'][0],
                'max_mean': data['mean_range'][1],
                'sensitivity': data['mean_sensitivity']
            })

    df = pd.DataFrame(sensitivity_df)
    df = df.sort_values('sensitivity', ascending=False)

    print("\nTop 5 Most Sensitive Combinations:")
    print("-" * 40)
    print(df.head().to_string(index=False))

    # Save results
    df.to_csv('sensitivity_analysis_results.csv', index=False)
    print("\nResults saved to sensitivity_analysis_results.csv")

    return sensitivity_results


def plot_sensitivity_heatmap(sensitivity_results):
    """
    Create a heatmap visualization of sensitivity results.
    """
    import plotly.graph_objects as go

    # Prepare data for heatmap
    params = list(sensitivity_results.keys())
    mechanisms = list(next(iter(sensitivity_results.values())).keys())

    z_data = []
    for param in params:
        row = []
        for mechanism in mechanisms:
            row.append(sensitivity_results[param][mechanism]['mean_sensitivity'])
        z_data.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=mechanisms,
        y=params,
        colorscale='RdBu',
        text=[[f"{val:.1%}" for val in row] for row in z_data],
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="Sensitivity")
    ))

    fig.update_layout(
        title="Sensitivity Analysis Heatmap",
        xaxis_title="Pricing Mechanism",
        yaxis_title="Parameter",
        width=800,
        height=600
    )

    fig.write_html('sensitivity_heatmap.html')
    print("Heatmap saved to sensitivity_heatmap.html")

    return fig


if __name__ == "__main__":
    results = run_sensitivity_analysis()
    plot_sensitivity_heatmap(results)