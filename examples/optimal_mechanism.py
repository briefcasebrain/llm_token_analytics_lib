"""
Optimal Mechanism Selection Example
====================================

Demonstrates how to select the optimal pricing mechanism
for different user profiles based on their characteristics.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path if running from examples folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_token_analytics import (
    TokenSimulator,
    SimulationConfig,
    CostAnalyzer
)


def select_optimal_mechanism():
    """
    Example of selecting optimal pricing mechanism for different user profiles.
    """
    print("\n" + "=" * 60)
    print("OPTIMAL MECHANISM SELECTION EXAMPLE")
    print("=" * 60)

    # First, run a simulation to get results
    config = SimulationConfig(
        n_simulations=50_000,
        mechanisms=['per_token', 'bundle', 'hybrid', 'cached', 'outcome']
    )

    simulator = TokenSimulator(config)
    results = simulator.run()

    # Define user profiles
    user_profiles = [
        {
            'name': 'Startup',
            'risk_tolerance': 'low',
            'usage_volume': 50_000,
            'predictability_preference': 0.9,
            'budget_constraint': 100
        },
        {
            'name': 'Enterprise',
            'risk_tolerance': 'medium',
            'usage_volume': 1_000_000,
            'predictability_preference': 0.7,
            'budget_constraint': 5000
        },
        {
            'name': 'Researcher',
            'risk_tolerance': 'high',
            'usage_volume': 200_000,
            'predictability_preference': 0.3,
            'budget_constraint': 500
        },
        {
            'name': 'Hobbyist',
            'risk_tolerance': 'low',
            'usage_volume': 10_000,
            'predictability_preference': 0.8,
            'budget_constraint': 20
        },
        {
            'name': 'Agency',
            'risk_tolerance': 'medium',
            'usage_volume': 500_000,
            'predictability_preference': 0.6,
            'budget_constraint': 2000
        },
        {
            'name': 'Developer',
            'risk_tolerance': 'high',
            'usage_volume': 100_000,
            'predictability_preference': 0.4,
            'budget_constraint': 150
        }
    ]

    print("\nUser Profile Analysis:")
    print("-" * 60)

    cost_analyzer = CostAnalyzer(results.mechanism_results)

    recommendations = []

    for profile in user_profiles:
        # Get recommendation
        recommended = cost_analyzer.optimal_mechanism_selection(profile)

        # Get stats for recommended mechanism
        stats = results.mechanism_results[recommended]

        print(f"\n{profile['name']}:")
        print(f"  Usage: {profile['usage_volume']:,} tokens/month")
        print(f"  Risk tolerance: {profile['risk_tolerance']}")
        print(f"  Budget: ${profile['budget_constraint']}")
        print(f"  → Recommended: {recommended.upper()}")
        print(f"     Expected cost: ${stats['mean']:.4f}")
        print(f"     Cost variance (CV): {stats['cv']:.2f}")
        print(f"     95% worst case: ${stats['p95']:.4f}")

        # Calculate potential savings vs per-token
        per_token_mean = results.mechanism_results['per_token']['mean']
        savings = (1 - stats['mean'] / per_token_mean) * 100 if per_token_mean > 0 else 0

        print(f"     Savings vs per-token: {savings:+.1f}%")

        recommendations.append({
            'profile': profile['name'],
            'recommended': recommended,
            'expected_cost': stats['mean'],
            'risk_cv': stats['cv'],
            'p95_cost': stats['p95'],
            'savings_pct': savings
        })

    # Create comparison table
    print("\n" + "=" * 60)
    print("RECOMMENDATION SUMMARY")
    print("=" * 60)

    df = pd.DataFrame(recommendations)
    print(df.to_string(index=False))

    # Save recommendations
    df.to_csv('mechanism_recommendations.csv', index=False)
    print("\nRecommendations saved to mechanism_recommendations.csv")

    # Create detailed analysis
    analyze_mechanism_tradeoffs(results.mechanism_results)

    return recommendations


def analyze_mechanism_tradeoffs(mechanism_results):
    """
    Analyze tradeoffs between different pricing mechanisms.
    """
    print("\n" + "=" * 60)
    print("MECHANISM TRADEOFFS ANALYSIS")
    print("=" * 60)

    # Create comparison matrix
    mechanisms = list(mechanism_results.keys())
    metrics = ['mean', 'median', 'std', 'cv', 'p95', 'var_95', 'cvar_95']

    comparison = []
    for mechanism in mechanisms:
        stats = mechanism_results[mechanism]
        comparison.append({
            'mechanism': mechanism,
            'mean_cost': stats['mean'],
            'predictability': 1 / (1 + stats['cv']),  # Higher is more predictable
            'risk_var95': stats.get('var_95', stats['p95']),
            'efficiency': 1 / stats['mean'] if stats['mean'] > 0 else 0
        })

    df = pd.DataFrame(comparison)

    # Normalize scores
    for col in ['predictability', 'efficiency']:
        df[f'{col}_score'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    # Calculate composite score (equal weighting)
    df['composite_score'] = (df['predictability_score'] + df['efficiency_score']) / 2

    # Sort by composite score
    df = df.sort_values('composite_score', ascending=False)

    print("\nMechanism Rankings:")
    print("-" * 40)
    print(f"{'Rank':<6} {'Mechanism':<15} {'Score':<10} {'Predictability':<15} {'Efficiency'}")
    print("-" * 40)

    for i, row in df.iterrows():
        rank = df.index.get_loc(i) + 1
        print(f"{rank:<6} {row['mechanism']:<15} "
              f"{row['composite_score']:<10.3f} "
              f"{row['predictability_score']:<15.3f} "
              f"{row['efficiency_score']:.3f}")

    # Create visualization
    create_tradeoff_visualization(df)

    return df


def create_tradeoff_visualization(df):
    """
    Create a scatter plot showing the tradeoff between predictability and efficiency.
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    # Add scatter points
    fig.add_trace(go.Scatter(
        x=df['efficiency_score'],
        y=df['predictability_score'],
        mode='markers+text',
        text=df['mechanism'],
        textposition="top center",
        marker=dict(
            size=df['composite_score'] * 20,
            color=df['composite_score'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Composite Score")
        ),
        hovertemplate='<b>%{text}</b><br>' +
                      'Efficiency: %{x:.3f}<br>' +
                      'Predictability: %{y:.3f}<br>' +
                      '<extra></extra>'
    ))

    # Add diagonal line (Pareto frontier approximation)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[1, 0],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.update_layout(
        title="Pricing Mechanism Tradeoffs",
        xaxis_title="Efficiency Score",
        yaxis_title="Predictability Score",
        width=800,
        height=600,
        showlegend=False
    )

    fig.write_html('mechanism_tradeoffs.html')
    print("\nTradeoff visualization saved to mechanism_tradeoffs.html")

    return fig


if __name__ == "__main__":
    select_optimal_mechanism()