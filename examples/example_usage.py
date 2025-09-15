"""
Example Scripts for LLM Token Analytics
========================================

Complete examples demonstrating common use cases.
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path if running from examples folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_token_analytics import (
    UnifiedCollector,
    CollectorConfig,
    TokenSimulator,
    SimulationConfig,
    TokenAnalyzer,
    SimulationVisualizer,
    create_dashboard
)


# ============================================================================
# EXAMPLE 1: Complete Pipeline from Collection to Analysis
# ============================================================================

def example_complete_pipeline():
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
    
    import json
    with open('analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("Report saved to analysis_report.json")
    
    return results


# ============================================================================
# EXAMPLE 2: Sensitivity Analysis
# ============================================================================

def example_sensitivity_analysis():
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
    
    return sensitivity_results


# ============================================================================
# EXAMPLE 3: Optimal Mechanism Selection
# ============================================================================

def example_optimal_mechanism_selection():
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
        }
    ]
    
    print("\nUser Profile Analysis:")
    print("-" * 60)
    
    from llm_token_analytics.analyzer import CostAnalyzer
    
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
        
        recommendations.append({
            'profile': profile['name'],
            'recommended': recommended,
            'expected_cost': stats['mean'],
            'risk_cv': stats['cv']
        })
    
    # Create comparison table
    print("\n" + "=" * 60)
    print("RECOMMENDATION SUMMARY")
    print("=" * 60)
    
    df = pd.DataFrame(recommendations)
    print(df.to_string(index=False))
    
    return recommendations


# ============================================================================
# EXAMPLE 4: Custom Pricing Mechanism
# ============================================================================

def example_custom_pricing():
    """
    Example of implementing and testing a custom pricing mechanism.
    """
    print("\n" + "=" * 60)
    print("CUSTOM PRICING MECHANISM EXAMPLE")
    print("=" * 60)
    
    from llm_token_analytics.simulator import PricingMechanism
    
    # Define custom tiered pricing mechanism
    class TieredPricing(PricingMechanism):
        """
        Custom tiered pricing with volume discounts.
        
        Tiers:
        - 0-10k tokens: $0.10/1k
        - 10k-100k tokens: $0.08/1k
        - 100k-1M tokens: $0.06/1k
        - 1M+ tokens: $0.04/1k
        """
        
        def calculate(self, input_tokens, output_tokens):
            total_tokens = input_tokens + output_tokens
            
            # Define tiers
            tiers = [
                (10_000, 0.10),
                (100_000, 0.08),
                (1_000_000, 0.06),
                (float('inf'), 0.04)
            ]
            
            # Calculate cost based on tiers
            if isinstance(total_tokens, np.ndarray):
                costs = np.zeros_like(total_tokens, dtype=float)
                
                for i, tokens in enumerate(total_tokens):
                    cost = 0
                    remaining = tokens
                    prev_limit = 0
                    
                    for limit, rate in tiers:
                        tier_tokens = min(remaining, limit - prev_limit)
                        cost += tier_tokens * rate / 1000
                        remaining -= tier_tokens
                        prev_limit = limit
                        
                        if remaining <= 0:
                            break
                    
                    costs[i] = cost
                
                return costs
            else:
                cost = 0
                remaining = total_tokens
                prev_limit = 0
                
                for limit, rate in tiers:
                    tier_tokens = min(remaining, limit - prev_limit)
                    cost += tier_tokens * rate / 1000
                    remaining -= tier_tokens
                    prev_limit = limit
                    
                    if remaining <= 0:
                        break
                
                return cost
    
    # Create simulator with custom mechanism
    config = SimulationConfig(
        n_simulations=50_000,
        mechanisms=['per_token', 'bundle', 'hybrid']
    )
    
    simulator = TokenSimulator(config)
    
    # Add custom mechanism
    simulator.pricing_mechanisms['tiered'] = TieredPricing({})
    config.mechanisms.append('tiered')
    
    print("Added custom tiered pricing mechanism")
    print("\nRunning simulation with custom pricing...")
    
    # Run simulation
    results = simulator.run()
    
    # Compare results
    print("\nComparison with Standard Mechanisms:")
    print("-" * 60)
    print(f"{'Mechanism':<15} {'Mean':<10} {'Median':<10} {'P95':<10} {'Savings vs Per-Token'}")
    print("-" * 60)
    
    per_token_mean = results.mechanism_results['per_token']['mean']
    
    for mechanism, stats in results.mechanism_results.items():
        savings = (1 - stats['mean'] / per_token_mean) * 100 if per_token_mean > 0 else 0
        
        print(f"{mechanism:<15} "
              f"${stats['mean']:<9.4f} "
              f"${stats['median']:<9.4f} "
              f"${stats['p95']:<9.4f} "
              f"{savings:+.1f}%")
    
    # Test with specific volumes
    print("\nTiered Pricing Examples:")
    print("-" * 40)
    
    test_volumes = [5_000, 50_000, 500_000, 2_000_000]
    tiered = TieredPricing({})
    
    for volume in test_volumes:
        # Assume 40% input, 60% output
        input_tokens = volume * 0.4
        output_tokens = volume * 0.6
        
        cost = tiered.calculate(input_tokens, output_tokens)
        effective_rate = (cost * 1000) / volume
        
        print(f"{volume:>10,} tokens: ${cost:.2f} "
              f"(${effective_rate:.4f}/1k tokens)")
    
    return results


# ============================================================================
# EXAMPLE 5: Real-time Monitoring Dashboard
# ============================================================================

def example_monitoring_dashboard():
    """
    Example of creating a real-time monitoring dashboard.
    """
    print("\n" + "=" * 60)
    print("MONITORING DASHBOARD EXAMPLE")
    print("=" * 60)
    
    try:
        import dash
        from dash import dcc, html, Input, Output
        import dash_bootstrap_components as dbc
        import plotly.graph_objects as go
    except ImportError:
        print("Dash not installed. Install with: pip install dash dash-bootstrap-components")
        return
    
    # Generate sample monitoring data
    np.random.seed(42)
    
    # Simulate hourly data for the past week
    hours = 24 * 7
    timestamps = pd.date_range(
        end=datetime.now(),
        periods=hours,
        freq='H'
    )
    
    # Generate usage patterns
    monitoring_data = pd.DataFrame({
        'timestamp': timestamps,
        'requests': np.random.poisson(100, hours),
        'input_tokens': np.random.lognormal(6, 0.5, hours),
        'output_tokens': np.random.lognormal(6.5, 0.7, hours),
        'cost': np.random.lognormal(0, 1, hours),
        'p95_latency': np.random.lognormal(5, 0.3, hours),
        'error_rate': np.random.beta(2, 100, hours)
    })
    
    # Calculate cumulative cost
    monitoring_data['cumulative_cost'] = monitoring_data['cost'].cumsum()
    
    print("Generated monitoring data for visualization")
    
    # Create Dash app
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("LLM Usage Monitoring Dashboard", className="text-center mb-4"),
                html.Hr()
            ])
        ]),
        
        # KPI Cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Total Requests", className="card-title"),
                        html.H2(f"{monitoring_data['requests'].sum():,}"),
                        html.P("Past 7 days", className="text-muted")
                    ])
                ])
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Total Cost", className="card-title"),
                        html.H2(f"${monitoring_data['cost'].sum():.2f}"),
                        html.P("Past 7 days", className="text-muted")
                    ])
                ])
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Avg Tokens/Request", className="card-title"),
                        html.H2(f"{monitoring_data['input_tokens'].mean():.0f}"),
                        html.P("Input tokens", className="text-muted")
                    ])
                ])
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Error Rate", className="card-title"),
                        html.H2(f"{monitoring_data['error_rate'].mean()*100:.2f}%"),
                        html.P("Past 7 days", className="text-muted")
                    ])
                ])
            ], width=3),
        ], className="mb-4"),
        
        # Charts
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='usage-timeline',
                    figure=go.Figure([
                        go.Scatter(
                            x=monitoring_data['timestamp'],
                            y=monitoring_data['requests'],
                            mode='lines',
                            name='Requests',
                            line=dict(color='blue')
                        )
                    ]).update_layout(
                        title='Request Volume Over Time',
                        xaxis_title='Time',
                        yaxis_title='Requests per Hour'
                    )
                )
            ], width=6),
            
            dbc.Col([
                dcc.Graph(
                    id='cost-timeline',
                    figure=go.Figure([
                        go.Scatter(
                            x=monitoring_data['timestamp'],
                            y=monitoring_data['cumulative_cost'],
                            mode='lines',
                            name='Cumulative Cost',
                            line=dict(color='green')
                        )
                    ]).update_layout(
                        title='Cumulative Cost',
                        xaxis_title='Time',
                        yaxis_title='Cost ($)'
                    )
                )
            ], width=6),
        ]),
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='token-distribution',
                    figure=go.Figure([
                        go.Box(y=monitoring_data['input_tokens'], name='Input'),
                        go.Box(y=monitoring_data['output_tokens'], name='Output')
                    ]).update_layout(
                        title='Token Distribution',
                        yaxis_title='Tokens'
                    )
                )
            ], width=6),
            
            dbc.Col([
                dcc.Graph(
                    id='latency-chart',
                    figure=go.Figure([
                        go.Scatter(
                            x=monitoring_data['timestamp'],
                            y=monitoring_data['p95_latency'],
                            mode='lines',
                            name='P95 Latency',
                            line=dict(color='red')
                        )
                    ]).update_layout(
                        title='P95 Latency',
                        xaxis_title='Time',
                        yaxis_title='Latency (ms)'
                    )
                )
            ], width=6),
        ])
    ], fluid=True)
    
    print("\nDashboard created successfully!")
    print("To view the dashboard, run: app.run_server(debug=True)")
    print("\nSample dashboard layout saved to 'monitoring_dashboard.html'")
    
    # Save static version
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Request Volume', 'Cumulative Cost', 
                       'Token Distribution', 'P95 Latency')
    )
    
    fig.add_trace(
        go.Scatter(x=monitoring_data['timestamp'], y=monitoring_data['requests'],
                  mode='lines', name='Requests'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=monitoring_data['timestamp'], y=monitoring_data['cumulative_cost'],
                  mode='lines', name='Cost'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Box(y=monitoring_data['input_tokens'], name='Input'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=monitoring_data['timestamp'], y=monitoring_data['p95_latency'],
                  mode='lines', name='Latency'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title='LLM Usage Monitoring Dashboard')
    fig.write_html('monitoring_dashboard.html')
    
    return app


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Run all examples.
    """
    print("LLM TOKEN ANALYTICS - EXAMPLE SCRIPTS")
    print("=" * 60)
    print("\nSelect an example to run:")
    print("1. Complete Pipeline (Collection → Analysis → Simulation)")
    print("2. Sensitivity Analysis")
    print("3. Optimal Mechanism Selection")
    print("4. Custom Pricing Mechanism")
    print("5. Monitoring Dashboard")
    print("6. Run All Examples")
    print("0. Exit")
    
    choice = input("\nEnter your choice (0-6): ").strip()
    
    if choice == '1':
        example_complete_pipeline()
    elif choice == '2':
        example_sensitivity_analysis()
    elif choice == '3':
        example_optimal_mechanism_selection()
    elif choice == '4':
        example_custom_pricing()
    elif choice == '5':
        example_monitoring_dashboard()
    elif choice == '6':
        print("\nRunning all examples...")
        example_complete_pipeline()
        example_sensitivity_analysis()
        example_optimal_mechanism_selection()
        example_custom_pricing()
        example_monitoring_dashboard()
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED")
        print("=" * 60)
    elif choice == '0':
        print("Exiting...")
        sys.exit(0)
    else:
        print("Invalid choice. Please run the script again.")


if __name__ == "__main__":
    main()
