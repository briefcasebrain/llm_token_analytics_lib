"""
Visualization Module
=====================

Tools for creating plots and dashboards.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class SimulationVisualizer:
    """Create visualizations for simulation results."""
    
    def __init__(self, results: Any):
        """
        Initialize visualizer with simulation results.
        
        Args:
            results: SimulationResults object or dictionary
        """
        self.results = results
        if hasattr(results, 'mechanism_results'):
            self.mechanism_results = results.mechanism_results
        else:
            self.mechanism_results = results
    
    def create_comparison_plot(self) -> go.Figure:
        """Create comprehensive comparison plot."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Mean Cost Comparison',
                'Cost Variance (CV)',
                'Percentile Distribution',
                'Tail Risk (P95/Median)'
            ),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'box'}, {'type': 'scatter'}]]
        )
        
        mechanisms = list(self.mechanism_results.keys())
        
        # Mean costs
        means = [self.mechanism_results[m]['mean'] for m in mechanisms]
        fig.add_trace(
            go.Bar(x=mechanisms, y=means, name='Mean Cost',
                  marker_color='lightblue'),
            row=1, col=1
        )
        
        # Coefficient of variation
        cvs = [self.mechanism_results[m]['cv'] for m in mechanisms]
        fig.add_trace(
            go.Bar(x=mechanisms, y=cvs, name='CV',
                  marker_color='lightcoral'),
            row=1, col=2
        )
        
        # Percentile box plots
        for mechanism in mechanisms:
            stats = self.mechanism_results[mechanism]
            percentiles = stats.get('percentiles', {})
            
            if percentiles:
                values = [percentiles.get(p, 0) for p in [1, 25, 50, 75, 99]]
                fig.add_trace(
                    go.Box(y=values, name=mechanism),
                    row=2, col=1
                )
        
        # Tail ratios
        tail_ratios = [self.mechanism_results[m].get('tail_ratio', 1) for m in mechanisms]
        fig.add_trace(
            go.Scatter(x=mechanisms, y=tail_ratios, mode='markers+lines',
                      name='Tail Ratio', marker=dict(size=10)),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Token Pricing Simulation Results',
            showlegend=False,
            height=800
        )
        
        # Update axes
        fig.update_xaxes(title_text="Mechanism", row=1, col=1)
        fig.update_xaxes(title_text="Mechanism", row=1, col=2)
        fig.update_xaxes(title_text="Mechanism", row=2, col=1)
        fig.update_xaxes(title_text="Mechanism", row=2, col=2)
        
        fig.update_yaxes(title_text="Cost ($)", row=1, col=1)
        fig.update_yaxes(title_text="CV", row=1, col=2)
        fig.update_yaxes(title_text="Cost ($)", row=2, col=1)
        fig.update_yaxes(title_text="Ratio", row=2, col=2)
        
        return fig
    
    def create_sensitivity_plot(self, sensitivity_results: Dict) -> go.Figure:
        """Create sensitivity analysis plot."""
        fig = make_subplots(
            rows=len(sensitivity_results),
            cols=1,
            subplot_titles=list(sensitivity_results.keys())
        )
        
        for i, (param, results) in enumerate(sensitivity_results.items(), 1):
            for mechanism, data in results.items():
                fig.add_trace(
                    go.Scatter(
                        x=data['parameter_values'],
                        y=data['means'],
                        mode='lines+markers',
                        name=f"{mechanism}",
                        legendgroup=mechanism
                    ),
                    row=i, col=1
                )
            
            fig.update_xaxes(title_text=param, row=i, col=1)
            fig.update_yaxes(title_text="Mean Cost ($)", row=i, col=1)
        
        fig.update_layout(
            title='Sensitivity Analysis',
            height=300 * len(sensitivity_results)
        )
        
        return fig
    
    def create_distribution_comparison(self) -> go.Figure:
        """Create distribution comparison plot."""
        fig = go.Figure()
        
        # Add histogram for each mechanism
        for mechanism, stats in self.mechanism_results.items():
            # Generate sample data based on statistics
            mean = stats['mean']
            std = stats['std']
            
            # Approximate distribution
            samples = np.random.lognormal(
                np.log(mean),
                std / mean,
                10000
            )
            
            fig.add_trace(
                go.Histogram(
                    x=samples,
                    name=mechanism,
                    opacity=0.6,
                    nbinsx=50,
                    histnorm='probability'
                )
            )
        
        fig.update_layout(
            title='Cost Distribution Comparison',
            xaxis_title='Cost ($)',
            yaxis_title='Probability',
            barmode='overlay',
            hovermode='x unified'
        )
        
        return fig


def plot_distributions(data: pd.DataFrame) -> go.Figure:
    """
    Plot token distributions from data.
    
    Args:
        data: DataFrame with input_tokens and output_tokens columns
    
    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Input Token Distribution',
            'Output Token Distribution',
            'Token Correlation',
            'Cost Distribution'
        )
    )
    
    # Input tokens histogram
    if 'input_tokens' in data.columns:
        fig.add_trace(
            go.Histogram(x=data['input_tokens'], nbinsx=50, name='Input'),
            row=1, col=1
        )
    
    # Output tokens histogram
    if 'output_tokens' in data.columns:
        fig.add_trace(
            go.Histogram(x=data['output_tokens'], nbinsx=50, name='Output'),
            row=1, col=2
        )
    
    # Correlation scatter
    if 'input_tokens' in data.columns and 'output_tokens' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data['input_tokens'],
                y=data['output_tokens'],
                mode='markers',
                marker=dict(size=3, opacity=0.5),
                name='Correlation'
            ),
            row=2, col=1
        )
    
    # Cost distribution
    if 'cost' in data.columns:
        fig.add_trace(
            go.Histogram(x=data['cost'], nbinsx=50, name='Cost'),
            row=2, col=2
        )
    
    # Update axes
    fig.update_xaxes(title_text="Input Tokens", row=1, col=1)
    fig.update_xaxes(title_text="Output Tokens", row=1, col=2)
    fig.update_xaxes(title_text="Input Tokens", row=2, col=1)
    fig.update_xaxes(title_text="Cost ($)", row=2, col=2)
    
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_yaxes(title_text="Output Tokens", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
    fig.update_layout(
        title='Token Usage Analysis',
        showlegend=False,
        height=800
    )
    
    return fig


def plot_cost_comparison(results: Dict) -> go.Figure:
    """
    Create cost comparison plot.
    
    Args:
        results: Dictionary of simulation results
    
    Returns:
        Plotly figure
    """
    # Prepare data
    mechanisms = []
    metrics = []
    values = []
    
    for mechanism, stats in results.items():
        for metric in ['mean', 'median', 'p95', 'p99']:
            if metric in stats:
                mechanisms.append(mechanism)
                metrics.append(metric.upper())
                values.append(stats[metric])
    
    df = pd.DataFrame({
        'Mechanism': mechanisms,
        'Metric': metrics,
        'Cost': values
    })
    
    # Create grouped bar chart
    fig = px.bar(
        df,
        x='Mechanism',
        y='Cost',
        color='Metric',
        barmode='group',
        title='Cost Comparison Across Pricing Mechanisms',
        labels={'Cost': 'Cost ($)'}
    )
    
    return fig


def create_dashboard(results: Any):
    """
    Create interactive Dash dashboard.
    
    Args:
        results: SimulationResults object
    
    Returns:
        Dash app
    """
    try:
        import dash
        from dash import dcc, html, Input, Output
        import dash_bootstrap_components as dbc
    except ImportError:
        logger.error("Dash not installed. Install with: pip install dash dash-bootstrap-components")
        return None
    
    # Initialize app
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # Create visualizer
    visualizer = SimulationVisualizer(results)
    
    # Layout
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("LLM Token Analytics Dashboard", className="text-center mb-4"),
                html.Hr()
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='comparison-plot',
                    figure=visualizer.create_comparison_plot()
                )
            ], width=12)
        ]),
        
        html.Hr(),
        
        dbc.Row([
            dbc.Col([
                html.H3("Distribution Comparison"),
                dcc.Graph(
                    id='distribution-plot',
                    figure=visualizer.create_distribution_comparison()
                )
            ], width=12)
        ]),
        
        html.Hr(),
        
        dbc.Row([
            dbc.Col([
                html.H3("Summary Statistics"),
                _create_summary_table(results)
            ], width=12)
        ])
    ], fluid=True)
    
    return app


def _create_summary_table(results):
    """Create summary statistics table for dashboard."""
    try:
        import dash_bootstrap_components as dbc
        from dash import html
    except ImportError:
        return html.Div("Table not available")
    
    # Extract data
    if hasattr(results, 'mechanism_results'):
        mechanism_results = results.mechanism_results
    else:
        mechanism_results = results
    
    # Create table rows
    rows = []
    for mechanism, stats in mechanism_results.items():
        rows.append(
            html.Tr([
                html.Td(mechanism),
                html.Td(f"${stats['mean']:.4f}"),
                html.Td(f"${stats['median']:.4f}"),
                html.Td(f"${stats['p95']:.4f}"),
                html.Td(f"{stats['cv']:.2f}"),
                html.Td(f"{stats.get('tail_ratio', 0):.2f}")
            ])
        )
    
    table = dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("Mechanism"),
                html.Th("Mean Cost"),
                html.Th("Median"),
                html.Th("P95"),
                html.Th("CV"),
                html.Th("Tail Ratio")
            ])
        ]),
        html.Tbody(rows)
    ], bordered=True, hover=True, responsive=True, striped=True)
    
    return table


def create_report_plots(data: pd.DataFrame, results: Optional[Dict] = None) -> Dict[str, go.Figure]:
    """
    Create all plots for a comprehensive report.
    
    Args:
        data: Token usage data
        results: Simulation results (optional)
    
    Returns:
        Dictionary of plot names to figures
    """
    plots = {}
    
    # Distribution plots
    plots['distributions'] = plot_distributions(data)
    
    # Time series if timestamp available
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Daily aggregation
        daily = data.groupby(data['timestamp'].dt.date).agg({
            'input_tokens': 'mean',
            'output_tokens': 'mean',
            'cost': 'sum'
        }).reset_index()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Daily Token Usage', 'Daily Cost')
        )
        
        fig.add_trace(
            go.Scatter(x=daily['timestamp'], y=daily['input_tokens'],
                      name='Input Tokens', mode='lines'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=daily['timestamp'], y=daily['output_tokens'],
                      name='Output Tokens', mode='lines'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=daily['timestamp'], y=daily['cost'],
                  name='Daily Cost'),
            row=2, col=1
        )
        
        fig.update_layout(title='Time Series Analysis', height=600)
        plots['time_series'] = fig
    
    # Model comparison if available
    if 'model' in data.columns:
        model_stats = data.groupby('model').agg({
            'input_tokens': 'mean',
            'output_tokens': 'mean',
            'cost': 'mean'
        }).reset_index()
        
        fig = px.bar(
            model_stats,
            x='model',
            y=['input_tokens', 'output_tokens'],
            title='Average Token Usage by Model',
            labels={'value': 'Tokens', 'variable': 'Type'}
        )
        
        plots['model_comparison'] = fig
    
    # Simulation results if provided
    if results:
        visualizer = SimulationVisualizer(results)
        plots['simulation_comparison'] = visualizer.create_comparison_plot()
        plots['cost_distributions'] = visualizer.create_distribution_comparison()
    
    return plots
