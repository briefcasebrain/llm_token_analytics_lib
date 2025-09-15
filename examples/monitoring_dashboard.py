"""
Monitoring Dashboard Example
============================

Creates a real-time monitoring dashboard for LLM usage and costs.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path if running from examples folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_monitoring_dashboard():
    """
    Example of creating a real-time monitoring dashboard.
    """
    print("\n" + "=" * 60)
    print("MONITORING DASHBOARD EXAMPLE")
    print("=" * 60)

    try:
        import dash
        from dash import dcc, html, Input, Output, callback
        import dash_bootstrap_components as dbc
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Dash not installed. Install with: pip install dash dash-bootstrap-components")
        print("\nGenerating static dashboard instead...")
        return create_static_dashboard()

    # Generate sample monitoring data
    np.random.seed(42)

    # Simulate hourly data for the past week
    hours = 24 * 7
    timestamps = pd.date_range(
        end=datetime.now(),
        periods=hours,
        freq='H'
    )

    # Generate usage patterns with daily seasonality
    hour_of_day = np.array([t.hour for t in timestamps])
    daily_pattern = np.exp(-((hour_of_day - 14) ** 2) / 50)  # Peak at 2 PM

    monitoring_data = pd.DataFrame({
        'timestamp': timestamps,
        'requests': np.random.poisson(100 * daily_pattern),
        'input_tokens': np.random.lognormal(6, 0.5, hours) * daily_pattern,
        'output_tokens': np.random.lognormal(6.5, 0.7, hours) * daily_pattern,
        'cost': np.random.lognormal(0, 1, hours) * daily_pattern,
        'p95_latency': np.random.lognormal(5, 0.3, hours),
        'error_rate': np.random.beta(2, 100, hours),
        'cache_hit_rate': np.random.beta(7, 3, hours)
    })

    # Calculate cumulative cost
    monitoring_data['cumulative_cost'] = monitoring_data['cost'].cumsum()

    # Calculate moving averages
    monitoring_data['requests_ma'] = monitoring_data['requests'].rolling(window=24).mean()
    monitoring_data['cost_ma'] = monitoring_data['cost'].rolling(window=24).mean()

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
                        html.H4("Cache Hit Rate", className="card-title"),
                        html.H2(f"{monitoring_data['cache_hit_rate'].mean()*100:.1f}%"),
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
                    figure=create_usage_timeline(monitoring_data)
                )
            ], width=6),

            dbc.Col([
                dcc.Graph(
                    id='cost-timeline',
                    figure=create_cost_timeline(monitoring_data)
                )
            ], width=6),
        ]),

        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='token-distribution',
                    figure=create_token_distribution(monitoring_data)
                )
            ], width=6),

            dbc.Col([
                dcc.Graph(
                    id='performance-metrics',
                    figure=create_performance_metrics(monitoring_data)
                )
            ], width=6),
        ]),

        # Auto-refresh interval
        dcc.Interval(
            id='interval-component',
            interval=5*1000,  # in milliseconds
            n_intervals=0
        )
    ], fluid=True)

    print("\nDashboard created successfully!")
    print("To view the dashboard, run: app.run_server(debug=True)")

    # Save static version
    save_static_dashboard(monitoring_data)

    return app


def create_usage_timeline(data):
    """Create usage timeline chart."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data['timestamp'],
        y=data['requests'],
        mode='lines',
        name='Requests',
        line=dict(color='blue', width=1),
        opacity=0.6
    ))

    fig.add_trace(go.Scatter(
        x=data['timestamp'],
        y=data['requests_ma'],
        mode='lines',
        name='24h Moving Avg',
        line=dict(color='darkblue', width=2)
    ))

    fig.update_layout(
        title='Request Volume Over Time',
        xaxis_title='Time',
        yaxis_title='Requests per Hour',
        height=400,
        hovermode='x unified'
    )

    return fig


def create_cost_timeline(data):
    """Create cost timeline chart."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=data['timestamp'],
            y=data['cost'],
            name='Hourly Cost',
            marker_color='lightgreen',
            opacity=0.6
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=data['timestamp'],
            y=data['cumulative_cost'],
            mode='lines',
            name='Cumulative Cost',
            line=dict(color='darkgreen', width=2)
        ),
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Hourly Cost ($)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Cost ($)", secondary_y=True)

    fig.update_layout(
        title='Cost Analysis',
        height=400,
        hovermode='x unified'
    )

    return fig


def create_token_distribution(data):
    """Create token distribution chart."""
    fig = go.Figure()

    fig.add_trace(go.Box(
        y=data['input_tokens'],
        name='Input Tokens',
        marker_color='lightblue'
    ))

    fig.add_trace(go.Box(
        y=data['output_tokens'],
        name='Output Tokens',
        marker_color='lightcoral'
    ))

    # Add violin plot overlay
    fig.add_trace(go.Violin(
        y=data['input_tokens'],
        name='Input Distribution',
        side='negative',
        opacity=0.3,
        marker_color='blue',
        showlegend=False
    ))

    fig.add_trace(go.Violin(
        y=data['output_tokens'],
        name='Output Distribution',
        side='positive',
        opacity=0.3,
        marker_color='red',
        showlegend=False
    ))

    fig.update_layout(
        title='Token Distribution Analysis',
        yaxis_title='Tokens',
        height=400,
        violinmode='overlay'
    )

    return fig


def create_performance_metrics(data):
    """Create performance metrics chart."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('P95 Latency', 'Cache Hit Rate'),
        vertical_spacing=0.15
    )

    # Latency
    fig.add_trace(
        go.Scatter(
            x=data['timestamp'],
            y=data['p95_latency'],
            mode='lines',
            name='P95 Latency',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )

    # Cache hit rate
    fig.add_trace(
        go.Scatter(
            x=data['timestamp'],
            y=data['cache_hit_rate'] * 100,
            mode='lines',
            name='Cache Hit Rate',
            line=dict(color='purple', width=2),
            fill='tozeroy',
            fillcolor='rgba(128, 0, 128, 0.2)'
        ),
        row=2, col=1
    )

    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Latency (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Hit Rate (%)", row=2, col=1)

    fig.update_layout(
        height=400,
        showlegend=False,
        title='Performance Metrics'
    )

    return fig


def create_static_dashboard():
    """Create a static HTML dashboard when Dash is not available."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    print("Creating static monitoring dashboard...")

    # Generate sample data
    np.random.seed(42)
    hours = 24 * 7
    timestamps = pd.date_range(end=datetime.now(), periods=hours, freq='H')

    hour_of_day = np.array([t.hour for t in timestamps])
    daily_pattern = np.exp(-((hour_of_day - 14) ** 2) / 50)

    monitoring_data = pd.DataFrame({
        'timestamp': timestamps,
        'requests': np.random.poisson(100 * daily_pattern),
        'input_tokens': np.random.lognormal(6, 0.5, hours) * daily_pattern,
        'output_tokens': np.random.lognormal(6.5, 0.7, hours) * daily_pattern,
        'cost': np.random.lognormal(0, 1, hours) * daily_pattern,
        'p95_latency': np.random.lognormal(5, 0.3, hours),
        'error_rate': np.random.beta(2, 100, hours),
        'cache_hit_rate': np.random.beta(7, 3, hours)
    })

    monitoring_data['cumulative_cost'] = monitoring_data['cost'].cumsum()

    # Create comprehensive dashboard
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Request Volume', 'Cumulative Cost',
            'Token Distribution', 'P95 Latency',
            'Cache Hit Rate', 'Error Rate'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "box"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.1
    )

    # Request volume
    fig.add_trace(
        go.Scatter(x=monitoring_data['timestamp'], y=monitoring_data['requests'],
                   mode='lines', name='Requests', line=dict(color='blue')),
        row=1, col=1
    )

    # Cumulative cost
    fig.add_trace(
        go.Scatter(x=monitoring_data['timestamp'], y=monitoring_data['cumulative_cost'],
                   mode='lines', name='Cost', line=dict(color='green')),
        row=1, col=2
    )

    # Token distribution
    fig.add_trace(
        go.Box(y=monitoring_data['input_tokens'], name='Input', marker_color='lightblue'),
        row=2, col=1
    )
    fig.add_trace(
        go.Box(y=monitoring_data['output_tokens'], name='Output', marker_color='lightcoral'),
        row=2, col=1
    )

    # P95 Latency
    fig.add_trace(
        go.Scatter(x=monitoring_data['timestamp'], y=monitoring_data['p95_latency'],
                   mode='lines', name='Latency', line=dict(color='red')),
        row=2, col=2
    )

    # Cache hit rate
    fig.add_trace(
        go.Scatter(x=monitoring_data['timestamp'], y=monitoring_data['cache_hit_rate'] * 100,
                   mode='lines', name='Cache Hit', line=dict(color='purple'),
                   fill='tozeroy', fillcolor='rgba(128, 0, 128, 0.2)'),
        row=3, col=1
    )

    # Error rate
    fig.add_trace(
        go.Scatter(x=monitoring_data['timestamp'], y=monitoring_data['error_rate'] * 100,
                   mode='lines', name='Error Rate', line=dict(color='orange'),
                   fill='tozeroy', fillcolor='rgba(255, 165, 0, 0.2)'),
        row=3, col=2
    )

    fig.update_layout(
        height=1200,
        title='LLM Usage Monitoring Dashboard',
        showlegend=False
    )

    # Update axes labels
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=2)
    fig.update_yaxes(title_text="Requests/hr", row=1, col=1)
    fig.update_yaxes(title_text="Cost ($)", row=1, col=2)
    fig.update_yaxes(title_text="Tokens", row=2, col=1)
    fig.update_yaxes(title_text="Latency (ms)", row=2, col=2)
    fig.update_yaxes(title_text="Cache Hit (%)", row=3, col=1)
    fig.update_yaxes(title_text="Error Rate (%)", row=3, col=2)

    fig.write_html('monitoring_dashboard.html')
    print("Static dashboard saved to monitoring_dashboard.html")

    # Print summary statistics
    print("\nDashboard Summary Statistics:")
    print("-" * 40)
    print(f"Total Requests: {monitoring_data['requests'].sum():,}")
    print(f"Total Cost: ${monitoring_data['cost'].sum():.2f}")
    print(f"Avg Input Tokens: {monitoring_data['input_tokens'].mean():.0f}")
    print(f"Avg Output Tokens: {monitoring_data['output_tokens'].mean():.0f}")
    print(f"Avg Cache Hit Rate: {monitoring_data['cache_hit_rate'].mean()*100:.1f}%")
    print(f"Avg Error Rate: {monitoring_data['error_rate'].mean()*100:.3f}%")
    print(f"Avg P95 Latency: {monitoring_data['p95_latency'].mean():.0f}ms")

    return fig


def save_static_dashboard(monitoring_data):
    """Save a static version of the dashboard."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

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
    fig.write_html('monitoring_dashboard_static.html')
    print("Static dashboard saved to monitoring_dashboard_static.html")


if __name__ == "__main__":
    app = create_monitoring_dashboard()
    if app and hasattr(app, 'run_server'):
        print("\nStarting dashboard server...")
        app.run_server(debug=True)