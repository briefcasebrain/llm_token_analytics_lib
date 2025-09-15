"""
Command-Line Interface for LLM Token Analytics
===============================================
Complete CLI for data collection, simulation, analysis, and visualization.
"""

import click
import pandas as pd
from pathlib import Path
import json
import logging

from .collectors import UnifiedCollector, DataProcessor
from .simulator import TokenSimulator, SimulationConfig, SimulationResults
from .analyzer import TokenAnalyzer, CostAnalyzer
from .visualizer import SimulationVisualizer, create_dashboard

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config', '-c', type=click.Path(exists=True), help='Config file path')
@click.pass_context
def cli(ctx, verbose, config):
    """LLM Token Analytics CLI - Analyze and optimize token pricing strategies."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if config:
        with open(config) as f:
            ctx.obj['config'] = json.load(f)
    else:
        ctx.obj['config'] = {}


@cli.group()
def collect():
    """Data collection commands."""


@collect.command('fetch')
@click.option('--provider', '-p', type=click.Choice(['openai', 'anthropic', 'google', 'all']),
              default='all', help='Provider to collect from')
@click.option('--start-date', '-s', type=click.DateTime(), help='Start date for collection')
@click.option('--end-date', '-e', type=click.DateTime(), help='End date for collection')
@click.option('--output', '-o', type=click.Path(), default='data.parquet',
              help='Output file path')
@click.pass_context
def collect_fetch(ctx, provider, start_date, end_date, output):
    """Fetch usage data from LLM providers."""
    click.echo(f"Collecting data from {provider}...")

    collector = UnifiedCollector()

    if provider == 'all':
        data = collector.collect_all()
    else:
        data = collector.collect_provider(provider)

    if len(data) == 0:
        click.echo("No data collected", err=True)
        return

    processor = DataProcessor()
    data = processor.clean_data(data)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output.endswith('.csv'):
        data.to_csv(output_path, index=False)
    else:
        data.to_parquet(output_path, index=False)

    stats = processor.calculate_statistics(data)
    click.echo(f"Collected {stats['total_records']} records")
    click.echo(f"Providers: {', '.join(stats['providers'])}")
    click.echo(f"Total cost: ${stats['cost_stats']['total']:.2f}")
    click.echo(f"Data saved to {output_path}")


@collect.command('process')
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--aggregate', '-a', type=click.Choice(['H', 'D', 'W', 'M']),
              help='Aggregation period')
def collect_process(input_file, output, aggregate):
    """Process and clean collected data."""
    click.echo(f"Processing {input_file}...")

    if input_file.endswith('.csv'):
        data = pd.read_csv(input_file)
    else:
        data = pd.read_parquet(input_file)

    processor = DataProcessor()
    data = processor.clean_data(data)

    if aggregate:
        data = processor.aggregate_by_period(data, aggregate)
        click.echo(f"Aggregated data by {aggregate}")

    if output:
        output_path = Path(output)
        if output.endswith('.csv'):
            data.to_csv(output_path, index=False)
        else:
            data.to_parquet(output_path, index=False)
        click.echo(f"Processed data saved to {output_path}")
    else:
        click.echo(data.head())


@cli.group()
def simulate():
    """Simulation commands."""


@simulate.command('run')
@click.option('--iterations', '-n', type=int, default=100000,
              help='Number of simulation iterations')
@click.option('--mechanisms', '-m', multiple=True,
              type=click.Choice(['per_token', 'bundle', 'hybrid', 'cached', 'outcome', 'dynamic']),
              help='Pricing mechanisms to simulate')
@click.option('--output', '-o', type=click.Path(), default='results.pkl',
              help='Output file for results')
@click.option('--seed', type=int, default=42, help='Random seed')
@click.pass_context
def simulate_run(ctx, iterations, mechanisms, output, seed):
    """Run pricing simulation."""
    click.echo(f"Running simulation with {iterations:,} iterations...")

    if not mechanisms:
        mechanisms = ['per_token', 'bundle', 'hybrid', 'cached']

    config = SimulationConfig(
        n_simulations=iterations,
        mechanisms=list(mechanisms),
        seed=seed
    )

    simulator = TokenSimulator(config)
    results = simulator.run()

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import pickle  # nosec: internal data serialization only
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)  # nosec: internal data serialization only

    click.echo("\nSimulation Results:")
    click.echo("-" * 40)
    for mechanism, stats in results.mechanism_results.items():
        click.echo(f"{mechanism:12} | Mean: ${stats['mean']:.4f} | CV: {stats['cv']:.2f} | P95: ${stats['p95']:.4f}")

    click.echo(f"\nResults saved to {output_path}")


@simulate.command('compare')
@click.argument('results_file', type=click.Path(exists=True))
@click.option('--format', '-f', type=click.Choice(['table', 'json']), default='table',
              help='Output format')
def simulate_compare(results_file, format):
    """Compare simulation results across mechanisms."""
    import pickle  # nosec: internal data serialization only
    with open(results_file, 'rb') as f:
        results = pickle.load(f)  # nosec: internal data serialization only

    if isinstance(results, SimulationResults):
        df = results.to_dataframe()
    else:
        click.echo("Invalid results file", err=True)
        return

    if format == 'table':
        click.echo("\nMechanism Comparison:")
        click.echo("=" * 80)
        click.echo(df.to_string())
    else:
        click.echo(df.to_json(orient='records', indent=2))


@simulate.command('sensitivity')
@click.option('--iterations', '-n', type=int, default=10000,
              help='Iterations per test')
@click.option('--parameter', '-p', type=click.Choice(['cache_hit_rate', 'correlation', 'bundle_size']),
              required=True, help='Parameter to test')
@click.option('--values', '-v', multiple=True, type=float,
              help='Values to test')
@click.option('--output', '-o', type=click.Path(), help='Output file')
def simulate_sensitivity(iterations, parameter, values, output):
    """Run sensitivity analysis."""
    click.echo(f"Running sensitivity analysis for {parameter}...")

    if not values:
        if parameter == 'cache_hit_rate':
            values = [0.3, 0.5, 0.7, 0.9]
        elif parameter == 'correlation':
            values = [0.0, 0.3, 0.6, 0.9]
        else:
            values = [50000, 100000, 200000]

    config = SimulationConfig(n_simulations=iterations)
    simulator = TokenSimulator(config)

    sensitivity = simulator.sensitivity_analysis({parameter: list(values)})

    for mechanism, data in sensitivity[parameter].items():
        click.echo(f"\n{mechanism}:")
        click.echo(f"  Sensitivity: {data['mean_sensitivity']:.1%}")
        click.echo(f"  Range: ${data['range'][0]:.4f} - ${data['range'][1]:.4f}")

    if output:
        with open(output, 'w') as f:
            json.dump(sensitivity, f, indent=2, default=str)
        click.echo(f"\nResults saved to {output}")


@cli.group()
def analyze():
    """Analysis commands."""


@analyze.command('distributions')
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
def analyze_distributions(data_file, output):
    """Analyze token distributions."""
    click.echo(f"Analyzing distributions in {data_file}...")

    if data_file.endswith('.csv'):
        data = pd.read_csv(data_file)
    else:
        data = pd.read_parquet(data_file)

    analyzer = TokenAnalyzer()
    results = analyzer.fit_distributions(
        data['input_tokens'].values,
        data['output_tokens'].values
    )

    click.echo("\nDistribution Analysis:")
    click.echo("-" * 40)

    for token_type, dist in results.items():
        click.echo(f"\n{token_type.upper()} tokens:")
        click.echo(f"  Best fit: {dist.distribution_type}")
        click.echo(f"  Parameters: {dist.parameters}")
        click.echo(f"  AIC: {dist.goodness_of_fit['aic']:.2f}")
        click.echo(f"  KS p-value: {dist.goodness_of_fit['ks_pvalue']:.4f}")

    if output:
        with open(output, 'w') as f:
            json.dump({
                k: {
                    'type': v.distribution_type,
                    'params': v.parameters,
                    'fit': v.goodness_of_fit
                }
                for k, v in results.items()
            }, f, indent=2)
        click.echo(f"\nResults saved to {output}")


@analyze.command('correlations')
@click.argument('data_file', type=click.Path(exists=True))
def analyze_correlations(data_file):
    """Analyze correlations between input and output tokens."""
    click.echo(f"Analyzing correlations in {data_file}...")

    if data_file.endswith('.csv'):
        data = pd.read_csv(data_file)
    else:
        data = pd.read_parquet(data_file)

    analyzer = TokenAnalyzer()
    results = analyzer.analyze_correlations(
        data['input_tokens'].values,
        data['output_tokens'].values
    )

    click.echo("\nCorrelation Analysis:")
    click.echo("-" * 40)

    linear = results['linear']
    click.echo(f"Pearson: {linear['pearson']['value']:.3f} (p={linear['pearson']['p_value']:.4f})")
    click.echo(f"Spearman: {linear['spearman']['value']:.3f} (p={linear['spearman']['p_value']:.4f})")
    click.echo(f"Kendall: {linear['kendall']['value']:.3f} (p={linear['kendall']['p_value']:.4f})")

    tail = results['tail_dependence']
    click.echo("\nTail Dependence:")
    click.echo(f"Upper tail: {tail['upper_tail']:.3f}")
    click.echo(f"Lower tail: {tail['lower_tail']:.3f}")


@analyze.command('risk')
@click.argument('results_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file')
def analyze_risk(results_file, output):
    """Calculate risk metrics for pricing mechanisms."""
    import pickle  # nosec: internal data serialization only
    with open(results_file, 'rb') as f:
        results = pickle.load(f)  # nosec: internal data serialization only

    if isinstance(results, SimulationResults):
        mechanism_results = results.mechanism_results
    else:
        mechanism_results = results

    analyzer = CostAnalyzer(mechanism_results)
    risk_metrics = analyzer.calculate_risk_metrics()

    click.echo("\nRisk Metrics:")
    click.echo("=" * 80)
    click.echo(risk_metrics.to_string())

    if output:
        if output.endswith('.csv'):
            risk_metrics.to_csv(output, index=False)
        else:
            risk_metrics.to_parquet(output, index=False)
        click.echo(f"\nResults saved to {output}")


@cli.group()
def visualize():
    """Visualization commands."""


@visualize.command('distributions-plot')
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output image file')
@click.option('--show', is_flag=True, help='Show plot interactively')
def visualize_distributions(data_file, output, show):
    """Plot token distributions."""
    click.echo(f"Plotting distributions from {data_file}...")

    if data_file.endswith('.csv'):
        data = pd.read_csv(data_file)
    else:
        data = pd.read_parquet(data_file)

    from .visualizer import plot_distributions
    fig = plot_distributions(data)

    if output:
        fig.write_html(output) if output.endswith('.html') else fig.write_image(output)
        click.echo(f"Plot saved to {output}")

    if show:
        fig.show()


@visualize.command('comparison')
@click.argument('results_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output image file')
@click.option('--show', is_flag=True, help='Show plot interactively')
def visualize_comparison(results_file, output, show):
    """Create comparison plots for simulation results."""
    import pickle  # nosec: internal data serialization only
    with open(results_file, 'rb') as f:
        results = pickle.load(f)  # nosec: internal data serialization only

    visualizer = SimulationVisualizer(results)
    fig = visualizer.create_comparison_plot()

    if output:
        fig.write_html(output) if output.endswith('.html') else fig.write_image(output)
        click.echo(f"Plot saved to {output}")

    if show:
        fig.show()


@visualize.command('dashboard')
@click.argument('results_file', type=click.Path(exists=True))
@click.option('--port', '-p', type=int, default=8050, help='Dashboard port')
@click.option('--serve', is_flag=True, help='Start dashboard server')
def visualize_dashboard(results_file, port, serve):
    """Create interactive dashboard."""
    click.echo(f"Creating dashboard from {results_file}...")

    import pickle  # nosec: internal data serialization only
    with open(results_file, 'rb') as f:
        results = pickle.load(f)  # nosec: internal data serialization only

    app = create_dashboard(results)

    if serve:
        click.echo(f"Starting dashboard on http://localhost:{port}")
        app.run_server(debug=False, port=port)
    else:
        click.echo("Dashboard created. Use --serve to start the server.")


@cli.command()
@click.option('--risk-tolerance', type=click.Choice(['low', 'medium', 'high']),
              default='medium', help='Risk tolerance level')
@click.option('--usage-volume', type=int, default=100000,
              help='Expected monthly token usage')
@click.option('--budget', type=float, help='Monthly budget constraint')
@click.option('--results-file', type=click.Path(exists=True),
              help='Simulation results file')
def optimize(risk_tolerance, usage_volume, budget, results_file):
    """Find optimal pricing mechanism based on profile."""
    click.echo("Finding optimal pricing mechanism...")

    if not results_file:
        config = SimulationConfig(n_simulations=50000)
        simulator = TokenSimulator(config)
        results = simulator.run()
        mechanism_results = results.mechanism_results
    else:
        import pickle  # nosec: internal data serialization only
        with open(results_file, 'rb') as f:
            results = pickle.load(f)  # nosec: internal data serialization only
            if isinstance(results, SimulationResults):
                mechanism_results = results.mechanism_results
            else:
                mechanism_results = results

    user_profile = {
        'risk_tolerance': risk_tolerance,
        'usage_volume': usage_volume,
        'predictability_preference': 0.7 if risk_tolerance == 'low' else 0.3,
        'budget_constraint': budget
    }

    analyzer = CostAnalyzer(mechanism_results)
    best_mechanism = analyzer.optimal_mechanism_selection(user_profile)

    click.echo("\nOptimal Pricing Recommendation:")
    click.echo("=" * 40)
    click.echo(f"Recommended mechanism: {best_mechanism.upper()}")

    if best_mechanism in mechanism_results:
        stats = mechanism_results[best_mechanism]
        monthly_cost = stats['mean'] * usage_volume / 1000
        click.echo(f"Expected monthly cost: ${monthly_cost:.2f}")
        click.echo(f"Cost variability (CV): {stats['cv']:.2f}")
        click.echo(f"95th percentile cost: ${stats['p95'] * usage_volume / 1000:.2f}")


if __name__ == '__main__':
    cli()
