# LLM Token Analytics Library

A comprehensive Python library for analyzing and simulating LLM token usage patterns and pricing mechanisms. This library provides tools for Monte Carlo simulations, statistical analysis, and visualization to help optimize LLM pricing strategies.

## Features

- **Multi-Provider Support**: Collect usage data from OpenAI, Anthropic, and Google AI
- **Advanced Simulation**: Monte Carlo simulations with 100,000+ iterations
- **Multiple Pricing Mechanisms**: Per-token, bundle, hybrid, cached, outcome-based, and dynamic pricing
- **Statistical Analysis**: Distribution fitting, correlation analysis, and risk metrics
- **Interactive Visualizations**: Dashboards and plots for insights
- **Production Ready**: Docker support, CLI interface, and REST API

## Installation

```bash
# Install from source
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Quick Start

### 1. Set Up Environment (Optional)

If you want to collect real usage data from LLM providers, set up API keys:

```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

### 2. Run Your First Simulation

```python
from llm_token_analytics import TokenSimulator, SimulationConfig

# Configure simulation
config = SimulationConfig(
    n_simulations=10_000,
    mechanisms=['per_token', 'bundle', 'hybrid']
)

# Run simulation
simulator = TokenSimulator(config)
results = simulator.run()

# Display results
for mechanism, stats in results.mechanism_results.items():
    print(f"{mechanism}: Mean=${stats['mean']:.4f}, P95=${stats['p95']:.4f}")
```

### 3. Visualize Results

```python
from llm_token_analytics import SimulationVisualizer

viz = SimulationVisualizer(results)
fig = viz.create_comparison_plot()
fig.show()
```

## CLI Usage

The library includes a comprehensive CLI for all operations:

```bash
# Collect data from providers
llm-analytics collect fetch --provider all --output data.parquet

# Run simulation
llm-analytics simulate run -n 100000 -m per_token bundle hybrid

# Analyze distributions
llm-analytics analyze distributions data.parquet

# Create visualizations
llm-analytics visualize dashboard results.pkl --serve

# Find optimal pricing
llm-analytics optimize --risk-tolerance low --usage-volume 100000
```

## API Usage

### Data Collection

```python
from llm_token_analytics import UnifiedCollector

collector = UnifiedCollector()
data = collector.collect_all()
```

### Statistical Analysis

```python
from llm_token_analytics import TokenAnalyzer

analyzer = TokenAnalyzer()
distributions = analyzer.fit_distributions(
    data['input_tokens'].values,
    data['output_tokens'].values
)
```

### Cost Analysis

```python
from llm_token_analytics import CostAnalyzer

analyzer = CostAnalyzer(results.mechanism_results)
best = analyzer.optimal_mechanism_selection({
    'risk_tolerance': 'low',
    'usage_volume': 100_000,
    'budget_constraint': 200
})
```

## Docker Deployment

```bash
# Start all services
docker-compose up -d

# Access services
# API: http://localhost:5000
# Dashboard: http://localhost:8050
```

## Project Structure

```
llm_token_analytics/
├── llm_token_analytics/     # Main library package
│   ├── __init__.py
│   ├── collectors.py        # Data collection from APIs
│   ├── simulator.py         # Monte Carlo simulation engine
│   ├── analyzer.py          # Statistical analysis tools
│   ├── visualizer.py        # Plotting and dashboards
│   └── cli.py              # Command-line interface
├── examples/                # Example scripts
├── tests/                   # Unit tests
├── config/                  # Configuration files
├── api_server.py           # REST API server
└── docker-compose.yml      # Docker orchestration
```

## Advanced Features

### Sensitivity Analysis

```python
sensitivity = simulator.sensitivity_analysis({
    'cache_hit_rate': [0.3, 0.5, 0.7, 0.9],
    'correlation': [0.0, 0.3, 0.6, 0.9]
})
```

### Custom Pricing Mechanisms

```python
from llm_token_analytics import PricingMechanism

class CustomPricing(PricingMechanism):
    def calculate(self, input_tokens, output_tokens):
        # Your custom pricing logic
        return custom_cost
```

### Risk Metrics

- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Sharpe Ratio
- Maximum Drawdown
- Tail Ratio Analysis

## Configuration

Configuration can be provided via:
- YAML files (`config/config.yaml`)
- Environment variables (`.env`)
- Command-line arguments
- Python API parameters

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=llm_token_analytics tests/
```

## Examples

See the [examples/](examples/) directory for complete working examples.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) for details.

---

Built with Python for data-driven LLM cost optimization.