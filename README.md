# LLM Token Analytics Library

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Non-Commercial](https://img.shields.io/badge/License-Non--Commercial-red.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](https://github.com/your-username/llm-token-analytics/actions)

A comprehensive Python library for analyzing and simulating LLM token usage patterns and pricing mechanisms. This library provides tools for Monte Carlo simulations, statistical analysis, and visualization to help optimize LLM pricing strategies.

## Features

- **Multi-Provider Support**: Collect usage data from OpenAI, Anthropic, and Google AI
- **Advanced Simulation**: Monte Carlo simulations with 100,000+ iterations
- **Multiple Pricing Mechanisms**: Per-token, bundle, hybrid, cached, outcome-based, and dynamic pricing
- **Statistical Analysis**: Distribution fitting, correlation analysis, and risk metrics
- **Interactive Visualizations**: Dashboards and plots for insights
- **Production Ready**: Docker support, CLI interface, and modular REST API
- **Modular Architecture**: Clean separation of concerns with blueprint-based API

## Installation

```bash
# Install from source
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
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

## REST API Server

The library includes a production-ready REST API server with modular architecture:

### Start the Server

```bash
# Development
python api_server.py

# Production with Docker
docker-compose up -d api
```

### API Endpoints

```bash
# Health check
GET /health

# Collect usage data
POST /collect
{
  "providers": ["openai", "anthropic"],
  "start_date": "2025-01-01",
  "end_date": "2025-01-31"
}

# Run simulation
POST /simulate
{
  "n_simulations": 100000,
  "mechanisms": ["per_token", "bundle", "hybrid"],
  "data_source": "synthetic"
}

# Analyze distributions
POST /analyze
{
  "collection_id": "collection_20250101_120000"
}

# Compare mechanisms
POST /compare
{
  "simulation_id": "simulation_20250101_120000"
}

# Optimize pricing
POST /optimize
{
  "simulation_id": "simulation_20250101_120000",
  "user_profile": {
    "risk_tolerance": "low",
    "usage_volume": 50000,
    "budget_constraint": 100
  }
}

# Generate visualization
GET /visualize/<simulation_id>
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
├── app/                     # REST API application
│   ├── __init__.py          # Flask app factory
│   ├── config.py            # Configuration management
│   ├── storage.py           # Results storage
│   ├── errors.py            # Error handlers
│   └── routes/              # API route blueprints
│       ├── health.py        # Health check endpoints
│       ├── collection.py    # Data collection endpoints
│       ├── simulation.py    # Simulation endpoints
│       ├── analysis.py      # Analysis endpoints
│       ├── comparison.py    # Comparison endpoints
│       └── visualization.py # Visualization endpoints
├── examples/                # Example scripts
├── tests/                   # Unit and integration tests
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── conftest.py         # Test configuration
├── config/                  # Configuration files
├── api_server.py           # REST API server entry point
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

Example configuration:

```yaml
# config/config.yaml
api:
  host: "0.0.0.0"
  port: 5000
  debug: false

simulation:
  default_iterations: 100000
  default_mechanisms: ["per_token", "bundle", "hybrid"]

providers:
  openai:
    model: "gpt-4"
  anthropic:
    model: "claude-3-opus"
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests

# Run with coverage
pytest --cov=llm_token_analytics --cov=app tests/

# Validate structure
pytest tests/test_structure_validation.py
```

## Development

### Setting up development environment

```bash
# Clone the repository
git clone https://github.com/your-username/llm-token-analytics.git
cd llm-token-analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .

# Run tests
pytest
```

### Code Quality

```bash
# Format code
black llm_token_analytics/ app/ tests/

# Lint code
flake8 llm_token_analytics/ app/ tests/

# Type checking
mypy llm_token_analytics/ app/
```

## Examples

See the [examples/](examples/) directory for complete working examples:
- `basic_simulation.py` - Simple simulation example
- `data_collection.py` - Collecting real usage data
- `custom_pricing.py` - Creating custom pricing mechanisms
- `api_client.py` - Using the REST API
- `advanced_analysis.py` - Advanced statistical analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Ensure all tests pass (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Roadmap

- [ ] Support for additional LLM providers (Azure OpenAI, AWS Bedrock)
- [ ] Real-time streaming analysis
- [ ] Machine learning-based usage prediction
- [ ] Advanced optimization algorithms
- [ ] GraphQL API support
- [ ] Kubernetes deployment examples

## License

This project is licensed under a Non-Commercial License - see [LICENSE](LICENSE) for details.

**Key Restrictions:**
- ❌ Commercial use prohibited
- ❌ Cannot be used in revenue-generating activities
- ❌ Cannot be incorporated into commercial products
- ✅ Personal, educational, and research use allowed
- ✅ Non-profit organizational use permitted

For commercial licensing inquiries, please contact the project maintainers.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{llm_token_analytics,
  title={LLM Token Analytics Library},
  author={Your Name},
  year={2025},
  url={https://github.com/your-username/llm-token-analytics}
}
```

---

Built with Python for data-driven LLM cost optimization.