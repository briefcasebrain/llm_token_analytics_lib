# LLM Token Analytics Examples

This directory contains example scripts that demonstrate how to use the LLM Token Analytics library for various tasks.

## Examples Overview

### 1. Basic Simulation (`01_basic_simulation.py`)
**Difficulty: Beginner**

Demonstrates how to run a basic Monte Carlo simulation comparing different pricing mechanisms. This is the best starting point for new users.

```bash
python 01_basic_simulation.py
```

**What you'll learn:**
- How to configure and run simulations
- Understanding different pricing mechanisms
- Interpreting simulation results
- Basic performance metrics

### 2. API Client (`02_api_client.py`)
**Difficulty: Intermediate**

Shows how to interact with the REST API server for running simulations and analysis remotely.

**Prerequisites:**
```bash
# Start the API server first
python api_server.py
```

```bash
python 02_api_client.py
```

**What you'll learn:**
- REST API integration
- Remote simulation execution
- Result retrieval and analysis
- Mechanism comparison and optimization
- Visualization download

### 3. Data Collection (`03_data_collection.py`)
**Difficulty: Intermediate**

Demonstrates collecting real usage data from LLM providers and analyzing usage patterns.

**Prerequisites:**
```bash
# Install provider dependencies
pip install llm-token-analytics[providers]

# Set up API keys (optional)
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_CLOUD_PROJECT="your-project"
```

```bash
python 03_data_collection.py
```

**What you'll learn:**
- Real data collection from LLM providers
- Synthetic data generation
- Statistical analysis of usage patterns
- Data storage and reuse

### 4. Custom Pricing (`custom_pricing.py`)
**Difficulty: Advanced**

Shows how to implement custom pricing mechanisms and integrate them into simulations.

```bash
python custom_pricing.py
```

**What you'll learn:**
- Creating custom pricing mechanisms
- Advanced simulation configuration
- Custom metrics and analysis

### 5. Complete Pipeline (`complete_pipeline.py`)
**Difficulty: Advanced**

Demonstrates a complete end-to-end workflow from data collection to optimization.

```bash
python complete_pipeline.py
```

**What you'll learn:**
- Full workflow integration
- Production-ready patterns
- Error handling and monitoring
- Result interpretation

## Running Examples

### Quick Start
```bash
# Install the library
pip install -e .

# Run the basic example
python examples/01_basic_simulation.py
```

### With API Server
```bash
# Terminal 1: Start API server
python api_server.py

# Terminal 2: Run API client example
python examples/02_api_client.py
```

### With Real Data
```bash
# Install provider dependencies
pip install llm-token-analytics[providers]

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Run data collection example
python examples/03_data_collection.py
```

## Example Output

### Basic Simulation
```
Starting Basic LLM Token Pricing Simulation
==================================================
Configuration:
  - Simulations: 50,000
  - Mechanisms: per_token, bundle, hybrid, cached
  - Data source: Synthetic

Simulation Results
--------------------------------------------------

PER_TOKEN PRICING:
  Mean Cost:    $0.0234
  Median Cost:  $0.0198
  95th Percentile: $0.0567
  Coefficient of Variation: 0.421

🏆 Best Mechanisms
------------------------------
Lowest Mean Cost: cached ($0.0187)
Most Predictable: bundle (CV: 0.234)
Best P95 Cost:   cached ($0.0423)
```

### API Client
```
LLM Token Analytics API Client Example
==================================================
1. Checking API health...
   API is healthy - Version: 1.0.0

2. Running simulation...
   Simulation completed - ID: simulation_20250115_143022

4. Finding optimal mechanisms...
   Budget-conscious startup:
      Recommended: per_token
      Expected cost: $0.0234
   Enterprise customer:
      Recommended: bundle
      Expected cost: $0.0456
```

## Troubleshooting

### Import Errors
If you get import errors, make sure the library is installed:
```bash
pip install -e .
```

### API Connection Errors
Make sure the API server is running:
```bash
python api_server.py
```

### Missing Dependencies
Install optional dependencies as needed:
```bash
pip install llm-token-analytics[providers,viz,api]
```

## Next Steps

After running these examples:

1. **Read the Documentation**: Check out the full API reference
2. **Modify Examples**: Try changing parameters to see different results
3. **Create Custom Examples**: Build your own analysis workflows
4. **Contribute**: Share your examples with the community

## Contributing Examples

We welcome contributions of new examples! Please:

1. Follow the naming convention: `##_description.py`
2. Include comprehensive docstrings
3. Add error handling and user-friendly output
4. Update this README with your example
5. Test with different configurations

For questions or suggestions, please open an issue on GitHub.