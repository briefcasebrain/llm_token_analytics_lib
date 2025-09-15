Getting Started
===============

Installation
------------

Basic Installation
~~~~~~~~~~~~~~~~~~

Install the core library:

.. code-block:: bash

   pip install llm-token-analytics

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~

For development with all dependencies:

.. code-block:: bash

   git clone https://github.com/your-username/llm-token-analytics.git
   cd llm-token-analytics
   pip install -e ".[dev,viz,api,providers]"

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~

The library supports optional feature sets:

.. code-block:: bash

   # Visualization features
   pip install llm-token-analytics[viz]

   # REST API server
   pip install llm-token-analytics[api]

   # Dashboard components
   pip install llm-token-analytics[dashboard]

   # LLM provider integrations
   pip install llm-token-analytics[providers]

   # All features
   pip install llm-token-analytics[all]

Configuration
-------------

Environment Variables
~~~~~~~~~~~~~~~~~~~~

Set up API keys for data collection:

.. code-block:: bash

   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   export GOOGLE_CLOUD_PROJECT="your-gcp-project"

Configuration Files
~~~~~~~~~~~~~~~~~

Create a configuration file at ``config/config.yaml``:

.. code-block:: yaml

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

First Steps
-----------

1. Basic Simulation
~~~~~~~~~~~~~~~~~~

Run a simple pricing simulation:

.. code-block:: python

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
       print(f"{mechanism}: Mean=${stats['mean']:.4f}")

2. Data Collection
~~~~~~~~~~~~~~~~~

Collect real usage data from providers:

.. code-block:: python

   from llm_token_analytics import UnifiedCollector

   collector = UnifiedCollector(['openai', 'anthropic'])
   data = collector.collect_all()
   print(f"Collected {len(data)} usage records")

3. Visualization
~~~~~~~~~~~~~~~

Create visualizations of your results:

.. code-block:: python

   from llm_token_analytics import SimulationVisualizer

   viz = SimulationVisualizer(results)
   fig = viz.create_comparison_plot()
   fig.show()

Next Steps
----------

- Explore the :doc:`api_reference` for detailed function documentation
- Check out :doc:`examples` for complete working examples
- Learn about the :doc:`api_server` for production deployments
- Read the :doc:`development` guide for contributing