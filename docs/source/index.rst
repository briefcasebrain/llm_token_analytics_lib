LLM Token Analytics Documentation
==================================

Welcome to the LLM Token Analytics library documentation. This library provides comprehensive tools for analyzing and simulating LLM token usage patterns and pricing mechanisms.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api_reference
   examples
   api_server
   development

Features
--------

- **Multi-Provider Support**: Collect usage data from OpenAI, Anthropic, and Google AI
- **Advanced Simulation**: Monte Carlo simulations with 100,000+ iterations
- **Multiple Pricing Mechanisms**: Per-token, bundle, hybrid, cached, outcome-based, and dynamic pricing
- **Statistical Analysis**: Distribution fitting, correlation analysis, and risk metrics
- **Interactive Visualizations**: Dashboards and plots for insights
- **Production Ready**: Docker support, CLI interface, and modular REST API

Quick Start
-----------

Install the library:

.. code-block:: bash

   pip install llm-token-analytics

Run your first simulation:

.. code-block:: python

   from llm_token_analytics import TokenSimulator, SimulationConfig

   config = SimulationConfig(
       n_simulations=10_000,
       mechanisms=['per_token', 'bundle', 'hybrid']
   )

   simulator = TokenSimulator(config)
   results = simulator.run()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`