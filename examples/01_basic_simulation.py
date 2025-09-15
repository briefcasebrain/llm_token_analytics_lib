#!/usr/bin/env python3
"""
Basic Simulation Example
=========================

This example demonstrates how to run a basic Monte Carlo simulation
comparing different pricing mechanisms for LLM token usage.
"""

from llm_token_analytics import TokenSimulator, SimulationConfig


def main():
    """Run a basic simulation comparing pricing mechanisms."""

    print("Starting Basic LLM Token Pricing Simulation")
    print("=" * 50)

    # Configure the simulation
    config = SimulationConfig(
        n_simulations=50_000,  # Number of Monte Carlo iterations
        mechanisms=['per_token', 'bundle', 'hybrid', 'cached'],
        use_empirical_data=False  # Use synthetic data for this example
    )

    print("Configuration:")
    print(f"  - Simulations: {config.n_simulations:,}")
    print(f"  - Mechanisms: {', '.join(config.mechanisms)}")
    print(f"  - Data source: {'Empirical' if config.use_empirical_data else 'Synthetic'}")
    print()

    # Create and run the simulator
    print("Running simulation...")
    simulator = TokenSimulator(config)
    results = simulator.run(progress_bar=True)

    # Display results
    print("\nSimulation Results")
    print("-" * 50)

    for mechanism, stats in results.mechanism_results.items():
        print(f"\n{mechanism.upper()} PRICING:")
        print(f"  Mean Cost:    ${stats['mean']:.4f}")
        print(f"  Median Cost:  ${stats['median']:.4f}")
        print(f"  Std Dev:      ${stats['std']:.4f}")
        print(f"  95th Percentile: ${stats['p95']:.4f}")
        print(f"  99th Percentile: ${stats['p99']:.4f}")
        print(f"  Coefficient of Variation: {stats['cv']:.3f}")

        if 'tail_ratio' in stats:
            print(f"  Tail Ratio:   {stats['tail_ratio']:.3f}")

    # Find the best mechanism for different criteria
    print("\nBest Mechanisms")
    print("-" * 30)

    # Lowest mean cost
    best_mean = min(results.mechanism_results.items(),
                    key=lambda x: x[1]['mean'])
    print(f"Lowest Mean Cost: {best_mean[0]} (${best_mean[1]['mean']:.4f})")

    # Most predictable (lowest CV)
    best_cv = min(results.mechanism_results.items(),
                  key=lambda x: x[1]['cv'])
    print(f"Most Predictable: {best_cv[0]} (CV: {best_cv[1]['cv']:.3f})")

    # Best P95 (good for budgeting)
    best_p95 = min(results.mechanism_results.items(),
                   key=lambda x: x[1]['p95'])
    print(f"Best P95 Cost:   {best_p95[0]} (${best_p95[1]['p95']:.4f})")

    print("\nSimulation completed successfully!")
    print(f"Total simulation time: {results.execution_time:.2f} seconds")

    return results


if __name__ == "__main__":
    results = main()
