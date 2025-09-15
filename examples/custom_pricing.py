"""
Custom Pricing Mechanism Example
=================================

Demonstrates how to implement and test custom pricing mechanisms
beyond the standard offerings.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path if running from examples folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_token_analytics import (
    TokenSimulator,
    SimulationConfig,
    PricingMechanism
)


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


class SubscriptionPricing(PricingMechanism):
    """
    Subscription-based pricing with multiple tiers.
    """

    def __init__(self, config):
        super().__init__(config)
        self.tiers = [
            {'name': 'Basic', 'price': 29, 'included_tokens': 50_000, 'overage_rate': 0.15},
            {'name': 'Pro', 'price': 99, 'included_tokens': 250_000, 'overage_rate': 0.10},
            {'name': 'Enterprise', 'price': 499, 'included_tokens': 2_000_000, 'overage_rate': 0.06}
        ]
        self.current_tier = config.get('subscription_tier', 1)  # Default to Pro

    def calculate(self, input_tokens, output_tokens):
        total_tokens = input_tokens + output_tokens
        tier = self.tiers[self.current_tier]

        if isinstance(total_tokens, np.ndarray):
            costs = np.full_like(total_tokens, tier['price'], dtype=float)
            overage = np.maximum(0, total_tokens - tier['included_tokens'])
            costs += overage * tier['overage_rate'] / 1000
            return costs
        else:
            cost = tier['price']
            overage = max(0, total_tokens - tier['included_tokens'])
            cost += overage * tier['overage_rate'] / 1000
            return cost


class UsageBasedCommitment(PricingMechanism):
    """
    Usage-based commitment pricing with prepaid credits.
    """

    def __init__(self, config):
        super().__init__(config)
        self.commitment_levels = [
            {'credits': 100, 'bonus': 0.00},    # No bonus
            {'credits': 500, 'bonus': 0.10},    # 10% bonus
            {'credits': 1000, 'bonus': 0.20},   # 20% bonus
            {'credits': 5000, 'bonus': 0.30}    # 30% bonus
        ]
        self.rate_per_1k = 0.08  # Base rate

    def calculate(self, input_tokens, output_tokens):
        total_tokens = input_tokens + output_tokens

        # Simulate different commitment levels
        commitment = self.commitment_levels[2]  # $1000 commitment
        effective_credits = commitment['credits'] * (1 + commitment['bonus'])
        tokens_per_credit = 1000 / self.rate_per_1k

        if isinstance(total_tokens, np.ndarray):
            costs = total_tokens * self.rate_per_1k / 1000
            # Apply discount based on commitment
            costs *= (1 - commitment['bonus'])
            return costs
        else:
            cost = total_tokens * self.rate_per_1k / 1000
            cost *= (1 - commitment['bonus'])
            return cost


def test_custom_pricing():
    """
    Example of implementing and testing custom pricing mechanisms.
    """
    print("\n" + "=" * 60)
    print("CUSTOM PRICING MECHANISM EXAMPLE")
    print("=" * 60)

    # Create simulator with standard mechanisms
    config = SimulationConfig(
        n_simulations=50_000,
        mechanisms=['per_token', 'bundle', 'hybrid']
    )

    simulator = TokenSimulator(config)

    # Add custom mechanisms
    simulator.pricing_mechanisms['tiered'] = TieredPricing({})
    simulator.pricing_mechanisms['subscription'] = SubscriptionPricing({'subscription_tier': 1})
    simulator.pricing_mechanisms['commitment'] = UsageBasedCommitment({})

    # Update config with custom mechanisms
    config.mechanisms.extend(['tiered', 'subscription', 'commitment'])

    print("Added custom pricing mechanisms:")
    print("  - Tiered: Volume-based discounts")
    print("  - Subscription: Monthly subscription with overage")
    print("  - Commitment: Prepaid credits with bonus")

    print("\nRunning simulation with custom pricing...")

    # Run simulation
    results = simulator.run()

    # Compare results
    print("\nComparison with Standard Mechanisms:")
    print("-" * 80)
    print(f"{'Mechanism':<15} {'Mean':<10} {'Median':<10} {'P95':<10} {'CV':<8} {'Savings vs Per-Token'}")
    print("-" * 80)

    per_token_mean = results.mechanism_results['per_token']['mean']

    for mechanism, stats in results.mechanism_results.items():
        savings = (1 - stats['mean'] / per_token_mean) * 100 if per_token_mean > 0 else 0

        print(f"{mechanism:<15} "
              f"${stats['mean']:<9.4f} "
              f"${stats['median']:<9.4f} "
              f"${stats['p95']:<9.4f} "
              f"{stats['cv']:<7.2f} "
              f"{savings:+.1f}%")

    # Test with specific volumes
    print("\nDetailed Pricing Examples:")
    print("-" * 60)

    test_volumes = [5_000, 50_000, 500_000, 2_000_000]
    mechanisms_to_test = {
        'tiered': TieredPricing({}),
        'subscription': SubscriptionPricing({'subscription_tier': 1}),
        'commitment': UsageBasedCommitment({})
    }

    for volume in test_volumes:
        print(f"\nVolume: {volume:,} tokens")
        print("-" * 40)

        # Assume 40% input, 60% output
        input_tokens = volume * 0.4
        output_tokens = volume * 0.6

        for name, mechanism in mechanisms_to_test.items():
            cost = mechanism.calculate(input_tokens, output_tokens)
            effective_rate = (cost * 1000) / volume if volume > 0 else 0

            print(f"  {name:<15}: ${cost:>8.2f} (${effective_rate:.4f}/1k tokens)")

    # Create break-even analysis
    create_breakeven_analysis(mechanisms_to_test)

    return results


def create_breakeven_analysis(mechanisms):
    """
    Create a break-even analysis comparing different pricing mechanisms.
    """
    import plotly.graph_objects as go

    volumes = np.logspace(3, 7, 100)  # 1k to 10M tokens
    fig = go.Figure()

    colors = ['blue', 'green', 'red', 'purple', 'orange']

    for i, (name, mechanism) in enumerate(mechanisms.items()):
        costs = []
        for volume in volumes:
            input_tokens = volume * 0.4
            output_tokens = volume * 0.6
            cost = mechanism.calculate(input_tokens, output_tokens)
            costs.append(cost)

        fig.add_trace(go.Scatter(
            x=volumes,
            y=costs,
            mode='lines',
            name=name.capitalize(),
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                          'Volume: %{x:,.0f} tokens<br>' +
                          'Cost: $%{y:.2f}<br>' +
                          '<extra></extra>'
        ))

    fig.update_layout(
        title="Pricing Mechanism Break-even Analysis",
        xaxis_title="Token Volume",
        yaxis_title="Cost ($)",
        xaxis_type="log",
        yaxis_type="log",
        width=900,
        height=600,
        hovermode='x unified'
    )

    fig.update_xaxis(
        tickformat='.0f',
        tickvals=[1e3, 1e4, 1e5, 1e6, 1e7],
        ticktext=['1K', '10K', '100K', '1M', '10M']
    )

    fig.write_html('breakeven_analysis.html')
    print("\nBreak-even analysis saved to breakeven_analysis.html")

    # Find crossover points
    print("\nCrossover Points:")
    print("-" * 40)

    test_points = [10_000, 50_000, 100_000, 500_000, 1_000_000]
    for volume in test_points:
        costs = {}
        input_tokens = volume * 0.4
        output_tokens = volume * 0.6

        for name, mechanism in mechanisms.items():
            costs[name] = mechanism.calculate(input_tokens, output_tokens)

        best = min(costs.items(), key=lambda x: x[1])
        print(f"At {volume:>8,} tokens: {best[0]:<15} is cheapest (${best[1]:.2f})")

    return fig


if __name__ == "__main__":
    test_custom_pricing()