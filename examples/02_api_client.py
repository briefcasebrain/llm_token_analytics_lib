#!/usr/bin/env python3
"""
REST API Client Example
========================

This example demonstrates how to interact with the LLM Token Analytics
REST API server for running simulations and analysis.
"""

import requests


class LLMAnalyticsAPIClient:
    """Client for interacting with the LLM Token Analytics API."""

    def __init__(self, base_url="http://localhost:5000"):
        """Initialize the API client."""
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

    def health_check(self):
        """Check if the API server is healthy."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Health check failed: {e}")

    def run_simulation(self, n_simulations=10000, mechanisms=None, data_source="synthetic"):
        """Run a pricing simulation via the API."""
        if mechanisms is None:
            mechanisms = ["per_token", "bundle", "hybrid"]

        payload = {
            "n_simulations": n_simulations,
            "mechanisms": mechanisms,
            "data_source": data_source
        }

        response = self.session.post(
            f"{self.base_url}/simulate",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()

    def get_results(self, result_id):
        """Get simulation results by ID."""
        response = self.session.get(f"{self.base_url}/results/{result_id}")
        response.raise_for_status()
        return response.json()

    def compare_mechanisms(self, simulation_id):
        """Compare pricing mechanisms from a simulation."""
        payload = {"simulation_id": simulation_id}

        response = self.session.post(
            f"{self.base_url}/compare",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()

    def optimize_mechanism(self, simulation_id, user_profile):
        """Find optimal pricing mechanism for a user profile."""
        payload = {
            "simulation_id": simulation_id,
            "user_profile": user_profile
        }

        response = self.session.post(
            f"{self.base_url}/optimize",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()

    def download_visualization(self, simulation_id, filename=None):
        """Download visualization for a simulation."""
        response = self.session.get(f"{self.base_url}/visualize/{simulation_id}")
        response.raise_for_status()

        if filename is None:
            filename = f"simulation_{simulation_id}_visualization.html"

        with open(filename, 'wb') as f:
            f.write(response.content)

        return filename


def main():
    """Demonstrate API client usage."""

    print("LLM Token Analytics API Client Example")
    print("=" * 50)

    # Initialize the API client
    client = LLMAnalyticsAPIClient()

    try:
        # 1. Health check
        print("1. Checking API health...")
        health = client.health_check()
        print(f"   API is healthy - Version: {health['version']}")

        # 2. Run simulation
        print("\n2. Running simulation...")
        simulation_response = client.run_simulation(
            n_simulations=25000,
            mechanisms=["per_token", "bundle", "hybrid", "cached"]
        )

        if simulation_response['success']:
            simulation_id = simulation_response['simulation_id']
            print(f"   Simulation completed - ID: {simulation_id}")

            # Display results
            print("\n   Results:")
            for mechanism, stats in simulation_response['results'].items():
                print(f"     {mechanism}: Mean=${stats['mean']:.4f}, "
                      f"P95=${stats['p95']:.4f}, CV={stats['cv']:.3f}")

            # 3. Compare mechanisms
            print("\n3. Comparing mechanisms...")
            comparison = client.compare_mechanisms(simulation_id)
            if comparison['success']:
                print("   Mechanism comparison completed")
                print(f"   Comparison data: {len(comparison['comparison'])} entries")

            # 4. Optimize for different user profiles
            print("\n4. Finding optimal mechanisms...")

            user_profiles = [
                {
                    "name": "Budget-conscious startup",
                    "profile": {
                        "risk_tolerance": "high",
                        "usage_volume": 10000,
                        "predictability_preference": 0.3,
                        "budget_constraint": 50
                    }
                },
                {
                    "name": "Enterprise customer",
                    "profile": {
                        "risk_tolerance": "low",
                        "usage_volume": 100000,
                        "predictability_preference": 0.9,
                        "budget_constraint": 500
                    }
                }
            ]

            for profile_info in user_profiles:
                optimization = client.optimize_mechanism(
                    simulation_id,
                    profile_info["profile"]
                )

                if optimization['success']:
                    print(f"   {profile_info['name']}:")
                    print(f"      Recommended: {optimization['recommended_mechanism']}")
                    print(f"      Expected cost: ${optimization['expected_cost']:.4f}")
                    print(f"      P95 cost: ${optimization['p95_cost']:.4f}")

            # 5. Download visualization
            print("\n5. Downloading visualization...")
            viz_filename = client.download_visualization(simulation_id)
            print(f"   Visualization saved as: {viz_filename}")

        else:
            print("   Simulation failed")

    except requests.RequestException as e:
        print(f"API request failed: {e}")
        print("\nMake sure the API server is running:")
        print("   python api_server.py")

    except Exception as e:
        print(f"Error: {e}")

    print("\nAPI client example completed!")


if __name__ == "__main__":
    main()
