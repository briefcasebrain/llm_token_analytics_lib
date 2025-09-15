"""
LLM Token Analytics Examples Runner
====================================

Main script to run all example use cases.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """
    Run all examples with an interactive menu.
    """
    print("=" * 60)
    print("LLM TOKEN ANALYTICS - EXAMPLE SCRIPTS")
    print("=" * 60)
    print("\nSelect an example to run:")
    print("1. Complete Pipeline (Collection → Analysis → Simulation)")
    print("2. Sensitivity Analysis")
    print("3. Optimal Mechanism Selection")
    print("4. Custom Pricing Mechanisms")
    print("5. Monitoring Dashboard")
    print("6. Run All Examples")
    print("0. Exit")

    while True:
        choice = input("\nEnter your choice (0-6): ").strip()

        if choice == '1':
            print("\n" + "=" * 60)
            print("Running Complete Pipeline Example...")
            print("=" * 60)
            from complete_pipeline import run_complete_pipeline
            run_complete_pipeline()

        elif choice == '2':
            print("\n" + "=" * 60)
            print("Running Sensitivity Analysis Example...")
            print("=" * 60)
            from sensitivity_analysis import run_sensitivity_analysis
            run_sensitivity_analysis()

        elif choice == '3':
            print("\n" + "=" * 60)
            print("Running Optimal Mechanism Selection Example...")
            print("=" * 60)
            from optimal_mechanism import select_optimal_mechanism
            select_optimal_mechanism()

        elif choice == '4':
            print("\n" + "=" * 60)
            print("Running Custom Pricing Example...")
            print("=" * 60)
            from custom_pricing import test_custom_pricing
            test_custom_pricing()

        elif choice == '5':
            print("\n" + "=" * 60)
            print("Running Monitoring Dashboard Example...")
            print("=" * 60)
            from monitoring_dashboard import create_monitoring_dashboard
            app = create_monitoring_dashboard()
            if app and hasattr(app, 'run_server'):
                print("\nWould you like to start the dashboard server? (y/n)")
                if input().strip().lower() == 'y':
                    app.run_server(debug=True)

        elif choice == '6':
            print("\n" + "=" * 60)
            print("Running All Examples...")
            print("=" * 60)

            print("\n[1/5] Complete Pipeline")
            print("-" * 40)
            from complete_pipeline import run_complete_pipeline
            run_complete_pipeline()

            print("\n[2/5] Sensitivity Analysis")
            print("-" * 40)
            from sensitivity_analysis import run_sensitivity_analysis
            run_sensitivity_analysis()

            print("\n[3/5] Optimal Mechanism Selection")
            print("-" * 40)
            from optimal_mechanism import select_optimal_mechanism
            select_optimal_mechanism()

            print("\n[4/5] Custom Pricing Mechanisms")
            print("-" * 40)
            from custom_pricing import test_custom_pricing
            test_custom_pricing()

            print("\n[5/5] Monitoring Dashboard")
            print("-" * 40)
            from monitoring_dashboard import create_monitoring_dashboard
            create_monitoring_dashboard()

            print("\n" + "=" * 60)
            print("ALL EXAMPLES COMPLETED")
            print("=" * 60)

        elif choice == '0':
            print("Exiting...")
            sys.exit(0)

        else:
            print("Invalid choice. Please enter a number between 0 and 6.")
            continue

        print("\nWould you like to run another example? (y/n)")
        if input().strip().lower() != 'y':
            print("Thank you for using LLM Token Analytics!")
            break


if __name__ == "__main__":
    main()