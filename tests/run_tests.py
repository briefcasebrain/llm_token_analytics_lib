#!/usr/bin/env python3
"""
Test Runner Script for LLM Token Analytics Library
==================================================
Comprehensive test runner with multiple execution modes and reporting.
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path
import time
import json


class TestRunner:
    """Test runner with various execution modes."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_dir = self.project_root / "tests"

    def run_quick_tests(self):
        """Run only fast unit tests."""
        print("🚀 Running quick tests (unit tests only)...")
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir),
            "-m", "not slow and not performance",
            "--tb=short",
            "-v",
            "--durations=5"
        ]
        return subprocess.run(cmd, cwd=self.project_root)

    def run_full_tests(self):
        """Run all tests including slow ones."""
        print("🔍 Running full test suite...")
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir),
            "-v",
            "--cov=llm_token_analytics",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--durations=10"
        ]
        return subprocess.run(cmd, cwd=self.project_root)

    def run_performance_tests(self):
        """Run only performance tests."""
        print("⚡ Running performance tests...")
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir),
            "-m", "performance",
            "-v",
            "--durations=0",
            "--tb=short"
        ]
        return subprocess.run(cmd, cwd=self.project_root)

    def run_integration_tests(self):
        """Run only integration tests."""
        print("🔗 Running integration tests...")
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir),
            "-m", "integration",
            "-v",
            "--tb=short"
        ]
        return subprocess.run(cmd, cwd=self.project_root)

    def run_contract_tests(self):
        """Run contract and schema validation tests."""
        print("📋 Running contract and schema tests...")
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir / "test_contracts.py"),
            str(self.test_dir / "test_schemas.py"),
            "-v",
            "--tb=short"
        ]
        return subprocess.run(cmd, cwd=self.project_root)

    def run_mock_tests(self):
        """Run tests with mocked dependencies."""
        print("🎭 Running mock tests...")
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir / "test_mocks.py"),
            "-v",
            "--tb=short"
        ]
        return subprocess.run(cmd, cwd=self.project_root)

    def run_syntax_tests(self):
        """Run syntax and import validation tests."""
        print("✅ Running syntax validation tests...")
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir / "test_syntax_imports.py"),
            "-v",
            "--tb=short"
        ]
        return subprocess.run(cmd, cwd=self.project_root)

    def run_parallel_tests(self, num_workers=None):
        """Run tests in parallel."""
        if num_workers is None:
            num_workers = "auto"

        print(f"🚀 Running tests in parallel with {num_workers} workers...")
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir),
            f"-n", str(num_workers),
            "-m", "not slow",
            "--tb=short",
            "-v"
        ]
        return subprocess.run(cmd, cwd=self.project_root)

    def run_with_coverage(self):
        """Run tests with detailed coverage reporting."""
        print("📊 Running tests with detailed coverage...")
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir),
            "--cov=llm_token_analytics",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--cov-report=xml",
            "--cov-fail-under=80",
            "-v"
        ]
        return subprocess.run(cmd, cwd=self.project_root)

    def run_specific_file(self, test_file):
        """Run tests from a specific file."""
        test_path = self.test_dir / test_file
        if not test_path.exists():
            print(f"❌ Test file not found: {test_path}")
            return subprocess.CompletedProcess([], 1)

        print(f"🎯 Running tests from {test_file}...")
        cmd = [
            "python", "-m", "pytest",
            str(test_path),
            "-v",
            "--tb=short"
        ]
        return subprocess.run(cmd, cwd=self.project_root)

    def run_with_profile(self):
        """Run tests with performance profiling."""
        print("📈 Running tests with performance profiling...")
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir),
            "--profile",
            "--profile-svg",
            "-m", "not slow",
            "-v"
        ]
        return subprocess.run(cmd, cwd=self.project_root)

    def validate_environment(self):
        """Validate test environment setup."""
        print("🔍 Validating test environment...")

        # Check if pytest is installed
        try:
            import pytest
            print(f"✅ pytest version: {pytest.__version__}")
        except ImportError:
            print("❌ pytest not installed")
            return False

        # Check if coverage is available
        try:
            import coverage
            print(f"✅ coverage version: {coverage.__version__}")
        except ImportError:
            print("⚠️  coverage not installed (optional)")

        # Check main library is importable
        try:
            import llm_token_analytics
            print(f"✅ llm_token_analytics importable: {llm_token_analytics.__version__}")
        except ImportError as e:
            print(f"❌ Cannot import llm_token_analytics: {e}")
            return False

        # Check test requirements
        test_requirements_file = self.test_dir / "test_requirements.txt"
        if test_requirements_file.exists():
            print(f"✅ Test requirements file found: {test_requirements_file}")
        else:
            print("⚠️  Test requirements file not found")

        print("✅ Environment validation complete")
        return True

    def generate_report(self):
        """Generate a comprehensive test report."""
        print("📝 Generating comprehensive test report...")

        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_results": {}
        }

        # Run different test categories
        test_categories = [
            ("syntax", self.run_syntax_tests),
            ("contracts", self.run_contract_tests),
            ("unit", self.run_quick_tests),
            ("integration", self.run_integration_tests),
            ("performance", self.run_performance_tests)
        ]

        for category, runner_func in test_categories:
            print(f"\n--- Running {category} tests ---")
            start_time = time.time()
            result = runner_func()
            end_time = time.time()

            report_data["test_results"][category] = {
                "return_code": result.returncode,
                "duration": end_time - start_time,
                "status": "PASSED" if result.returncode == 0 else "FAILED"
            }

        # Save report
        report_file = self.project_root / "test_report.json"
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)

        print(f"\n📊 Test report saved to: {report_file}")

        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        for category, data in report_data["test_results"].items():
            status_emoji = "✅" if data["status"] == "PASSED" else "❌"
            print(f"{status_emoji} {category.upper()}: {data['status']} ({data['duration']:.1f}s)")

        return report_data


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="LLM Token Analytics Test Runner")
    parser.add_argument("mode", nargs="?", default="quick",
                        choices=["quick", "full", "performance", "integration",
                                "contracts", "mocks", "syntax", "parallel",
                                "coverage", "profile", "validate", "report"],
                        help="Test execution mode")
    parser.add_argument("--file", "-f", help="Run specific test file")
    parser.add_argument("--workers", "-w", type=int, help="Number of parallel workers")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    runner = TestRunner()

    print("🧪 LLM Token Analytics Test Runner")
    print("=" * 50)

    if args.mode == "validate":
        if not runner.validate_environment():
            sys.exit(1)
        return

    if args.file:
        result = runner.run_specific_file(args.file)
    elif args.mode == "quick":
        result = runner.run_quick_tests()
    elif args.mode == "full":
        result = runner.run_full_tests()
    elif args.mode == "performance":
        result = runner.run_performance_tests()
    elif args.mode == "integration":
        result = runner.run_integration_tests()
    elif args.mode == "contracts":
        result = runner.run_contract_tests()
    elif args.mode == "mocks":
        result = runner.run_mock_tests()
    elif args.mode == "syntax":
        result = runner.run_syntax_tests()
    elif args.mode == "parallel":
        result = runner.run_parallel_tests(args.workers)
    elif args.mode == "coverage":
        result = runner.run_with_coverage()
    elif args.mode == "profile":
        result = runner.run_with_profile()
    elif args.mode == "report":
        report_data = runner.generate_report()
        # Exit with error if any tests failed
        failed_tests = [cat for cat, data in report_data["test_results"].items()
                       if data["status"] == "FAILED"]
        if failed_tests:
            print(f"\n❌ Failed test categories: {', '.join(failed_tests)}")
            sys.exit(1)
        return
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)

    # Exit with the same code as pytest
    if hasattr(result, 'returncode'):
        print(f"\n{'✅ Tests passed!' if result.returncode == 0 else '❌ Tests failed!'}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()