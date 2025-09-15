# Test Suite for LLM Token Analytics Library

This directory contains a comprehensive test suite that validates functionality, enforces contracts, and ensures code quality for the LLM Token Analytics library.

## Test Structure

```
tests/
├── conftest.py                     # Pytest configuration and shared fixtures
├── pytest.ini                     # Pytest settings and markers
├── test_requirements.txt           # Testing dependencies
├── run_tests.py                   # Test runner script
├── test_schemas.py                # Dataclass and schema validation tests
├── test_syntax_imports.py         # Syntax and import validation tests
├── test_contracts.py              # Contract enforcement and interface tests
├── test_integration.py            # End-to-end integration tests
├── test_mocks.py                  # Mock tests for external dependencies
├── test_performance_edge_cases.py # Performance and edge case tests
└── README.md                      # This file
```

## Test Categories

### 1. Schema Validation Tests (`test_schemas.py`)
- **Purpose**: Validate dataclass structures and type enforcement
- **Coverage**: SimulationConfig, SimulationResults, TokenDistribution, CollectorConfig
- **Key Features**:
  - Field validation and defaults
  - Type checking and coercion
  - Method behavior testing
  - Edge case handling

### 2. Syntax and Import Tests (`test_syntax_imports.py`)
- **Purpose**: Ensure all modules have valid Python syntax and proper imports
- **Coverage**: All Python modules in the package
- **Key Features**:
  - AST parsing for syntax validation
  - Import dependency checking
  - Code quality checks (no print statements, proper docstrings)
  - Python version compatibility

### 3. Contract Enforcement Tests (`test_contracts.py`)
- **Purpose**: Validate abstract base classes and interface contracts
- **Coverage**: BaseCollector, PricingMechanism, and their implementations
- **Key Features**:
  - Abstract method implementation requirements
  - Interface consistency
  - Polymorphic behavior
  - Method signature validation

### 4. Integration Tests (`test_integration.py`)
- **Purpose**: Test end-to-end workflows and component interactions
- **Coverage**: Complete simulation workflows, data processing pipelines
- **Key Features**:
  - Full simulation workflows
  - Data collection and processing
  - Configuration loading
  - Results persistence

### 5. Mock Tests (`test_mocks.py`)
- **Purpose**: Test external dependencies with mocked responses
- **Coverage**: API collectors, file I/O, network operations
- **Key Features**:
  - HTTP request mocking
  - File system mocking
  - Environment variable mocking
  - Error condition simulation

### 6. Performance and Edge Case Tests (`test_performance_edge_cases.py`)
- **Purpose**: Test performance characteristics and boundary conditions
- **Coverage**: Large datasets, extreme values, resource constraints
- **Key Features**:
  - Performance scaling tests
  - Memory usage monitoring
  - Edge case handling
  - Numerical stability

## Quick Start

### Install Test Dependencies

```bash
# Install testing requirements
pip install -r tests/test_requirements.txt
```

### Run Tests

Using the test runner script (recommended):

```bash
# Quick tests (fast unit tests only)
python tests/run_tests.py quick

# Full test suite
python tests/run_tests.py full

# Specific categories
python tests/run_tests.py performance
python tests/run_tests.py integration
python tests/run_tests.py contracts

# With coverage
python tests/run_tests.py coverage

# Parallel execution
python tests/run_tests.py parallel

# Validate environment
python tests/run_tests.py validate
```

Using pytest directly:

```bash
# All tests
pytest

# Quick tests only
pytest -m "not slow"

# With coverage
pytest --cov=llm_token_analytics --cov-report=html

# Specific test file
pytest tests/test_schemas.py

# Parallel execution
pytest -n auto
```

## Test Markers

The test suite uses markers to categorize tests:

- `unit`: Unit tests for individual components
- `integration`: Integration tests for component interactions
- `performance`: Performance and benchmark tests
- `mock`: Tests using mocked dependencies
- `slow`: Slow-running tests (> 5 seconds)
- `network`: Tests requiring network access
- `api`: Tests that make actual API calls
- `edge_case`: Edge case and boundary condition tests
- `contract`: Contract and interface enforcement tests
- `schema`: Data schema validation tests

### Running Specific Categories

```bash
# Only fast tests
pytest -m "not slow"

# Only performance tests
pytest -m performance

# Only unit tests
pytest -m unit

# Exclude network tests
pytest -m "not network"
```

## Configuration

### pytest.ini
- Test discovery settings
- Coverage configuration
- Logging setup
- Warning filters

### conftest.py
- Shared fixtures for all tests
- Session-level configuration
- Data generation utilities
- Mock setups

## Test Data and Fixtures

### Available Fixtures

- `basic_simulation_config`: Standard simulation configuration
- `sample_token_data`: Representative token usage data
- `problematic_token_data`: Data with various issues for testing cleaning
- `mock_simulation_results`: Pre-generated simulation results
- `performance_timer`: Timer for performance tests
- `memory_monitor`: Memory usage monitoring

### Creating Test Data

```python
def test_my_feature(sample_token_data):
    # Use pre-generated sample data
    processor = DataProcessor()
    result = processor.clean_data(sample_token_data)
    assert len(result) > 0
```

## Writing New Tests

### Best Practices

1. **Use descriptive test names**: `test_distribution_fitting_with_lognormal_data()`
2. **Follow AAA pattern**: Arrange, Act, Assert
3. **Use appropriate fixtures**: Leverage existing fixtures for common setup
4. **Add markers**: Use `@pytest.mark.unit`, `@pytest.mark.slow`, etc.
5. **Test edge cases**: Include boundary conditions and error cases

### Example Test Structure

```python
import pytest
from llm_token_analytics import SimulationConfig

class TestMyComponent:
    """Test suite for MyComponent."""

    def test_basic_functionality(self, basic_simulation_config):
        """Test basic component functionality."""
        # Arrange
        component = MyComponent(basic_simulation_config)

        # Act
        result = component.process()

        # Assert
        assert result is not None
        assert isinstance(result, expected_type)

    @pytest.mark.slow
    def test_performance_with_large_dataset(self, large_token_dataset):
        """Test performance with large dataset."""
        # Performance test implementation
        pass

    @pytest.mark.edge_case
    def test_empty_input_handling(self):
        """Test handling of empty input."""
        # Edge case test implementation
        pass
```

## Coverage Requirements

- **Minimum coverage**: 80%
- **Target coverage**: 90%+
- **Coverage reports**: Generated in `htmlcov/` directory

### Viewing Coverage

```bash
# Generate coverage report
pytest --cov=llm_token_analytics --cov-report=html

# Open in browser (macOS)
open htmlcov/index.html
```

## Continuous Integration

The test suite is designed for CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r tests/test_requirements.txt
    python tests/run_tests.py full
```

## Performance Monitoring

Performance tests monitor:
- Execution time scaling
- Memory usage
- Resource constraints
- Large dataset processing

### Performance Thresholds

- Simulation (10k iterations): < 5 seconds
- Distribution fitting (10k samples): < 2 seconds
- Data processing (50k records): < 10 seconds
- Memory usage increase: < 500MB

## Debugging Tests

### Verbose Output
```bash
pytest -v --tb=long
```

### Debug Specific Test
```bash
pytest tests/test_schemas.py::TestSimulationConfig::test_default_values -v -s
```

### Logging
Test logs are saved to `tests.log` and displayed in console when enabled.

## Contributing

When adding new features:

1. **Add corresponding tests** for new functionality
2. **Update existing tests** if interfaces change
3. **Add performance tests** for computationally intensive features
4. **Include edge case tests** for boundary conditions
5. **Update test documentation** if new test patterns are introduced

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure package is installed in development mode (`pip install -e .`)
2. **Missing dependencies**: Install test requirements (`pip install -r tests/test_requirements.txt`)
3. **Slow tests**: Use `pytest -m "not slow"` to skip performance tests
4. **Memory issues**: Run tests with `--forked` or reduce dataset sizes

### Environment Issues

```bash
# Validate test environment
python tests/run_tests.py validate

# Check package installation
python -c "import llm_token_analytics; print(llm_token_analytics.__version__)"
```

---

This comprehensive test suite ensures the reliability, performance, and maintainability of the LLM Token Analytics library through rigorous validation of all components and workflows.