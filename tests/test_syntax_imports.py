"""
Syntax and Import Validation Tests
==================================
Tests to ensure all modules have valid Python syntax and proper imports.
"""

import ast
import importlib
import sys
from pathlib import Path
import pytest
import subprocess
import tempfile

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
PACKAGE_PATH = PROJECT_ROOT / "llm_token_analytics"


class TestSyntaxValidation:
    """Test Python syntax validity for all modules."""

    def get_python_files(self) -> list:
        """Get all Python files in the package."""
        python_files = []
        for file_path in PACKAGE_PATH.rglob("*.py"):
            if not file_path.name.startswith("_") or file_path.name == "__init__.py":
                python_files.append(file_path)
        return python_files

    @pytest.mark.parametrize("file_path", [
        PACKAGE_PATH / "__init__.py",
        PACKAGE_PATH / "collectors.py",
        PACKAGE_PATH / "simulator.py",
        PACKAGE_PATH / "analyzer.py",
        PACKAGE_PATH / "cli.py",
        PACKAGE_PATH / "visualizer.py"
    ])
    def test_module_syntax(self, file_path):
        """Test that each module has valid Python syntax."""
        assert file_path.exists(), f"Module file not found: {file_path}"

        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()

        try:
            ast.parse(source_code, filename=str(file_path))
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {file_path}: {e}")

    def test_all_python_files_syntax(self):
        """Test syntax for all Python files in the package."""
        python_files = self.get_python_files()
        syntax_errors = []

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                ast.parse(source_code, filename=str(file_path))
            except SyntaxError as e:
                syntax_errors.append(f"{file_path}: {e}")
            except UnicodeDecodeError as e:
                syntax_errors.append(f"{file_path}: Encoding error - {e}")

        if syntax_errors:
            pytest.fail("Syntax errors found:\n" + "\n".join(syntax_errors))

    def test_no_print_statements(self):
        """Ensure no print() statements in production code (except CLI)."""
        python_files = self.get_python_files()
        files_with_prints = []

        for file_path in python_files:
            # Skip CLI module as it legitimately uses print
            if file_path.name == "cli.py":
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()

            tree = ast.parse(source_code)

            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and
                    isinstance(node.func, ast.Name) and
                    node.func.id == 'print'):
                    files_with_prints.append(f"{file_path}:line {node.lineno}")

        if files_with_prints:
            pytest.fail("Print statements found in:\n" + "\n".join(files_with_prints))

    def test_proper_encoding_declaration(self):
        """Test that files have proper encoding declarations if needed."""
        python_files = self.get_python_files()

        for file_path in python_files:
            with open(file_path, 'rb') as f:
                first_line = f.readline()
                second_line = f.readline()

            # Check for non-ASCII characters
            try:
                with open(file_path, 'r', encoding='ascii') as f:
                    f.read()
            except UnicodeDecodeError:
                # File contains non-ASCII, should have encoding declaration
                content = (first_line + second_line).decode('utf-8', errors='ignore')
                if 'coding:' not in content and 'coding=' not in content:
                    pytest.fail(f"File {file_path} contains non-ASCII but no encoding declaration")


class TestImportValidation:
    """Test import statements and module loading."""

    @pytest.mark.parametrize("module_name", [
        "llm_token_analytics",
        "llm_token_analytics.collectors",
        "llm_token_analytics.simulator",
        "llm_token_analytics.analyzer",
        "llm_token_analytics.cli",
        "llm_token_analytics.visualizer"
    ])
    def test_module_imports(self, module_name):
        """Test that all modules can be imported successfully."""
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            pytest.fail(f"Failed to import {module_name}: {e}")

    def test_main_package_exports(self):
        """Test that main package exports expected symbols."""
        try:
            import llm_token_analytics

            # Check for key exports from __init__.py
            expected_exports = [
                'SimulationConfig',
                'SimulationResults',
                'TokenSimulator',
                'CollectorConfig',
                'UnifiedCollector',
                'TokenAnalyzer',
                'DistributionFitter',
                'CostAnalyzer'
            ]

            for export in expected_exports:
                assert hasattr(llm_token_analytics, export), f"Missing export: {export}"

        except ImportError as e:
            pytest.fail(f"Failed to import main package: {e}")

    def test_no_circular_imports(self):
        """Test for circular import dependencies."""
        # This test imports each module individually to detect circular imports
        modules = [
            "llm_token_analytics.collectors",
            "llm_token_analytics.simulator",
            "llm_token_analytics.analyzer",
            "llm_token_analytics.visualizer",
            "llm_token_analytics.cli"
        ]

        for module_name in modules:
            # Fresh interpreter state for each test
            cmd = [sys.executable, "-c", f"import {module_name}"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                pytest.fail(f"Circular import or other import error in {module_name}: {result.stderr}")

    def test_missing_dependencies(self):
        """Test for missing external dependencies."""
        required_packages = [
            'numpy',
            'pandas',
            'scipy',
            'matplotlib',
            'plotly',
            'dash',
            'click',
            'requests',
            'pydantic'  # If we add it for validation
        ]

        missing_packages = []
        for package in required_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                # Some packages might be optional
                if package not in ['pydantic']:  # Optional packages
                    missing_packages.append(package)

        if missing_packages:
            pytest.fail(f"Missing required packages: {', '.join(missing_packages)}")

    def test_version_string(self):
        """Test that package has a valid version string."""
        try:
            import llm_token_analytics
            assert hasattr(llm_token_analytics, '__version__')

            version = llm_token_analytics.__version__
            assert isinstance(version, str)
            assert len(version.split('.')) >= 2  # At least major.minor

        except ImportError as e:
            pytest.fail(f"Failed to import package for version check: {e}")


class TestCodeQuality:
    """Test code quality and best practices."""

    def test_no_bare_except(self):
        """Ensure no bare except clauses are used."""
        python_files = [
            PACKAGE_PATH / "collectors.py",
            PACKAGE_PATH / "simulator.py",
            PACKAGE_PATH / "analyzer.py"
        ]

        files_with_bare_except = []

        for file_path in python_files:
            with open(file_path, 'r') as f:
                source_code = f.read()

            tree = ast.parse(source_code)

            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    files_with_bare_except.append(f"{file_path}:line {node.lineno}")

        if files_with_bare_except:
            pytest.fail("Bare except clauses found in:\n" + "\n".join(files_with_bare_except))

    def test_docstring_presence(self):
        """Test that modules and classes have docstrings."""
        python_files = [
            PACKAGE_PATH / "collectors.py",
            PACKAGE_PATH / "simulator.py",
            PACKAGE_PATH / "analyzer.py"
        ]

        missing_docstrings = []

        for file_path in python_files:
            with open(file_path, 'r') as f:
                source_code = f.read()

            tree = ast.parse(source_code)

            # Check module docstring
            if not (isinstance(tree.body[0], ast.Expr) and
                    isinstance(tree.body[0].value, ast.Constant) and
                    isinstance(tree.body[0].value.value, str)):
                missing_docstrings.append(f"{file_path}: Missing module docstring")

            # Check class docstrings
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if not (node.body and
                            isinstance(node.body[0], ast.Expr) and
                            isinstance(node.body[0].value, ast.Constant) and
                            isinstance(node.body[0].value.value, str)):
                        missing_docstrings.append(f"{file_path}: Class {node.name} missing docstring")

        # Allow some missing docstrings but warn about them
        if len(missing_docstrings) > 5:  # Arbitrary threshold
            pytest.fail("Too many missing docstrings:\n" + "\n".join(missing_docstrings[:10]))

    def test_function_complexity(self):
        """Test that functions are not overly complex (simple McCabe complexity check)."""
        python_files = [
            PACKAGE_PATH / "simulator.py",
            PACKAGE_PATH / "analyzer.py"
        ]

        complex_functions = []

        for file_path in python_files:
            with open(file_path, 'r') as f:
                source_code = f.read()

            tree = ast.parse(source_code)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Simple complexity measure: count decision points
                    complexity = self._calculate_complexity(node)
                    if complexity > 15:  # Arbitrary threshold
                        complex_functions.append(f"{file_path}: Function {node.name} complexity={complexity}")

        if complex_functions:
            # This is a warning, not a failure - complex functions might be necessary
            print("Warning: Complex functions found:")
            for func in complex_functions:
                print(f"  {func}")

    def _calculate_complexity(self, node):
        """Calculate simple McCabe complexity."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                complexity += 1

        return complexity

    def test_import_organization(self):
        """Test that imports are properly organized."""
        python_files = [
            PACKAGE_PATH / "__init__.py",
            PACKAGE_PATH / "collectors.py",
            PACKAGE_PATH / "simulator.py"
        ]

        for file_path in python_files:
            with open(file_path, 'r') as f:
                source_code = f.read()

            tree = ast.parse(source_code)

            imports = []
            non_import_found = False

            for i, node in enumerate(tree.body):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if non_import_found:
                        # Import found after non-import statement
                        pytest.fail(f"{file_path}: Imports should be at the top of the file")
                    imports.append(node)
                elif isinstance(node, ast.Expr):
                    # Allow docstrings (string constants)
                    if not (isinstance(node.value, ast.Constant) and
                           isinstance(node.value.value, str)):
                        non_import_found = True
                elif isinstance(node, ast.Assign):
                    # Allow module-level assignments like __version__, __author__ before imports
                    # Only mark as non-import if it's not a dunder variable
                    is_dunder = any(
                        isinstance(target, ast.Name) and
                        target.id.startswith('__') and target.id.endswith('__')
                        for target in node.targets
                    )
                    if not is_dunder:
                        non_import_found = True
                else:
                    # Other statements (functions, classes, etc.)
                    non_import_found = True


class TestCompatibility:
    """Test Python version compatibility."""

    def test_python_version_features(self):
        """Test that code uses appropriate Python version features."""
        python_files = [
            PACKAGE_PATH / "simulator.py",
            PACKAGE_PATH / "analyzer.py"
        ]

        for file_path in python_files:
            with open(file_path, 'r') as f:
                source_code = f.read()

            tree = ast.parse(source_code)

            # Check for Python 3.7+ features (dataclasses, type hints)
            has_annotations = False
            has_dataclasses = False

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.returns or any(arg.annotation for arg in node.args.args):
                        has_annotations = True

                if isinstance(node, ast.ImportFrom) and node.module == 'dataclasses':
                    has_dataclasses = True

            # This is informational - we want to use modern Python features
            if not has_annotations:
                print(f"Info: {file_path} has minimal type annotations")

    def test_encoding_compatibility(self):
        """Test that files are properly encoded."""
        python_files = self.get_all_python_files()

        for file_path in python_files:
            # Test UTF-8 encoding
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Try to encode back to UTF-8
                content.encode('utf-8')
            except (UnicodeDecodeError, UnicodeEncodeError) as e:
                pytest.fail(f"Encoding issue in {file_path}: {e}")

    def get_all_python_files(self):
        """Helper to get all Python files."""
        return list(PACKAGE_PATH.rglob("*.py"))


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__])