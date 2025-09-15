# Makefile for LLM Token Analytics Library
# =========================================

.PHONY: help install test clean build run docker-build docker-up docker-down docs lint format

# Variables
PYTHON := python3
PIP := pip3
PROJECT_NAME := llm-token-analytics
DOCKER_IMAGE := $(PROJECT_NAME):latest
PORT := 5000

# Default target
help:
	@echo "LLM Token Analytics - Available Commands"
	@echo "========================================"
	@echo ""
	@echo "Development:"
	@echo "  make install       Install the library and dependencies"
	@echo "  make dev-install   Install with development dependencies"
	@echo "  make test          Run unit tests"
	@echo "  make lint          Run code linting"
	@echo "  make format        Format code with black"
	@echo "  make clean         Clean build artifacts"
	@echo ""
	@echo "Running:"
	@echo "  make run-api       Run the API server"
	@echo "  make run-cli       Run the CLI interface"
	@echo "  make run-examples  Run example scripts"
	@echo "  make run-dashboard Run the dashboard"
	@echo ""
	@echo "Data Collection:"
	@echo "  make collect       Collect data from all providers"
	@echo "  make simulate      Run a simulation"
	@echo "  make analyze       Analyze collected data"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-up     Start all services"
	@echo "  make docker-down   Stop all services"
	@echo "  make docker-logs   View service logs"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs          Generate documentation"
	@echo "  make serve-docs    Serve documentation locally"
	@echo ""

# Installation targets
install:
	$(PIP) install -e .
	@echo "✓ Library installed successfully"

dev-install:
	$(PIP) install -e ".[dev,viz]"
	@echo "✓ Development dependencies installed"

requirements:
	$(PIP) install -r requirements.txt
	@echo "✓ Requirements installed"

# Testing targets
test:
	$(PYTHON) -m pytest tests/ -v --cov=llm_token_analytics --cov-report=html
	@echo "✓ Tests completed. Coverage report in htmlcov/index.html"

test-quick:
	$(PYTHON) -m pytest tests/ -v -k "not integration"
	@echo "✓ Quick tests completed"

test-integration:
	$(PYTHON) -m pytest tests/ -v -k "integration"
	@echo "✓ Integration tests completed"

# Code quality targets
lint:
	$(PYTHON) -m pylint llm_token_analytics/
	$(PYTHON) -m mypy llm_token_analytics/
	@echo "✓ Linting completed"

format:
	$(PYTHON) -m black llm_token_analytics/ tests/ examples/
	$(PYTHON) -m isort llm_token_analytics/ tests/ examples/
	@echo "✓ Code formatted"

check-format:
	$(PYTHON) -m black --check llm_token_analytics/ tests/ examples/
	@echo "✓ Format check completed"

# Cleaning targets
clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache .coverage htmlcov/
	rm -rf *.pyc */*.pyc */*/*.pyc
	rm -rf data/*.parquet data/*.csv data/*.json
	rm -rf logs/*.log
	rm -rf cache/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleaned build artifacts"

clean-data:
	rm -rf data/* results/* cache/*
	@echo "✓ Cleaned data files"

# Running targets
run-api:
	$(PYTHON) api_server.py
	@echo "✓ API server started on http://localhost:$(PORT)"

run-cli:
	$(PYTHON) -m llm_token_analytics.cli
	@echo "✓ CLI started"

run-examples:
	$(PYTHON) examples/example_usage.py
	@echo "✓ Examples completed"

run-dashboard:
	$(PYTHON) -c "from llm_token_analytics.cli import main; main()" visualize dashboard --serve
	@echo "✓ Dashboard started on http://localhost:8050"

# Data operations
collect:
	$(PYTHON) -m llm_token_analytics.cli collect fetch --provider all --output data/collected.parquet
	@echo "✓ Data collection completed"

collect-openai:
	$(PYTHON) -m llm_token_analytics.cli collect fetch --provider openai --output data/openai.parquet
	@echo "✓ OpenAI data collected"

collect-anthropic:
	$(PYTHON) -m llm_token_analytics.cli collect fetch --provider anthropic --output data/anthropic.parquet
	@echo "✓ Anthropic data collected"

simulate:
	$(PYTHON) -m llm_token_analytics.cli simulate run \
		--iterations 100000 \
		--mechanisms per_token bundle hybrid cached \
		--output results/simulation.json
	@echo "✓ Simulation completed"

simulate-quick:
	$(PYTHON) -m llm_token_analytics.cli simulate run \
		--iterations 10000 \
		--mechanisms per_token bundle \
		--output results/quick_simulation.json
	@echo "✓ Quick simulation completed"

analyze:
	$(PYTHON) -m llm_token_analytics.cli analyze distributions data/collected.parquet \
		--output results/analysis.json
	@echo "✓ Analysis completed"

# Docker targets
docker-build:
	docker build -t $(DOCKER_IMAGE) .
	@echo "✓ Docker image built: $(DOCKER_IMAGE)"

docker-up:
	docker-compose up -d
	@echo "✓ Services started"
	@echo "  API: http://localhost:5000"
	@echo "  Dashboard: http://localhost:8050"

docker-down:
	docker-compose down
	@echo "✓ Services stopped"

docker-logs:
	docker-compose logs -f

docker-shell:
	docker run -it --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/results:/app/results \
		$(DOCKER_IMAGE) /bin/bash

docker-clean:
	docker-compose down -v
	docker rmi $(DOCKER_IMAGE)
	@echo "✓ Docker resources cleaned"

# Documentation targets
docs:
	$(PYTHON) -m sphinx-apidoc -o docs/source llm_token_analytics
	cd docs && make html
	@echo "✓ Documentation built in docs/_build/html"

serve-docs:
	cd docs/_build/html && $(PYTHON) -m http.server 8000
	@echo "✓ Documentation served at http://localhost:8000"

# Database targets
db-init:
	$(PYTHON) scripts/init_db.py
	@echo "✓ Database initialized"

db-migrate:
	$(PYTHON) scripts/migrate_db.py
	@echo "✓ Database migrated"

db-backup:
	pg_dump -h localhost -U analytics llm_analytics > backups/db_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "✓ Database backed up"

# Environment setup
env-setup:
	cp .env.example .env
	@echo "✓ Environment file created. Please edit .env with your API keys"

check-env:
	@$(PYTHON) -c "import os; \
		keys = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY']; \
		missing = [k for k in keys if not os.getenv(k)]; \
		print('✗ Missing keys:', missing) if missing else print('✓ All API keys configured')"

# Release targets
version:
	@$(PYTHON) -c "from llm_token_analytics import __version__; print(__version__)"

bump-patch:
	bumpversion patch

bump-minor:
	bumpversion minor

bump-major:
	bumpversion major

release: clean test
	$(PYTHON) setup.py sdist bdist_wheel
	@echo "✓ Release packages built in dist/"

publish-test:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*
	@echo "✓ Published to TestPyPI"

publish:
	twine upload dist/*
	@echo "✓ Published to PyPI"

# Monitoring targets
monitor:
	@while true; do \
		clear; \
		echo "=== LLM Token Analytics Monitor ==="; \
		echo ""; \
		echo "API Status:"; \
		curl -s http://localhost:5000/health | jq .; \
		echo ""; \
		echo "Recent Logs:"; \
		tail -n 10 logs/llm_analytics.log 2>/dev/null || echo "No logs"; \
		echo ""; \
		echo "Data Files:"; \
		ls -lh data/*.parquet 2>/dev/null || echo "No data files"; \
		sleep 5; \
	done

# Development shortcuts
dev: dev-install
	@echo "✓ Development environment ready"

ci: clean lint test
	@echo "✓ CI checks passed"

all: clean install test lint docs
	@echo "✓ Full build completed"

# Performance profiling
profile:
	$(PYTHON) -m cProfile -o profile.stats examples/example_usage.py
	$(PYTHON) -m pstats profile.stats
	@echo "✓ Profiling completed"

benchmark:
	$(PYTHON) scripts/benchmark.py
	@echo "✓ Benchmark completed"

# Utility targets
count-lines:
	@echo "Lines of code:"
	@find llm_token_analytics -name "*.py" | xargs wc -l | tail -1

check-security:
	$(PYTHON) -m bandit -r llm_token_analytics/
	@echo "✓ Security check completed"

update-deps:
	$(PIP) list --outdated
	$(PIP) install --upgrade pip setuptools wheel
	@echo "✓ Dependencies checked"

.DEFAULT_GOAL := help
