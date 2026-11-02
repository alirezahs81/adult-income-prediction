# Makefile for Adult Income Prediction
# Usage: make <target>

.PHONY: help install install-dev setup download-data preprocess train test test-api docker-build docker-run docker-test clean lint format type-check

# Default target
help:
	@echo "Adult Income Prediction - Available commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  install        Install production dependencies"
	@echo "  install-dev    Install development dependencies"
	@echo "  setup          Full setup (install + download data + preprocess + train)"
	@echo ""
	@echo "Data & Training:"
	@echo "  download-data  Download dataset from UCI"
	@echo "  preprocess     Run data preprocessing"
	@echo "  train          Train all models"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  test           Run unit tests"
	@echo "  test-api       Run API tests (requires running server)"
	@echo "  lint           Run ruff linter"
	@echo "  format         Format code with ruff"
	@echo "  type-check     Run mypy type checking"
	@echo ""
	@echo "API & Docker:"
	@echo "  api            Run FastAPI server locally"
	@echo "  docker-build   Build Docker image"
	@echo "  docker-run     Run with docker-compose"
	@echo "  docker-test    Run API tests in Docker"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean          Remove cache, build artifacts"

# Installation
install:
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements-dev.txt

# Full setup pipeline
setup: install download-data preprocess train
	@echo "Setup complete! Run 'make api' to start the server."

# Data downloading
download-data:
	@echo "Downloading dataset from UCI..."
	python scripts/download_data.py

# Preprocessing
preprocess:
	@echo "Preprocessing data..."
	python src/preprocess.py

# Training
train:
	@echo "Training models..."
	python src/train.py

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-api:
	python test_api.py

# Code quality
lint:
	ruff check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/

type-check:
	mypy src/

# API
api:
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Docker
docker-build:
	docker build -t adult-income-api .

docker-run:
	docker-compose up --build

docker-test:
	docker-compose up --build --abort-on-container-exit test

# Cleanup
clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache
	rm -rf src/__pycache__ tests/__pycache__ scripts/__pycache__
	rm -rf *.egg-info dist build
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete