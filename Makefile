.PHONY: help install install-dev setup download-data train test test-api docker-build docker-run clean lint format type-check api

help:
	@echo "Adult Income Prediction"
	@echo "install       Install runtime dependencies"
	@echo "install-dev   Install development dependencies"
	@echo "setup         Install dependencies, download data, and train"
	@echo "download-data Download the UCI Adult dataset"
	@echo "train         Train and evaluate the baseline model"
	@echo "test          Run unit tests with coverage"
	@echo "lint          Run Ruff"
	@echo "api           Start the FastAPI service"
	@echo "docker-build  Build the API image"
	@echo "docker-run    Start the API with Docker Compose"

install:
	python -m pip install -r requirements.txt

install-dev: install
	python -m pip install -r requirements-dev.txt

setup: install-dev download-data train

download-data:
	python scripts/download_data.py

train:
	python -m src.train

test:
	pytest --cov=src --cov-report=term-missing

test-api: test

lint:
	ruff check src scripts tests

format:
	ruff format src scripts tests

type-check:
	mypy src

api:
	uvicorn src.api:app --host 127.0.0.1 --port 8000

docker-build:
	docker build -t adult-income-api .

docker-run:
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
