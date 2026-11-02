# Adult Income Prediction

[![CI/CD](https://github.com/alirezahs81/adult-income-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/alirezahs81/adult-income-prediction/actions/workflows/ci.yml)
[![Docker Image](https://img.shields.io/badge/docker-ghcr.io-blue)](https://github.com/alirezahs81/adult-income-prediction/pkgs/container/adult-income-prediction)
[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

This project predicts whether an individual's annual income exceeds $50K based on the UCI Adult Census dataset.

## Model Results

| Model | Validation AUC | Test AUC | Test Accuracy |
|-------|----------------|----------|---------------|
| **GradientBoosting** (Best) | **0.9279** | **0.9256** | **0.8714** |
| RandomForest | 0.9196 | 0.9134 | 0.8130 |
| LogisticRegression | 0.8934 | 0.8863 | 0.7938 |

> **Best Model:** GradientBoosting Classifier with AUC 0.9256 on test data.

## Project Structure

```
adult-income-prediction/
├── .github/workflows/     # GitHub Actions CI/CD
├── src/                   # Main source code
│   ├── preprocess.py      # Data preprocessing pipeline
│   ├── train.py           # Model training and comparison
│   └── api.py             # FastAPI REST API
├── data/                  # Data files
│   ├── adult.csv          # Raw combined data (48,842 samples)
│   └── processed/         # Processed train/val/test splits
├── models/                # Trained model artifacts
│   ├── best_model.pkl     # Best model (GradientBoosting)
│   ├── preprocessor.pkl   # Fitted preprocessor
│   ├── label_encoder.pkl  # Label encoder
│   └── model_comparison.csv # Model comparison metrics
├── notebooks/             # Jupyter notebooks for EDA
├── tests/                 # Unit and integration tests
├── scripts/               # Utility scripts
├── Dockerfile             # Docker image definition
├── docker-compose.yml     # Docker Compose configuration
├── test_api.py            # API test suite
├── requirements.txt       # Production dependencies
├── requirements-dev.txt   # Development dependencies
├── pyproject.toml         # Project metadata and tool config
├── Makefile               # Automation commands
├── .env.example           # Environment variable template
└── README.md              # This file
```

## Quick Start

### Prerequisites
- Python 3.11+
- Docker (optional, for containerized deployment)

### 1. Local Installation
```bash
# Clone repository
git clone https://github.com/alirezahs81/adult-income-prediction.git
cd adult-income-prediction

# Create virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download data (if not present)
python scripts/download_data.py

# Preprocess data
python src/preprocess.py

# Train models
python src/train.py

# Run API
python src/api.py
# Or: uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### 2. Docker Deployment
```bash
# Build and run with docker-compose
docker-compose up --build

# Or build and run manually
docker build -t adult-income-api .
docker run -p 8000:8000 adult-income-api
```

API will be available at `http://localhost:8000`.

## API Documentation

### Swagger UI
Interactive documentation at: `http://localhost:8000/docs`

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| GET | `/model-info` | Model metadata |
| GET | `/example` | Example input payload |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch prediction (max 1000) |

### Example Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 39,
    "workclass": "State-gov",
    "fnlwgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
  }'
```

### Example Response
```json
{
  "prediction": "<=50K",
  "probability": 0.1234,
  "confidence": "high"
}
```

## Available Commands (Makefile)

```bash
make install          # Install production dependencies
make install-dev      # Install development dependencies
make setup            # Full setup: install + download + preprocess + train
make download-data    # Download dataset from UCI
make preprocess       # Run data preprocessing
make train            # Train all models
make test             # Run unit tests with coverage
make test-api         # Run API tests (requires running server)
make lint             # Run ruff linter
make format           # Format code with ruff
make type-check       # Run mypy type checking
make api              # Run FastAPI server locally
make docker-build     # Build Docker image
make docker-run       # Run with docker-compose
make clean            # Remove cache and build artifacts
```

## Testing

```bash
# Unit tests
pytest tests/ -v

# API tests (server must be running)
python test_api.py

# Docker test suite
docker-compose up --build --abort-on-container-exit test
```

## Data

Dataset sourced from [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult):
- 48,842 samples (32,561 train + 16,281 test)
- 14 features (6 numerical + 8 categorical)
- Binary target: >50K or <=50K

### Features
| Type | Features |
|------|----------|
| Numerical | age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week |
| Categorical | workclass, education, marital-status, occupation, relationship, race, sex, native-country |

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`) includes:
1. **Lint & Test** - Ruff linting, MyPy type checking, Pytest with coverage
2. **Build Docker** - Multi-stage build, push to GHCR with semantic tags
3. **Deploy Staging** - Automatic deployment from develop branch
4. **Deploy Production** - Deployment on GitHub Release publication

## Configuration

Environment variables (see `.env.example`):
- `MODEL_PATH`, `PREPROCESSOR_PATH`, `LABEL_ENCODER_PATH` - Model artifact paths
- `HOST`, `PORT` - Server binding
- `LOG_LEVEL` - Logging verbosity
- `ENVIRONMENT` - development/staging/production

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -m "feat: add new feature"`)
4. Push to branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## Changelog

### v1.0.0 (2026-11-02)
- Complete data preprocessing pipeline
- Model training and comparison (LogisticRegression, RandomForest, GradientBoosting)
- Hyperparameter optimization ready (Optuna integration)
- FastAPI REST API with validation and batch prediction
- Docker and Docker Compose support
- GitHub Actions CI/CD pipeline
- Unit and integration test suite
- Exploratory data analysis notebook
- Automated data download script
- Comprehensive documentation

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult) for the dataset
- [scikit-learn](https://scikit-learn.org/) for machine learning algorithms
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
- [Optuna](https://optuna.org/) for hyperparameter optimization