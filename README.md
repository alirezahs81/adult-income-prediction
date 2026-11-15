# Adult Income Prediction

A reproducible machine-learning project that estimates whether an adult's annual income is above $50K using the UCI Adult Census dataset.

> **Responsible-use note:** This dataset contains sensitive demographic attributes and historical societal biases. Predictions are educational/research outputs, not facts about a person's worth, ability, or future income. Do not use this model for hiring, lending, insurance, eligibility, or other consequential decisions.

## Current implementation

- Leakage-aware scikit-learn Pipeline: imputers, scaling, and one-hot encoding are fitted only on the training split.
- Logistic Regression baseline with class balancing and a fixed random seed.
- Separate validation and untouched official UCI test evaluation.
- FastAPI single and batch prediction endpoints.
- Unit tests and GitHub Actions CI.
- Docker image that does not require datasets or model artifacts to be committed.

No benchmark score is hard-coded as a project result. Run training to generate `models/metrics.json`; reported metrics depend on the dataset and installed library versions.

## Project structure

~~~text
adult-income-prediction/
├── .github/workflows/
│   ├── ci.yml
│   └── model-evaluation.yml
├── scripts/
│   └── download_data.py
├── src/
│   ├── api.py
│   ├── preprocess.py
│   └── train.py
├── tests/
│   ├── test_api.py
│   ├── test_preprocess.py
│   └── test_train.py
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
~~~

## Setup (Python 3.11+)

~~~bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
python scripts/download_data.py
python -m src.train
pytest
ruff check src scripts tests
~~~

The downloader saves the canonical UCI files to `data/raw/adult.data` and `data/raw/adult.test`. Dataset files are intentionally not committed.

Training writes:
- `models/adult_income_pipeline.joblib`
- `models/metrics.json`

The official test set is only used for final evaluation. It is not used to fit preprocessing, train the classifier, or select model settings.

## API

Start the service after training:

~~~bash
uvicorn src.api:app --host 127.0.0.1 --port 8000
~~~

- GET `/health` — liveness check (API process responds).
- GET `/ready` — readiness check (HTTP 200 only when the model is loaded; otherwise HTTP 503).
- GET `/model-info`
- GET `/example`
- POST `/predict`
- POST `/predict/batch` (1–100 instances)
- GET `/docs` (interactive OpenAPI documentation)

The API returns a predicted class and model probability. Probability is not a guarantee or a calibrated personal assessment. The response does not label probability as model confidence.

## Docker

Train the model locally first, then:

~~~bash
docker compose up --build
~~~

The model directory is mounted read-only at runtime. The container healthcheck uses `/ready`, so the container is not marked healthy until a model is loaded. If the artifact is absent, the API process remains live but readiness and prediction endpoints return HTTP 503.

**Network exposure:** Compose publishes the API on `127.0.0.1:8000` by default, so it is reachable only from the Docker host. Do not expose this unauthenticated API directly to a public network. A production deployment needs an authenticated gateway/reverse proxy, TLS, request limits, and appropriate monitoring.

**Model artifact security:** `joblib.load` uses Python pickle-based deserialization and can execute code. Only load model artifacts that you created or obtained from a trusted, integrity-verified source. Do not accept uploaded or third-party model files as `MODEL_PATH`.

## Data

Source: [UCI Adult dataset](https://archive.ics.uci.edu/dataset/2/adult)

The task is binary classification: <=50K versus >50K. The dataset includes six numeric and eight categorical input features. The UCI test labels contain a trailing period, which the loader normalizes.

## Limitations and fairness

The data reflects historical census patterns and includes protected or sensitive attributes such as sex and race. The baseline is not fairness-certified, probabilities are not calibrated, and the model may reproduce or amplify dataset biases. Review subgroup performance and legal/ethical requirements before any research deployment; do not use it for individual decisions.

## CI

GitHub Actions runs Ruff and pytest on pushes to the repair branch and pull requests targeting main. CI tests do not download the dataset or require a trained model. A separate workflow downloads the official UCI data, trains the baseline, and evaluates the untouched official test split.

## License

MIT. See LICENSE. Dataset terms and attribution are governed by UCI.
