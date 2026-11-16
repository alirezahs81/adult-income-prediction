# Adult Income Prediction

A reproducible end-to-end machine-learning project that predicts whether an adult's annual income is above **$50,000** using the [UCI Adult (Census Income) dataset](https://archive.ics.uci.edu/dataset/2/adult). The repository demonstrates leakage-aware model training, evaluation on the official test split, subgroup diagnostics, a REST API, Docker packaging, and automated CI.

> **Responsible-use warning:** This is an educational/research project. The dataset contains sensitive demographic attributes and reflects historical societal inequalities. Model outputs are not facts about a person's worth, ability, or future income. Do not use this model for hiring, lending, insurance, public benefits, eligibility, or any other consequential individual decision.

## Highlights

- **Leakage-aware preprocessing:** imputation, scaling, and one-hot encoding are fitted inside a scikit-learn Pipeline on training data only.
- **Reproducible baseline:** class-balanced Logistic Regression with a fixed random seed.
- **Honest evaluation:** validation data is kept separate from the untouched official UCI test set.
- **Fairness diagnostics:** descriptive sex, race, and intersectional subgroup metrics plus validation-only threshold comparisons.
- **Production-style interface:** FastAPI health/readiness checks and single/batch prediction endpoints.
- **Containerization:** Docker image, non-root runtime user, health check, and read-only model mount in Compose.
- **Automated quality gates:** Ruff, pytest, package installation/import check, Docker build, and live API smoke test in GitHub Actions.

## Latest official test results

The following results were produced by the successful real-data GitHub Actions evaluation on the official UCI test split, using the default decision threshold of **0.50**.

| Metric | Official test result |
|---|---:|
| Accuracy | **80.71%** |
| Precision | 56.10% |
| Recall | 84.30% |
| F1-score | 67.37% |
| ROC-AUC | **90.37%** |

**Evaluation split sizes**

| Split | Rows |
|---|---:|
| Training | 26,048 |
| Validation | 6,513 |
| Official UCI test | 16,281 |

The official test set is used only for final evaluation; it is not used to fit preprocessing, train the classifier, or select model settings. Metrics are a reproducible baseline snapshot, not a guarantee of performance on other populations or future data.

- [Latest real-data training and evaluation workflow](https://github.com/alirezahs81/adult-income-prediction/actions/runs/43910000422)
- [Latest CI workflow: tests, package build, Docker and API smoke test](https://github.com/alirezahs81/adult-income-prediction/actions/runs/43910000378)
- [Detailed evaluation report](reports/model_evaluation.md)

## Project structure

~~~text
adult-income-prediction/
├── .env.example
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── model-evaluation.yml
├── scripts/
│   ├── analyze_fairness.py
│   └── download_data.py
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── preprocess.py
│   └── train.py
├── tests/
│   ├── test_api.py
│   ├── test_fairness.py
│   ├── test_preprocess.py
│   └── test_train.py
├── reports/
│   └── model_evaluation.md
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── LICENSE
├── Makefile
├── pyproject.toml
├── README.md
├── requirements-dev.txt
└── requirements.txt
~~~

Generated datasets, trained model artifacts, and runtime outputs are created locally and are not committed as part of the source tree.

## Getting started

Python **3.11 or newer** is required.

### 1. Create an environment and install dependencies

~~~bash
python -m venv .venv
~~~

Activate it:

~~~powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
~~~

~~~bash
# macOS / Linux
source .venv/bin/activate
~~~

Install runtime and development dependencies:

~~~bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
~~~

### 2. Download the official dataset and train

~~~bash
python scripts/download_data.py
python -m src.train
~~~

The downloader saves the canonical UCI files under `data/raw/`. Dataset files are intentionally not committed to the repository.

Training creates:

- `models/adult_income_pipeline.joblib`
- `models/metrics.json`

### 3. Run tests and lint

~~~bash
pytest
ruff check src scripts tests
~~~

The latest verified CI run completed **18 tests successfully** and reported **90% statement coverage**. Ruff reported no issues. These are results from the linked CI run; local results may differ if dependencies or code change.

## Fairness and threshold diagnostics

Run the validation-only analysis after downloading the dataset:

~~~bash
python -m scripts.analyze_fairness --data-dir data/raw --output-dir reports/fairness
~~~

It generates:

- `reports/fairness/subgroup_metrics.csv`: descriptive metrics for sufficiently large sex, race, and sex-by-race groups.
- `reports/fairness/threshold_metrics.csv`: validation metrics across candidate thresholds from 0.20 to 0.80.
- `reports/fairness/summary.json`: split sizes and the threshold with the highest validation F1.

The script creates a stratified train/validation split and fits preprocessing and the classifier on the training partition only. The official UCI test set is not used for threshold selection. Groups smaller than the configured minimum are omitted because their estimates may be unstable.

These diagnostics are **not a fairness certification**, proof of legal compliance, or evidence that the model is suitable for consequential decisions. No alternative threshold is automatically deployed; the API/model baseline retains its documented default threshold unless a separately justified change is made.

## API

After training the model, start the service:

~~~bash
uvicorn src.api:app --host 127.0.0.1 --port 8000
~~~

Available endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/ready` | Readiness check; returns HTTP 503 if the model is unavailable |
| GET | `/model-info` | Model metadata |
| GET | `/example` | Example input |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch prediction (1–100 instances) |
| GET | `/docs` | Interactive OpenAPI documentation |

The API returns a predicted class and model probability. A probability is not a guarantee or a calibrated personal assessment, and should not be interpreted as an individual's intrinsic likelihood of earning a particular income.

## Docker

Train the model locally first, then build and start the service:

~~~bash
docker compose up --build
~~~

The model directory is mounted read-only at runtime. The container health check uses `/ready`; the service is not considered ready until a model is loaded. If the model artifact is missing, liveness can still succeed while readiness and prediction requests return HTTP 503.

**Network security:** Compose binds the API to `127.0.0.1:8000` by default. Do not expose this unauthenticated API directly to a public network. A production deployment would require authentication, TLS, request limits, monitoring, and a documented privacy/security review.

**Model artifact security:** `joblib.load` uses pickle-based deserialization and can execute code. Load only trusted, integrity-verified artifacts. Never accept arbitrary uploaded model files as `MODEL_PATH`.

## Dataset and limitations

The UCI Adult dataset is a historical census-derived dataset with a binary target (`<=50K` versus `>50K`). It includes six numeric and eight categorical input features. The official test labels use a trailing period, which the data loader normalizes.

Important limitations:

- Historical data may encode sampling, measurement, and societal biases.
- Subgroup metrics can be affected by group size and dataset composition.
- The baseline is not fairness-certified and its probabilities are not established as calibrated.
- Performance on the UCI test split does not establish generalization to other countries, time periods, or populations.
- This model is for learning and experimentation, not individual-level decision-making.

## Continuous integration

GitHub Actions runs Ruff, pytest, a package build/install/import check, and a Docker build with a live API smoke test. The Docker smoke test creates a deterministic temporary model, starts the container, checks `/health` and `/ready`, and sends a real request to `/predict`. CI does not need the full dataset for these checks.

A separate workflow downloads the official UCI data, trains the baseline, evaluates the untouched official test split, runs validation-only fairness diagnostics, validates the metrics report, and uploads the generated evaluation artifacts.

## License and attribution

The project code is licensed under the [MIT License](LICENSE). The dataset is provided by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult); consult UCI's dataset page for source details and applicable terms.
