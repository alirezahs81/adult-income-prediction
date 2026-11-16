# Model Evaluation Report

## Evaluation provenance

- Repository: [alirezahs81/adult-income-prediction](https://github.com/alirezahs81/adult-income-prediction)
- Branch: `main`
- Evaluated commit: `545a6f8ca6cbe4be7ff3f9145bc168aaa75c91e4`
- Workflow run: [Real Dataset Model Evaluation #43906059980](https://github.com/alirezahs81/adult-income-prediction/actions/runs/43906059980)
- Dataset: [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult)
- Model: class-balanced Logistic Regression baseline
- Decision threshold: `0.50`
- Random state: `42`

This report records the metrics emitted by the linked successful GitHub Actions run. It is a snapshot for reproducibility and portfolio documentation; rerunning training or changing dependencies/code may produce different results.

## Data split

| Split | Rows | Use |
|---|---:|---|
| Training | 26,048 | Fit preprocessing and classifier |
| Validation | 6,513 | Validation/model-development evaluation |
| Official UCI test | 16,281 | Final held-out evaluation |

Preprocessing is contained in a scikit-learn Pipeline and is fitted on training rows only. The official test set is not used to fit transformations, train the classifier, or select model settings.

## Official test metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.8070757 (80.71%) |
| Precision | 0.5609967 (56.10%) |
| Recall | 0.8429537 (84.30%) |
| F1-score | 0.6736623 (67.37%) |
| ROC-AUC | 0.9036971 (90.37%) |

The relatively high recall alongside lower precision indicates that, at the 0.50 threshold, the model identifies a large share of positive cases but also produces a meaningful number of false positives. The operating point should not be interpreted as appropriate for any real-world eligibility or screening decision.

## Validation metrics

| Metric | Value |
|---|---:|
| Accuracy | 80.75% |
| Precision | 56.65% |
| Recall | 85.27% |
| F1-score | 68.08% |
| ROC-AUC | 90.76% |

## Automated verification

The same evaluation workflow completed successfully, including:

- Download and verification of the official UCI data files.
- Training and evaluation without test leakage.
- Validation-only subgroup and threshold analysis.
- Validation of the generated metrics report.
- Upload of the trained pipeline and evaluation report as a workflow artifact.

The companion [CI run #43906060010](https://github.com/alirezahs81/adult-income-prediction/actions/runs/43906060010) also completed successfully: 18 tests passed, reported statement coverage was 90%, Ruff passed, package build/install succeeded, and the Docker build plus live API smoke test passed.

## Fairness and responsible-use caveats

Subgroup metrics are descriptive diagnostics, not a fairness certification, legal-compliance assessment, or proof that disparate impacts are absent. Group estimates depend on dataset composition and sample size. The threshold analysis uses validation data only; it does not authorize automatically changing the production/API threshold.

The UCI Adult dataset reflects historical census patterns and contains sensitive demographic attributes, including sex and race. This baseline is intended for education and experimentation only. Do not use it for hiring, lending, insurance, public benefits, eligibility, or other consequential decisions about individuals.

## Reproducing the evaluation

From the repository root, install dependencies, download the canonical UCI dataset, and run:

~~~bash
python scripts/download_data.py
python -m src.train
python -m scripts.analyze_fairness --data-dir data/raw --output-dir reports/fairness
pytest
ruff check src scripts tests
~~~

See the main [README](../README.md) for environment setup, API usage, and Docker instructions.
