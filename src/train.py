"""Train and evaluate a reproducible Adult income classification pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import (
    FEATURES,
    TARGET,
    build_preprocessor,
    load_adult_csv,
    split_features_target,
)

RANDOM_STATE = 42


def evaluate(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    probabilities = model.predict_proba(x)[:, 1]
    predictions = (probabilities >= 0.5).astype("int8")
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
    }


def train(
    data_dir: Path = Path("data/raw"),
    model_dir: Path = Path("models"),
) -> dict[str, object]:
    train_path = data_dir / "adult.data"
    test_path = data_dir / "adult.test"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            "Dataset files are missing. Run: python scripts/download_data.py"
        )

    training_frame = load_adult_csv(train_path)
    official_test_frame = load_adult_csv(test_path, is_test=True)
    x, y = split_features_target(training_frame)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    validation_metrics = evaluate(pipeline, x_valid, y_valid)

    # The official test set is held out from every fitting/model-selection step.
    x_official, y_official = split_features_target(official_test_frame)
    official_metrics = evaluate(pipeline, x_official, y_official)

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / "adult_income_pipeline.joblib")
    metadata = {
        "model_name": "LogisticRegression",
        "features": FEATURES,
        "target": TARGET,
        "random_state": RANDOM_STATE,
        "validation_metrics": validation_metrics,
        "official_test_metrics": official_metrics,
        "training_rows": int(len(x_train)),
        "validation_rows": int(len(x_valid)),
        "official_test_rows": int(len(x_official)),
        "threshold": 0.5,
        "notes": (
            "Preprocessing is fitted inside a Pipeline on training rows only; "
            "the official test set is never used for fitting or model selection."
        ),
    }
    (model_dir / "metrics.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
