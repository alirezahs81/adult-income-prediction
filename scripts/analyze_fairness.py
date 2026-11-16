"""Validation-only subgroup fairness and decision-threshold analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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

from src.preprocess import FEATURES, build_preprocessor, load_adult_csv

RANDOM_STATE = 42
DEFAULT_THRESHOLDS = tuple(round(value / 100, 2) for value in range(20, 81, 5))
SENSITIVE_COLUMNS = ("sex", "race")


def _binary_metrics(y_true: pd.Series, probabilities, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype("int8")
    result = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "positive_prediction_rate": float(predictions.mean()),
    }
    if y_true.nunique() == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    else:
        result["roc_auc"] = float("nan")
    return result


def subgroup_metrics(
    frame: pd.DataFrame,
    y_true: pd.Series,
    probabilities,
    *,
    threshold: float = 0.5,
    min_group_size: int = 30,
) -> pd.DataFrame:
    """Calculate descriptive metrics for sex, race, and intersectional groups."""
    if min_group_size < 1:
        raise ValueError("min_group_size must be at least 1")
    predictions = (probabilities >= threshold).astype("int8")
    rows: list[dict[str, object]] = []
    group_columns = list(SENSITIVE_COLUMNS)
    if all(column in frame.columns for column in SENSITIVE_COLUMNS):
        group_columns.append("sex_race")

    working = frame.copy()
    if "sex_race" in group_columns:
        working["sex_race"] = (
            working["sex"].astype(str).str.strip()
            + " | "
            + working["race"].astype(str).str.strip()
        )

    for column in group_columns:
        if column not in working:
            continue
        for group_value, indices in working.groupby(column, dropna=False).groups.items():
            positions = working.index.get_indexer(indices)
            if len(positions) < min_group_size:
                continue
            group_y = y_true.iloc[positions]
            group_probabilities = probabilities[positions]
            group_predictions = predictions[positions]
            tp = int(((group_y.to_numpy() == 1) & (group_predictions == 1)).sum())
            fp = int(((group_y.to_numpy() == 0) & (group_predictions == 1)).sum())
            tn = int(((group_y.to_numpy() == 0) & (group_predictions == 0)).sum())
            fn = int(((group_y.to_numpy() == 1) & (group_predictions == 0)).sum())
            metrics = _binary_metrics(group_y, group_probabilities, threshold)
            rows.append(
                {
                    "attribute": column,
                    "group": str(group_value),
                    "count": len(positions),
                    "threshold": threshold,
                    **metrics,
                    "true_positive_rate": tp / (tp + fn) if tp + fn else float("nan"),
                    "false_positive_rate": fp / (fp + tn) if fp + tn else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def threshold_metrics(
    y_true: pd.Series,
    probabilities,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Compare candidate thresholds on validation data only."""
    if not thresholds or any(not 0 < threshold < 1 for threshold in thresholds):
        raise ValueError("thresholds must be non-empty and between 0 and 1")
    rows = [
        {"threshold": threshold, **_binary_metrics(y_true, probabilities, threshold)}
        for threshold in thresholds
    ]
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def run_analysis(
    data_dir: Path = Path("data/raw"),
    output_dir: Path = Path("reports/fairness"),
    *,
    min_group_size: int = 30,
) -> dict[str, object]:
    """Fit on a training partition and analyze only the held-out validation partition."""
    dataset = load_adult_csv(data_dir / "adult.data")
    train_frame, validation_frame = train_test_split(
        dataset,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=dataset["income"],
    )
    model = Pipeline(
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
    model.fit(train_frame[FEATURES], train_frame["income"])
    probabilities = model.predict_proba(validation_frame[FEATURES])[:, 1]
    y_validation = validation_frame["income"].reset_index(drop=True)
    validation_features = validation_frame.reset_index(drop=True)

    groups = subgroup_metrics(
        validation_features,
        y_validation,
        probabilities,
        threshold=0.5,
        min_group_size=min_group_size,
    )
    thresholds = threshold_metrics(y_validation, probabilities)
    best_row = thresholds.loc[thresholds["f1"].idxmax()]
    summary = {
        "dataset": "UCI Adult",
        "random_state": RANDOM_STATE,
        "training_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "min_group_size": min_group_size,
        "baseline_threshold": 0.5,
        "threshold_selection_metric": "validation_f1",
        "best_validation_f1_threshold": float(best_row["threshold"]),
        "best_validation_f1": float(best_row["f1"]),
        "warning": (
            "Descriptive group metrics are not a fairness certification. "
            "Threshold selection uses validation data only; do not treat the "
            "selected threshold as suitable for consequential decisions."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    groups.to_csv(output_dir / "subgroup_metrics.csv", index=False)
    thresholds.to_csv(output_dir / "threshold_metrics.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/fairness"))
    parser.add_argument("--min-group-size", type=int, default=30)
    args = parser.parse_args()
    result = run_analysis(
        args.data_dir,
        args.output_dir,
        min_group_size=args.min_group_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
