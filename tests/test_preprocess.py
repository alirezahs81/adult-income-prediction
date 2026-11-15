from pathlib import Path

import pandas as pd
import pytest

from src.preprocess import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_preprocessor,
    load_adult_csv,
    split_features_target,
)


def test_load_adult_test_strips_period(tmp_path: Path) -> None:
    row = [
        "39", " State-gov", "77516", " Bachelors", "13", " Never-married",
        " Adm-clerical", " Not-in-family", " White", " Male", "2174", "0",
        "40", " United-States", " >50K.",
    ]
    path = tmp_path / "adult.test"
    path.write_text("|1x3 Cross validator\n" + ", ".join(row) + "\n", encoding="utf-8")
    frame = load_adult_csv(path, is_test=True)
    assert frame.loc[0, TARGET] == 1


def test_split_features_target() -> None:
    frame = pd.DataFrame([{**{name: 1 for name in FEATURES}, TARGET: 0}])
    x, y = split_features_target(frame)
    assert list(x.columns) == FEATURES
    assert y.tolist() == [0]


def test_missing_column_is_rejected() -> None:
    frame = pd.DataFrame([{TARGET: 0}])
    with pytest.raises(ValueError, match="missing required columns"):
        split_features_target(frame)


def test_preprocessor_ignores_unseen_categories() -> None:
    row = {
        **{name: 1 for name in NUMERIC_FEATURES},
        **{name: "category-a" for name in CATEGORICAL_FEATURES},
    }
    training = pd.DataFrame([row])
    validation = training.copy()
    validation.loc[0, "workclass"] = "previously-unseen"
    preprocessor = build_preprocessor()
    preprocessor.fit(training)
    transformed = preprocessor.transform(validation)
    assert transformed.shape[0] == 1


def test_loader_preserves_official_uci_column_order(tmp_path: Path) -> None:
    row = [
        "39", "Private", "77516", "Bachelors", "13", "Never-married",
        "Adm-clerical", "Not-in-family", "White", "Male", "2174", "0",
        "40", "United-States", "<=50K",
    ]
    path = tmp_path / "adult.data"
    path.write_text(", ".join(row) + "\n", encoding="utf-8")

    frame = load_adult_csv(path)

    assert frame.loc[0, "age"] == 39
    assert frame.loc[0, "workclass"] == "Private"
    assert frame.loc[0, "fnlwgt"] == 77516
    assert frame.loc[0, TARGET] == 0
