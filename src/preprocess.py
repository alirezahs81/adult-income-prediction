"""Data loading and leakage-safe preprocessing for the UCI Adult dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET: Final[str] = "income"
NUMERIC_FEATURES: Final[list[str]] = [
    "age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"
]
CATEGORICAL_FEATURES: Final[list[str]] = [
    "workclass", "education", "marital-status", "occupation", "relationship",
    "race", "sex", "native-country"
]
FEATURES: Final[list[str]] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
COLUMNS: Final[list[str]] = FEATURES + [TARGET]


def load_adult_csv(path: str | Path, *, is_test: bool = False) -> pd.DataFrame:
    """Load an official UCI Adult data file and normalize its target labels."""
    frame = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        na_values=["?", " ?"],
        skipinitialspace=True,
        comment="|",
        skiprows=1 if is_test else 0,
    )
    frame = frame.dropna(subset=[TARGET]).copy()
    labels = frame[TARGET].astype(str).str.strip().str.removesuffix(".")
    mapping = {"<=50K": 0, ">50K": 1}
    unknown = sorted(set(labels) - set(mapping))
    if unknown:
        raise ValueError(f"Unexpected income labels: {unknown}")
    frame[TARGET] = labels.map(mapping).astype("int8")
    return frame


def build_preprocessor() -> ColumnTransformer:
    """Fit transformations only on training rows when used inside a Pipeline."""
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and target without fitting any transformation."""
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return frame[FEATURES].copy(), frame[TARGET].astype("int8").copy()
