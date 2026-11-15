from pathlib import Path

import pandas as pd

from src.preprocess import COLUMNS
from src.train import train


def _sample_row(index: int, income: str) -> dict[str, object]:
    high = index % 2 == 1
    return {
        "age": 25 + index,
        "workclass": "Private" if high else "State-gov",
        "fnlwgt": 100000 + index,
        "education": "Bachelors" if high else "HS-grad",
        "education-num": 13 if high else 9,
        "marital-status": "Married-civ-spouse" if high else "Never-married",
        "occupation": "Exec-managerial" if high else "Adm-clerical",
        "relationship": "Husband" if high else "Not-in-family",
        "race": "White",
        "sex": "Male" if high else "Female",
        "capital-gain": 5000 if income == ">50K" else 0,
        "capital-loss": 0,
        "hours-per-week": 40 + index % 10,
        "native-country": "United-States",
        "income": income,
    }


def test_training_saves_pipeline_and_metrics(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models"
    data_dir.mkdir()

    train_rows = [
        _sample_row(index, ">50K" if index % 2 else "<=50K")
        for index in range(30)
    ]
    test_rows = [
        _sample_row(index + 40, ">50K" if index % 2 else "<=50K")
        for index in range(10)
    ]

    pd.DataFrame(train_rows, columns=COLUMNS).to_csv(
        data_dir / "adult.data", index=False, header=False
    )
    test_csv = pd.DataFrame(test_rows, columns=COLUMNS).to_csv(
        index=False, header=False
    )
    (data_dir / "adult.test").write_text(
        "|1x3 Cross validator\n" + test_csv, encoding="utf-8"
    )

    result = train(data_dir=data_dir, model_dir=model_dir)

    assert result["training_rows"] == 24
    assert result["validation_rows"] == 6
    assert result["official_test_rows"] == 10
    assert 0.0 <= result["official_test_metrics"]["roc_auc"] <= 1.0
    assert (model_dir / "adult_income_pipeline.joblib").is_file()
    assert (model_dir / "metrics.json").is_file()
