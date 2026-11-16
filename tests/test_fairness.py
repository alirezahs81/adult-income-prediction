import numpy as np
import pandas as pd
import pytest

from scripts.analyze_fairness import subgroup_metrics, threshold_metrics


def test_threshold_metrics_compares_candidate_cutoffs() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.6, 0.4, 0.9])

    result = threshold_metrics(y_true, probabilities, (0.3, 0.5, 0.7))

    assert result["threshold"].tolist() == [0.3, 0.5, 0.7]
    assert result.loc[result["f1"].idxmax(), "f1"] == 0.8


def test_subgroup_metrics_reports_sex_race_and_intersections() -> None:
    frame = pd.DataFrame(
        {
            "sex": ["Female", "Female", "Male", "Male"],
            "race": ["White", "Black", "White", "Black"],
        }
    )
    y_true = pd.Series([0, 1, 1, 0])
    probabilities = np.array([0.2, 0.8, 0.7, 0.3])

    result = subgroup_metrics(
        frame, y_true, probabilities, min_group_size=1
    )

    assert set(result["attribute"]) == {"sex", "race", "sex_race"}
    assert (result["count"] == 2).all()
    assert {"true_positive_rate", "false_positive_rate"} <= set(result.columns)


def test_subgroup_metrics_excludes_small_groups() -> None:
    frame = pd.DataFrame({"sex": ["Female", "Male"], "race": ["White", "White"]})
    result = subgroup_metrics(
        frame, pd.Series([0, 1]), np.array([0.2, 0.8]), min_group_size=2
    )
    assert result.empty


def test_invalid_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        threshold_metrics(pd.Series([0, 1]), np.array([0.2, 0.8]), (1.0,))
