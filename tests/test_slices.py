import numpy as np

from return_risk.slices import disparity_summary, slice_metrics


def test_slice_metrics_suppresses_small_groups_and_calculates_rates():
    result = slice_metrics(
        y_true=[0, 1, 0, 1, 1],
        probabilities=[0.1, 0.8, 0.7, 0.4, 0.9],
        groups=["A", "A", "A", "A", "B"],
        threshold=0.5,
        min_group_size=2,
    )
    row = result["eligible_groups"][0]
    assert row["group"] == "A"
    assert row["flagged_rate"] == 0.5
    assert row["precision"] == 0.5
    assert row["recall"] == 0.5
    assert result["suppressed_groups"] == [{"group": "B", "orders": 1}]


def test_disparity_summary_ignores_unavailable_metrics():
    rows = [
        {"group": "A", "flagged_rate": 0.2, "recall": None, "false_positive_rate": 0.1},
        {"group": "B", "flagged_rate": 0.5, "recall": None, "false_positive_rate": 0.4},
    ]
    result = disparity_summary(rows)
    assert np.isclose(result["flagged_rate"]["absolute_gap"], 0.3)
    assert result["recall"] is None
    assert result["false_positive_rate"]["maximum_group"] == "B"
