import pytest

from return_risk.robustness import (
    fixed_capacity_audit,
    quality_guardrail,
    threshold_category_audit,
)


def test_fixed_capacity_audit_flags_exact_top_scores():
    result = fixed_capacity_audit(
        y_true=[0, 1, 1, 0],
        probabilities=[0.1, 0.9, 0.8, 0.7],
        groups=["Books", "Clothing", "Clothing", "Books"],
        orders_to_flag=2,
    )
    assert result["orders_flagged"] == 2
    assert result["precision"] == 1.0
    assert result["largest_flagged_group"] == "Clothing"
    assert result["largest_group_share"] == 1.0


def test_fixed_capacity_audit_rejects_invalid_capacity():
    with pytest.raises(ValueError, match="orders_to_flag"):
        fixed_capacity_audit([0, 1], [0.2, 0.8], ["A", "B"], 0)


def test_quality_guardrail_allows_small_losses_only():
    benchmark = {"roc_auc": 0.61, "average_precision": 0.42}
    passing = {"roc_auc": 0.601, "average_precision": 0.411}
    failing = {"roc_auc": 0.59, "average_precision": 0.41}
    assert quality_guardrail(benchmark, passing)["passed"] is True
    assert quality_guardrail(benchmark, failing)["passed"] is False


def test_threshold_category_audit_handles_empty_and_diverse_queues():
    empty = threshold_category_audit([0, 1], [0.2, 0.8], ["A", "B"], 0.9)
    assert empty["orders_flagged"] == 0
    assert empty["precision"] is None
    assert empty["largest_group_share"] is None

    diverse = threshold_category_audit([0, 1], [0.7, 0.8], ["A", "B"], 0.5)
    assert diverse["categories_represented"] == 2
    assert diverse["largest_group_share"] == 0.5
