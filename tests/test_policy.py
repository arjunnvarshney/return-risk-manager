from __future__ import annotations

import numpy as np

from return_risk.policy import (
    CostAssumptions,
    build_policy_frontier,
    evaluate_policy_frontier,
    expected_loss_exposure,
    policy_cost,
    select_cost_threshold,
)


def test_perfect_ranking_can_reduce_cost() -> None:
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.10, 0.90, 0.95])
    assumptions = CostAssumptions(
        verification_cost=5,
        false_positive_friction_cost=20,
        missed_return_cost=100,
        intervention_effectiveness=1.0,
    )
    policy = select_cost_threshold(y, probabilities, assumptions)
    assert policy["savings_vs_no_intervention"] > 0


def test_no_intervention_cost_equals_missed_positive_cost() -> None:
    y = np.array([0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.8])
    assumptions = CostAssumptions(missed_return_cost=200)
    assert policy_cost(y, probabilities, 1.000001, assumptions) == 400


def test_policy_frontier_respects_review_capacity() -> None:
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.10, 0.90, 0.95])
    frontier = build_policy_frontier(y, probabilities, grid_size=11)
    result = evaluate_policy_frontier(
        frontier,
        CostAssumptions(
            verification_cost=5,
            false_positive_friction_cost=20,
            missed_return_cost=100,
            intervention_effectiveness=1.0,
        ),
        max_review_rate=0.5,
    )
    assert result["selected"]["flagged_rate"] <= 0.5
    assert result["selected"]["true_positive"] == 2
    assert result["selected"]["false_positive"] == 0
    assert result["selected"]["savings_per_1000_orders"] > 0


def test_expected_loss_exposure_is_probability_times_loss() -> None:
    assert expected_loss_exposure(0.4, 250) == 100.0
