from __future__ import annotations

import numpy as np

from return_risk.policy import CostAssumptions, policy_cost_breakdown, standard_cost_scenarios
from return_risk.uncertainty import bootstrap_policy_intervals, percentile_interval


def test_standard_scenarios_increase_false_positive_cost() -> None:
    scenarios = standard_cost_scenarios()
    assert (
        scenarios["low_false_positive_cost"].false_positive_friction_cost
        < scenarios["base_case"].false_positive_friction_cost
        < scenarios["high_false_positive_cost"].false_positive_friction_cost
    )


def test_policy_breakdown_counts_every_order() -> None:
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.7, 0.2])
    breakdown = policy_cost_breakdown(
        y,
        probabilities,
        0.5,
        CostAssumptions(),
    )
    classified = sum(
        breakdown[name]
        for name in ("true_positive", "false_positive", "false_negative", "true_negative")
    )
    assert classified == len(y)


def test_bootstrap_policy_intervals_are_ordered() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.4, 0.5, 0.7, 0.9])
    result = bootstrap_policy_intervals(
        y,
        probabilities,
        0.5,
        CostAssumptions(),
        replicates=50,
    )
    interval = result["precision"]
    assert interval["lower"] <= interval["median"] <= interval["upper"]


def test_percentile_interval_uses_expected_median() -> None:
    interval = percentile_interval([1, 2, 3, 4, 5])
    assert interval["median"] == 3
