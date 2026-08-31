from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CostAssumptions:
    verification_cost: float = 25.0
    false_positive_friction_cost: float = 50.0
    missed_return_cost: float = 200.0
    intervention_effectiveness: float = 0.50


def standard_cost_scenarios() -> dict[str, CostAssumptions]:
    """Return transparent hypothetical scenarios for sensitivity analysis."""
    return {
        "low_false_positive_cost": CostAssumptions(
            verification_cost=15.0,
            false_positive_friction_cost=20.0,
            missed_return_cost=200.0,
            intervention_effectiveness=0.50,
        ),
        "base_case": CostAssumptions(
            verification_cost=25.0,
            false_positive_friction_cost=50.0,
            missed_return_cost=200.0,
            intervention_effectiveness=0.50,
        ),
        "high_false_positive_cost": CostAssumptions(
            verification_cost=40.0,
            false_positive_friction_cost=100.0,
            missed_return_cost=200.0,
            intervention_effectiveness=0.50,
        ),
    }


def policy_cost(y_true, probabilities, threshold: float, assumptions: CostAssumptions) -> float:
    y = np.asarray(y_true, dtype=int)
    flagged = np.asarray(probabilities, dtype=float) >= threshold
    cost = np.zeros(len(y), dtype=float)
    cost[flagged] += assumptions.verification_cost
    cost[flagged & (y == 0)] += assumptions.false_positive_friction_cost
    cost[flagged & (y == 1)] += (
        1 - assumptions.intervention_effectiveness
    ) * assumptions.missed_return_cost
    cost[(~flagged) & (y == 1)] += assumptions.missed_return_cost
    return float(cost.sum())


def policy_cost_breakdown(
    y_true,
    probabilities,
    threshold: float,
    assumptions: CostAssumptions,
) -> dict:
    y = np.asarray(y_true, dtype=int)
    flagged = np.asarray(probabilities, dtype=float) >= threshold
    true_positive = int((flagged & (y == 1)).sum())
    false_positive = int((flagged & (y == 0)).sum())
    false_negative = int(((~flagged) & (y == 1)).sum())
    true_negative = int(((~flagged) & (y == 0)).sum())
    total_cost = policy_cost(y, probabilities, threshold, assumptions)
    no_intervention_cost = float(y.sum() * assumptions.missed_return_cost)
    savings = no_intervention_cost - total_cost
    return {
        "orders": int(len(y)),
        "orders_flagged": int(flagged.sum()),
        "flagged_rate": round(float(flagged.mean()), 6),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "total_cost": round(total_cost, 2),
        "cost_per_1000_orders": round(total_cost / len(y) * 1000, 2),
        "savings_vs_no_intervention": round(savings, 2),
        "savings_per_1000_orders": round(savings / len(y) * 1000, 2),
    }


def select_cost_threshold(
    y_true,
    probabilities,
    assumptions: CostAssumptions | None = None,
    grid_size: int = 501,
) -> dict:
    assumptions = assumptions or CostAssumptions()
    thresholds = np.linspace(0.0, 1.0, grid_size)
    costs = np.array(
        [policy_cost(y_true, probabilities, threshold, assumptions) for threshold in thresholds]
    )
    best_index = int(np.argmin(costs))
    best_threshold = float(thresholds[best_index])
    no_intervention_cost = policy_cost(y_true, probabilities, 1.000001, assumptions)
    verify_all_cost = policy_cost(y_true, probabilities, 0.0, assumptions)
    return {
        "assumptions": asdict(assumptions),
        "selected_threshold": round(best_threshold, 6),
        "selected_total_cost": round(float(costs[best_index]), 2),
        "selected_cost_per_1000_orders": round(
            float(costs[best_index] / len(y_true) * 1000), 2
        ),
        "no_intervention_total_cost": round(no_intervention_cost, 2),
        "verify_all_total_cost": round(verify_all_cost, 2),
        "savings_vs_no_intervention": round(no_intervention_cost - float(costs[best_index]), 2),
        "selected_policy": policy_cost_breakdown(
            y_true,
            probabilities,
            best_threshold,
            assumptions,
        ),
    }


def build_policy_frontier(y_true, probabilities, grid_size: int = 501) -> list[dict]:
    """Aggregate validation outcomes across thresholds without retaining row-level data."""
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if len(y) == 0 or len(y) != len(scores):
        raise ValueError("Labels and probabilities must have the same non-zero length.")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2.")

    frontier = []
    for threshold in np.linspace(0.0, 1.0, grid_size):
        flagged = scores >= threshold
        true_positive = int((flagged & (y == 1)).sum())
        false_positive = int((flagged & (y == 0)).sum())
        false_negative = int(((~flagged) & (y == 1)).sum())
        true_negative = int(((~flagged) & (y == 0)).sum())
        frontier.append(
            {
                "threshold": round(float(threshold), 6),
                "orders": int(len(y)),
                "orders_flagged": int(flagged.sum()),
                "flagged_rate": round(float(flagged.mean()), 6),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
            }
        )
    return frontier


def load_policy_frontier(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("evaluation_partition") != "validation_only":
        raise RuntimeError("Policy frontier must be derived from validation data only.")
    if payload.get("test_set_accessed_for_selection") is not False:
        raise RuntimeError("Policy frontier cannot use held-out test results for selection.")
    if not payload.get("frontier"):
        raise RuntimeError("Policy frontier is empty.")
    return payload


def evaluate_policy_frontier(
    frontier: list[dict],
    assumptions: CostAssumptions,
    max_review_rate: float = 1.0,
) -> dict:
    """Choose the lowest-cost validation policy subject to review capacity."""
    if not 0 <= max_review_rate <= 1:
        raise ValueError("max_review_rate must be between 0 and 1.")
    if not 0 <= assumptions.intervention_effectiveness <= 1:
        raise ValueError("intervention_effectiveness must be between 0 and 1.")
    if min(
        assumptions.verification_cost,
        assumptions.false_positive_friction_cost,
        assumptions.missed_return_cost,
    ) < 0:
        raise ValueError("Cost assumptions cannot be negative.")

    curve = []
    for point in frontier:
        if point["flagged_rate"] > max_review_rate + 1e-12:
            continue
        flagged = point["true_positive"] + point["false_positive"]
        positive_orders = point["true_positive"] + point["false_negative"]
        total_cost = (
            flagged * assumptions.verification_cost
            + point["false_positive"] * assumptions.false_positive_friction_cost
            + point["true_positive"]
            * (1 - assumptions.intervention_effectiveness)
            * assumptions.missed_return_cost
            + point["false_negative"] * assumptions.missed_return_cost
        )
        no_intervention_cost = positive_orders * assumptions.missed_return_cost
        savings = no_intervention_cost - total_cost
        precision = point["true_positive"] / flagged if flagged else 0.0
        recall = point["true_positive"] / positive_orders if positive_orders else 0.0
        curve.append(
            {
                **point,
                "precision": round(float(precision), 6),
                "recall": round(float(recall), 6),
                "total_cost": round(float(total_cost), 2),
                "savings_vs_no_intervention": round(float(savings), 2),
                "savings_per_1000_orders": round(
                    float(savings / point["orders"] * 1000), 2
                ),
            }
        )
    if not curve:
        raise ValueError("No policy frontier point satisfies the review capacity.")

    selected = min(curve, key=lambda point: (point["total_cost"], point["threshold"]))
    return {
        "assumptions": asdict(assumptions),
        "max_review_rate": max_review_rate,
        "selected": selected,
        "curve": curve,
    }


def expected_loss_exposure(risk_score: float, assumed_return_loss: float) -> float:
    """Convert a probability into a transparent, non-causal exposure estimate."""
    if not 0 <= risk_score <= 1:
        raise ValueError("risk_score must be between 0 and 1.")
    if assumed_return_loss < 0:
        raise ValueError("assumed_return_loss cannot be negative.")
    return round(float(risk_score * assumed_return_loss), 2)
