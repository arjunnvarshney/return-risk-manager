from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from return_risk.policy import CostAssumptions, policy_cost


def percentile_interval(values, confidence: float = 0.95) -> dict:
    array = np.asarray(values, dtype=float)
    alpha = 1 - confidence
    return {
        "lower": round(float(np.quantile(array, alpha / 2)), 6),
        "median": round(float(np.median(array)), 6),
        "upper": round(float(np.quantile(array, 1 - alpha / 2)), 6),
    }


def bootstrap_ranking_intervals(
    y_true,
    probabilities,
    replicates: int = 2_000,
    seed: int = 42,
) -> dict:
    y = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    rng = np.random.default_rng(seed)
    roc_auc_values = []
    average_precision_values = []
    top_10_precision_values = []
    top_count = max(1, int(round(len(y) * 0.10)))
    for _ in range(replicates):
        indices = rng.integers(0, len(y), size=len(y))
        sample_y = y[indices]
        sample_probabilities = probabilities[indices]
        if len(np.unique(sample_y)) < 2:
            continue
        roc_auc_values.append(roc_auc_score(sample_y, sample_probabilities))
        average_precision_values.append(
            average_precision_score(sample_y, sample_probabilities)
        )
        ranking = np.argsort(-sample_probabilities, kind="mergesort")
        top_10_precision_values.append(float(sample_y[ranking[:top_count]].mean()))
    return {
        "method": "nonparametric_order_bootstrap",
        "requested_replicates": replicates,
        "successful_replicates": len(roc_auc_values),
        "confidence": 0.95,
        "roc_auc": percentile_interval(roc_auc_values),
        "average_precision": percentile_interval(average_precision_values),
        "top_10_percent_precision": percentile_interval(top_10_precision_values),
    }


def bootstrap_policy_intervals(
    y_true,
    probabilities,
    threshold: float,
    assumptions: CostAssumptions,
    replicates: int = 2_000,
    seed: int = 42,
) -> dict:
    y = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    rng = np.random.default_rng(seed)
    values = {
        "precision": [],
        "recall": [],
        "f1": [],
        "flagged_rate": [],
        "savings_per_1000_orders": [],
    }
    for _ in range(replicates):
        indices = rng.integers(0, len(y), size=len(y))
        sample_y = y[indices]
        sample_probabilities = probabilities[indices]
        predictions = sample_probabilities >= threshold
        values["precision"].append(
            precision_score(sample_y, predictions, zero_division=0)
        )
        values["recall"].append(recall_score(sample_y, predictions, zero_division=0))
        values["f1"].append(f1_score(sample_y, predictions, zero_division=0))
        values["flagged_rate"].append(float(predictions.mean()))
        selected_cost = policy_cost(
            sample_y,
            sample_probabilities,
            threshold,
            assumptions,
        )
        no_intervention_cost = float(sample_y.sum() * assumptions.missed_return_cost)
        values["savings_per_1000_orders"].append(
            (no_intervention_cost - selected_cost) / len(sample_y) * 1000
        )
    return {
        "method": "fixed_threshold_nonparametric_order_bootstrap",
        "replicates": replicates,
        "confidence": 0.95,
        "threshold_reselected_per_replicate": False,
        **{name: percentile_interval(samples) for name, samples in values.items()},
    }
