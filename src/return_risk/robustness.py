from __future__ import annotations

import numpy as np
import pandas as pd


def fixed_capacity_audit(
    y_true,
    probabilities,
    groups,
    orders_to_flag: int,
) -> dict:
    """Audit an exact top-score capacity, avoiding threshold tie ambiguity."""
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    group_values = pd.Series(groups).fillna("Missing").astype(str).to_numpy()
    if not 1 <= orders_to_flag <= len(y):
        raise ValueError("orders_to_flag must be between 1 and the number of orders.")

    ranking = np.argsort(-scores, kind="mergesort")
    flagged = np.zeros(len(y), dtype=bool)
    flagged[ranking[:orders_to_flag]] = True
    tp = int(np.sum(flagged & (y == 1)))
    fp = int(np.sum(flagged & (y == 0)))
    category_counts = pd.Series(group_values[flagged]).value_counts()
    group_distribution = {
        str(group): {
            "orders_flagged": int(count),
            "share_of_flags": round(float(count / orders_to_flag), 6),
        }
        for group, count in category_counts.items()
    }

    false_positive_rates = {}
    for group in sorted(np.unique(group_values)):
        group_mask = group_values == group
        negatives = group_mask & (y == 0)
        false_positive_rates[group] = (
            round(float(np.sum(flagged & negatives) / np.sum(negatives)), 6)
            if np.sum(negatives)
            else None
        )

    largest_group = str(category_counts.index[0])
    return {
        "orders_flagged": int(orders_to_flag),
        "flagged_rate": round(float(orders_to_flag / len(y)), 6),
        "precision": round(float(tp / orders_to_flag), 6),
        "recall": round(float(tp / max(1, np.sum(y))), 6),
        "true_positive": tp,
        "false_positive": fp,
        "largest_flagged_group": largest_group,
        "largest_group_share": round(
            float(category_counts.iloc[0] / orders_to_flag),
            6,
        ),
        "group_distribution": group_distribution,
        "group_false_positive_rates": false_positive_rates,
    }


def quality_guardrail(
    benchmark: dict,
    candidate: dict,
    maximum_auc_loss: float = 0.01,
    maximum_ap_loss: float = 0.01,
) -> dict:
    auc_loss = benchmark["roc_auc"] - candidate["roc_auc"]
    ap_loss = benchmark["average_precision"] - candidate["average_precision"]
    return {
        "maximum_allowed_auc_loss": maximum_auc_loss,
        "maximum_allowed_average_precision_loss": maximum_ap_loss,
        "observed_auc_loss": round(float(auc_loss), 6),
        "observed_average_precision_loss": round(float(ap_loss), 6),
        "passed": bool(auc_loss <= maximum_auc_loss and ap_loss <= maximum_ap_loss),
    }


def threshold_category_audit(y_true, probabilities, groups, threshold: float) -> dict:
    """Describe quality and category concentration for one score threshold."""
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    group_values = pd.Series(groups).fillna("Missing").astype(str).to_numpy()
    flagged = scores >= threshold
    flagged_count = int(flagged.sum())
    tp = int(np.sum(flagged & (y == 1)))
    fp = int(np.sum(flagged & (y == 0)))

    if flagged_count:
        counts = pd.Series(group_values[flagged]).value_counts()
        distribution = {
            str(group): {
                "orders_flagged": int(count),
                "share_of_flags": round(float(count / flagged_count), 6),
            }
            for group, count in counts.items()
        }
        largest_group = str(counts.index[0])
        largest_share = round(float(counts.iloc[0] / flagged_count), 6)
    else:
        distribution = {}
        largest_group = None
        largest_share = None

    return {
        "threshold": float(threshold),
        "orders_flagged": flagged_count,
        "flagged_rate": round(float(flagged.mean()), 6),
        "precision": round(float(tp / flagged_count), 6) if flagged_count else None,
        "recall": round(float(tp / max(1, np.sum(y))), 6),
        "true_positive": tp,
        "false_positive": fp,
        "categories_represented": int(len(distribution)),
        "largest_flagged_group": largest_group,
        "largest_group_share": largest_share,
        "group_distribution": distribution,
    }
