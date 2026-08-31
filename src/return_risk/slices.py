from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(float(numerator / denominator), 6)


def slice_metrics(
    y_true,
    probabilities,
    groups,
    threshold: float,
    min_group_size: int = 20,
) -> dict:
    """Calculate descriptive threshold metrics for sufficiently large groups."""
    audit = pd.DataFrame(
        {
            "target": np.asarray(y_true, dtype=int),
            "probability": np.asarray(probabilities, dtype=float),
            "group": pd.Series(groups).fillna("Missing").astype(str).to_numpy(),
        }
    )
    rows = []
    suppressed = []
    for group, frame in audit.groupby("group", sort=True):
        if len(frame) < min_group_size:
            suppressed.append({"group": group, "orders": int(len(frame))})
            continue

        y = frame["target"].to_numpy()
        scores = frame["probability"].to_numpy()
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (y == 1)))
        fp = int(np.sum(predicted & (y == 0)))
        tn = int(np.sum(~predicted & (y == 0)))
        fn = int(np.sum(~predicted & (y == 1)))
        row = {
            "group": group,
            "orders": int(len(frame)),
            "prevalence": round(float(y.mean()), 6),
            "mean_score": round(float(scores.mean()), 6),
            "flagged_rate": round(float(predicted.mean()), 6),
            "precision": _safe_ratio(tp, tp + fp),
            "recall": _safe_ratio(tp, tp + fn),
            "false_positive_rate": _safe_ratio(fp, fp + tn),
            "false_negative_rate": _safe_ratio(fn, fn + tp),
            "confusion_matrix": {
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
            },
            "roc_auc": None,
            "average_precision": None,
        }
        if np.unique(y).size == 2:
            row["roc_auc"] = round(float(roc_auc_score(y, scores)), 6)
            row["average_precision"] = round(float(average_precision_score(y, scores)), 6)
        rows.append(row)

    return {
        "minimum_group_size": int(min_group_size),
        "eligible_groups": rows,
        "suppressed_groups": suppressed,
        "suppressed_orders": int(sum(item["orders"] for item in suppressed)),
    }


def disparity_summary(rows: list[dict], metrics: tuple[str, ...] | None = None) -> dict:
    """Summarize observed max-minus-min gaps without asserting causal fairness."""
    metrics = metrics or ("flagged_rate", "recall", "false_positive_rate")
    result = {}
    for metric in metrics:
        available = [(row["group"], row[metric]) for row in rows if row[metric] is not None]
        if len(available) < 2:
            result[metric] = None
            continue
        minimum = min(available, key=lambda item: item[1])
        maximum = max(available, key=lambda item: item[1])
        result[metric] = {
            "minimum_group": minimum[0],
            "minimum": minimum[1],
            "maximum_group": maximum[0],
            "maximum": maximum[1],
            "absolute_gap": round(float(maximum[1] - minimum[1]), 6),
        }
    return result
