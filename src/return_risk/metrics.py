from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def capacity_metrics(y_true, probabilities, capacities=(0.10, 0.20, 0.30)) -> dict:
    y = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    ranking = np.argsort(-probabilities, kind="mergesort")
    result = {}
    for capacity in capacities:
        count = max(1, int(round(len(y) * capacity)))
        selected = y[ranking[:count]]
        result[f"top_{int(capacity * 100)}_percent"] = {
            "orders_flagged": int(count),
            "precision": round(float(selected.mean()), 6),
            "recall": round(float(selected.sum() / max(1, y.sum())), 6),
            "lift_over_prevalence": round(float(selected.mean() / y.mean()), 6),
        }
    return result


def probability_metrics(y_true, probabilities) -> dict:
    y = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    return {
        "prevalence": round(float(y.mean()), 6),
        "roc_auc": round(float(roc_auc_score(y, probabilities)), 6),
        "average_precision": round(float(average_precision_score(y, probabilities)), 6),
        "brier_score": round(float(brier_score_loss(y, probabilities)), 6),
        "log_loss": round(float(log_loss(y, probabilities, labels=[0, 1])), 6),
        "capacity": capacity_metrics(y, probabilities),
    }


def threshold_metrics(y_true, probabilities, threshold: float) -> dict:
    y = np.asarray(y_true, dtype=int)
    predictions = np.asarray(probabilities) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 6),
        "precision": round(float(precision_score(y, predictions, zero_division=0)), 6),
        "recall": round(float(recall_score(y, predictions, zero_division=0)), 6),
        "f1": round(float(f1_score(y, predictions, zero_division=0)), 6),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }

