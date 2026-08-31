from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_sigmoid_calibrator(raw_scores, y_true) -> LogisticRegression:
    scores = np.asarray(raw_scores, dtype=float).reshape(-1, 1)
    target = np.asarray(y_true, dtype=int)
    if len(scores) != len(target):
        raise ValueError("Scores and labels must have the same length")
    if len(np.unique(target)) != 2:
        raise ValueError("Calibration requires both target classes")
    calibrator = LogisticRegression(C=1_000_000, solver="lbfgs", max_iter=2_000)
    calibrator.fit(scores, target)
    return calibrator


def calibrated_probabilities(calibrator: LogisticRegression, raw_scores) -> np.ndarray:
    scores = np.asarray(raw_scores, dtype=float).reshape(-1, 1)
    return calibrator.predict_proba(scores)[:, 1]


def reliability_summary(y_true, probabilities, n_bins: int = 10) -> dict:
    y = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(y) != len(probabilities):
        raise ValueError("Probabilities and labels must have the same length")
    if n_bins < 2:
        raise ValueError("n_bins must be at least two")

    order = np.argsort(probabilities, kind="mergesort")
    bins = []
    weighted_error = 0.0
    maximum_error = 0.0
    for bin_number, indices in enumerate(np.array_split(order, n_bins), start=1):
        if len(indices) == 0:
            continue
        mean_probability = float(probabilities[indices].mean())
        observed_rate = float(y[indices].mean())
        absolute_error = abs(mean_probability - observed_rate)
        weighted_error += len(indices) / len(y) * absolute_error
        maximum_error = max(maximum_error, absolute_error)
        bins.append(
            {
                "bin": bin_number,
                "orders": int(len(indices)),
                "mean_probability": round(mean_probability, 6),
                "observed_return_rate": round(observed_rate, 6),
                "absolute_error": round(absolute_error, 6),
            }
        )
    return {
        "binning": "equal_frequency",
        "bin_count": len(bins),
        "expected_calibration_error": round(weighted_error, 6),
        "maximum_calibration_error": round(maximum_error, 6),
        "bins": bins,
    }
