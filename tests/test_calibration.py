from __future__ import annotations

import numpy as np

from return_risk.calibration import (
    calibrated_probabilities,
    fit_sigmoid_calibrator,
    reliability_summary,
)


def test_sigmoid_calibration_outputs_monotonic_probabilities() -> None:
    raw_scores = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    calibrator = fit_sigmoid_calibrator(raw_scores, labels)
    probabilities = calibrated_probabilities(calibrator, raw_scores)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.all(np.diff(probabilities) > 0)


def test_reliability_bins_cover_all_orders() -> None:
    labels = np.array([0, 0, 1, 1, 0, 1])
    probabilities = np.array([0.1, 0.2, 0.7, 0.8, 0.4, 0.6])
    summary = reliability_summary(labels, probabilities, n_bins=3)
    assert sum(item["orders"] for item in summary["bins"]) == len(labels)
    assert 0 <= summary["expected_calibration_error"] <= 1
