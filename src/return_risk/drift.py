from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from return_risk.config import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES

EPSILON = 1e-6


def _distribution(values, interior_edges: list[float]) -> list[float]:
    array = np.asarray(values, dtype=float)
    bins = np.digitize(array, interior_edges, right=False)
    counts = np.bincount(bins, minlength=len(interior_edges) + 1).astype(float)
    return (counts / max(1, counts.sum())).tolist()


def build_drift_reference(frame: pd.DataFrame) -> dict:
    missing = sorted(set(MODEL_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Reference frame is missing model features: {missing}")
    numeric = {}
    for feature in NUMERIC_FEATURES:
        values = frame[feature].to_numpy(dtype=float)
        candidate_edges = np.quantile(values, np.linspace(0.1, 0.9, 9))
        edges = sorted({float(value) for value in candidate_edges})
        numeric[feature] = {
            "interior_edges": edges,
            "expected_distribution": _distribution(values, edges),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    categorical = {}
    for feature in CATEGORICAL_FEATURES:
        frequencies = frame[feature].astype(str).value_counts(normalize=True)
        categorical[feature] = {
            "expected_distribution": {
                str(category): float(value) for category, value in frequencies.items()
            }
        }
    return {
        "reference_orders": int(len(frame)),
        "features": MODEL_FEATURES,
        "numeric": numeric,
        "categorical": categorical,
        "psi_thresholds": {"warning": 0.1, "high": 0.25},
        "threshold_note": "Illustrative PSI thresholds; calibrate with merchant history.",
    }


def load_drift_reference(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Drift reference is missing: {path}")
    reference = json.loads(path.read_text(encoding="utf-8"))
    if reference.get("features") != MODEL_FEATURES:
        raise ValueError("Drift reference does not match the active feature allowlist.")
    return reference


def _psi(expected, observed) -> float:
    expected_array = np.clip(np.asarray(expected, dtype=float), EPSILON, None)
    observed_array = np.clip(np.asarray(observed, dtype=float), EPSILON, None)
    return float(
        np.sum((observed_array - expected_array) * np.log(observed_array / expected_array))
    )


def _severity(value: float, thresholds: dict) -> str:
    if value >= thresholds["high"]:
        return "high"
    if value >= thresholds["warning"]:
        return "warning"
    return "stable"


def drift_report(frame: pd.DataFrame, reference: dict) -> dict:
    if frame.empty:
        raise ValueError("Drift monitoring requires at least one valid order.")
    thresholds = reference["psi_thresholds"]
    features = {}
    for feature, details in reference["numeric"].items():
        observed = _distribution(frame[feature], details["interior_edges"])
        psi = _psi(details["expected_distribution"], observed)
        out_of_range = (
            (frame[feature] < details["minimum"])
            | (frame[feature] > details["maximum"])
        ).mean()
        features[feature] = {
            "psi": round(psi, 6),
            "severity": _severity(psi, thresholds),
            "out_of_reference_range_rate": round(float(out_of_range), 6),
        }

    for feature, details in reference["categorical"].items():
        expected_map = details["expected_distribution"]
        observed_frequencies = frame[feature].astype(str).value_counts(normalize=True)
        categories = list(expected_map)
        expected = [expected_map[category] for category in categories] + [EPSILON]
        observed = [float(observed_frequencies.get(category, 0.0)) for category in categories]
        other_rate = float(
            observed_frequencies[
                ~observed_frequencies.index.astype(str).isin(categories)
            ].sum()
        )
        observed.append(other_rate)
        psi = _psi(expected, observed)
        features[feature] = {
            "psi": round(psi, 6),
            "severity": _severity(psi, thresholds),
            "unseen_category_rate": round(other_rate, 6),
        }

    maximum_psi = max(item["psi"] for item in features.values())
    severity = _severity(maximum_psi, thresholds)
    sample_status = "sufficient" if len(frame) >= 30 else "insufficient_below_30_orders"
    return {
        "orders": int(len(frame)),
        "sample_status": sample_status,
        "overall_severity": severity if sample_status == "sufficient" else "insufficient_data",
        "maximum_psi": round(maximum_psi, 6),
        "shadow_mode_required": True,
        "features": features,
        "interpretation": (
            "PSI is descriptive. Shadow mode remains mandatory regardless of drift status."
        ),
    }
