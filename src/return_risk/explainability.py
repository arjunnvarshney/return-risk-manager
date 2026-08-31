from __future__ import annotations

import numpy as np


def global_shap_importance(feature_names, contributions) -> list[dict]:
    values = np.asarray(contributions, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError("SHAP contributions must have one column per feature.")
    importance = np.mean(np.abs(values), axis=0)
    ranking = sorted(zip(feature_names, importance, strict=True), key=lambda item: -item[1])
    return [
        {"feature": feature, "mean_absolute_shap": round(float(value), 6)}
        for feature, value in ranking
    ]


def _display_value(value) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.2f}"
    return str(value)


def local_explanation(
    feature_names,
    feature_values,
    contributions,
    expected_value: float,
    raw_prediction: float,
    probability: float,
    top_k: int = 5,
) -> dict:
    """Return model-behavior reason codes; SHAP values are in raw log-odds space."""
    rows = []
    for feature, value, contribution in zip(
        feature_names,
        feature_values,
        contributions,
        strict=True,
    ):
        direction = "raised" if contribution >= 0 else "lowered"
        rows.append(
            {
                "feature": feature,
                "value": _display_value(value),
                "shap_log_odds": round(float(contribution), 6),
                "direction": direction,
                "reason": (
                    f"{feature} = {_display_value(value)} {direction} the model's return-risk score"
                ),
            }
        )
    rows.sort(key=lambda row: abs(row["shap_log_odds"]), reverse=True)
    reconstructed = float(expected_value + np.sum(contributions))
    return {
        "probability": round(float(probability), 6),
        "expected_raw_value": round(float(expected_value), 6),
        "raw_prediction": round(float(raw_prediction), 6),
        "reconstruction_error": round(abs(reconstructed - float(raw_prediction)), 10),
        "top_reasons": rows[:top_k],
        "interpretation_note": (
            "Positive SHAP values raise model log-odds and negative values lower them; "
            "these describe model behavior, not causal effects."
        ),
    }
