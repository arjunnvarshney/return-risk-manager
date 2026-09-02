from __future__ import annotations

import numpy as np
import pandas as pd

from return_risk.config import CATEGORICAL_FEATURES

CONTEXT_NUMERIC_FEATURES = [
    "Product_Price",
    "Order_Quantity",
    "Discount_Applied",
    "Order_Value",
]
CONTEXT_TEMPORAL_FEATURES = ["order_year", "order_month", "order_day_of_week"]
CONTEXT_LABELS = {
    "Product_Price": "product-price",
    "Order_Quantity": "quantity",
    "Discount_Applied": "discount",
    "Order_Value": "order-value",
    "order_year": "year",
    "order_month": "month",
    "order_day_of_week": "day-of-week",
    "Product_Category": "product category",
    "Shipping_Method": "shipping method",
    "Payment_Method": "payment method",
}


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


def _group_rates(values: pd.Series, labels: np.ndarray) -> dict[str, dict]:
    grouped = (
        pd.DataFrame({"value": values.astype(str), "returned": labels})
        .groupby("value")["returned"]
        .agg(["mean", "count"])
    )
    return {
        str(value): {"return_rate": round(float(row["mean"]), 6), "orders": int(row["count"])}
        for value, row in grouped.iterrows()
    }


def build_target_context(frame: pd.DataFrame, labels) -> dict:
    """Build aggregate, development-only outcome context for human-readable reasons."""
    label_array = np.asarray(labels, dtype=int)
    if len(frame) != len(label_array) or len(frame) == 0:
        raise ValueError("Context features and labels must be non-empty and have equal length.")
    required = set(CATEGORICAL_FEATURES + CONTEXT_NUMERIC_FEATURES + CONTEXT_TEMPORAL_FEATURES)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Context frame is missing features: {missing}")

    numeric = {}
    for feature in CONTEXT_NUMERIC_FEATURES:
        values = frame[feature].to_numpy(dtype=float)
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        edges = sorted(
            {
                float(value)
                for value in np.quantile(values, np.linspace(0.1, 0.9, 9))
                if minimum < float(value) < maximum
            }
        )
        bin_ids = np.digitize(values, edges, right=False)
        bins = []
        for bin_id in range(len(edges) + 1):
            mask = bin_ids == bin_id
            lower = minimum if bin_id == 0 else edges[bin_id - 1]
            upper = maximum if bin_id == len(edges) else edges[bin_id]
            bins.append(
                {
                    "lower": round(lower, 6),
                    "upper": round(upper, 6),
                    "return_rate": (
                        round(float(label_array[mask].mean()), 6) if mask.any() else None
                    ),
                    "orders": int(mask.sum()),
                }
            )
        numeric[feature] = {"interior_edges": edges, "bins": bins}

    return {
        "orders": int(len(frame)),
        "overall_return_rate": round(float(label_array.mean()), 6),
        "categorical": {
            feature: _group_rates(frame[feature], label_array) for feature in CATEGORICAL_FEATURES
        },
        "temporal": {
            feature: _group_rates(frame[feature], label_array)
            for feature in CONTEXT_TEMPORAL_FEATURES
        },
        "numeric": numeric,
        "interpretation": (
            "Aggregate associations from training and validation only; they describe "
            "development data and do not establish causality."
        ),
    }


def contextual_reason(feature: str, value, direction: str, context: dict) -> str:
    """Explain a SHAP direction with aggregate development evidence when available."""
    overall = float(context["overall_return_rate"])
    label = CONTEXT_LABELS.get(feature, feature.replace("_", " ").lower())
    key = str(int(value)) if feature in CONTEXT_TEMPORAL_FEATURES else str(value)

    if feature in context["categorical"]:
        group = context["categorical"][feature].get(key)
        if group:
            return (
                f"Development orders with {label} {key} returned at "
                f"{group['return_rate']:.1%} versus {overall:.1%} overall "
                f"(n={group['orders']:,}). This association {direction} the model score; "
                "it does not prove causation."
            )

    if feature in context["temporal"]:
        group = context["temporal"][feature].get(key)
        if group:
            return (
                f"Development orders in {label} {key} returned at "
                f"{group['return_rate']:.1%} versus {overall:.1%} overall "
                f"(n={group['orders']:,}). This time association {direction} the score; "
                "it may reflect drift rather than a causal effect."
            )
        known = sorted(int(item) for item in context["temporal"][feature])
        if feature == "order_year":
            return (
                f"No {key} orders were used in development ({known[0]}–{known[-1]}). "
                "The model routed this date through its latest learned time branch, so the "
                "contribution may reflect time drift and is less reliable—not a causal effect."
            )

    if feature in context["numeric"]:
        details = context["numeric"][feature]
        bin_id = int(np.digitize([float(value)], details["interior_edges"], right=False)[0])
        selected = details["bins"][bin_id]
        if selected["orders"] == 0:
            return (
                f"No development orders fell in the same {label} band "
                f"({selected['lower']:,.2f}–{selected['upper']:,.2f}). The model still "
                f"assigned a contribution that {direction} the score, but this sparse-region "
                "explanation is less reliable and does not imply causation."
            )
        return (
            f"Development orders in the same {label} band "
            f"({selected['lower']:,.2f}–{selected['upper']:,.2f}) returned at "
            f"{selected['return_rate']:.1%} versus {overall:.1%} overall "
            f"(n={selected['orders']:,}). This association {direction} the model score; "
            "it does not prove causation."
        )

    return (
        f"This value {direction} the model score relative to its development baseline. "
        "SHAP describes model behavior, not causation."
    )


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
