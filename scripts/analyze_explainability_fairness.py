# ruff: noqa: E402
from __future__ import annotations

import json
import os
from pathlib import Path

MATPLOTLIB_CONFIG_DIR = Path(__file__).resolve().parents[1] / "work" / "matplotlib"
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from return_risk.config import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    MODELS_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
)
from return_risk.data import binary_target, model_input_frame
from return_risk.explainability import global_shap_importance, local_explanation
from return_risk.slices import disparity_summary, slice_metrics

BASE_THRESHOLD = 0.426
MIN_GROUP_SIZE = 20


def _audit_slice(y, probabilities, groups, threshold: float) -> dict:
    result = slice_metrics(
        y,
        probabilities,
        groups,
        threshold,
        min_group_size=MIN_GROUP_SIZE,
    )
    result["observed_disparities"] = disparity_summary(result["eligible_groups"])
    return result


def _save_global_plot(importance: list[dict], output_path: Path) -> None:
    top = list(reversed(importance[:10]))
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(
        [row["feature"] for row in top],
        [row["mean_absolute_shap"] for row in top],
        color="#2563EB",
    )
    axis.set(
        title="What most influenced return-risk predictions?",
        xlabel="Mean absolute SHAP value (model log-odds)",
        ylabel="",
    )
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _save_local_plot(explanation: dict, output_path: Path) -> None:
    reasons = list(reversed(explanation["top_reasons"]))
    values = [row["shap_log_odds"] for row in reasons]
    labels = [f"{row['feature']} = {row['value']}" for row in reasons]
    colors = ["#DC2626" if value >= 0 else "#16A34A" for value in values]
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(labels, values, color=colors)
    axis.axvline(0, color="#111827", linewidth=0.8)
    axis.set(
        title=(
            "Why the highest-risk validation order received its score "
            f"({explanation['probability']:.1%})"
        ),
        xlabel="SHAP contribution to model log-odds",
        ylabel="",
    )
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _save_slice_plot(slices: dict, output_path: Path) -> None:
    selected = ["User_Gender", "Age_Band", "Product_Category", "Payment_Method"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, slice_name in zip(axes.flat, selected, strict=True):
        rows = slices[slice_name]["eligible_groups"]
        labels = [row["group"] for row in rows]
        rates = [row["false_positive_rate"] or 0.0 for row in rows]
        axis.bar(labels, rates, color="#7C3AED")
        axis.set_title(slice_name.replace("_", " "))
        axis.set_ylabel("False-positive rate")
        axis.set_ylim(0, max(0.12, max(rates, default=0) * 1.2))
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Validation false-positive rates at the base threshold\n"
        "Sensitive attributes are audit-only, not model inputs",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    validation_path = PROCESSED_DIR / "validation.csv"
    model_path = MODELS_DIR / "return_risk_final.cbm"
    if not validation_path.exists():
        raise FileNotFoundError("Validation split missing. Run scripts/prepare_data.py first.")
    if not model_path.exists():
        raise FileNotFoundError("Selected CatBoost model is missing. Run train_catboost.py first.")

    # Intentionally read validation only. The held-out test partition remains locked.
    validation = pd.read_csv(validation_path)
    x_validation = model_input_frame(validation)
    y_validation = binary_target(validation)
    model = CatBoostClassifier()
    model.load_model(model_path)
    pool = Pool(x_validation, y_validation, cat_features=CATEGORICAL_FEATURES)
    probabilities = model.predict_proba(pool)[:, 1]
    raw_predictions = np.asarray(model.predict(pool, prediction_type="RawFormulaVal"))
    shap_matrix = np.asarray(model.get_feature_importance(pool, type="ShapValues"))
    contributions = shap_matrix[:, :-1]
    expected_values = shap_matrix[:, -1]
    reconstruction_errors = np.abs(
        expected_values + contributions.sum(axis=1) - raw_predictions
    )
    importance = global_shap_importance(MODEL_FEATURES, contributions)

    positions = {
        "highest_risk": int(np.argmax(probabilities)),
        "median_risk": int(np.argmin(np.abs(probabilities - np.median(probabilities)))),
        "lowest_risk": int(np.argmin(probabilities)),
    }
    local_examples = {}
    for label, position in positions.items():
        local_examples[label] = {
            "validation_row_position": position,
            **local_explanation(
                MODEL_FEATURES,
                x_validation.iloc[position].tolist(),
                contributions[position],
                expected_values[position],
                raw_predictions[position],
                probabilities[position],
                top_k=8,
            ),
        }

    age_band = pd.cut(
        validation["User_Age"],
        bins=[17, 29, 44, 60, np.inf],
        labels=["18-29", "30-44", "45-60", "61+"],
    )
    group_columns = {
        "Product_Category": validation["Product_Category"],
        "Payment_Method": validation["Payment_Method"],
        "Shipping_Method": validation["Shipping_Method"],
        "User_Location": validation["User_Location"],
        "User_Gender": validation["User_Gender"],
        "Age_Band": age_band,
    }
    slices = {
        name: _audit_slice(y_validation, probabilities, groups, BASE_THRESHOLD)
        for name, groups in group_columns.items()
    }

    report = {
        "evaluation_partition": "validation_only",
        "test_set_accessed": False,
        "model": "promoted_catboost_without_location_raw_probabilities",
        "decision_threshold": BASE_THRESHOLD,
        "model_features": MODEL_FEATURES,
        "sensitive_attributes_in_model": False,
        "sensitive_audit_note": (
            "Age band and gender are used only after scoring to inspect outcomes. "
            "They are not model inputs. Location was removed from the promoted model and "
            "is retained only for proxy auditing."
        ),
        "shap": {
            "method": "CatBoost native TreeSHAP",
            "space": "raw model log-odds",
            "maximum_reconstruction_error": round(float(reconstruction_errors.max()), 10),
            "global_importance": importance,
            "local_examples": local_examples,
        },
        "slice_audit": slices,
        "limitations": [
            "All findings are validation-only; the final test partition remains locked.",
            "SHAP explains model behavior and does not establish causal effects.",
            "Observed group gaps are descriptive and do not by themselves prove discrimination.",
            "Groups below the minimum sample size are suppressed because rates are unstable.",
            "The threshold was selected on validation data under hypothetical merchant costs.",
            "The source dataset is synthetic and cannot establish production fairness or utility.",
        ],
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    figure_dir = REPORTS_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "explainability_fairness_validation.json"
    global_path = figure_dir / "global_shap_importance.png"
    local_path = figure_dir / "high_risk_local_shap.png"
    slice_path = figure_dir / "validation_slice_false_positive_rates.png"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _save_global_plot(importance, global_path)
    _save_local_plot(local_examples["highest_risk"], local_path)
    _save_slice_plot(slices, slice_path)

    print(json.dumps(report, indent=2))
    print(f"\nSaved explainability and slice report to {report_path}")
    print(f"Saved global SHAP plot to {global_path}")
    print(f"Saved local SHAP plot to {local_path}")
    print(f"Saved slice audit plot to {slice_path}")


if __name__ == "__main__":
    main()
