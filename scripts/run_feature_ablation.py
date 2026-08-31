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
from catboost import CatBoostClassifier

from return_risk.config import (
    BENCHMARK_CATEGORICAL_FEATURES,
    BENCHMARK_MODEL_FEATURES,
    MODELS_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
)
from return_risk.data import benchmark_input_frame, binary_target
from return_risk.metrics import probability_metrics, threshold_metrics
from return_risk.modeling import untuned_catboost_model
from return_risk.policy import select_cost_threshold
from return_risk.robustness import fixed_capacity_audit, quality_guardrail

CURRENT_BASE_THRESHOLD = 0.438
VARIANTS = {
    "full_model": BENCHMARK_MODEL_FEATURES,
    "without_order_year": [
        feature for feature in BENCHMARK_MODEL_FEATURES if feature != "order_year"
    ],
    "without_location": [
        feature for feature in BENCHMARK_MODEL_FEATURES if feature != "User_Location"
    ],
    "without_year_and_location": [
        feature
        for feature in BENCHMARK_MODEL_FEATURES
        if feature not in {"order_year", "User_Location"}
    ],
}
PREFERENCE_ORDER = [
    "without_year_and_location",
    "without_order_year",
    "without_location",
    "full_model",
]


def _train_variant(x_train, y_train, x_validation, y_validation, features):
    categoricals = [
        feature for feature in BENCHMARK_CATEGORICAL_FEATURES if feature in features
    ]
    model = untuned_catboost_model()
    model.fit(
        x_train[features],
        y_train,
        cat_features=categoricals,
        eval_set=(x_validation[features], y_validation),
        early_stopping_rounds=75,
        use_best_model=True,
        verbose=False,
    )
    return model


def _save_comparison_plot(results: dict, output_path: Path) -> None:
    labels = list(results)
    auc = [results[name]["probability_metrics"]["roc_auc"] for name in labels]
    ap = [results[name]["probability_metrics"]["average_precision"] for name in labels]
    concentration = [
        results[name]["fixed_capacity_audit"]["largest_group_share"] for name in labels
    ]
    display_labels = [label.replace("_", "\n") for label in labels]
    positions = np.arange(len(labels))

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    width = 0.36
    axes[0].bar(positions - width / 2, auc, width, label="ROC-AUC", color="#2563EB")
    axes[0].bar(positions + width / 2, ap, width, label="Average precision", color="#16A34A")
    axes[0].set_ylim(0.35, 0.65)
    axes[0].set_title("Validation ranking quality")
    axes[0].set_xticks(positions, display_labels)
    axes[0].set_ylabel("Score")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(positions, concentration, color="#7C3AED")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Largest product-category share among top flags")
    axes[1].set_xticks(positions, display_labels)
    axes[1].set_ylabel("Share of flagged orders")
    axes[1].grid(axis="y", alpha=0.2)
    figure.suptitle("Feature-ablation robustness check (validation only)", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    train_path = PROCESSED_DIR / "train.csv"
    validation_path = PROCESSED_DIR / "validation.csv"
    selected_model_path = MODELS_DIR / "catboost_untuned.cbm"
    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError("Prepared splits missing. Run scripts/prepare_data.py first.")
    if not selected_model_path.exists():
        raise FileNotFoundError("Selected CatBoost model missing. Run train_catboost.py first.")

    # Intentionally load train and validation only; do not inspect the locked test partition.
    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    x_train = benchmark_input_frame(train)
    y_train = binary_target(train)
    x_validation = benchmark_input_frame(validation)
    y_validation = binary_target(validation)

    current_model = CatBoostClassifier()
    current_model.load_model(selected_model_path)
    current_probabilities = current_model.predict_proba(x_validation)[:, 1]
    fixed_capacity = int(np.sum(current_probabilities >= CURRENT_BASE_THRESHOLD))
    if fixed_capacity == 0:
        raise RuntimeError("Current base threshold flags no validation orders.")

    models = {"full_model": current_model}
    for name, features in VARIANTS.items():
        if name == "full_model":
            continue
        models[name] = _train_variant(
            x_train,
            y_train,
            x_validation,
            y_validation,
            features,
        )

    results = {}
    for name, features in VARIANTS.items():
        model = models[name]
        probabilities = model.predict_proba(x_validation[features])[:, 1]
        metrics = probability_metrics(y_validation, probabilities)
        policy = select_cost_threshold(y_validation, probabilities)
        results[name] = {
            "features": features,
            "removed_features": sorted(set(BENCHMARK_MODEL_FEATURES) - set(features)),
            "tree_count": int(model.tree_count_),
            "probability_metrics": metrics,
            "base_cost_policy": policy,
            "threshold_metrics": threshold_metrics(
                y_validation,
                probabilities,
                policy["selected_threshold"],
            ),
            "fixed_capacity_audit": fixed_capacity_audit(
                y_validation,
                probabilities,
                validation["Product_Category"],
                fixed_capacity,
            ),
        }

    benchmark_metrics = results["full_model"]["probability_metrics"]
    guardrails = {}
    for name, result in results.items():
        guardrails[name] = quality_guardrail(
            benchmark_metrics,
            result["probability_metrics"],
        )

    recommended = next(name for name in PREFERENCE_ORDER if guardrails[name]["passed"])
    rationale = (
        "Among variants within 0.01 of the full model on both ROC-AUC and average "
        "precision, prefer removing both year and location, then year only, then location "
        "only. This reduces temporal fragility and an unauditable location proxy without "
        "accepting a material validation-quality loss. Category concentration is reported "
        "separately and cannot be treated as solved by this rule."
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        if name != "full_model":
            model.save_model(MODELS_DIR / f"catboost_ablation_{name}.cbm")

    report = {
        "evaluation_partition": "train_and_validation_only",
        "test_set_accessed": False,
        "fixed_capacity_orders": fixed_capacity,
        "quality_guardrail": {
            "rule": (
                "Candidate ROC-AUC and average-precision losses must each be no more than "
                "0.01 versus the full validation model."
            ),
            "results": guardrails,
        },
        "selection_preference": PREFERENCE_ORDER,
        "recommended_variant": recommended,
        "recommendation_rationale": rationale,
        "variants": results,
        "limitations": [
            "The recommendation is validation-only and is not a final test claim.",
            "Feature removal cannot by itself establish fairness or causal robustness.",
            "The fixed-capacity audit compares exactly the same number of top-ranked orders.",
            "The source dataset is synthetic.",
        ],
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    figure_dir = REPORTS_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "feature_ablation_validation.json"
    figure_path = figure_dir / "validation_feature_ablation.png"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _save_comparison_plot(results, figure_path)

    print(json.dumps(report, indent=2))
    print(f"\nSaved feature-ablation report to {report_path}")
    print(f"Saved comparison plot to {figure_path}")


if __name__ == "__main__":
    main()
