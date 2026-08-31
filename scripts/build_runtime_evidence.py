from __future__ import annotations

import json

import pandas as pd
from catboost import CatBoostClassifier, Pool

from return_risk.config import CATEGORICAL_FEATURES, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from return_risk.data import binary_target, model_input_frame
from return_risk.policy import build_policy_frontier


def read_validation_report(name: str) -> dict:
    report = json.loads((REPORTS_DIR / name).read_text(encoding="utf-8"))
    if report.get("evaluation_partition") != "validation_only":
        raise RuntimeError(f"{name} is not a validation-only report.")
    if report.get("test_set_accessed") is not False:
        raise RuntimeError(f"{name} cannot be used for runtime model-selection evidence.")
    return report


def build_frontier_artifact() -> dict:
    validation = pd.read_csv(PROCESSED_DIR / "validation.csv")
    model = CatBoostClassifier()
    model.load_model(MODELS_DIR / "return_risk_final.cbm")
    frame = model_input_frame(validation)
    labels = binary_target(validation)
    probabilities = model.predict_proba(Pool(frame, cat_features=CATEGORICAL_FEATURES))[:, 1]
    return {
        "evaluation_partition": "validation_only",
        "test_set_accessed_for_selection": False,
        "purpose": "aggregate_counterfactual_policy_simulation",
        "contains_row_level_data": False,
        "orders": int(len(validation)),
        "grid_size": 501,
        "frontier": build_policy_frontier(labels, probabilities, grid_size=501),
        "limitations": [
            "Results are hypothetical validation estimates, not production claims.",
            "Intervention effectiveness is an assumption rather than an observed effect.",
            "The held-out test result does not participate in threshold selection.",
            "The operational release remains locked to shadow monitoring.",
        ],
    }


def build_model_selection_summary() -> dict:
    comparison = read_validation_report("catboost_validation_comparison.json")
    tuning = read_validation_report("catboost_tuning_validation.json")
    calibration = read_validation_report("catboost_calibration_validation.json")

    logistic = comparison["models"]["logistic_regression"]["probability_metrics"]
    champion = calibration["raw_validation"]["probability_metrics"]
    tuned = tuning["tuned_catboost"]["validation_probability_metrics"]
    calibrated = calibration["calibrated_validation"]["probability_metrics"]

    def row(name: str, status: str, metrics: dict) -> dict:
        return {
            "model": name,
            "status": status,
            "roc_auc": metrics["roc_auc"],
            "average_precision": metrics["average_precision"],
            "brier_score": metrics["brier_score"],
            "top_10_percent_precision": metrics["capacity"]["top_10_percent"][
                "precision"
            ],
        }

    return {
        "evaluation_partition": "validation_only",
        "test_set_accessed_for_selection": False,
        "selection_metric": "validation_average_precision_with_safety_guardrails",
        "selected_model": "Frozen CatBoost (no location)",
        "models": [
            row("Logistic regression", "Baseline", logistic),
            row("Frozen CatBoost (no location)", "Champion", champion),
            row("Optuna-tuned CatBoost", "Rejected: weaker AP", tuned),
            row("Calibrated CatBoost", "Rejected: worse calibration", calibrated),
        ],
        "decision": (
            "The no-location raw CatBoost was frozen because it retained the strongest "
            "validation average precision without using the unauditable location proxy."
        ),
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    frontier_path = MODELS_DIR / "policy_frontier.json"
    selection_path = MODELS_DIR / "model_selection_summary.json"
    frontier_path.write_text(
        json.dumps(build_frontier_artifact(), indent=2), encoding="utf-8"
    )
    selection_path.write_text(
        json.dumps(build_model_selection_summary(), indent=2), encoding="utf-8"
    )
    print(f"Saved {frontier_path}")
    print(f"Saved {selection_path}")


if __name__ == "__main__":
    main()
