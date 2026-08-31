from __future__ import annotations

import json

import pandas as pd

from return_risk.config import (
    CATEGORICAL_FEATURES,
    MODELS_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
)
from return_risk.data import binary_target, model_input_frame
from return_risk.metrics import probability_metrics, threshold_metrics
from return_risk.modeling import untuned_catboost_model
from return_risk.policy import select_cost_threshold


def main() -> None:
    train_path = PROCESSED_DIR / "train.csv"
    validation_path = PROCESSED_DIR / "validation.csv"
    baseline_report_path = REPORTS_DIR / "baseline_validation.json"
    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError("Prepared splits not found. Run scripts/prepare_data.py first.")
    if not baseline_report_path.exists():
        raise FileNotFoundError("Baseline report not found. Run scripts/train_baseline.py first.")

    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    x_train = model_input_frame(train)
    y_train = binary_target(train)
    x_validation = model_input_frame(validation)
    y_validation = binary_target(validation)

    model = untuned_catboost_model()
    model.fit(
        x_train,
        y_train,
        cat_features=CATEGORICAL_FEATURES,
        eval_set=(x_validation, y_validation),
        early_stopping_rounds=75,
        use_best_model=True,
        verbose=False,
    )
    probabilities = model.predict_proba(x_validation)[:, 1]
    policy = select_cost_threshold(y_validation, probabilities)
    catboost_result = {
        "benchmark_type": "untuned_conservative_defaults",
        "best_iteration": int(model.get_best_iteration()),
        "tree_count": int(model.tree_count_),
        "probability_metrics": probability_metrics(y_validation, probabilities),
        "cost_policy": policy,
        "threshold_metrics": threshold_metrics(
            y_validation,
            probabilities,
            policy["selected_threshold"],
        ),
    }

    baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    logistic_result = baseline_report["models"]["logistic_regression"]
    logistic_probability = logistic_result["probability_metrics"]
    catboost_probability = catboost_result["probability_metrics"]
    comparison = {
        "evaluation_partition": "validation_only",
        "test_set_accessed": False,
        "selection_metric": "validation_average_precision",
        "models": {
            "logistic_regression": logistic_result,
            "catboost_untuned": catboost_result,
        },
        "catboost_minus_logistic": {
            "roc_auc": round(
                catboost_probability["roc_auc"] - logistic_probability["roc_auc"], 6
            ),
            "average_precision": round(
                catboost_probability["average_precision"]
                - logistic_probability["average_precision"],
                6,
            ),
            "top_10_percent_precision": round(
                catboost_probability["capacity"]["top_10_percent"]["precision"]
                - logistic_probability["capacity"]["top_10_percent"]["precision"],
                6,
            ),
            "cost_savings_vs_no_intervention": round(
                catboost_result["cost_policy"]["savings_vs_no_intervention"]
                - logistic_result["cost_policy"]["savings_vs_no_intervention"],
                2,
            ),
        },
    }
    comparison["validation_winner"] = max(
        ("logistic_regression", "catboost_untuned"),
        key=lambda name: comparison["models"][name]["probability_metrics"][
            "average_precision"
        ],
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "catboost_validation_comparison.json"
    model_path = MODELS_DIR / "catboost_untuned.cbm"
    report_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    model.save_model(model_path)

    print(json.dumps(comparison, indent=2))
    print(f"\nSaved comparison report to {report_path}")
    print(f"Saved untuned CatBoost model to {model_path}")


if __name__ == "__main__":
    main()

