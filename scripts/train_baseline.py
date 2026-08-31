from __future__ import annotations

import json

import joblib
import pandas as pd

from return_risk.config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from return_risk.data import binary_target, model_input_frame
from return_risk.metrics import probability_metrics, threshold_metrics
from return_risk.modeling import baseline_models
from return_risk.policy import select_cost_threshold


def main() -> None:
    train_path = PROCESSED_DIR / "train.csv"
    validation_path = PROCESSED_DIR / "validation.csv"
    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError("Prepared splits not found. Run scripts/prepare_data.py first.")

    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    x_train = model_input_frame(train)
    y_train = binary_target(train)
    x_validation = model_input_frame(validation)
    y_validation = binary_target(validation)

    report = {
        "evaluation_partition": "validation_only",
        "test_set_accessed": False,
        "model_features": list(x_train.columns),
        "models": {},
    }
    trained_models = {}
    for name, model in baseline_models().items():
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_validation)[:, 1]
        policy = select_cost_threshold(y_validation, probabilities)
        report["models"][name] = {
            "probability_metrics": probability_metrics(y_validation, probabilities),
            "cost_policy": policy,
            "threshold_metrics": threshold_metrics(
                y_validation, probabilities, policy["selected_threshold"]
            ),
        }
        trained_models[name] = model

    selected_name = max(
        report["models"],
        key=lambda name: report["models"][name]["probability_metrics"]["average_precision"],
    )
    report["selected_baseline"] = selected_name
    report["selection_metric"] = "validation_average_precision"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "baseline_validation.json"
    model_path = MODELS_DIR / "baseline_model.joblib"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "model": trained_models[selected_name],
            "model_name": selected_name,
            "threshold": report["models"][selected_name]["cost_policy"]["selected_threshold"],
            "features": list(x_train.columns),
        },
        model_path,
    )

    print(json.dumps(report, indent=2))
    print(f"\nSaved validation report to {report_path}")
    print(f"Saved selected baseline to {model_path}")


if __name__ == "__main__":
    main()

