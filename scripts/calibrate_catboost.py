# ruff: noqa: E402
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib

MATPLOTLIB_CONFIG_DIR = Path(__file__).resolve().parents[1] / "work" / "matplotlib"
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from return_risk.calibration import (
    calibrated_probabilities,
    fit_sigmoid_calibrator,
    reliability_summary,
)
from return_risk.config import (
    CATEGORICAL_FEATURES,
    MODELS_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
)
from return_risk.data import binary_target, model_input_frame
from return_risk.metrics import probability_metrics
from return_risk.modeling import fixed_untuned_catboost_model
from return_risk.policy import select_cost_threshold
from return_risk.splits import expanding_date_folds


def plot_reliability(raw_summary: dict, calibrated_summary: dict, output_path) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect calibration")
    for label, summary, color in (
        ("Raw CatBoost", raw_summary, "#6B7280"),
        ("Sigmoid calibrated", calibrated_summary, "#2563EB"),
    ):
        axis.plot(
            [item["mean_probability"] for item in summary["bins"]],
            [item["observed_return_rate"] for item in summary["bins"]],
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )
    axis.set(
        title="Validation reliability curve",
        xlabel="Mean predicted return probability",
        ylabel="Observed return rate",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    train_path = PROCESSED_DIR / "train.csv"
    validation_path = PROCESSED_DIR / "validation.csv"
    model_path = MODELS_DIR / "return_risk_final.cbm"
    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError("Prepared splits not found. Run scripts/prepare_data.py first.")
    if not model_path.exists():
        message = "Promoted CatBoost model missing. Run scripts/run_feature_ablation.py first."
        raise FileNotFoundError(message)

    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    frozen_model = CatBoostClassifier()
    frozen_model.load_model(model_path)
    frozen_tree_count = int(frozen_model.tree_count_)
    folds = expanding_date_folds(train, n_splits=3, initial_train_fraction=0.40)
    out_of_fold_scores = []
    out_of_fold_labels = []
    fold_details = []
    for fold_number, (fold_train, fold_validation) in enumerate(folds, start=1):
        model = fixed_untuned_catboost_model(iterations=frozen_tree_count)
        x_fold_train = model_input_frame(fold_train)
        y_fold_train = binary_target(fold_train)
        x_fold_validation = model_input_frame(fold_validation)
        y_fold_validation = binary_target(fold_validation)
        model.fit(
            x_fold_train,
            y_fold_train,
            cat_features=CATEGORICAL_FEATURES,
            verbose=False,
        )
        raw_scores = model.predict(x_fold_validation, prediction_type="RawFormulaVal")
        out_of_fold_scores.extend(np.asarray(raw_scores, dtype=float))
        out_of_fold_labels.extend(y_fold_validation.to_numpy(dtype=int))
        fold_details.append(
            {
                "fold": fold_number,
                "training_orders": int(len(fold_train)),
                "calibration_orders": int(len(fold_validation)),
                "training_date_max": str(pd.to_datetime(fold_train["Order_Date"]).max().date()),
                "calibration_date_min": str(
                    pd.to_datetime(fold_validation["Order_Date"]).min().date()
                ),
                "calibration_date_max": str(
                    pd.to_datetime(fold_validation["Order_Date"]).max().date()
                ),
            }
        )

    calibrator = fit_sigmoid_calibrator(out_of_fold_scores, out_of_fold_labels)
    x_validation = model_input_frame(validation)
    y_validation = binary_target(validation)
    validation_raw_scores = frozen_model.predict(
        x_validation, prediction_type="RawFormulaVal"
    )
    raw_probabilities = frozen_model.predict_proba(x_validation)[:, 1]
    calibrated = calibrated_probabilities(calibrator, validation_raw_scores)

    raw_metrics = probability_metrics(y_validation, raw_probabilities)
    calibrated_metrics = probability_metrics(y_validation, calibrated)
    raw_reliability = reliability_summary(y_validation, raw_probabilities)
    calibrated_reliability = reliability_summary(y_validation, calibrated)
    raw_policy = select_cost_threshold(y_validation, raw_probabilities)
    calibrated_policy = select_cost_threshold(y_validation, calibrated)
    calibration_preferred = (
        calibrated_metrics["log_loss"] < raw_metrics["log_loss"]
        and calibrated_metrics["brier_score"] < raw_metrics["brier_score"]
    )

    report = {
        "evaluation_partition": "validation_only",
        "test_set_accessed": False,
        "calibration_method": "sigmoid_platt_scaling",
        "model": "promoted_catboost_without_location",
        "frozen_tree_count": frozen_tree_count,
        "calibration_training": {
            "source": "chronological_out_of_fold_training_predictions",
            "orders": int(len(out_of_fold_labels)),
            "positive_rate": round(float(np.mean(out_of_fold_labels)), 6),
            "folds": fold_details,
            "sigmoid_coefficient": round(float(calibrator.coef_[0, 0]), 6),
            "sigmoid_intercept": round(float(calibrator.intercept_[0]), 6),
        },
        "raw_validation": {
            "probability_metrics": raw_metrics,
            "reliability": raw_reliability,
            "cost_policy": raw_policy,
        },
        "calibrated_validation": {
            "probability_metrics": calibrated_metrics,
            "reliability": calibrated_reliability,
            "cost_policy": calibrated_policy,
        },
        "calibrated_minus_raw": {
            "brier_score": round(
                calibrated_metrics["brier_score"] - raw_metrics["brier_score"], 6
            ),
            "log_loss": round(calibrated_metrics["log_loss"] - raw_metrics["log_loss"], 6),
            "expected_calibration_error": round(
                calibrated_reliability["expected_calibration_error"]
                - raw_reliability["expected_calibration_error"],
                6,
            ),
        },
        "calibration_preferred": calibration_preferred,
        "decision_rule": (
            "Keep calibration only if both validation Brier score and log loss improve."
        ),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    figure_dir = REPORTS_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "catboost_calibration_validation.json"
    calibrator_path = MODELS_DIR / "catboost_sigmoid_calibrator_candidate.joblib"
    figure_path = figure_dir / "validation_reliability_curve.png"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump(calibrator, calibrator_path)
    plot_reliability(raw_reliability, calibrated_reliability, figure_path)

    print(json.dumps(report, indent=2))
    print(f"\nSaved calibration report to {report_path}")
    print(f"Saved sigmoid calibrator to {calibrator_path}")
    print(f"Saved reliability plot to {figure_path}")


if __name__ == "__main__":
    main()
