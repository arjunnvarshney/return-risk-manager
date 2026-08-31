# ruff: noqa: E402
from __future__ import annotations

import json
import os
from dataclasses import asdict
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

from return_risk.config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from return_risk.data import binary_target, model_input_frame
from return_risk.metrics import probability_metrics, threshold_metrics
from return_risk.policy import (
    policy_cost,
    select_cost_threshold,
    standard_cost_scenarios,
)
from return_risk.uncertainty import (
    bootstrap_policy_intervals,
    bootstrap_ranking_intervals,
)

BOOTSTRAP_REPLICATES = 2_000


def main() -> None:
    validation_path = PROCESSED_DIR / "validation.csv"
    model_path = MODELS_DIR / "return_risk_final.cbm"
    if not validation_path.exists():
        raise FileNotFoundError("Validation split missing. Run scripts/prepare_data.py first.")
    if not model_path.exists():
        raise FileNotFoundError("Untuned CatBoost model missing. Run train_catboost.py first.")

    validation = pd.read_csv(validation_path)
    x_validation = model_input_frame(validation)
    y_validation = binary_target(validation)
    model = CatBoostClassifier()
    model.load_model(model_path)
    probabilities = model.predict_proba(x_validation)[:, 1]

    scenarios = standard_cost_scenarios()
    ranking_intervals = bootstrap_ranking_intervals(
        y_validation,
        probabilities,
        replicates=BOOTSTRAP_REPLICATES,
    )
    scenario_results = {}
    thresholds = np.linspace(0.0, 1.0, 501)
    cost_curves = {}
    for name, assumptions in scenarios.items():
        policy = select_cost_threshold(y_validation, probabilities, assumptions)
        threshold = policy["selected_threshold"]
        scenario_results[name] = {
            "assumptions": asdict(assumptions),
            "selected_threshold": threshold,
            "policy": policy,
            "threshold_metrics": threshold_metrics(
                y_validation,
                probabilities,
                threshold,
            ),
            "bootstrap_intervals": bootstrap_policy_intervals(
                y_validation,
                probabilities,
                threshold,
                assumptions,
                replicates=BOOTSTRAP_REPLICATES,
            ),
        }
        cost_curves[name] = [
            policy_cost(y_validation, probabilities, threshold_value, assumptions)
            / len(y_validation)
            * 1000
            for threshold_value in thresholds
        ]

    report = {
        "evaluation_partition": "validation_only",
        "test_set_accessed": False,
        "model": "promoted_catboost_without_location_raw_probabilities",
        "probability_metrics": probability_metrics(y_validation, probabilities),
        "ranking_bootstrap_intervals": ranking_intervals,
        "cost_assumptions_status": (
            "Hypothetical sensitivity scenarios; replace with merchant-observed costs "
            "before production use."
        ),
        "scenarios": scenario_results,
        "limitations": [
            "Thresholds were selected on validation data and are not final test claims.",
            "Bootstrap intervals condition on the selected fixed threshold.",
            "Intervention effectiveness is assumed rather than observed.",
            "The source dataset is synthetic.",
        ],
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    figure_dir = REPORTS_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "cost_sensitivity_validation.json"
    figure_path = figure_dir / "validation_cost_threshold_sensitivity.png"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(9, 6))
    colors = {
        "low_false_positive_cost": "#16A34A",
        "base_case": "#2563EB",
        "high_false_positive_cost": "#DC2626",
    }
    labels = {
        "low_false_positive_cost": "Low false-positive cost",
        "base_case": "Base case",
        "high_false_positive_cost": "High false-positive cost",
    }
    for name, costs in cost_curves.items():
        selected_threshold = scenario_results[name]["selected_threshold"]
        axis.plot(thresholds, costs, color=colors[name], label=labels[name], linewidth=2)
        axis.axvline(selected_threshold, color=colors[name], linestyle=":", alpha=0.8)
    axis.set(
        title="Validation cost sensitivity by decision threshold",
        xlabel="Return-risk threshold",
        ylabel="Estimated cost per 1,000 orders",
        xlim=(0, 1),
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)

    print(json.dumps(report, indent=2))
    print(f"\nSaved cost-sensitivity report to {report_path}")
    print(f"Saved threshold-cost plot to {figure_path}")


if __name__ == "__main__":
    main()
