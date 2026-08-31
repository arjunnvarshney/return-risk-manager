# ruff: noqa: E402
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
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
from sklearn.metrics import precision_recall_curve

from return_risk.config import DATE_COLUMN, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from return_risk.data import binary_target, model_input_frame
from return_risk.metrics import probability_metrics, threshold_metrics
from return_risk.policy import CostAssumptions, policy_cost_breakdown
from return_risk.release import load_and_validate_release
from return_risk.robustness import threshold_category_audit
from return_risk.slices import disparity_summary, slice_metrics
from return_risk.uncertainty import (
    bootstrap_policy_intervals,
    bootstrap_ranking_intervals,
)

BOOTSTRAP_REPLICATES = 2_000


def _audit_slice(y, probabilities, groups, threshold: float) -> dict:
    result = slice_metrics(y, probabilities, groups, threshold, min_group_size=20)
    result["observed_disparities"] = disparity_summary(result["eligible_groups"])
    return result


def _save_figure(y, probabilities, threshold: float, metrics: dict, output_path: Path) -> None:
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    threshold_index = int(np.argmin(np.abs(thresholds - threshold)))
    confusion = metrics["confusion_matrix"]
    matrix = np.array(
        [
            [confusion["true_negative"], confusion["false_positive"]],
            [confusion["false_negative"], confusion["true_positive"]],
        ]
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(recall, precision, color="#2563EB", linewidth=2)
    axes[0].scatter(
        recall[threshold_index],
        precision[threshold_index],
        color="#DC2626",
        s=65,
        label=f"Frozen threshold {threshold:.3f}",
        zorder=3,
    )
    axes[0].set(
        title="Held-out precision-recall curve",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    image = axes[1].imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axes[1].text(column, row, str(matrix[row, column]), ha="center", va="center")
    axes[1].set(
        title="Held-out confusion matrix",
        xlabel="Predicted class",
        ylabel="Actual class",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Not returned", "Flagged"],
        yticklabels=["Not returned", "Returned"],
    )
    figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    figure.suptitle("Frozen return-risk release: final test evaluation", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    test_path = PROCESSED_DIR / "test.csv"
    model_path = MODELS_DIR / "return_risk_final.cbm"
    manifest_path = MODELS_DIR / "release_manifest.json"
    report_path = REPORTS_DIR / "final_test_evaluation.json"
    figure_path = REPORTS_DIR / "figures" / "final_test_evaluation.png"
    if report_path.exists():
        raise RuntimeError(
            "Final test report already exists. Refusing to evaluate the held-out test twice."
        )
    if not test_path.exists():
        raise FileNotFoundError("Locked test partition is missing.")

    # Validate every frozen choice before opening the held-out test partition.
    manifest = load_and_validate_release(manifest_path, model_path)
    model = CatBoostClassifier()
    model.load_model(model_path)
    if list(model.feature_names_) != manifest["model_features"]:
        raise RuntimeError("Frozen model feature names do not match the release manifest.")

    # This is the sole intentional read of test.csv in the project workflow.
    test = pd.read_csv(test_path)
    x_test = model_input_frame(test)
    y_test = binary_target(test)
    probabilities = model.predict_proba(x_test)[:, 1]
    threshold = float(manifest["decision_threshold"])
    assumptions = CostAssumptions()
    threshold_result = threshold_metrics(y_test, probabilities, threshold)
    policy = policy_cost_breakdown(y_test, probabilities, threshold, assumptions)
    ranking_intervals = bootstrap_ranking_intervals(
        y_test,
        probabilities,
        replicates=BOOTSTRAP_REPLICATES,
    )
    policy_intervals = bootstrap_policy_intervals(
        y_test,
        probabilities,
        threshold,
        assumptions,
        replicates=BOOTSTRAP_REPLICATES,
    )
    age_band = pd.cut(
        test["User_Age"],
        bins=[17, 29, 44, 60, np.inf],
        labels=["18-29", "30-44", "45-60", "61+"],
    )
    group_columns = {
        "Product_Category": test["Product_Category"],
        "Payment_Method": test["Payment_Method"],
        "Shipping_Method": test["Shipping_Method"],
        "User_Location": test["User_Location"],
        "User_Gender": test["User_Gender"],
        "Age_Band": age_band,
    }
    slices = {
        name: _audit_slice(y_test, probabilities, groups, threshold)
        for name, groups in group_columns.items()
    }

    report = {
        "evaluation_partition": "held_out_chronological_test",
        "test_set_accessed": True,
        "evaluation_number": 1,
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "release": manifest,
        "test_data": {
            "orders": int(len(test)),
            "date_min": str(pd.to_datetime(test[DATE_COLUMN]).min().date()),
            "date_max": str(pd.to_datetime(test[DATE_COLUMN]).max().date()),
            "prevalence": round(float(y_test.mean()), 6),
        },
        "probability_metrics": probability_metrics(y_test, probabilities),
        "fixed_threshold_metrics": threshold_result,
        "fixed_threshold_policy": {
            "cost_assumptions": assumptions.__dict__,
            "point_estimate": policy,
            "bootstrap_intervals": policy_intervals,
        },
        "ranking_bootstrap_intervals": ranking_intervals,
        "category_concentration": threshold_category_audit(
            y_test,
            probabilities,
            test["Product_Category"],
            threshold,
        ),
        "slice_audit": slices,
        "interpretation": {
            "allowed_use": "Soft verification or human review only.",
            "not_allowed": [
                "Automatic order rejection",
                "Automatic restriction of return rights",
            ],
            "metric_status": "Final held-out metrics; no further model selection is allowed.",
        },
        "limitations": [
            "The source dataset is synthetic.",
            "Cost assumptions and intervention effectiveness are hypothetical.",
            "Bootstrap intervals describe sampling uncertainty, not production drift.",
            "Slice comparisons are descriptive and do not establish causal fairness.",
        ],
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _save_figure(y_test, probabilities, threshold, threshold_result, figure_path)

    summary = {
        "release_id": manifest["release_id"],
        "orders": len(test),
        "probability_metrics": report["probability_metrics"],
        "fixed_threshold_metrics": threshold_result,
        "policy": policy,
        "ranking_bootstrap_intervals": ranking_intervals,
        "policy_bootstrap_intervals": policy_intervals,
        "category_concentration": report["category_concentration"],
    }
    print(json.dumps(summary, indent=2))
    print(f"\nSaved the one-time final test report to {report_path}")
    print(f"Saved final evaluation plot to {figure_path}")


if __name__ == "__main__":
    main()
