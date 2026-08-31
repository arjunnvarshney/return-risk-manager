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

from return_risk.config import MODEL_FEATURES, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from return_risk.data import binary_target, model_input_frame
from return_risk.policy import CostAssumptions, policy_cost_breakdown, select_cost_threshold
from return_risk.robustness import threshold_category_audit

MODEL_NAME = "catboost_ablation_without_location"
CANDIDATE_FEATURES = MODEL_FEATURES
MAX_BALANCED_FLAG_RATE = 0.30
MAX_CATEGORY_SHARE = 0.75


def _with_cost(audit: dict, y, probabilities, assumptions: CostAssumptions) -> dict:
    cost = policy_cost_breakdown(y, probabilities, audit["threshold"], assumptions)
    return {**audit, "cost": cost}


def _closest_rate(rows: list[dict], target: float) -> dict:
    return min(rows, key=lambda row: abs(row["flagged_rate"] - target))


def _first_with_categories(rows: list[dict], count: int) -> dict | None:
    eligible = [row for row in rows if row["categories_represented"] >= count]
    return min(eligible, key=lambda row: row["flagged_rate"]) if eligible else None


def _save_plot(rows: list[dict], prevalence: float, output_path: Path) -> None:
    nonempty = [row for row in rows if row["orders_flagged"]]
    rates = np.array([row["flagged_rate"] for row in nonempty])
    precision = np.array([row["precision"] for row in nonempty])
    recall = np.array([row["recall"] for row in nonempty])
    savings = np.array([row["cost"]["savings_per_1000_orders"] for row in nonempty])
    shares = np.array([row["largest_group_share"] for row in nonempty])
    categories = np.array([row["categories_represented"] for row in nonempty])

    order = np.argsort(rates)
    rates = rates[order]
    precision = precision[order]
    recall = recall[order]
    savings = savings[order]
    shares = shares[order]
    categories = categories[order]

    figure, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    axes[0].plot(rates, precision, label="Precision", color="#2563EB", linewidth=2)
    axes[0].plot(rates, recall, label="Recall", color="#16A34A", linewidth=2)
    axes[0].axhline(prevalence, color="#6B7280", linestyle=":", label="Return prevalence")
    axes[0].set_ylabel("Rate")
    axes[0].set_title("Detection quality")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(rates, savings, color="#D97706", linewidth=2)
    axes[1].axhline(0, color="#111827", linewidth=0.8)
    axes[1].set_ylabel("Estimated savings per 1,000 orders (₹)")
    axes[1].set_title("Hypothetical base-cost outcome")
    axes[1].grid(alpha=0.2)

    axes[2].plot(rates, shares, color="#7C3AED", linewidth=2, label="Largest category share")
    axes[2].axhline(
        MAX_CATEGORY_SHARE,
        color="#DC2626",
        linestyle=":",
        label="Illustrative 75% concentration guardrail",
    )
    category_axis = axes[2].twinx()
    category_axis.step(
        rates,
        categories,
        color="#0891B2",
        where="post",
        alpha=0.7,
        label="Categories represented",
    )
    axes[2].set_ylabel("Largest category share")
    category_axis.set_ylabel("Categories represented")
    category_axis.set_yticks(range(1, int(categories.max()) + 1))
    axes[2].set_xlabel("Share of validation orders sent to review")
    axes[2].set_title("Review-queue product-category concentration")
    lines, labels = axes[2].get_legend_handles_labels()
    extra_lines, extra_labels = category_axis.get_legend_handles_labels()
    axes[2].legend(lines + extra_lines, labels + extra_labels, loc="best")
    axes[2].grid(alpha=0.2)

    figure.suptitle("Threshold trade-offs for the no-location candidate", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    validation_path = PROCESSED_DIR / "validation.csv"
    model_path = MODELS_DIR / f"{MODEL_NAME}.cbm"
    if not validation_path.exists():
        raise FileNotFoundError("Validation split missing. Run scripts/prepare_data.py first.")
    if not model_path.exists():
        raise FileNotFoundError("No-location candidate missing. Run run_feature_ablation.py first.")

    # Intentionally read validation only. The locked final test is not accessed.
    validation = pd.read_csv(validation_path)
    x_validation = model_input_frame(validation)[CANDIDATE_FEATURES]
    y_validation = binary_target(validation)
    categories = validation["Product_Category"]
    model = CatBoostClassifier()
    model.load_model(model_path)
    probabilities = model.predict_proba(x_validation)[:, 1]
    assumptions = CostAssumptions()

    thresholds = np.r_[np.nextafter(probabilities.max(), np.inf), np.unique(probabilities)]
    rows = [
        _with_cost(
            threshold_category_audit(
                y_validation,
                probabilities,
                categories,
                float(threshold),
            ),
            y_validation,
            probabilities,
            assumptions,
        )
        for threshold in thresholds
    ]

    cost_policy = select_cost_threshold(y_validation, probabilities, assumptions)
    cost_optimal = _with_cost(
        threshold_category_audit(
            y_validation,
            probabilities,
            categories,
            cost_policy["selected_threshold"],
        ),
        y_validation,
        probabilities,
        assumptions,
    )
    milestones = {
        "cost_optimal": cost_optimal,
        "approximately_top_10_percent": _closest_rate(rows, 0.10),
        "approximately_top_20_percent": _closest_rate(rows, 0.20),
        "approximately_top_30_percent": _closest_rate(rows, 0.30),
        "first_threshold_with_2_categories": _first_with_categories(rows, 2),
        "first_threshold_with_3_categories": _first_with_categories(rows, 3),
        "first_threshold_with_all_categories": _first_with_categories(
            rows,
            int(categories.nunique()),
        ),
    }

    prevalence = float(y_validation.mean())
    balanced = [
        row
        for row in rows
        if row["orders_flagged"]
        and row["flagged_rate"] <= MAX_BALANCED_FLAG_RATE
        and row["largest_group_share"] <= MAX_CATEGORY_SHARE
        and row["precision"] >= prevalence
        and row["cost"]["savings_vs_no_intervention"] > 0
    ]
    balanced_candidate = (
        max(balanced, key=lambda row: row["cost"]["savings_vs_no_intervention"])
        if balanced
        else None
    )

    report = {
        "evaluation_partition": "validation_only",
        "test_set_accessed": False,
        "model": MODEL_NAME,
        "features": CANDIDATE_FEATURES,
        "cost_assumptions": assumptions.__dict__,
        "thresholds_evaluated": int(len(rows)),
        "milestones": milestones,
        "illustrative_balanced_guardrails": {
            "maximum_flagged_rate": MAX_BALANCED_FLAG_RATE,
            "maximum_largest_category_share": MAX_CATEGORY_SHARE,
            "minimum_precision": round(prevalence, 6),
            "requires_positive_estimated_savings": True,
            "candidate": balanced_candidate,
            "status": (
                "candidate_found" if balanced_candidate is not None else "no_candidate_found"
            ),
        },
        "recommended_action_design": {
            "high_risk": "Soft verification or human review only; never automatic rejection.",
            "lower_risk": "Return the score for monitoring without customer friction.",
            "category_quotas": "Not recommended; quotas would deliberately flag lower-risk orders.",
            "threshold_status": (
                "Validation demo setting only; merchant costs and review capacity are required "
                "before production use."
            ),
        },
        "limitations": [
            "Thresholds were explored on validation data and are not final test claims.",
            "The 75% category-share and 30% capacity limits are illustrative, not merchant input.",
            "Cost assumptions and intervention effectiveness are hypothetical.",
            "Category concentration may reflect synthetic target construction.",
            "The source dataset is synthetic.",
        ],
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    figure_dir = REPORTS_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "threshold_concentration_validation.json"
    figure_path = figure_dir / "validation_threshold_concentration.png"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _save_plot(rows, prevalence, figure_path)

    print(json.dumps(report, indent=2))
    print(f"\nSaved threshold-concentration report to {report_path}")
    print(f"Saved threshold trade-off plot to {figure_path}")


if __name__ == "__main__":
    main()
