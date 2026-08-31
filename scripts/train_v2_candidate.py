from __future__ import annotations

import json
from statistics import mean, median, pstdev

import joblib
import pandas as pd
from catboost import CatBoostClassifier

from return_risk.config import (
    CATEGORICAL_FEATURES,
    MODELS_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
)
from return_risk.data import binary_target, model_input_frame
from return_risk.metrics import probability_metrics
from return_risk.modeling import fixed_untuned_catboost_model, untuned_catboost_model
from return_risk.splits import expanding_date_folds
from return_risk.v2 import (
    V2_CATEGORICAL_FEATURES,
    V2_MODEL_FEATURES,
    v2_catboost_model,
    v2_input_frame,
    v2_logistic_model,
)

CATBOOST_CANDIDATES = {"v1_catboost", "v2_catboost"}


def candidate_frame(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    if name == "v1_catboost":
        return model_input_frame(frame)
    return v2_input_frame(frame)


def candidate_categories(name: str) -> list[str]:
    return CATEGORICAL_FEATURES if name == "v1_catboost" else V2_CATEGORICAL_FEATURES


def new_candidate(name: str, iterations: int | None = None):
    if name == "v1_catboost":
        return (
            fixed_untuned_catboost_model(iterations)
            if iterations is not None
            else untuned_catboost_model()
        )
    if name == "v2_catboost":
        return v2_catboost_model(iterations or 1_000)
    if name == "v2_logistic":
        return v2_logistic_model()
    raise ValueError(f"Unknown candidate: {name}")


def evaluate_training_folds(train: pd.DataFrame) -> dict[str, dict]:
    folds = expanding_date_folds(train, n_splits=3, initial_train_fraction=0.40)
    results: dict[str, dict] = {}
    for name in ("v1_catboost", "v2_catboost", "v2_logistic"):
        fold_results = []
        best_iterations = []
        for fold_number, (fold_train, fold_validation) in enumerate(folds, start=1):
            x_train = candidate_frame(name, fold_train)
            y_train = binary_target(fold_train)
            x_validation = candidate_frame(name, fold_validation)
            y_validation = binary_target(fold_validation)
            model = new_candidate(name)
            if name in CATBOOST_CANDIDATES:
                model.fit(
                    x_train,
                    y_train,
                    cat_features=candidate_categories(name),
                    eval_set=(x_validation, y_validation),
                    early_stopping_rounds=60,
                    use_best_model=True,
                    verbose=False,
                )
                best_iterations.append(max(1, int(model.tree_count_)))
            else:
                model.fit(x_train, y_train)
            probabilities = model.predict_proba(x_validation)[:, 1]
            metrics = probability_metrics(y_validation, probabilities)
            fold_results.append(
                {
                    "fold": fold_number,
                    "train_rows": len(fold_train),
                    "validation_rows": len(fold_validation),
                    "train_date_max": str(pd.to_datetime(fold_train["Order_Date"]).max().date()),
                    "validation_date_min": str(
                        pd.to_datetime(fold_validation["Order_Date"]).min().date()
                    ),
                    "average_precision": metrics["average_precision"],
                    "roc_auc": metrics["roc_auc"],
                    "top_10_percent_precision": metrics["capacity"]["top_10_percent"][
                        "precision"
                    ],
                }
            )
        average_precision = [row["average_precision"] for row in fold_results]
        roc_auc = [row["roc_auc"] for row in fold_results]
        results[name] = {
            "feature_set": "official_v1" if name == "v1_catboost" else "v2_checkout_only",
            "fold_results": fold_results,
            "mean_average_precision": round(mean(average_precision), 6),
            "average_precision_std": round(pstdev(average_precision), 6),
            "mean_roc_auc": round(mean(roc_auc), 6),
            "recommended_iterations": (
                int(round(median(best_iterations))) if best_iterations else None
            ),
        }
    return results


def fit_for_exploratory_validation(
    name: str,
    iterations: int | None,
    train: pd.DataFrame,
    validation: pd.DataFrame,
):
    model = new_candidate(name, iterations)
    x_train = candidate_frame(name, train)
    y_train = binary_target(train)
    x_validation = candidate_frame(name, validation)
    y_validation = binary_target(validation)
    if name in CATBOOST_CANDIDATES:
        model.fit(
            x_train,
            y_train,
            cat_features=candidate_categories(name),
            verbose=False,
        )
    else:
        model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_validation)[:, 1]
    return model, probability_metrics(y_validation, probabilities)


def save_research_model(name: str, model) -> str:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if isinstance(model, CatBoostClassifier):
        path = MODELS_DIR / "research_selected_candidate.cbm"
        model.save_model(path)
    else:
        path = MODELS_DIR / "research_selected_candidate.joblib"
        joblib.dump(model, path)
    return path.name


def main() -> None:
    train_path = PROCESSED_DIR / "train.csv"
    validation_path = PROCESSED_DIR / "validation.csv"
    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError("Prepared splits not found. Run scripts/prepare_data.py first.")

    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    cross_validation = evaluate_training_folds(train)
    selected = max(
        cross_validation,
        key=lambda name: cross_validation[name]["mean_average_precision"],
    )

    exploratory_validation = {}
    fitted_selected_model = None
    for name, candidate_result in cross_validation.items():
        model, metrics = fit_for_exploratory_validation(
            name,
            candidate_result["recommended_iterations"],
            train,
            validation,
        )
        exploratory_validation[name] = metrics
        if name == selected:
            fitted_selected_model = model

    artifact_name = save_research_model(selected, fitted_selected_model)
    v1_cv_ap = cross_validation["v1_catboost"]["mean_average_precision"]
    selected_cv_ap = cross_validation[selected]["mean_average_precision"]
    report = {
        "research_status": "post_test_exploratory_not_eligible_for_promotion",
        "official_release_untouched": True,
        "held_out_test_accessed_by_script": False,
        "selection_partition": "original_train_expanding_date_folds_only",
        "selection_metric": "mean_training_fold_average_precision",
        "validation_status": "reused_exploratory_check_not_a_new_holdout",
        "candidates": cross_validation,
        "selected_research_candidate": selected,
        "selected_artifact": artifact_name,
        "selected_minus_v1_cv_average_precision": round(selected_cv_ap - v1_cv_ap, 6),
        "exploratory_reused_validation": exploratory_validation,
        "v2_feature_count": len(V2_MODEL_FEATURES),
        "limitations": [
            "The official held-out test result was already known before this research began.",
            "The original validation period had already been used in earlier development.",
            "Synthetic data may not reproduce real merchant return behavior.",
            "This artifact is not loaded by the API or dashboard.",
        ],
        "promotion_requirements": [
            "Collect a new untouched chronologically later outcome set.",
            "Freeze the candidate and decision policy before opening that set.",
            "Confirm ranking quality, calibration, false-positive cost, and slice safety.",
        ],
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "v2_research_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved post-test research report to {report_path}")
    print(f"Saved isolated research artifact to {MODELS_DIR / artifact_name}")


if __name__ == "__main__":
    main()
