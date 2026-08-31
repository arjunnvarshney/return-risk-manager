from __future__ import annotations

import json
from statistics import mean, pstdev

import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score

from return_risk.config import (
    CATEGORICAL_FEATURES,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
)
from return_risk.data import binary_target, model_input_frame
from return_risk.metrics import probability_metrics, threshold_metrics
from return_risk.policy import select_cost_threshold
from return_risk.splits import expanding_date_folds

N_TRIALS = 25


def sampled_parameters(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2.0, 20.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
    }


def build_model(parameters: dict) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=1_200,
        loss_function="Logloss",
        eval_metric="AUC",
        border_count=64,
        random_seed=RANDOM_SEED,
        allow_writing_files=False,
        verbose=False,
        **parameters,
    )


def main() -> None:
    train_path = PROCESSED_DIR / "train.csv"
    validation_path = PROCESSED_DIR / "validation.csv"
    comparison_path = REPORTS_DIR / "catboost_validation_comparison.json"
    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError("Prepared splits not found. Run scripts/prepare_data.py first.")
    if not comparison_path.exists():
        raise FileNotFoundError("Untuned comparison missing. Run scripts/train_catboost.py first.")

    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    folds = expanding_date_folds(train, n_splits=3, initial_train_fraction=0.40)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        parameters = sampled_parameters(trial)
        fold_scores = []
        fold_iterations = []
        for fold_train, fold_validation in folds:
            x_fold_train = model_input_frame(fold_train)
            y_fold_train = binary_target(fold_train)
            x_fold_validation = model_input_frame(fold_validation)
            y_fold_validation = binary_target(fold_validation)
            model = build_model(parameters)
            model.fit(
                x_fold_train,
                y_fold_train,
                cat_features=CATEGORICAL_FEATURES,
                eval_set=(x_fold_validation, y_fold_validation),
                early_stopping_rounds=60,
                use_best_model=True,
                verbose=False,
            )
            probabilities = model.predict_proba(x_fold_validation)[:, 1]
            fold_scores.append(float(average_precision_score(y_fold_validation, probabilities)))
            fold_iterations.append(int(model.get_best_iteration()))

        trial.set_user_attr("fold_average_precision", fold_scores)
        trial.set_user_attr("fold_best_iterations", fold_iterations)
        trial.set_user_attr("average_precision_std", pstdev(fold_scores))
        return mean(fold_scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        study_name="catboost_time_aware_average_precision",
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_parameters = study.best_trial.params
    tuned_model = build_model(best_parameters)
    x_train = model_input_frame(train)
    y_train = binary_target(train)
    x_validation = model_input_frame(validation)
    y_validation = binary_target(validation)
    tuned_model.fit(
        x_train,
        y_train,
        cat_features=CATEGORICAL_FEATURES,
        eval_set=(x_validation, y_validation),
        early_stopping_rounds=75,
        use_best_model=True,
        verbose=False,
    )
    validation_probabilities = tuned_model.predict_proba(x_validation)[:, 1]
    policy = select_cost_threshold(y_validation, validation_probabilities)
    tuned_result = {
        "benchmark_type": "25_trial_time_aware_optuna",
        "best_parameters": best_parameters,
        "best_iteration": int(tuned_model.get_best_iteration()),
        "tree_count": int(tuned_model.tree_count_),
        "training_cv": {
            "folds": 3,
            "strategy": "expanding_date_windows",
            "best_mean_average_precision": round(float(study.best_value), 6),
            "best_fold_average_precision": [
                round(float(value), 6)
                for value in study.best_trial.user_attrs["fold_average_precision"]
            ],
            "best_average_precision_std": round(
                float(study.best_trial.user_attrs["average_precision_std"]), 6
            ),
        },
        "validation_probability_metrics": probability_metrics(
            y_validation, validation_probabilities
        ),
        "validation_cost_policy": policy,
        "validation_threshold_metrics": threshold_metrics(
            y_validation,
            validation_probabilities,
            policy["selected_threshold"],
        ),
    }

    previous_comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    untuned_result = previous_comparison["models"]["catboost_untuned"]
    untuned_probability = untuned_result["probability_metrics"]
    tuned_probability = tuned_result["validation_probability_metrics"]
    report = {
        "evaluation_partition": "validation_only",
        "test_set_accessed": False,
        "trial_count": N_TRIALS,
        "selection_metric": "training_time_aware_cv_average_precision",
        "untuned_catboost": untuned_result,
        "tuned_catboost": tuned_result,
        "tuned_minus_untuned_on_validation": {
            "roc_auc": round(tuned_probability["roc_auc"] - untuned_probability["roc_auc"], 6),
            "average_precision": round(
                tuned_probability["average_precision"] - untuned_probability["average_precision"],
                6,
            ),
            "top_10_percent_precision": round(
                tuned_probability["capacity"]["top_10_percent"]["precision"]
                - untuned_probability["capacity"]["top_10_percent"]["precision"],
                6,
            ),
        },
    }
    report["preferred_catboost_variant"] = (
        "tuned"
        if tuned_probability["average_precision"] >= untuned_probability["average_precision"]
        else "untuned"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "catboost_tuning_validation.json"
    trials_path = REPORTS_DIR / "catboost_tuning_trials.csv"
    model_path = MODELS_DIR / "catboost_tuned.cbm"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    study.trials_dataframe().to_csv(trials_path, index=False)
    tuned_model.save_model(model_path)

    print(json.dumps(report, indent=2))
    print(f"\nSaved tuning report to {report_path}")
    print(f"Saved trial history to {trials_path}")
    print(f"Saved tuned CatBoost model to {model_path}")


if __name__ == "__main__":
    main()

