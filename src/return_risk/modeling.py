from __future__ import annotations

from catboost import CatBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from return_risk.config import RANDOM_SEED
from return_risk.features import build_preprocessor

UNTUNED_CATBOOST_PARAMETERS = {
    "learning_rate": 0.05,
    "depth": 6,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "l2_leaf_reg": 5.0,
    "random_seed": RANDOM_SEED,
    "allow_writing_files": False,
    "verbose": False,
}


def baseline_models() -> dict[str, Pipeline]:
    return {
        "dummy_prior": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", DummyClassifier(strategy="prior")),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        max_iter=2_000,
                        random_state=RANDOM_SEED,
                        solver="liblinear",
                    ),
                ),
            ]
        ),
    }


def untuned_catboost_model() -> CatBoostClassifier:
    """Return a conservative benchmark, not a tuned production model."""
    return CatBoostClassifier(
        iterations=1_000,
        **UNTUNED_CATBOOST_PARAMETERS,
    )


def fixed_untuned_catboost_model(iterations: int = 101) -> CatBoostClassifier:
    """Return the frozen CatBoost structure used for out-of-fold scoring."""
    return CatBoostClassifier(
        iterations=iterations,
        **UNTUNED_CATBOOST_PARAMETERS,
    )
