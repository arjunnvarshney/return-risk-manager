from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from return_risk.config import BLOCKED_MODEL_COLUMNS, RANDOM_SEED
from return_risk.data import DatasetContractError, add_prediction_time_features
from return_risk.modeling import UNTUNED_CATBOOST_PARAMETERS

V2_NUMERIC_FEATURES = [
    "Product_Price",
    "Order_Quantity",
    "Discount_Applied",
    "Order_Value",
    "order_year",
    "order_month",
    "order_day_of_week",
    "Gross_Order_Value",
    "Net_Unit_Price",
    "Discount_Amount",
    "Log_Product_Price",
    "Log_Order_Value",
    "order_quarter",
    "order_is_weekend",
    "order_month_index",
]

V2_CATEGORICAL_FEATURES = [
    "Product_Category",
    "Shipping_Method",
    "Payment_Method",
    "category_shipping",
    "category_payment",
    "category_discount_band",
]

V2_MODEL_FEATURES = V2_NUMERIC_FEATURES + V2_CATEGORICAL_FEATURES


def v2_input_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Create deployment-safe interactions from checkout-time fields only."""
    blocked_overlap = BLOCKED_MODEL_COLUMNS.intersection(V2_MODEL_FEATURES)
    if blocked_overlap:
        raise RuntimeError(f"Blocked columns entered v2: {sorted(blocked_overlap)}")

    enriched = add_prediction_time_features(frame)
    required = {
        "Product_Price",
        "Order_Quantity",
        "Discount_Applied",
        "Order_Value",
        "Product_Category",
        "Shipping_Method",
        "Payment_Method",
    }
    missing = sorted(required - set(enriched.columns))
    if missing:
        raise DatasetContractError(f"Prepared data is missing v2 inputs: {missing}")

    price = pd.to_numeric(enriched["Product_Price"], errors="raise")
    quantity = pd.to_numeric(enriched["Order_Quantity"], errors="raise")
    discount = pd.to_numeric(enriched["Discount_Applied"], errors="raise")
    order_value = pd.to_numeric(enriched["Order_Value"], errors="raise")

    enriched["Gross_Order_Value"] = price * quantity
    enriched["Net_Unit_Price"] = price * (1 - discount / 100)
    enriched["Discount_Amount"] = enriched["Gross_Order_Value"] - order_value
    enriched["Log_Product_Price"] = np.log1p(price.clip(lower=0))
    enriched["Log_Order_Value"] = np.log1p(order_value.clip(lower=0))
    enriched["order_quarter"] = ((enriched["order_month"] - 1) // 3 + 1).astype(int)
    enriched["order_is_weekend"] = (enriched["order_day_of_week"] >= 5).astype(int)
    enriched["order_month_index"] = (
        (enriched["order_year"] - 2022) * 12 + enriched["order_month"] - 1
    ).astype(int)

    category = enriched["Product_Category"].astype(str)
    shipping = enriched["Shipping_Method"].astype(str)
    payment = enriched["Payment_Method"].astype(str)
    discount_band = pd.cut(
        discount,
        bins=[-np.inf, 10, 25, 40, np.inf],
        labels=["minimal", "moderate", "high", "very_high"],
        include_lowest=True,
    ).astype(str)
    enriched["category_shipping"] = category + "__" + shipping
    enriched["category_payment"] = category + "__" + payment
    enriched["category_discount_band"] = category + "__" + discount_band

    return enriched[V2_MODEL_FEATURES].copy()


def build_v2_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=True),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, V2_NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, V2_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def v2_logistic_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_v2_preprocessor()),
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
    )


def v2_catboost_model(iterations: int = 1_000) -> CatBoostClassifier:
    return CatBoostClassifier(iterations=iterations, **UNTUNED_CATBOOST_PARAMETERS)
