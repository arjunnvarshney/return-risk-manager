from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from return_risk.config import (
    BENCHMARK_CATEGORICAL_FEATURES,
    BENCHMARK_MODEL_FEATURES,
    BLOCKED_MODEL_COLUMNS,
    DATE_COLUMN,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    POSITIVE_LABEL,
    REQUIRED_RAW_COLUMNS,
    ROW_ID_COLUMN,
    TARGET_COLUMN,
)


class DatasetContractError(ValueError):
    """Raised when the input dataset violates the expected data contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")
    frame = pd.read_csv(path)
    validate_raw_data(frame)
    return frame


def validate_raw_data(frame: pd.DataFrame) -> None:
    duplicate_columns = frame.columns[frame.columns.duplicated()].tolist()
    if duplicate_columns:
        raise DatasetContractError(f"Duplicate columns: {duplicate_columns}")

    missing_columns = sorted(REQUIRED_RAW_COLUMNS - set(frame.columns))
    if missing_columns:
        raise DatasetContractError(f"Missing required columns: {missing_columns}")

    allowed_targets = {"Returned", "Not Returned"}
    actual_targets = set(frame[TARGET_COLUMN].dropna().unique())
    if actual_targets != allowed_targets:
        message = (
            f"Unexpected target labels: {sorted(actual_targets)}; "
            f"expected {sorted(allowed_targets)}"
        )
        raise DatasetContractError(message)

    if frame[ROW_ID_COLUMN].isna().any() or not frame[ROW_ID_COLUMN].is_unique:
        raise DatasetContractError(f"{ROW_ID_COLUMN} must be non-null and unique")

    parsed_dates = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
    if parsed_dates.isna().any():
        raise DatasetContractError(f"{DATE_COLUMN} contains unparseable dates")

    raw_model_columns = set(NUMERIC_FEATURES[:4] + BENCHMARK_CATEGORICAL_FEATURES)
    columns_with_nulls = sorted(c for c in raw_model_columns if frame[c].isna().any())
    if columns_with_nulls:
        raise DatasetContractError(f"Model-input columns contain nulls: {columns_with_nulls}")

    expected_order_value = (
        frame["Product_Price"]
        * frame["Order_Quantity"]
        * (1 - frame["Discount_Applied"] / 100)
    )
    if not np.allclose(frame["Order_Value"], expected_order_value, rtol=1e-9, atol=1e-6):
        raise DatasetContractError("Order_Value is inconsistent with price, quantity, and discount")


def add_prediction_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    dates = pd.to_datetime(enriched[DATE_COLUMN], errors="raise")
    enriched["order_year"] = dates.dt.year.astype(int)
    enriched["order_month"] = dates.dt.month.astype(int)
    enriched["order_day_of_week"] = dates.dt.dayofweek.astype(int)
    return enriched


def model_input_frame(frame: pd.DataFrame) -> pd.DataFrame:
    blocked_overlap = BLOCKED_MODEL_COLUMNS.intersection(MODEL_FEATURES)
    if blocked_overlap:
        message = f"Blocked columns entered the feature allowlist: {sorted(blocked_overlap)}"
        raise RuntimeError(message)

    enriched = add_prediction_time_features(frame)
    missing = sorted(set(MODEL_FEATURES) - set(enriched.columns))
    if missing:
        raise DatasetContractError(f"Prepared data is missing model features: {missing}")
    return enriched[MODEL_FEATURES].copy()


def benchmark_input_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the superseded full feature set for reproducible ablation comparisons."""
    enriched = add_prediction_time_features(frame)
    missing = sorted(set(BENCHMARK_MODEL_FEATURES) - set(enriched.columns))
    if missing:
        raise DatasetContractError(f"Prepared data is missing benchmark features: {missing}")
    return enriched[BENCHMARK_MODEL_FEATURES].copy()


def binary_target(frame: pd.DataFrame) -> pd.Series:
    return (frame[TARGET_COLUMN] == POSITIVE_LABEL).astype(int)
