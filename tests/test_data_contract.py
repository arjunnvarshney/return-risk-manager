from __future__ import annotations

import pandas as pd
import pytest

from return_risk.config import BLOCKED_MODEL_COLUMNS, MODEL_FEATURES
from return_risk.data import DatasetContractError, model_input_frame, validate_raw_data


def minimal_valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Order_ID": ["ORD1", "ORD2"],
            "Product_ID": ["PROD1", "PROD2"],
            "User_ID": ["USER1", "USER2"],
            "Order_Date": ["2024-01-01", "2024-01-02"],
            "Product_Category": ["Books", "Clothing"],
            "Product_Price": [100.0, 200.0],
            "Order_Quantity": [1, 2],
            "Discount_Applied": [10.0, 0.0],
            "Shipping_Method": ["Standard", "Express"],
            "Payment_Method": ["Wallet", "COD"],
            "User_Age": [30, 40],
            "User_Gender": ["Female", "Male"],
            "User_Location": ["City1", "City2"],
            "Return_Status": ["Not Returned", "Returned"],
            "Return_Reason": ["No Return", "Changed Mind"],
            "Days_to_Return": [0, 5],
            "Order_Value": [90.0, 400.0],
            "Return_Cost": [0, 200],
            "Profit_Loss": [90.0, 200.0],
            "CO2_Emissions": [1.0, 2.0],
            "Packaging_Waste": [0.2, 0.4],
            "CO2_Saved": [1.0, 0.0],
            "Waste_Avoided": [0.2, 0.0],
        }
    )


def test_model_allowlist_contains_no_blocked_columns() -> None:
    assert not BLOCKED_MODEL_COLUMNS.intersection(MODEL_FEATURES)


def test_model_input_contains_only_allowed_features() -> None:
    frame = minimal_valid_frame()
    validate_raw_data(frame)
    prepared = model_input_frame(frame)
    assert list(prepared.columns) == MODEL_FEATURES


def test_missing_required_column_fails_contract() -> None:
    frame = minimal_valid_frame().drop(columns="Return_Reason")
    with pytest.raises(DatasetContractError, match="Missing required columns"):
        validate_raw_data(frame)


def test_inconsistent_order_value_fails_contract() -> None:
    frame = minimal_valid_frame()
    frame.loc[0, "Order_Value"] = 999.0
    with pytest.raises(DatasetContractError, match="Order_Value is inconsistent"):
        validate_raw_data(frame)

