from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from return_risk.config import BLOCKED_MODEL_COLUMNS
from return_risk.v2 import V2_MODEL_FEATURES, v2_input_frame


def example_order() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Product_Price": 1_200.0,
                "Order_Quantity": 2,
                "Discount_Applied": 25.0,
                "Order_Value": 1_800.0,
                "Product_Category": "Clothing",
                "Shipping_Method": "Standard",
                "Payment_Method": "Credit Card",
                "Order_Date": "2025-08-16",
            }
        ]
    )


def test_v2_allowlist_has_no_blocked_or_sensitive_columns() -> None:
    assert not BLOCKED_MODEL_COLUMNS.intersection(V2_MODEL_FEATURES)


def test_v2_features_are_checkout_time_derivations() -> None:
    transformed = v2_input_frame(example_order())
    row = transformed.iloc[0]

    assert list(transformed.columns) == V2_MODEL_FEATURES
    assert row["Gross_Order_Value"] == pytest.approx(2_400.0)
    assert row["Net_Unit_Price"] == pytest.approx(900.0)
    assert row["Discount_Amount"] == pytest.approx(600.0)
    assert row["order_quarter"] == 3
    assert row["order_is_weekend"] == 1
    assert row["category_shipping"] == "Clothing__Standard"
    assert row["category_discount_band"] == "Clothing__moderate"


def test_v2_research_script_does_not_load_held_out_test() -> None:
    script = Path("scripts/train_v2_candidate.py").read_text(encoding="utf-8")
    assert "test.csv" not in script
    assert "final_test_evaluation" not in script
