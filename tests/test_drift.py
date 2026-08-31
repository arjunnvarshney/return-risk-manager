import numpy as np
import pandas as pd

from return_risk.drift import build_drift_reference, drift_report


def reference_frame(rows: int = 100) -> pd.DataFrame:
    index = np.arange(rows)
    price = 100 + index * 10
    quantity = index % 5 + 1
    discount = index % 50
    return pd.DataFrame(
        {
            "Product_Price": price,
            "Order_Quantity": quantity,
            "Discount_Applied": discount,
            "Order_Value": price * quantity * (1 - discount / 100),
            "order_year": 2022 + index % 3,
            "order_month": index % 12 + 1,
            "order_day_of_week": index % 7,
            "Product_Category": np.resize(
                ["Books", "Clothing", "Electronics", "Home Appliances", "Toys"],
                rows,
            ),
            "Shipping_Method": np.resize(["Standard", "Express", "Next-Day"], rows),
            "Payment_Method": np.resize(
                ["COD", "Credit Card", "Debit Card", "Wallet"],
                rows,
            ),
        }
    )


def test_identical_distribution_is_stable():
    frame = reference_frame()
    report = drift_report(frame, build_drift_reference(frame))
    assert report["overall_severity"] == "stable"
    assert report["maximum_psi"] < 0.1
    assert report["shadow_mode_required"] is True


def test_concentrated_category_distribution_detects_high_drift():
    frame = reference_frame()
    shifted = frame.copy()
    shifted["Product_Category"] = "Clothing"
    report = drift_report(shifted, build_drift_reference(frame))
    assert report["features"]["Product_Category"]["severity"] == "high"
    assert report["overall_severity"] == "high"


def test_small_batch_reports_insufficient_data():
    frame = reference_frame()
    report = drift_report(frame.head(10), build_drift_reference(frame))
    assert report["sample_status"] == "insufficient_below_30_orders"
    assert report["overall_severity"] == "insufficient_data"
