import numpy as np
import pandas as pd

from return_risk.explainability import (
    build_target_context,
    contextual_reason,
    global_shap_importance,
    local_explanation,
)


def test_global_shap_importance_orders_by_mean_absolute_value():
    result = global_shap_importance(["a", "b"], [[1.0, 4.0], [-1.0, 0.0]])
    assert [row["feature"] for row in result] == ["b", "a"]


def test_local_explanation_reconstructs_raw_prediction():
    result = local_explanation(
        ["price", "payment"],
        [100.0, "COD"],
        [0.3, -0.1],
        expected_value=-0.4,
        raw_prediction=-0.2,
        probability=0.45,
        top_k=2,
    )
    assert np.isclose(result["reconstruction_error"], 0.0)
    assert result["top_reasons"][0]["direction"] == "raised"


def test_contextual_reason_reports_development_association_without_causal_claim():
    frame = pd.DataFrame(
        {
            "Product_Price": [100, 200, 300, 400],
            "Order_Quantity": [1, 2, 3, 4],
            "Discount_Applied": [0, 10, 40, 50],
            "Order_Value": [100, 360, 540, 800],
            "order_year": [2022, 2023, 2024, 2024],
            "order_month": [1, 2, 3, 4],
            "order_day_of_week": [0, 1, 2, 3],
            "Product_Category": ["Books", "Books", "Clothing", "Clothing"],
            "Shipping_Method": ["Express", "Standard", "Express", "Standard"],
            "Payment_Method": ["COD", "Wallet", "COD", "Wallet"],
        }
    )
    context = build_target_context(frame, [0, 0, 1, 1])
    reason = contextual_reason("Product_Category", "Clothing", "raised", context)
    assert "100.0% versus 50.0% overall" in reason
    assert "n=2" in reason
    assert "does not prove causation" in reason


def test_contextual_reason_discloses_unseen_year_as_time_drift():
    frame = pd.DataFrame(
        {
            "Product_Price": [100, 200],
            "Order_Quantity": [1, 2],
            "Discount_Applied": [0, 50],
            "Order_Value": [100, 200],
            "order_year": [2023, 2024],
            "order_month": [1, 2],
            "order_day_of_week": [0, 1],
            "Product_Category": ["Books", "Clothing"],
            "Shipping_Method": ["Express", "Standard"],
            "Payment_Method": ["COD", "Wallet"],
        }
    )
    context = build_target_context(frame, [0, 1])
    reason = contextual_reason("order_year", 2025, "raised", context)
    assert "No 2025 orders were used in development (2023–2024)" in reason
    assert "time drift" in reason
