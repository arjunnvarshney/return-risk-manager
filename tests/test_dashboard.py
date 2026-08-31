import pandas as pd

from return_risk.dashboard import (
    batch_distribution_items,
    demo_batch_frame,
    merchant_risk_report_html,
    prepare_batch_review_frame,
)


def test_demo_batch_is_deterministic_and_schema_shaped() -> None:
    first = demo_batch_frame()
    second = demo_batch_frame()

    assert first.equals(second)
    assert len(first) == 40
    assert first["order_reference"].is_unique
    assert first["product_category"].nunique() == 5
    assert first["discount_applied"].nunique() == 4
    assert list(first.columns) == [
        "order_reference",
        "product_category",
        "product_price",
        "order_quantity",
        "discount_applied",
        "shipping_method",
        "payment_method",
        "order_date",
    ]


def test_batch_review_ranks_expected_loss_not_probability_alone() -> None:
    input_frame = pd.DataFrame(
        {
            "product_category": ["Electronics", "Books", "Clothing"],
        }
    )
    results = [
        {
            "row_number": 2,
            "risk_score": 0.40,
            "computed_order_value": 1000.0,
            "would_flag_under_frozen_policy": False,
        },
        {
            "row_number": 3,
            "risk_score": 0.60,
            "computed_order_value": 100.0,
            "would_flag_under_frozen_policy": True,
        },
        {
            "row_number": 4,
            "risk_score": 0.45,
            "computed_order_value": 500.0,
            "would_flag_under_frozen_policy": True,
        },
    ]
    scored = prepare_batch_review_frame(results, input_frame, 100, 25)

    assert scored.iloc[0]["row_number"] == 2
    assert scored.iloc[0]["expected_loss_exposure"] == 140.0
    assert list(scored["priority_rank"]) == [1, 2, 3]
    assert list(scored["product_category"]) == ["Electronics", "Clothing", "Books"]

    risk_items, category_items = batch_distribution_items(scored)
    assert sum(value for _, value, _ in risk_items) == 3
    assert any(label == "Clothing" and "1 flagged" in detail for label, _, detail in category_items)


def test_merchant_report_is_complete_and_escapes_order_references() -> None:
    input_frame = pd.DataFrame(
        {"product_category": ["Clothing", "Books"]}
    )
    results = [
        {
            "row_number": 2,
            "order_reference": "<script>alert(1)</script>",
            "risk_score": 0.60,
            "decision_threshold": 0.426,
            "computed_order_value": 1000.0,
            "would_flag_under_frozen_policy": True,
        },
        {
            "row_number": 3,
            "order_reference": "ORDER-2",
            "risk_score": 0.20,
            "decision_threshold": 0.426,
            "computed_order_value": 500.0,
            "would_flag_under_frozen_policy": False,
        },
    ]
    scored = prepare_batch_review_frame(results, input_frame, 100, 25)
    report = merchant_risk_report_html(
        scored,
        batch_summary={
            "total_rows": 2,
            "scored_rows": 2,
            "invalid_rows": 0,
            "would_flag_count": 1,
        },
        drift={
            "overall_severity": "warning",
            "features": {
                "Product_Price": {"psi": 0.2, "severity": "warning"},
            },
        },
        release_id="return-risk-test",
        reverse_logistics_cost=100,
        merchandise_loss_rate=25,
        generated_at_utc="2026-08-30 12:00 UTC",
    )

    assert "Merchant return-risk report" in report
    assert "Monitor only" in report
    assert "return-risk-test" in report
    assert "₹255" in report
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in report
    assert "<script>alert(1)</script>" not in report
