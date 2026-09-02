import asyncio

import httpx
import pytest

from return_risk.api import create_app
from return_risk.monitoring import ShadowMonitoringStore
from return_risk.scoring import FrozenReturnRiskScorer

VALID_ORDER = {
    "product_category": "Clothing",
    "product_price": 1200.0,
    "order_quantity": 1,
    "discount_applied": 35.0,
    "shipping_method": "Express",
    "payment_method": "Credit Card",
    "order_date": "2025-08-15",
}


@pytest.fixture
def app(tmp_path):
    return create_app(
        scorer=FrozenReturnRiskScorer.from_project(),
        store=ShadowMonitoringStore(tmp_path / "monitoring.db"),
    )


def request(app, method: str, path: str, **kwargs) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def test_health_verifies_frozen_release(app):
    response = request(app, "GET", "/health")
    assert response.status_code == 200
    assert response.json()["model_hash_verified"] is True
    assert response.json()["release_id"].startswith("return-risk-")
    assert response.json()["monitoring_storage"] == "ready"


def test_score_returns_safe_explainable_response(app):
    response = request(app, "POST", "/v1/score", json=VALID_ORDER)
    assert response.status_code == 200
    result = response.json()
    assert len(result["prediction_id"]) == 36
    assert 0 <= result["risk_score"] <= 1
    assert result["computed_order_value"] == 780.0
    assert result["would_flag_under_frozen_policy"] == (
        result["risk_score"] >= result["decision_threshold"]
    )
    assert result["deployment_mode"] == "shadow"
    assert result["actual_action"] == "monitor_only"
    assert 1 <= len(result["reasons"]) <= 5
    assert not {"User_Location", "User_Age", "User_Gender"}.intersection(
        reason["feature"] for reason in result["reasons"]
    )
    assert "Shadow mode is enforced" in result["safety_notice"]


def test_score_rejects_invalid_or_extra_inputs(app):
    invalid = {**VALID_ORDER, "discount_applied": 120, "user_location": "City1"}
    response = request(app, "POST", "/v1/score", json=invalid)
    assert response.status_code == 422


def test_score_warns_when_valid_inputs_are_outside_reference_ranges(app):
    unusual = {
        **VALID_ORDER,
        "product_price": 5000,
        "order_quantity": 10,
        "discount_applied": 80,
        "order_date": "2026-01-15",
    }
    response = request(app, "POST", "/v1/score", json=unusual)
    assert response.status_code == 200
    result = response.json()
    warnings = result["warnings"]
    assert any("Product price" in warning for warning in warnings)
    assert any("Order quantity" in warning for warning in warnings)
    assert any("Discount" in warning for warning in warnings)
    assert any("model-development date range" in warning for warning in warnings)
    assert result["recommended_action"].startswith("Abstain from review")
    assert result["actual_action"] == "monitor_only"


def test_elevated_order_is_a_simulated_human_review_candidate(app):
    elevated = {
        "product_category": "Clothing",
        "product_price": 1200,
        "order_quantity": 1,
        "discount_applied": 50,
        "shipping_method": "Standard",
        "payment_method": "COD",
        "order_date": "2024-12-01",
    }
    response = request(app, "POST", "/v1/score", json=elevated)
    assert response.status_code == 200
    result = response.json()
    assert result["would_flag_under_frozen_policy"] is True
    assert result["recommended_action"].startswith("Human-review candidate")
    assert result["actual_action"] == "monitor_only"


def test_model_card_discloses_negative_held_out_result(app):
    response = request(app, "GET", "/model-card")
    assert response.status_code == 200
    card = response.json()
    assert card["held_out_metrics"]["estimated_savings_per_1000"] < 0
    assert card["status"].startswith("demonstration_only")
    assert card["deployment_mode"] == "shadow"


def test_batch_csv_scores_valid_rows_and_reports_drift(app):
    header = (
        "order_reference,product_category,product_price,order_quantity,discount_applied,"
        "shipping_method,payment_method,order_date\n"
    )
    csv = (
        header
        + "ORD-1,Clothing,1200,1,35,Express,Credit Card,2025-08-15\n"
        + "ORD-2,Books,450,2,5,Standard,COD,2025-08-16\n"
    )
    response = request(
        app,
        "POST",
        "/v1/score/batch",
        files={"file": ("orders.csv", csv.encode(), "text/csv")},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["deployment_mode"] == "shadow"
    assert result["scored_rows"] == 2
    assert result["invalid_rows"] == 0
    assert result["drift"]["sample_status"] == "insufficient_below_30_orders"
    assert all(row["actual_action"] == "monitor_only" for row in result["results"])
    assert all(row["prediction_id"] for row in result["results"])
    assert request(app, "GET", "/v1/monitoring/summary").json()["total_predictions"] == 2


def test_batch_csv_rejects_schema_errors(app):
    csv = "product_category,product_price\nClothing,1000\n"
    response = request(
        app,
        "POST",
        "/v1/score/batch",
        files={"file": ("orders.csv", csv.encode(), "text/csv")},
    )
    assert response.status_code == 400
    assert "missing required columns" in response.json()["detail"]


def test_delayed_outcome_updates_live_metrics_without_taking_action(app):
    scored = request(app, "POST", "/v1/score", json=VALID_ORDER).json()
    outcome = request(
        app,
        "POST",
        "/v1/outcomes",
        json={
            "prediction_id": scored["prediction_id"],
            "returned": False,
        },
    )
    assert outcome.status_code == 201

    summary = request(app, "GET", "/v1/monitoring/summary").json()
    assert summary["total_predictions"] == 1
    assert summary["completed_outcomes"] == 1
    assert summary["actual_interventions"] == 0
    assert summary["confusion_matrix"]["false_positive"] == 1

    duplicate = request(
        app,
        "POST",
        "/v1/outcomes",
        json={"prediction_id": scored["prediction_id"], "returned": True},
    )
    assert duplicate.status_code == 409
