import sqlite3

import pytest

from return_risk.monitoring import (
    OutcomeAlreadyRecordedError,
    PredictionNotFoundError,
    ShadowMonitoringStore,
)


def _prediction(store, score: float, source: str = "test") -> str:
    return store.record_prediction(
        release_id="test-release",
        source=source,
        risk_score=score,
        decision_threshold=0.5,
        would_flag=score >= 0.5,
        computed_order_value=100.0,
    )


def test_store_schema_omits_raw_order_and_customer_attributes(tmp_path):
    store = ShadowMonitoringStore(tmp_path / "monitoring.db")
    _prediction(store, 0.8)

    with sqlite3.connect(store.database_path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(predictions)").fetchall()
        }

    assert "order_reference" not in columns
    assert "user_id" not in columns
    assert "user_age" not in columns
    assert "user_gender" not in columns
    assert "user_location" not in columns
    assert "product_category" not in columns


def test_summary_calculates_completed_shadow_metrics(tmp_path):
    store = ShadowMonitoringStore(tmp_path / "monitoring.db")
    examples = [(0.8, True), (0.7, False), (0.4, True), (0.2, False)]
    for score, returned in examples:
        prediction_id = _prediction(store, score)
        store.record_outcome(prediction_id, returned=returned)

    summary = store.summary()
    assert summary["total_predictions"] == 4
    assert summary["completed_outcomes"] == 4
    assert summary["actual_interventions"] == 0
    assert summary["precision"] == 0.5
    assert summary["recall"] == 0.5
    assert summary["f1"] == 0.5
    assert summary["confusion_matrix"] == {
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_positive": 1,
    }


def test_outcomes_are_immutable_and_require_a_known_prediction(tmp_path):
    store = ShadowMonitoringStore(tmp_path / "monitoring.db")
    prediction_id = _prediction(store, 0.8)
    store.record_outcome(prediction_id, returned=True, observed_return_cost=150)

    with pytest.raises(OutcomeAlreadyRecordedError):
        store.record_outcome(prediction_id, returned=False)
    with pytest.raises(PredictionNotFoundError):
        store.record_outcome("00000000-0000-0000-0000-000000000000", returned=True)
