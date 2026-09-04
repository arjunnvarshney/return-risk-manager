from pathlib import Path

from streamlit.testing.v1 import AppTest

from return_risk.monitoring import ShadowMonitoringStore


def test_streamlit_percentage_columns_use_supported_format() -> None:
    demo_path = Path(__file__).resolve().parents[1] / "demo" / "app.py"
    source = demo_path.read_text(encoding="utf-8")
    assert 'format="%.1%%"' not in source
    assert source.count('format="percent"') == 1


def test_policy_lab_uses_interactive_chart_and_model_cards() -> None:
    demo_path = Path(__file__).resolve().parents[1] / "demo" / "app.py"
    source = demo_path.read_text(encoding="utf-8")
    assert "st.vega_lite_chart(" in source
    assert 'key="interactive_policy_frontier"' in source
    assert '"tooltip": tooltip' in source
    assert "HOVER OR TAP A DOT" in source
    assert "model_comparison_cards(model_selection[\"models\"])" in source
    assert "Why this model won:" in source


def test_streamlit_tab_contrast_targets_rendered_tab_roles() -> None:
    demo_path = Path(__file__).resolve().parents[1] / "demo" / "app.py"
    source = demo_path.read_text(encoding="utf-8")
    assert 'div[data-testid="stTabs"] [role="tab"]' in source
    assert 'div[data-testid="stTabs"] [role="tab"] p' in source
    assert 'div[data-testid="stTabs"] button {' not in source
    assert "min-height:42px" in source
    assert 'div[data-testid="stTabs"] [role="tablist"]' in source
    assert "border-bottom:1px solid rgba(128,153,194,.42)" in source
    assert ".react-aria-SelectionIndicator {display:none!important;}" in source
    assert "border-color:#6f99ed" in source


def test_streamlit_score_layout_uses_full_width_for_explanations() -> None:
    demo_path = Path(__file__).resolve().parents[1] / "demo" / "app.py"
    source = demo_path.read_text(encoding="utf-8")
    assert 'input_col, result_col = st.columns([1, 1], gap="large")' in source
    assert (
        ".rr-reason-list {display:grid; grid-template-columns:repeat(3,minmax(0,1fr))"
        in source
    )
    assert 'with st.expander("Technical evidence (optional)")' in source


def test_streamlit_demo_renders_without_exceptions(tmp_path, monkeypatch):
    monkeypatch.setenv("RETURN_RISK_DB_PATH", str(tmp_path / "dashboard-monitoring.db"))
    demo_path = Path(__file__).resolve().parents[1] / "demo" / "app.py"
    application = AppTest.from_file(demo_path).run(timeout=20)
    assert not application.exception
    assert any("Return Risk Manager" in item.value for item in application.markdown)
    assert any(
        "What shadow mode means" in item.value
        and "cannot block or delay an order" in item.value
        and "NO CUSTOMER ACTION" in item.value
        for item in application.markdown
    )
    assert len(application.tabs) == 6
    assert any(tab.label == "Judge overview" for tab in application.tabs)
    assert any(tab.label == "Batch risk review" for tab in application.tabs)
    assert any(tab.label == "Policy Lab" for tab in application.tabs)
    assert any(
        "underlying validation order rankings and confusion counts remain frozen"
        in item.value
        for item in application.caption
    )
    assert any(metric.label == "Permitted action" for metric in application.metric)
    assert any(
        "Understand the detector, its held-out evidence" in item.value
        for item in application.markdown
    )
    assert any(
        "System architecture" in item.value
        and "FastAPI /v1/score" in item.value
        and "No automatic retraining" in item.value
        and "Blocked from scoring" in item.value
        for item in application.markdown
    )
    assert any(
        "Estimate return probability before shipment" in item.value
        for item in application.markdown
    )
    assert any(
        "Live pilot metrics — separate from held-out evidence" in item.value
        for item in application.markdown
    )
    assert any(
        "rr-model-grid" in item.value
        and "Frozen CatBoost (no location)" in item.value
        and "Calibrated CatBoost" in item.value
        for item in application.markdown
    )
    assert any(
        "rr-evidence-verdict" in item.value
        and "useful risk signal, but not safe for customer action" in item.value
        and "detected 17 of 294 returns" in item.value
        for item in application.markdown
    )
    assert any(
        "rr-evidence-groups" in item.value
        and "Detection quality" in item.value
        and "Economic safety" in item.value
        for item in application.markdown
    )
    assert any(
        "Leakage-safe scoring boundary" in item.value
        and "Never used for prediction" in item.value
        for item in application.markdown
    )
    assert any(
        "Held-out test metrics—not validation results" in item.value
        for item in application.markdown
    )
    assert any(
        "rr-detail-grid" in item.value
        and "Held-out ROC-AUC" in item.value
        and "28.8% held-out return-rate" in item.value
        and "Economic safety warning" in item.value
        and "Coverage warning" in item.value
        for item in application.markdown
    )
    assert any(
        "Evaluation safeguards" in item.value
        and "Known limitations" in item.value
        and "Synthetic source data" in item.value
        for item in application.markdown
    )
    assert any(
        "Auditable frozen release" in item.value
        and "Permitted action: monitor only" in item.value
        for item in application.markdown
    )
    assert not any("rr-model-card" in item.value for item in application.code)
    assert any("0 OF 30 OUTCOMES" in item.value for item in application.markdown)
    assert any("Not measurable yet" in item.value for item in application.markdown)
    assert any(
        "Estimate financial exposure—not return probability" in item.value
        and "never change the model's risk score" in item.value
        for item in application.markdown
    )
    assert any(
        "01 · SET COSTS" in item.value
        and "02 · SCORE ORDERS" in item.value
        and "03 · REVIEW OUTPUT" in item.value
        for item in application.markdown
    )
    assert any(
        "rr-exposure-formula" in item.value
        and "Cost calculation:" in item.value
        and "₹100 reverse-logistics cost + 25%" in item.value
        for item in application.markdown
    )
    assert len(application.button) == 5

    product_price = next(
        widget for widget in application.number_input if widget.label == "Product price (₹)"
    )
    product_price.set_value(5000.0)
    submit = next(
        button for button in application.button if button.label == "Calculate return risk  →"
    )
    submit.click().run(timeout=20)
    assert not application.exception
    assert any(
        "rr-action-grid" in item.value
        and "Abstain — reliability warning" in item.value
        and "Monitor only" in item.value
        for item in application.markdown
    )
    assert any(
        "outside the development reference range" in warning.value
        for warning in application.warning
    )


def test_streamlit_demo_preset_populates_reproducible_order(tmp_path, monkeypatch):
    monkeypatch.setenv("RETURN_RISK_DB_PATH", str(tmp_path / "preset-monitoring.db"))
    demo_path = Path(__file__).resolve().parents[1] / "demo" / "app.py"
    application = AppTest.from_file(demo_path).run(timeout=20)

    preset = next(
        button for button in application.button if button.label == "Load elevated-risk order"
    )
    preset.click().run(timeout=20)

    assert not application.exception
    category = next(
        widget for widget in application.selectbox if widget.label == "Product category"
    )
    payment = next(
        widget for widget in application.selectbox if widget.label == "Payment method"
    )
    discount = next(widget for widget in application.slider if widget.label == "Discount (%)")
    assert category.value == "Clothing"
    assert payment.value == "COD"
    assert discount.value == 50.0

    submit = next(
        button for button in application.button if button.label == "Calculate return risk  →"
    )
    submit.click().run(timeout=20)
    assert any(
        "rr-action-grid" in item.value
        and "Human-review candidate" in item.value
        and "Monitor only" in item.value
        for item in application.markdown
    )
    assert any(
        "rr-reason-list" in item.value
        and "This order: Clothing" in item.value
        and "Similar orders returned more often than average in past data" in item.value
        for item in application.markdown
    )
    assert any(
        "This order scored 47.1%" in item.value
        and "The review line is" in item.value
        and "permitted action remains monitor only" in item.value
        for item in application.markdown
    )
    assert any(
        "Development orders with product category Clothing returned" in item.value
        and "does not prove causation" in item.value
        for item in application.markdown
    )


def test_streamlit_pilot_metrics_explain_undefined_values(tmp_path, monkeypatch):
    database_path = tmp_path / "pilot-monitoring.db"
    store = ShadowMonitoringStore(database_path)
    for index, returned in enumerate([True, False, True, False, False]):
        prediction_id = store.record_prediction(
            release_id="test-release",
            source="test",
            risk_score=0.20 + (index * 0.01),
            decision_threshold=0.50,
            would_flag=False,
            computed_order_value=500.0,
        )
        store.record_outcome(prediction_id, returned=returned)

    monkeypatch.setenv("RETURN_RISK_DB_PATH", str(database_path))
    demo_path = Path(__file__).resolve().parents[1] / "demo" / "app.py"
    application = AppTest.from_file(demo_path).run(timeout=20)

    assert not application.exception
    assert any("5 OF 30 OUTCOMES" in item.value for item in application.markdown)
    assert any(
        item.value.count("Not measurable yet") == 2
        and '>0.0%</strong>' in item.value
        and '>40.0%</strong>' in item.value
        for item in application.markdown
    )
    assert any(
        "flagged 0 of 5 completed orders" in item.value
        and "Precision becomes measurable" in item.value
        and "No customer intervention was performed" in item.value
        for item in application.markdown
    )


def test_streamlit_demo_batch_runs_without_upload(tmp_path, monkeypatch):
    monkeypatch.setenv("RETURN_RISK_DB_PATH", str(tmp_path / "batch-monitoring.db"))
    demo_path = Path(__file__).resolve().parents[1] / "demo" / "app.py"
    application = AppTest.from_file(demo_path).run(timeout=20)

    run_demo = next(
        button
        for button in application.button
        if button.label == "Run the 40-order reviewer demo"
    )
    run_demo.click().run(timeout=30)

    assert not application.exception
    assert any(
        metric.label == "Scored orders" and metric.value == "40"
        for metric in application.metric
    )
    assert any(
        button.label == "Download merchant risk report"
        for button in application.download_button
    )
