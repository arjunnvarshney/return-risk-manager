from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_percentage_columns_use_supported_format() -> None:
    demo_path = Path(__file__).resolve().parents[1] / "demo" / "app.py"
    source = demo_path.read_text(encoding="utf-8")
    assert 'format="%.1%%"' not in source
    assert source.count('format="percent"') == 2


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
        "Estimate return probability before shipment" in item.value
        for item in application.markdown
    )
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
