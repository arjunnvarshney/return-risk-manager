from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_demo_renders_without_exceptions(tmp_path, monkeypatch):
    monkeypatch.setenv("RETURN_RISK_DB_PATH", str(tmp_path / "dashboard-monitoring.db"))
    demo_path = Path(__file__).resolve().parents[1] / "demo" / "app.py"
    application = AppTest.from_file(demo_path).run(timeout=20)
    assert not application.exception
    assert any("Return Risk Manager" in item.value for item in application.markdown)
    assert len(application.tabs) == 6
    assert any(tab.label == "Judge overview" for tab in application.tabs)
    assert any(tab.label == "Batch review" for tab in application.tabs)
    assert any(tab.label == "Policy Lab" for tab in application.tabs)
    assert any(metric.label == "Permitted action" for metric in application.metric)
    assert any(
        "Understand the detector, its held-out evidence" in item.value
        for item in application.markdown
    )
    assert any(
        "Estimate return probability before shipment" in item.value
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
        metric.label == "Safe action" and metric.value == "Monitor only"
        for metric in application.metric
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


def test_streamlit_demo_batch_runs_without_upload(tmp_path, monkeypatch):
    monkeypatch.setenv("RETURN_RISK_DB_PATH", str(tmp_path / "batch-monitoring.db"))
    demo_path = Path(__file__).resolve().parents[1] / "demo" / "app.py"
    application = AppTest.from_file(demo_path).run(timeout=20)

    run_demo = next(
        button for button in application.button if button.label == "Run 40-order demo batch"
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
