import numpy as np

from return_risk.explainability import global_shap_importance, local_explanation


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
