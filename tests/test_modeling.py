from __future__ import annotations

from return_risk.modeling import untuned_catboost_model


def test_catboost_benchmark_is_reproducible_and_quiet() -> None:
    model = untuned_catboost_model()
    params = model.get_params()
    assert params["random_seed"] == 42
    assert params["allow_writing_files"] is False
    assert params["verbose"] is False

