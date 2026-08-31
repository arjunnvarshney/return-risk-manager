import json
import shutil
from pathlib import Path

from return_risk.config import PROJECT_ROOT
from return_risk.preflight import (
    container_hardening_checks,
    policy_safety_checks,
    release_integrity_checks,
    static_preflight_checks,
)


def test_repository_static_preflight_contract_passes() -> None:
    failures = [result for result in static_preflight_checks(PROJECT_ROOT) if not result.passed]
    assert failures == []


def test_release_check_detects_changed_model_bytes(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    shutil.copy(PROJECT_ROOT / "models" / "return_risk_final.cbm", models)
    shutil.copy(PROJECT_ROOT / "models" / "release_manifest.json", models)

    with (models / "return_risk_final.cbm").open("ab") as handle:
        handle.write(b"tampered")

    results = release_integrity_checks(tmp_path)
    assert any(
        result.name == "Frozen model hash matches manifest" and not result.passed
        for result in results
    )


def test_policy_check_rejects_customer_friction(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    manifest = {"release_id": "return-risk-test"}
    unsafe_policy = {
        "release_id": "return-risk-test",
        "deployment_mode": "shadow",
        "actual_action": "monitor_only",
        "would_flag_field_is_counterfactual": True,
        "customer_friction_allowed": True,
        "automatic_rejection_allowed": False,
        "return_rights_restriction_allowed": False,
    }
    (models / "release_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (models / "operational_policy.json").write_text(json.dumps(unsafe_policy), encoding="utf-8")

    result = policy_safety_checks(tmp_path)[0]
    assert not result.passed


def test_container_check_detects_missing_hardening(tmp_path: Path) -> None:
    (tmp_path / "compose.yaml").write_text("services: {}", encoding="utf-8")
    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim", encoding="utf-8")
    (tmp_path / ".dockerignore").write_text("data/", encoding="utf-8")

    results = container_hardening_checks(tmp_path)
    assert all(not result.passed for result in results)
