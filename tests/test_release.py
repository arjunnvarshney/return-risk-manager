import json
from pathlib import Path

import pytest

from return_risk.config import CATEGORICAL_FEATURES, MODEL_FEATURES
from return_risk.data import sha256_file
from return_risk.release import (
    ReleaseValidationError,
    load_and_validate_operational_policy,
    load_and_validate_release,
)


def _write_manifest(path: Path, model_path: Path) -> None:
    manifest = {
        "release_id": "test-release",
        "model_filename": model_path.name,
        "model_sha256": sha256_file(model_path),
        "model_features": MODEL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "decision_threshold": 0.426,
        "probability_output": "raw_catboost_probability",
        "allowed_action": "soft_verification_or_human_review",
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_release_manifest_validates_hash_and_allowlist(tmp_path):
    model_path = tmp_path / "model.cbm"
    model_path.write_bytes(b"frozen-model")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, model_path)
    result = load_and_validate_release(manifest_path, model_path)
    assert result["decision_threshold"] == 0.426


def test_release_manifest_rejects_modified_model(tmp_path):
    model_path = tmp_path / "model.cbm"
    model_path.write_bytes(b"original")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, model_path)
    model_path.write_bytes(b"modified")
    with pytest.raises(ReleaseValidationError, match="SHA-256"):
        load_and_validate_release(manifest_path, model_path)


def test_operational_policy_enforces_shadow_monitoring(tmp_path):
    path = tmp_path / "policy.json"
    policy = {
        "release_id": "release-1",
        "deployment_mode": "shadow",
        "actual_action": "monitor_only",
        "customer_friction_allowed": False,
        "automatic_rejection_allowed": False,
        "return_rights_restriction_allowed": False,
    }
    path.write_text(json.dumps(policy), encoding="utf-8")
    assert load_and_validate_operational_policy(path, "release-1")["actual_action"] == (
        "monitor_only"
    )


def test_operational_policy_rejects_customer_friction(tmp_path):
    path = tmp_path / "policy.json"
    policy = {
        "release_id": "release-1",
        "deployment_mode": "shadow",
        "actual_action": "monitor_only",
        "customer_friction_allowed": True,
        "automatic_rejection_allowed": False,
        "return_rights_restriction_allowed": False,
    }
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ReleaseValidationError, match="prohibited"):
        load_and_validate_operational_policy(path, "release-1")
