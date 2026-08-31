from __future__ import annotations

import json
from pathlib import Path

from return_risk.config import CATEGORICAL_FEATURES, MODEL_FEATURES
from return_risk.data import sha256_file


class ReleaseValidationError(ValueError):
    """Raised when the frozen model artifact and manifest disagree."""


def load_and_validate_release(manifest_path: Path, model_path: Path) -> dict:
    if not manifest_path.exists() or not model_path.exists():
        raise FileNotFoundError("Frozen release manifest or model artifact is missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "release_id",
        "model_filename",
        "model_sha256",
        "model_features",
        "categorical_features",
        "decision_threshold",
        "probability_output",
        "allowed_action",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ReleaseValidationError(f"Release manifest is missing fields: {missing}")
    if manifest["model_filename"] != model_path.name:
        raise ReleaseValidationError("Manifest model filename does not match the artifact.")
    if manifest["model_features"] != MODEL_FEATURES:
        raise ReleaseValidationError("Manifest features do not match the active allowlist.")
    if manifest["categorical_features"] != CATEGORICAL_FEATURES:
        raise ReleaseValidationError("Manifest categorical features do not match the allowlist.")
    if manifest["model_sha256"].lower() != sha256_file(model_path).lower():
        raise ReleaseValidationError("Model SHA-256 does not match the frozen manifest.")
    threshold = manifest["decision_threshold"]
    if not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        raise ReleaseValidationError("Decision threshold must be between zero and one.")
    if manifest["probability_output"] != "raw_catboost_probability":
        raise ReleaseValidationError("Only the validation-approved raw probability is allowed.")
    if manifest["allowed_action"] != "soft_verification_or_human_review":
        raise ReleaseValidationError("Release action must remain defense-only and non-blocking.")
    return manifest


def load_and_validate_operational_policy(path: Path, release_id: str) -> dict:
    if not path.exists():
        raise FileNotFoundError("Operational policy artifact is missing.")
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("release_id") != release_id:
        raise ReleaseValidationError("Operational policy targets a different release.")
    if policy.get("deployment_mode") != "shadow":
        raise ReleaseValidationError("This release must remain in shadow mode.")
    if policy.get("actual_action") != "monitor_only":
        raise ReleaseValidationError("Shadow mode permits monitoring only.")
    prohibited_flags = (
        "customer_friction_allowed",
        "automatic_rejection_allowed",
        "return_rights_restriction_allowed",
    )
    if any(policy.get(flag) is not False for flag in prohibited_flags):
        raise ReleaseValidationError("Operational policy enables a prohibited action.")
    return policy
