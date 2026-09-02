from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from return_risk.config import BLOCKED_MODEL_COLUMNS, CATEGORICAL_FEATURES, MODEL_FEATURES

REQUIRED_SUBMISSION_ARTIFACTS = (
    "models/return_risk_final.cbm",
    "models/release_manifest.json",
    "models/operational_policy.json",
    "models/drift_reference.json",
    "models/policy_frontier.json",
    "models/model_selection_summary.json",
    "reports/final_test_evaluation.json",
    "reports/figures/final_test_evaluation.png",
    "reports/figures/global_shap_importance.png",
    "compose.yaml",
    "Dockerfile",
    ".dockerignore",
    "pyproject.toml",
    "README.md",
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def _check(name: str, condition: bool, success: str, failure: str) -> CheckResult:
    return CheckResult(name=name, passed=condition, detail=success if condition else failure)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_artifact_checks(project_root: Path) -> list[CheckResult]:
    missing = [
        item for item in REQUIRED_SUBMISSION_ARTIFACTS if not (project_root / item).is_file()
    ]
    return [
        _check(
            "Submission evidence is complete",
            not missing,
            f"Found all {len(REQUIRED_SUBMISSION_ARTIFACTS)} required files.",
            f"Missing: {', '.join(missing)}",
        )
    ]


def release_integrity_checks(project_root: Path) -> list[CheckResult]:
    manifest_path = project_root / "models" / "release_manifest.json"
    model_path = project_root / "models" / "return_risk_final.cbm"
    try:
        manifest = _load_json(manifest_path)
        actual_hash = sha256_path(model_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [CheckResult("Frozen release integrity", False, str(exc))]

    expected_hash = manifest.get("model_sha256")
    release_id = manifest.get("release_id")
    blocked = sorted(BLOCKED_MODEL_COLUMNS.intersection(manifest.get("model_features", [])))
    return [
        _check(
            "Frozen model hash matches manifest",
            actual_hash == expected_hash,
            f"SHA-256 {actual_hash[:12]}... matches.",
            f"Expected {expected_hash}; calculated {actual_hash}.",
        ),
        _check(
            "Release identifier is content-addressed",
            release_id == f"return-risk-{actual_hash[:12]}",
            f"Release {release_id} is tied to the model bytes.",
            f"Release ID {release_id!r} does not match model hash {actual_hash[:12]}.",
        ),
        _check(
            "Frozen feature contract matches code",
            manifest.get("model_features") == MODEL_FEATURES
            and manifest.get("categorical_features") == CATEGORICAL_FEATURES,
            f"Verified {len(MODEL_FEATURES)} checkout-time features.",
            "Manifest features differ from the reviewed code allowlist.",
        ),
        _check(
            "Leakage and identity columns are excluded",
            not blocked,
            "No post-return, identity, or sensitive audit fields enter the model.",
            f"Blocked model features found: {', '.join(blocked)}",
        ),
        _check(
            "Release was frozen before test access",
            manifest.get("final_test_status_at_freeze") == "unaccessed",
            "Manifest records the held-out test as unaccessed at freeze time.",
            "The manifest does not prove that the model preceded final-test access.",
        ),
    ]


def evaluation_evidence_checks(project_root: Path) -> list[CheckResult]:
    try:
        manifest = _load_json(project_root / "models" / "release_manifest.json")
        evaluation = _load_json(project_root / "reports" / "final_test_evaluation.json")
        split = _load_json(project_root / "data" / "processed" / "split_manifest.json")
        drift_reference = _load_json(project_root / "models" / "drift_reference.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [CheckResult("Held-out evaluation evidence", False, str(exc))]

    evaluated_release = evaluation.get("release", {})
    test_data = evaluation.get("test_data", {})
    split_test = split.get("splits", {}).get("test", {})
    fixed_policy = evaluation.get("fixed_threshold_policy", {})
    savings_interval = fixed_policy.get("bootstrap_intervals", {}).get(
        "savings_per_1000_orders", {}
    )
    same_partition = (
        test_data.get("orders") == split_test.get("rows")
        and test_data.get("date_min") == split_test.get("date_min")
        and test_data.get("date_max") == split_test.get("date_max")
    )
    development_rows = sum(
        int(split.get("splits", {}).get(partition, {}).get("rows", 0))
        for partition in ("train", "validation")
    )
    target_context = drift_reference.get("target_context", {})
    return [
        _check(
            "Chronological test split remains locked",
            split.get("test_locked") is True
            and split.get("strategy") == "chronological_unique_date_60_20_20",
            f"Locked test partition contains {split_test.get('rows')} later orders.",
            "Test lock or chronological split declaration is missing.",
        ),
        _check(
            "Evaluation points to the frozen release",
            evaluation.get("evaluation_number") == 1
            and evaluation.get("test_set_accessed") is True
            and evaluated_release.get("release_id") == manifest.get("release_id")
            and evaluated_release.get("model_sha256") == manifest.get("model_sha256"),
            "The single final evaluation references the immutable release.",
            "Final evaluation metadata does not match the frozen release.",
        ),
        _check(
            "Evaluation matches the declared test partition",
            same_partition,
            (
                f"Verified {test_data.get('orders')} orders from "
                f"{test_data.get('date_min')} to {test_data.get('date_max')}."
            ),
            "Evaluation row count or date boundaries differ from the split manifest.",
        ),
        _check(
            "False-positive economics include uncertainty",
            isinstance(savings_interval.get("upper"), (int, float))
            and savings_interval.get("upper") < 0,
            "The full 95% savings interval is negative, requiring shadow mode.",
            "A negative upper savings bound was not found in held-out evidence.",
        ),
        _check(
            "Explanation context excludes the held-out test",
            drift_reference.get("source_partitions")
            == ["chronological_train", "chronological_validation"]
            and drift_reference.get("test_set_accessed") is False
            and target_context.get("orders") == development_rows,
            f"Context contains {development_rows} training-plus-validation orders only.",
            "Explanation aggregates are missing or do not match the development partitions.",
        ),
    ]


def policy_safety_checks(project_root: Path) -> list[CheckResult]:
    try:
        manifest = _load_json(project_root / "models" / "release_manifest.json")
        policy = _load_json(project_root / "models" / "operational_policy.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [CheckResult("Defense-only operational policy", False, str(exc))]

    safe = (
        policy.get("release_id") == manifest.get("release_id")
        and policy.get("deployment_mode") == "shadow"
        and policy.get("actual_action") == "monitor_only"
        and policy.get("would_flag_field_is_counterfactual") is True
        and policy.get("customer_friction_allowed") is False
        and policy.get("automatic_rejection_allowed") is False
        and policy.get("return_rights_restriction_allowed") is False
    )
    return [
        _check(
            "Defense-only operational policy is enforced",
            safe,
            "Shadow monitoring only; blocking and return-right restrictions are disabled.",
            "Operational policy is missing a required monitor-only safety control.",
        )
    ]


def container_hardening_checks(project_root: Path) -> list[CheckResult]:
    try:
        compose = (project_root / "compose.yaml").read_text(encoding="utf-8").lower()
        dockerfile = (project_root / "Dockerfile").read_text(encoding="utf-8").lower()
        dockerignore = (project_root / ".dockerignore").read_text(encoding="utf-8").lower()
    except OSError as exc:
        return [CheckResult("Container safety contract", False, str(exc))]

    runtime_controls = (
        all(
            token in compose
            for token in (
                "read_only: true",
                "no-new-privileges:true",
                "tmpfs:",
                "healthcheck:",
                '"8000:8000"',
                '"8501:8501"',
            )
        )
        and compose.count("healthcheck:") >= 2
    )
    non_root = "user app" in dockerfile and "useradd --system" in dockerfile
    data_excluded = all(
        token in dockerignore for token in ("data/", "scripts/", "tests/", "models/*", "reports/*")
    )
    return [
        _check(
            "Containers use hardened runtime controls",
            runtime_controls,
            "Read-only filesystem, no-new-privileges, tmpfs, health checks, and fixed ports found.",
            "One or more required Compose hardening controls are missing.",
        ),
        _check(
            "Runtime image drops root privileges",
            non_root,
            "Docker image runs as the dedicated app user.",
            "Dockerfile does not prove a non-root runtime user.",
        ),
        _check(
            "Training and raw data are excluded from image context",
            data_excluded,
            "Raw data, scripts, tests, and non-release artifacts are excluded.",
            ".dockerignore is missing one or more sensitive-development exclusions.",
        ),
    ]


def static_preflight_checks(project_root: Path) -> list[CheckResult]:
    checks: list[CheckResult] = []
    checks.extend(required_artifact_checks(project_root))
    checks.extend(release_integrity_checks(project_root))
    checks.extend(evaluation_evidence_checks(project_root))
    checks.extend(policy_safety_checks(project_root))
    checks.extend(container_hardening_checks(project_root))
    return checks


def command_check(name: str, command: list[str], project_root: Path) -> CheckResult:
    completed = subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    if completed.returncode == 0:
        final_line = output.splitlines()[-1] if output else "Command completed successfully."
        return CheckResult(name, True, final_line)
    return CheckResult(name, False, output or f"Exited with code {completed.returncode}.")


def run_preflight(project_root: Path, *, run_quality: bool = True) -> list[CheckResult]:
    results = static_preflight_checks(project_root)
    if run_quality:
        results.append(
            command_check(
                "Ruff static analysis",
                [sys.executable, "-m", "ruff", "check", "."],
                project_root,
            )
        )
        results.append(
            command_check(
                "Complete pytest suite",
                [sys.executable, "-m", "pytest"],
                project_root,
            )
        )
    return results


def _print_results(results: list[CheckResult]) -> None:
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}")
        for line in result.detail.splitlines():
            print(f"       {line}")
    passed = sum(result.passed for result in results)
    print(f"\nPreflight summary: {passed}/{len(results)} checks passed.")


def main(argv: list[str] | None = None) -> int:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Verify the immutable submission release.")
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument(
        "--skip-quality",
        action="store_true",
        help="Skip Ruff and pytest (intended only for focused preflight unit tests).",
    )
    args = parser.parse_args(argv)
    results = run_preflight(args.project_root.resolve(), run_quality=not args.skip_quality)
    _print_results(results)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
