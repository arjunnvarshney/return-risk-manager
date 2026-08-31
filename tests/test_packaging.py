from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_docker_image_is_non_root_and_copies_only_runtime_artifacts():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "USER app" in dockerfile
    assert "COPY ." not in dockerfile
    assert "data/" not in dockerfile
    assert "models/return_risk_final.cbm" in dockerfile
    assert "models/release_manifest.json" in dockerfile
    assert "models/operational_policy.json" in dockerfile
    assert "models/drift_reference.json" in dockerfile
    assert "models/policy_frontier.json" in dockerfile
    assert "models/model_selection_summary.json" in dockerfile
    assert "reports/final_test_evaluation.json" in dockerfile


def test_docker_context_excludes_private_and_experimental_assets():
    exclusions = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")

    for entry in ("data/", "notebooks/", "scripts/", "tests/", "work/", "outputs/"):
        assert entry in exclusions


def test_compose_exposes_both_services_with_health_and_safety_controls():
    compose = (PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8")

    assert "  api:" in compose
    assert "  dashboard:" in compose
    assert "8000:8000" in compose
    assert "8501:8501" in compose
    assert compose.count("healthcheck:") == 2
    assert "read_only: true" in compose
    assert "no-new-privileges:true" in compose
    assert "RETURN_RISK_DB_PATH: /runtime/shadow.db" in compose
    assert "shadow-monitoring-data:/runtime" in compose
