from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime

from catboost import CatBoostClassifier

from return_risk.config import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    MODELS_DIR,
    REPORTS_DIR,
)
from return_risk.data import sha256_file
from return_risk.release import load_and_validate_release


def main() -> None:
    source_model = MODELS_DIR / "catboost_ablation_without_location.cbm"
    final_model = MODELS_DIR / "return_risk_final.cbm"
    manifest_path = MODELS_DIR / "release_manifest.json"
    ablation_path = REPORTS_DIR / "feature_ablation_validation.json"
    threshold_path = REPORTS_DIR / "threshold_concentration_validation.json"
    calibration_path = REPORTS_DIR / "catboost_calibration_validation.json"
    required_paths = [source_model, ablation_path, threshold_path, calibration_path]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze release; missing artifacts: {missing}")

    ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
    threshold_report = json.loads(threshold_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if ablation["recommended_variant"] != "without_location":
        raise RuntimeError("Ablation report does not recommend the no-location model.")
    if calibration["calibration_preferred"]:
        raise RuntimeError("Calibration was preferred; raw-probability release cannot be frozen.")

    threshold = float(threshold_report["milestones"]["cost_optimal"]["threshold"])
    shutil.copy2(source_model, final_model)
    model = CatBoostClassifier()
    model.load_model(final_model)
    model_hash = sha256_file(final_model)
    manifest = {
        "release_id": f"return-risk-{model_hash[:12]}",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "model_filename": final_model.name,
        "model_sha256": model_hash,
        "model_type": "CatBoostClassifier",
        "tree_count": int(model.tree_count_),
        "training_partition": "chronological_train_only_with_validation_early_stopping",
        "model_features": MODEL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "excluded_proxy_feature": "User_Location",
        "decision_threshold": threshold,
        "threshold_source": "validation_base_cost_optimum",
        "probability_output": "raw_catboost_probability",
        "allowed_action": "soft_verification_or_human_review",
        "prohibited_actions": [
            "automatic_order_rejection",
            "automatic_return_rights_restriction",
        ],
        "final_test_status_at_freeze": "unaccessed",
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    load_and_validate_release(manifest_path, final_model)
    print(json.dumps(manifest, indent=2))
    print(f"\nFrozen release manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
