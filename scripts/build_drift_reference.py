from __future__ import annotations

import json

import pandas as pd

from return_risk.config import MODELS_DIR, PROCESSED_DIR
from return_risk.data import model_input_frame
from return_risk.drift import build_drift_reference


def main() -> None:
    train_path = PROCESSED_DIR / "train.csv"
    validation_path = PROCESSED_DIR / "validation.csv"
    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError("Prepared train and validation partitions are required.")
    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    development = pd.concat([train, validation], ignore_index=True)
    reference = build_drift_reference(model_input_frame(development))
    reference["source_partitions"] = ["chronological_train", "chronological_validation"]
    reference["test_set_accessed"] = False
    output_path = MODELS_DIR / "drift_reference.json"
    output_path.write_text(json.dumps(reference, indent=2), encoding="utf-8")
    print(json.dumps(reference, indent=2))
    print(f"\nSaved drift reference to {output_path}")


if __name__ == "__main__":
    main()
