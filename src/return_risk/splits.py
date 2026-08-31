from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from return_risk.config import (
    DATE_COLUMN,
    ROW_ID_COLUMN,
    TARGET_COLUMN,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
)
from return_risk.data import sha256_file


def _hash_ids(frame: pd.DataFrame) -> str:
    joined = "\n".join(frame[ROW_ID_COLUMN].astype(str).sort_values())
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def chronological_split(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if abs(TRAIN_FRACTION + VALIDATION_FRACTION + TEST_FRACTION - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to one")

    ordered = frame.assign(_parsed_date=pd.to_datetime(frame[DATE_COLUMN])).sort_values(
        ["_parsed_date", ROW_ID_COLUMN], kind="mergesort"
    )
    unique_dates = ordered["_parsed_date"].drop_duplicates().sort_values().to_numpy()
    train_date_index = max(1, int(len(unique_dates) * TRAIN_FRACTION))
    validation_date_index = max(
        train_date_index + 1,
        int(len(unique_dates) * (TRAIN_FRACTION + VALIDATION_FRACTION)),
    )
    train_end = unique_dates[train_date_index - 1]
    validation_end = unique_dates[min(validation_date_index - 1, len(unique_dates) - 1)]

    train = ordered.loc[ordered["_parsed_date"] <= train_end].drop(columns="_parsed_date")
    validation = ordered.loc[
        (ordered["_parsed_date"] > train_end) & (ordered["_parsed_date"] <= validation_end)
    ].drop(columns="_parsed_date")
    test = ordered.loc[ordered["_parsed_date"] > validation_end].drop(columns="_parsed_date")

    splits = {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
    if any(split.empty for split in splits.values()):
        raise ValueError("Chronological split produced an empty partition")
    return splits


def expanding_date_folds(
    frame: pd.DataFrame,
    n_splits: int = 3,
    initial_train_fraction: float = 0.40,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Create expanding folds without placing the same date in both sides."""
    if n_splits < 1:
        raise ValueError("n_splits must be positive")
    if not 0 < initial_train_fraction < 1:
        raise ValueError("initial_train_fraction must be between zero and one")

    working = frame.assign(_parsed_date=pd.to_datetime(frame[DATE_COLUMN]))
    unique_dates = np.sort(working["_parsed_date"].unique())
    initial_dates = max(1, int(len(unique_dates) * initial_train_fraction))
    remaining_dates = len(unique_dates) - initial_dates
    if remaining_dates < n_splits:
        raise ValueError("Not enough unique dates for the requested folds")

    fold_date_blocks = np.array_split(unique_dates[initial_dates:], n_splits)
    folds = []
    train_dates = unique_dates[:initial_dates]
    for validation_dates in fold_date_blocks:
        train = working.loc[working["_parsed_date"].isin(train_dates)].drop(
            columns="_parsed_date"
        )
        validation = working.loc[working["_parsed_date"].isin(validation_dates)].drop(
            columns="_parsed_date"
        )
        folds.append((train.reset_index(drop=True), validation.reset_index(drop=True)))
        train_dates = np.concatenate([train_dates, validation_dates])
    return folds


def write_splits(
    splits: dict[str, pd.DataFrame],
    output_dir: Path,
    source_path: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "strategy": "chronological_unique_date_60_20_20",
        "source_file": source_path.name,
        "source_sha256": sha256_file(source_path),
        "test_locked": True,
        "splits": {},
    }
    for name, split in splits.items():
        split.to_csv(output_dir / f"{name}.csv", index=False)
        dates = pd.to_datetime(split[DATE_COLUMN])
        manifest["splits"][name] = {
            "rows": int(len(split)),
            "date_min": dates.min().date().isoformat(),
            "date_max": dates.max().date().isoformat(),
            "return_rate": round(float((split[TARGET_COLUMN] == "Returned").mean()), 6),
            "order_id_sha256": _hash_ids(split),
        }

    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
