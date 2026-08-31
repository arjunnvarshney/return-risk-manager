from __future__ import annotations

import pandas as pd

from return_risk.splits import chronological_split, expanding_date_folds


def test_chronological_partitions_do_not_overlap_dates() -> None:
    frame = pd.DataFrame(
        {
            "Order_ID": [f"ORD{i}" for i in range(20)],
            "Order_Date": pd.date_range("2024-01-01", periods=20, freq="D").astype(str),
            "Return_Status": ["Returned", "Not Returned"] * 10,
        }
    )
    splits = chronological_split(frame)
    train_dates = pd.to_datetime(splits["train"]["Order_Date"])
    validation_dates = pd.to_datetime(splits["validation"]["Order_Date"])
    test_dates = pd.to_datetime(splits["test"]["Order_Date"])
    assert train_dates.max() < validation_dates.min()
    assert validation_dates.max() < test_dates.min()
    assert sum(len(split) for split in splits.values()) == len(frame)


def test_expanding_folds_preserve_time_and_expand_training() -> None:
    frame = pd.DataFrame(
        {
            "Order_ID": [f"ORD{i}" for i in range(30)],
            "Order_Date": pd.date_range("2024-01-01", periods=30, freq="D").astype(str),
            "Return_Status": ["Returned", "Not Returned"] * 15,
        }
    )
    folds = expanding_date_folds(frame, n_splits=3, initial_train_fraction=0.4)
    assert len(folds) == 3
    previous_train_size = 0
    for fold_train, fold_validation in folds:
        train_dates = pd.to_datetime(fold_train["Order_Date"])
        validation_dates = pd.to_datetime(fold_validation["Order_Date"])
        assert train_dates.max() < validation_dates.min()
        assert len(fold_train) > previous_train_size
        previous_train_size = len(fold_train)
