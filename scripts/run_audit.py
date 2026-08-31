from __future__ import annotations

import json

import numpy as np
import pandas as pd

from return_risk.config import (
    POST_OUTCOME_COLUMNS,
    RAW_DATA_PATH,
    REPORTS_DIR,
    TARGET_COLUMN,
)
from return_risk.data import load_raw_data, sha256_file


def main() -> None:
    frame = load_raw_data(RAW_DATA_PATH)
    target = (frame[TARGET_COLUMN] == "Returned").astype(int)
    dates = pd.to_datetime(frame["Order_Date"])

    group_rates = {}
    for column in ["Product_Category", "Shipping_Method", "Payment_Method", "User_Gender"]:
        values = frame.assign(_target=target).groupby(column)["_target"].agg(["count", "mean"])
        group_rates[column] = {
            str(index): {
                "orders": int(row["count"]),
                "return_rate": round(float(row["mean"]), 6),
            }
            for index, row in values.iterrows()
        }

    leakage_agreement = {}
    rules = {
        "Return_Reason": frame["Return_Reason"] != "No Return",
        "Days_to_Return": frame["Days_to_Return"] > 0,
        "Return_Cost": frame["Return_Cost"] > 0,
        "Profit_Loss": ~np.isclose(frame["Profit_Loss"], frame["Order_Value"]),
        "CO2_Saved": frame["CO2_Saved"] == 0,
        "Waste_Avoided": frame["Waste_Avoided"] == 0,
    }
    for column, inferred_return in rules.items():
        leakage_agreement[column] = round(float((inferred_return.astype(int) == target).mean()), 6)

    product_category_counts = frame.groupby("Product_ID")["Product_Category"].nunique()
    report = {
        "source": {
            "file": RAW_DATA_PATH.name,
            "sha256": sha256_file(RAW_DATA_PATH),
        },
        "shape": {"rows": int(len(frame)), "columns": int(frame.shape[1])},
        "target": {
            "returned": int(target.sum()),
            "not_returned": int((1 - target).sum()),
            "return_rate": round(float(target.mean()), 6),
        },
        "quality": {
            "duplicate_rows": int(frame.duplicated().sum()),
            "missing_cells": int(frame.isna().sum().sum()),
            "date_min": dates.min().date().isoformat(),
            "date_max": dates.max().date().isoformat(),
            "unique_order_ids": int(frame["Order_ID"].nunique()),
            "unique_user_ids": int(frame["User_ID"].nunique()),
            "unique_product_ids": int(frame["Product_ID"].nunique()),
            "products_assigned_multiple_categories": int((product_category_counts > 1).sum()),
        },
        "blocked_post_outcome_columns": POST_OUTCOME_COLUMNS,
        "target_agreement_of_obvious_leakage_rules": leakage_agreement,
        "group_return_rates": group_rates,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / "data_audit.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved audit report to {output_path}")


if __name__ == "__main__":
    main()

