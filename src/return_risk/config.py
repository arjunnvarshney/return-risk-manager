import os
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get("RETURN_RISK_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve()
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "returns_sustainability_dataset.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

TARGET_COLUMN = "Return_Status"
POSITIVE_LABEL = "Returned"
DATE_COLUMN = "Order_Date"
ROW_ID_COLUMN = "Order_ID"

NUMERIC_FEATURES = [
    "Product_Price",
    "Order_Quantity",
    "Discount_Applied",
    "Order_Value",
    "order_year",
    "order_month",
    "order_day_of_week",
]

BENCHMARK_CATEGORICAL_FEATURES = [
    "Product_Category",
    "Shipping_Method",
    "Payment_Method",
    "User_Location",
]

CATEGORICAL_FEATURES = [
    "Product_Category",
    "Shipping_Method",
    "Payment_Method",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
BENCHMARK_MODEL_FEATURES = NUMERIC_FEATURES + BENCHMARK_CATEGORICAL_FEATURES

SENSITIVE_AUDIT_COLUMNS = ["User_Age", "User_Gender"]

POST_OUTCOME_COLUMNS = [
    "Return_Reason",
    "Days_to_Return",
    "Return_Cost",
    "Profit_Loss",
    "CO2_Emissions",
    "Packaging_Waste",
    "CO2_Saved",
    "Waste_Avoided",
]

RAW_IDENTIFIER_COLUMNS = ["Order_ID", "Product_ID", "User_ID"]
BLOCKED_MODEL_COLUMNS = set(POST_OUTCOME_COLUMNS + RAW_IDENTIFIER_COLUMNS + SENSITIVE_AUDIT_COLUMNS)

REQUIRED_RAW_COLUMNS = {
    TARGET_COLUMN,
    DATE_COLUMN,
    ROW_ID_COLUMN,
    *NUMERIC_FEATURES[:4],
    *BENCHMARK_CATEGORICAL_FEATURES,
    *POST_OUTCOME_COLUMNS,
    *RAW_IDENTIFIER_COLUMNS,
    *SENSITIVE_AUDIT_COLUMNS,
}

RANDOM_SEED = 42
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
TEST_FRACTION = 0.20
