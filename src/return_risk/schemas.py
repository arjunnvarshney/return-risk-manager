from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


class OrderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    order_reference: str | None = Field(default=None, max_length=100)
    product_category: Literal[
        "Books",
        "Clothing",
        "Electronics",
        "Home Appliances",
        "Toys",
    ]
    product_price: float = Field(gt=0, le=1_000_000)
    order_quantity: int = Field(ge=1, le=100)
    discount_applied: float = Field(ge=0, le=100)
    shipping_method: Literal["Express", "Next-Day", "Standard"]
    payment_method: Literal["COD", "Credit Card", "Debit Card", "Wallet"]
    order_date: date

    @computed_field
    @property
    def order_value(self) -> float:
        return round(
            self.product_price * self.order_quantity * (1 - self.discount_applied / 100),
            2,
        )


class ReasonCode(BaseModel):
    feature: str
    value: str
    shap_log_odds: float
    direction: Literal["raised", "lowered"]
    reason: str


class ScoreResponse(BaseModel):
    prediction_id: str | None = None
    order_reference: str | None
    release_id: str
    risk_score: float
    decision_threshold: float
    would_flag_under_frozen_policy: bool
    deployment_mode: Literal["shadow"]
    actual_action: Literal["monitor_only"]
    risk_band: Literal["below_threshold", "above_threshold"]
    recommended_action: str
    computed_order_value: float
    reasons: list[ReasonCode]
    warnings: list[str]
    safety_notice: str


class BatchScoreRow(BaseModel):
    prediction_id: str | None = None
    row_number: int
    order_reference: str | None
    risk_score: float
    decision_threshold: float
    would_flag_under_frozen_policy: bool
    deployment_mode: Literal["shadow"] = "shadow"
    actual_action: Literal["monitor_only"] = "monitor_only"
    computed_order_value: float


class BatchValidationError(BaseModel):
    row_number: int
    errors: list[str]


class BatchScoreResponse(BaseModel):
    release_id: str
    deployment_mode: Literal["shadow"] = "shadow"
    total_rows: int
    scored_rows: int
    invalid_rows: int
    would_flag_count: int
    drift: dict[str, Any]
    results: list[BatchScoreRow]
    validation_errors: list[BatchValidationError]
    safety_notice: str


class OutcomeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: str = Field(pattern=r"^[0-9a-f]{8}-[0-9a-f-]{27}$")
    returned: bool
    observed_return_cost: float | None = Field(default=None, ge=0, le=1_000_000)


class OutcomeResponse(BaseModel):
    prediction_id: str
    returned: bool
    observed_return_cost: float | None
    recorded_at_utc: str
    status: Literal["recorded"]


class RecentPrediction(BaseModel):
    prediction_id: str
    created_at_utc: str
    release_id: str
    source: str
    risk_score: float
    decision_threshold: float
    would_flag: bool
    actual_action: Literal["monitor_only"]
    returned: bool | None
    observed_return_cost: float | None
    recorded_at_utc: str | None


class MonitoringSummary(BaseModel):
    total_predictions: int
    completed_outcomes: int
    pending_outcomes: int
    labeled_coverage: float
    would_flag_count: int
    actual_interventions: Literal[0]
    observed_return_rate: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    confusion_matrix: dict[str, int]
    counterfactual_frozen_policy: dict[str, Any] | None
    observed_return_cost_total: float | None
    safety_notice: str
