from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from pydantic import ValidationError

from return_risk.config import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    MODELS_DIR,
    REPORTS_DIR,
)
from return_risk.data import model_input_frame
from return_risk.drift import drift_report, load_drift_reference
from return_risk.explainability import contextual_reason, local_explanation
from return_risk.release import (
    load_and_validate_operational_policy,
    load_and_validate_release,
)
from return_risk.schemas import (
    BatchScoreResponse,
    BatchScoreRow,
    BatchValidationError,
    OrderRequest,
    ScoreResponse,
)

DEVELOPMENT_DATE_MIN = date(2022, 1, 1)
DEVELOPMENT_DATE_MAX = date(2024, 12, 14)
BATCH_REQUIRED_COLUMNS = {
    "product_category",
    "product_price",
    "order_quantity",
    "discount_applied",
    "shipping_method",
    "payment_method",
    "order_date",
}
BATCH_OPTIONAL_COLUMNS = {"order_reference"}
MAX_BATCH_ROWS = 1_000
INPUT_RANGE_LABELS = {
    "Product_Price": "Product price",
    "Order_Quantity": "Order quantity",
    "Discount_Applied": "Discount",
    "Order_Value": "Calculated order value",
}


class FrozenReturnRiskScorer:
    def __init__(self, model_path: Path, manifest_path: Path) -> None:
        self.manifest = load_and_validate_release(manifest_path, model_path)
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)
        if list(self.model.feature_names_) != MODEL_FEATURES:
            raise RuntimeError("Frozen model feature names do not match the active allowlist.")
        self.operational_policy = load_and_validate_operational_policy(
            MODELS_DIR / "operational_policy.json",
            self.manifest["release_id"],
        )
        self.drift_reference = load_drift_reference(MODELS_DIR / "drift_reference.json")

    @classmethod
    def from_project(cls) -> FrozenReturnRiskScorer:
        return cls(
            MODELS_DIR / "return_risk_final.cbm",
            MODELS_DIR / "release_manifest.json",
        )

    def _frame(self, order: OrderRequest) -> pd.DataFrame:
        raw = pd.DataFrame(
            [
                {
                    "Product_Category": order.product_category,
                    "Product_Price": order.product_price,
                    "Order_Quantity": order.order_quantity,
                    "Discount_Applied": order.discount_applied,
                    "Shipping_Method": order.shipping_method,
                    "Payment_Method": order.payment_method,
                    "Order_Date": order.order_date.isoformat(),
                    "Order_Value": order.order_value,
                }
            ]
        )
        return model_input_frame(raw)

    def _input_range_warnings(self, frame: pd.DataFrame) -> list[str]:
        warnings = []
        for feature, label in INPUT_RANGE_LABELS.items():
            value = float(frame.iloc[0][feature])
            reference = self.drift_reference["numeric"][feature]
            minimum = float(reference["minimum"])
            maximum = float(reference["maximum"])
            if value < minimum or value > maximum:
                warnings.append(
                    f"{label} ({value:,.2f}) is outside the development reference range "
                    f"({minimum:,.2f}–{maximum:,.2f}); prediction reliability may be lower."
                )
        return warnings

    def score_batch(self, input_frame: pd.DataFrame) -> BatchScoreResponse:
        if not 1 <= len(input_frame) <= MAX_BATCH_ROWS:
            raise ValueError(f"Batch must contain between 1 and {MAX_BATCH_ROWS} rows.")
        columns = set(input_frame.columns)
        missing = sorted(BATCH_REQUIRED_COLUMNS - columns)
        extra = sorted(columns - BATCH_REQUIRED_COLUMNS - BATCH_OPTIONAL_COLUMNS)
        if missing:
            raise ValueError(f"Batch CSV is missing required columns: {missing}")
        if extra:
            raise ValueError(f"Batch CSV contains unsupported columns: {extra}")

        orders = []
        errors = []
        row_numbers = []
        for position, (_, row) in enumerate(input_frame.iterrows(), start=2):
            payload = row.to_dict()
            if pd.isna(payload.get("order_reference")):
                payload["order_reference"] = None
            payload["order_date"] = str(payload["order_date"])
            try:
                orders.append(OrderRequest.model_validate(payload))
                row_numbers.append(position)
            except ValidationError as error:
                messages = [
                    f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
                    for item in error.errors()
                ]
                errors.append(BatchValidationError(row_number=position, errors=messages))

        if not orders:
            return BatchScoreResponse(
                release_id=self.manifest["release_id"],
                total_rows=len(input_frame),
                scored_rows=0,
                invalid_rows=len(errors),
                would_flag_count=0,
                drift={"status": "unavailable_no_valid_rows"},
                results=[],
                validation_errors=errors,
                safety_notice="Shadow mode enforced; no operational actions were taken.",
            )

        model_frame = pd.concat([self._frame(order) for order in orders], ignore_index=True)
        probabilities = self.model.predict_proba(model_frame)[:, 1]
        threshold = float(self.manifest["decision_threshold"])
        results = [
            BatchScoreRow(
                row_number=row_number,
                order_reference=order.order_reference,
                risk_score=round(float(probability), 6),
                decision_threshold=threshold,
                would_flag_under_frozen_policy=bool(probability >= threshold),
                computed_order_value=order.order_value,
            )
            for row_number, order, probability in zip(
                row_numbers,
                orders,
                probabilities,
                strict=True,
            )
        ]
        return BatchScoreResponse(
            release_id=self.manifest["release_id"],
            total_rows=len(input_frame),
            scored_rows=len(results),
            invalid_rows=len(errors),
            would_flag_count=sum(row.would_flag_under_frozen_policy for row in results),
            drift=drift_report(model_frame, self.drift_reference),
            results=results,
            validation_errors=errors,
            safety_notice=(
                "Shadow mode enforced: scores were recorded, but no customer-facing or "
                "operational actions were taken."
            ),
        )

    def score(self, order: OrderRequest) -> ScoreResponse:
        frame = self._frame(order)
        pool = Pool(frame, cat_features=CATEGORICAL_FEATURES)
        probability = float(self.model.predict_proba(pool)[0, 1])
        raw_prediction = float(self.model.predict(pool, prediction_type="RawFormulaVal")[0])
        shap_values = np.asarray(self.model.get_feature_importance(pool, type="ShapValues"))
        explanation = local_explanation(
            MODEL_FEATURES,
            frame.iloc[0].tolist(),
            shap_values[0, :-1],
            shap_values[0, -1],
            raw_prediction,
            probability,
            top_k=5,
        )
        for reason in explanation["top_reasons"]:
            reason["reason"] = contextual_reason(
                reason["feature"],
                frame.iloc[0][reason["feature"]],
                reason["direction"],
                self.drift_reference["target_context"],
            )
        threshold = float(self.manifest["decision_threshold"])
        flagged = probability >= threshold
        warnings = self._input_range_warnings(frame)
        if not DEVELOPMENT_DATE_MIN <= order.order_date <= DEVELOPMENT_DATE_MAX:
            warnings.append(
                "Order date is outside the model-development date range; drift risk is elevated."
            )
        has_reliability_warning = bool(warnings)
        warnings.append(
            "Held-out testing found negative estimated savings at the frozen threshold."
        )
        if has_reliability_warning:
            recommended_action = (
                "Abstain from review because input reliability is uncertain; inspect the "
                "order data and continue shadow monitoring."
            )
        elif flagged:
            recommended_action = (
                "Human-review candidate under the frozen threshold (simulation only); "
                "the permitted operational action remains monitor only."
            )
        else:
            recommended_action = (
                "No review signal under the frozen threshold; continue shadow monitoring."
            )
        return ScoreResponse(
            order_reference=order.order_reference,
            release_id=self.manifest["release_id"],
            risk_score=round(probability, 6),
            decision_threshold=threshold,
            would_flag_under_frozen_policy=flagged,
            deployment_mode="shadow",
            actual_action="monitor_only",
            risk_band="above_threshold" if flagged else "below_threshold",
            recommended_action=recommended_action,
            computed_order_value=order.order_value,
            reasons=explanation["top_reasons"],
            warnings=warnings,
            safety_notice=(
                "Shadow mode is enforced because held-out economics were negative. Never "
                "automatically reject an order, add friction, or restrict return rights."
            ),
        )

    def model_card(self) -> dict:
        report_path = REPORTS_DIR / "final_test_evaluation.json"
        if not report_path.exists():
            raise FileNotFoundError("Final held-out evaluation report is missing.")
        import json

        report = json.loads(report_path.read_text(encoding="utf-8"))
        return {
            "release_id": self.manifest["release_id"],
            "model_sha256": self.manifest["model_sha256"],
            "features": MODEL_FEATURES,
            "excluded_features": [
                "User_Location",
                "User_Age",
                "User_Gender",
                "post-return outcome columns",
            ],
            "held_out_metrics": {
                "orders": report["test_data"]["orders"],
                "prevalence": report["test_data"]["prevalence"],
                "roc_auc": report["probability_metrics"]["roc_auc"],
                "average_precision": report["probability_metrics"]["average_precision"],
                "precision": report["fixed_threshold_metrics"]["precision"],
                "recall": report["fixed_threshold_metrics"]["recall"],
                "f1": report["fixed_threshold_metrics"]["f1"],
                "confusion_matrix": report["fixed_threshold_metrics"]["confusion_matrix"],
                "orders_flagged": report["fixed_threshold_policy"]["point_estimate"][
                    "orders_flagged"
                ],
                "estimated_savings_per_1000": report["fixed_threshold_policy"][
                    "point_estimate"
                ]["savings_per_1000_orders"],
                "savings_per_1000_interval": report["fixed_threshold_policy"][
                    "bootstrap_intervals"
                ]["savings_per_1000_orders"],
                "largest_flagged_group": report["category_concentration"][
                    "largest_flagged_group"
                ],
                "largest_group_share": report["category_concentration"][
                    "largest_group_share"
                ],
            },
            "status": "demonstration_only_not_approved_for_customer_facing_intervention",
            "deployment_mode": "shadow",
            "actual_action": "monitor_only",
            "frozen_policy_allowed_action": self.manifest["allowed_action"],
            "operational_override_reason": self.operational_policy["reason"],
            "prohibited_actions": self.manifest["prohibited_actions"],
            "api_documentation": "/docs",
        }
