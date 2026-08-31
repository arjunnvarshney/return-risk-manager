from __future__ import annotations

from io import BytesIO
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

from return_risk.monitoring import (
    OutcomeAlreadyRecordedError,
    PredictionNotFoundError,
    ShadowMonitoringStore,
)
from return_risk.schemas import (
    BatchScoreResponse,
    MonitoringSummary,
    OrderRequest,
    OutcomeRequest,
    OutcomeResponse,
    RecentPrediction,
    ScoreResponse,
)
from return_risk.scoring import FrozenReturnRiskScorer

MAX_UPLOAD_BYTES = 2 * 1024 * 1024


def create_app(
    scorer: FrozenReturnRiskScorer | None = None,
    store: ShadowMonitoringStore | None = None,
) -> FastAPI:
    service = scorer or FrozenReturnRiskScorer.from_project()
    monitoring = store or ShadowMonitoringStore.from_environment()
    application = FastAPI(
        title="Return Risk Manager",
        version="1.1.0",
        description=(
            "Defense-only return-risk scoring with an enforced shadow-monitoring action gate."
        ),
    )

    @application.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "release_id": service.manifest["release_id"],
            "model_hash_verified": True,
            "deployment_mode": service.operational_policy["deployment_mode"],
            "actual_action": service.operational_policy["actual_action"],
            "monitoring_storage": "ready" if monitoring.healthcheck() else "unavailable",
        }

    @application.get("/model-card")
    def model_card() -> dict:
        return service.model_card()

    @application.post("/v1/score", response_model=ScoreResponse)
    def score(order: OrderRequest) -> ScoreResponse:
        result = service.score(order)
        prediction_id = monitoring.record_prediction(
            release_id=result.release_id,
            source="api_single",
            risk_score=result.risk_score,
            decision_threshold=result.decision_threshold,
            would_flag=result.would_flag_under_frozen_policy,
            computed_order_value=result.computed_order_value,
        )
        return result.model_copy(update={"prediction_id": prediction_id})

    @application.post("/v1/score/batch", response_model=BatchScoreResponse)
    async def score_batch(file: Annotated[UploadFile, File()]) -> BatchScoreResponse:
        if file.content_type not in {"text/csv", "application/csv", "application/vnd.ms-excel"}:
            raise HTTPException(status_code=415, detail="Upload must be a CSV file.")
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="CSV file exceeds the 2 MB limit.")
        try:
            frame = pd.read_csv(BytesIO(content))
            batch = service.score_batch(frame)
            logged_rows = []
            for row in batch.results:
                prediction_id = monitoring.record_prediction(
                    release_id=batch.release_id,
                    source="api_batch",
                    risk_score=row.risk_score,
                    decision_threshold=row.decision_threshold,
                    would_flag=row.would_flag_under_frozen_policy,
                    computed_order_value=row.computed_order_value,
                )
                logged_rows.append(row.model_copy(update={"prediction_id": prediction_id}))
            return batch.model_copy(update={"results": logged_rows})
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @application.post("/v1/outcomes", response_model=OutcomeResponse, status_code=201)
    def record_outcome(outcome: OutcomeRequest) -> dict:
        try:
            return monitoring.record_outcome(
                outcome.prediction_id,
                returned=outcome.returned,
                observed_return_cost=outcome.observed_return_cost,
            )
        except PredictionNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except OutcomeAlreadyRecordedError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @application.get("/v1/monitoring/summary", response_model=MonitoringSummary)
    def monitoring_summary() -> dict:
        return monitoring.summary()

    @application.get("/v1/monitoring/recent", response_model=list[RecentPrediction])
    def recent_predictions(limit: int = 50) -> list[dict]:
        if not 1 <= limit <= 200:
            raise HTTPException(status_code=400, detail="Limit must be between 1 and 200.")
        return monitoring.recent_predictions(limit)

    return application


app = create_app()
