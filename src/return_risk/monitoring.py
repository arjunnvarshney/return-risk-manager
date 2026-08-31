from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from return_risk.config import PROJECT_ROOT
from return_risk.policy import CostAssumptions, policy_cost_breakdown


class PredictionNotFoundError(LookupError):
    """Raised when an outcome references an unknown prediction."""


class OutcomeAlreadyRecordedError(ValueError):
    """Raised when an immutable outcome has already been recorded."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class ShadowMonitoringStore:
    """Privacy-minimized audit storage for shadow predictions and delayed outcomes."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @classmethod
    def from_environment(cls) -> ShadowMonitoringStore:
        configured = os.environ.get("RETURN_RISK_DB_PATH")
        path = Path(configured) if configured else PROJECT_ROOT / "runtime" / "shadow.db"
        return cls(path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    created_at_utc TEXT NOT NULL,
                    release_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    risk_score REAL NOT NULL CHECK (risk_score BETWEEN 0 AND 1),
                    decision_threshold REAL NOT NULL CHECK (decision_threshold BETWEEN 0 AND 1),
                    would_flag INTEGER NOT NULL CHECK (would_flag IN (0, 1)),
                    actual_action TEXT NOT NULL CHECK (actual_action = 'monitor_only'),
                    computed_order_value REAL NOT NULL CHECK (computed_order_value >= 0)
                );

                CREATE TABLE IF NOT EXISTS outcomes (
                    prediction_id TEXT PRIMARY KEY,
                    returned INTEGER NOT NULL CHECK (returned IN (0, 1)),
                    observed_return_cost REAL CHECK (observed_return_cost >= 0),
                    recorded_at_utc TEXT NOT NULL,
                    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
                );

                CREATE INDEX IF NOT EXISTS idx_predictions_created
                    ON predictions(created_at_utc DESC);
                """
            )

    def healthcheck(self) -> bool:
        with self._connect() as connection:
            return connection.execute("SELECT 1").fetchone()[0] == 1

    def record_prediction(
        self,
        *,
        release_id: str,
        source: str,
        risk_score: float,
        decision_threshold: float,
        would_flag: bool,
        computed_order_value: float,
    ) -> str:
        prediction_id = str(uuid4())
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO predictions (
                    prediction_id, created_at_utc, release_id, source, risk_score,
                    decision_threshold, would_flag, actual_action, computed_order_value
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'monitor_only', ?)
                """,
                (
                    prediction_id,
                    _utc_now(),
                    release_id,
                    source,
                    float(risk_score),
                    float(decision_threshold),
                    int(would_flag),
                    float(computed_order_value),
                ),
            )
        return prediction_id

    def record_outcome(
        self,
        prediction_id: str,
        *,
        returned: bool,
        observed_return_cost: float | None = None,
    ) -> dict:
        with self._connect() as connection:
            prediction = connection.execute(
                "SELECT prediction_id FROM predictions WHERE prediction_id = ?",
                (prediction_id,),
            ).fetchone()
            if prediction is None:
                raise PredictionNotFoundError("Prediction ID was not found.")
            existing = connection.execute(
                "SELECT prediction_id FROM outcomes WHERE prediction_id = ?",
                (prediction_id,),
            ).fetchone()
            if existing is not None:
                raise OutcomeAlreadyRecordedError(
                    "An outcome is already recorded for this prediction."
                )
            recorded_at = _utc_now()
            connection.execute(
                """
                INSERT INTO outcomes (
                    prediction_id, returned, observed_return_cost, recorded_at_utc
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    prediction_id,
                    int(returned),
                    observed_return_cost,
                    recorded_at,
                ),
            )
        return {
            "prediction_id": prediction_id,
            "returned": returned,
            "observed_return_cost": observed_return_cost,
            "recorded_at_utc": recorded_at,
            "status": "recorded",
        }

    def recent_predictions(self, limit: int = 50) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    p.prediction_id, p.created_at_utc, p.release_id, p.source,
                    p.risk_score, p.decision_threshold, p.would_flag,
                    p.actual_action, o.returned, o.observed_return_cost,
                    o.recorded_at_utc
                FROM predictions AS p
                LEFT JOIN outcomes AS o USING (prediction_id)
                ORDER BY p.created_at_utc DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                **dict(row),
                "would_flag": bool(row["would_flag"]),
                "returned": None if row["returned"] is None else bool(row["returned"]),
            }
            for row in rows
        ]

    def summary(self) -> dict:
        with self._connect() as connection:
            total_predictions = connection.execute(
                "SELECT COUNT(*) FROM predictions"
            ).fetchone()[0]
            would_flag_count = connection.execute(
                "SELECT COUNT(*) FROM predictions WHERE would_flag = 1"
            ).fetchone()[0]
            rows = connection.execute(
                """
                SELECT p.risk_score, p.decision_threshold, o.returned,
                       o.observed_return_cost
                FROM predictions AS p
                INNER JOIN outcomes AS o USING (prediction_id)
                ORDER BY p.created_at_utc
                """
            ).fetchall()

        completed = len(rows)
        result = {
            "total_predictions": total_predictions,
            "completed_outcomes": completed,
            "pending_outcomes": total_predictions - completed,
            "labeled_coverage": round(completed / total_predictions, 6)
            if total_predictions
            else 0.0,
            "would_flag_count": would_flag_count,
            "actual_interventions": 0,
            "observed_return_rate": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "confusion_matrix": {
                "true_negative": 0,
                "false_positive": 0,
                "false_negative": 0,
                "true_positive": 0,
            },
            "counterfactual_frozen_policy": None,
            "observed_return_cost_total": None,
            "safety_notice": (
                "Metrics describe completed shadow outcomes only. No customer-facing "
                "interventions were performed."
            ),
        }
        if not rows:
            return result

        scores = np.asarray([row["risk_score"] for row in rows], dtype=float)
        thresholds = np.asarray([row["decision_threshold"] for row in rows], dtype=float)
        labels = np.asarray([row["returned"] for row in rows], dtype=int)
        flagged = scores >= thresholds
        true_positive = int((flagged & (labels == 1)).sum())
        false_positive = int((flagged & (labels == 0)).sum())
        false_negative = int(((~flagged) & (labels == 1)).sum())
        true_negative = int(((~flagged) & (labels == 0)).sum())
        precision = true_positive / (true_positive + false_positive) if flagged.any() else None
        recall = true_positive / (true_positive + false_negative) if labels.any() else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else None
        )
        observed_costs = [
            float(row["observed_return_cost"])
            for row in rows
            if row["observed_return_cost"] is not None
        ]
        result.update(
            {
                "observed_return_rate": round(float(labels.mean()), 6),
                "precision": round(precision, 6) if precision is not None else None,
                "recall": round(recall, 6) if recall is not None else None,
                "f1": round(f1, 6) if f1 is not None else None,
                "confusion_matrix": {
                    "true_negative": true_negative,
                    "false_positive": false_positive,
                    "false_negative": false_negative,
                    "true_positive": true_positive,
                },
                "counterfactual_frozen_policy": policy_cost_breakdown(
                    labels,
                    flagged.astype(float),
                    0.5,
                    CostAssumptions(),
                ),
                "observed_return_cost_total": round(sum(observed_costs), 2)
                if observed_costs
                else None,
            }
        )
        return result
