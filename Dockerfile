FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    RETURN_RISK_PROJECT_ROOT=/app \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN groupadd --system app \
    && useradd --system --gid app --create-home app \
    && mkdir -p /runtime \
    && chown app:app /runtime

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN python -m pip install --no-cache-dir ".[runtime]"

# Copy only frozen runtime evidence. Raw datasets and experiment outputs are excluded.
COPY --chown=app:app demo/ ./demo/
COPY --chown=app:app .streamlit/ ./.streamlit/
COPY --chown=app:app models/return_risk_final.cbm ./models/return_risk_final.cbm
COPY --chown=app:app models/release_manifest.json ./models/release_manifest.json
COPY --chown=app:app models/operational_policy.json ./models/operational_policy.json
COPY --chown=app:app models/drift_reference.json ./models/drift_reference.json
COPY --chown=app:app models/policy_frontier.json ./models/policy_frontier.json
COPY --chown=app:app models/model_selection_summary.json ./models/model_selection_summary.json
COPY --chown=app:app reports/final_test_evaluation.json ./reports/final_test_evaluation.json

USER app

EXPOSE 8000 8501

CMD ["python", "-m", "uvicorn", "return_risk.api:app", "--host", "0.0.0.0", "--port", "8000"]
