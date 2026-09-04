# Return Risk Manager

A defense-only return-risk scoring prototype for Razorpay Buildathon Track 02.
It estimates return probability from information available at checkout or
pre-shipment, then maps the score to a bounded verification policy.

Judge-facing resources: [submission brief](docs/SUBMISSION.md) and
[2½-minute demo script](docs/DEMO_SCRIPT.md).

## Safety and evaluation commitments

- Post-return fields are blocked from model input by code and tests.
- Raw customer, product, and order identifiers are not model features.
- Gender and age are excluded from production scoring and reserved for bias audits.
- Model and threshold selection use training/validation data only.
- The chronological test set was opened once, only after the release was frozen.
- The deployed demo is hard-locked to shadow monitoring: no verification friction,
  rejection, or return-rights restriction is permitted.

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Place `returns_sustainability_dataset.csv` in `data/raw/`. The raw CSV is intentionally
excluded from Git. Dataset source: <https://www.kaggle.com/datasets/sowmihari/returns-management>

## One-command Docker startup

After installing Docker Desktop, run this from the project folder:

```powershell
docker compose up --build
```

Then open:

- Dashboard: <http://127.0.0.1:8501>
- Interactive API: <http://127.0.0.1:8000/docs>
- API health: <http://127.0.0.1:8000/health>

Check service health with `docker compose ps`, follow logs with
`docker compose logs -f`, and stop the stack with `docker compose down`.
Both services use the same non-root, read-only runtime image. The image includes only the
frozen model, its release manifest, the enforced shadow-mode policy, the drift reference,
the aggregate validation policy frontier, the model-selection summary, and the final
evaluation summary. Raw datasets, processed splits, notebooks, tuning
artifacts, tests, and experimental reports are excluded from the Docker build context.
Predictions and delayed outcomes are stored in the named `shadow-monitoring-data` volume,
so the privacy-minimized audit history survives ordinary container restarts.

## First commands

```powershell
python scripts/prepare_data.py
python scripts/run_audit.py
python scripts/train_baseline.py
python scripts/train_catboost.py
python scripts/tune_catboost.py
python scripts/calibrate_catboost.py
python scripts/analyze_cost_sensitivity.py
python scripts/analyze_explainability_fairness.py
python scripts/run_feature_ablation.py
python scripts/analyze_threshold_concentration.py
python scripts/build_runtime_evidence.py
pytest
ruff check .
```

`train_baseline.py` reports validation metrics only. The guarded final evaluator refuses
to run again while its final report exists.

## Current validation benchmark

The untuned CatBoost benchmark is currently ahead of logistic regression on the locked
chronological validation split:

| Model | ROC-AUC | Average precision | Top-10% precision |
|---|---:|---:|---:|
| Logistic regression | 0.580 | 0.392 | 45.5% |
| Untuned CatBoost | 0.609 | 0.412 | 51.5% |

These are validation results, not final test claims. CatBoost used early stopping and kept
101 trees. The test partition remains unaccessed by the training scripts.

A bounded Optuna study then tested 25 configurations across three expanding chronological
folds. Its selected configuration reached 0.404 validation average precision, below the
untuned model's 0.412, and reduced top-10% precision from 51.5% to 45.5%. The untuned
CatBoost therefore remains the preferred model. This negative tuning result is retained as
evidence against validation overfitting; no further hyperparameter search is planned.

Sigmoid calibration was also trained from 1,771 chronological out-of-fold training scores.
It slightly worsened validation Brier score (0.2062 to 0.2067), log loss (0.6016 to 0.6030),
and expected calibration error (0.0407 to 0.0476). The calibrator is therefore retained only
as a rejected experiment; raw CatBoost probabilities remain the operational output.

Cost sensitivity on the initial full model confirmed that the operating policy must be
merchant-configurable. With
low assumed false-positive cost, threshold 0.300 flags 53.0% of orders, reaches 38.2%
precision and 65.8% recall, and estimates ₹5,744 savings per 1,000 orders (95% bootstrap
interval ₹3,223 to ₹8,438). The base threshold 0.438 flags 5.7%, reaches 57.9% precision and
10.6% recall, but its savings interval crosses zero (₹-446 to ₹1,786 per 1,000). Under high
false-positive cost, the cost-minimizing policy flags no orders. These are hypothetical
validation scenarios, not production claims.

Across 2,000 bootstrap resamples, validation ROC-AUC has a 95% interval of 0.571–0.648,
average precision 0.364–0.470, and top-10% precision 42.6%–61.4%. 

## Explainability and slice audit

Native TreeSHAP explanations reconstruct every promoted-model validation prediction with
zero numerical error. The leading global influences are product category (mean absolute
SHAP 0.210), discount applied (0.117), and order year (0.115). These describe model behavior, not
causal drivers of returns. The strong year contribution is also a temporal-drift warning
that was tested by ablation; year was retained but requires drift monitoring.

At the promoted threshold of 0.426, all 56 flagged validation orders are Clothing orders.
This produces a concentrated false-positive burden for Clothing and 0% for every other
category. That
concentration makes the base policy unsuitable for automatic adverse action; it should
remain a soft-verification or human-review signal only.

Age and gender remain excluded from model inputs and are used only for auditing. Observed
validation false-positive-rate gaps were 1.3 percentage points by gender and 2.7 points across age
bands. These validation-only gaps are descriptive, not evidence of causal discrimination.
Every location group had fewer than 20 validation orders, so location-level rates were
suppressed rather than reporting unstable comparisons.

## Feature-ablation decision

A controlled ablation compared the full model with variants excluding order year,
location, or both. The decision rule was fixed before comparison: each candidate could
lose at most 0.01 ROC-AUC and 0.01 average precision relative to the full model, with a
preference for removing temporal or proxy-risk features when that quality guardrail passed.

Removing `User_Location` passed: validation ROC-AUC was 0.606 instead of 0.609, average
precision was 0.413 instead of 0.412, and precision among the same top 57 orders improved
from 57.9% to 59.6%. This is the recommended deployment candidate because location-level
behavior cannot be reliably audited with the available sample. Removing `order_year`
missed the AUC guardrail (0.597), while removing both reduced AUC to 0.589 and average
precision to 0.395. Year therefore remains for now, with explicit drift monitoring.

All four variants still placed every top-57 flag in Clothing. Feature ablation did not
solve category concentration, so no version should automatically reject orders or change
return rights. This prompted a separate sweep of review capacity and lower,
merchant-configurable thresholds.

## Threshold and concentration decision

The no-location candidate was evaluated at every distinct validation score. Lowering the
threshold does not provide a defensible category-balancing fix. The base-cost optimum of
0.426 flags 5.6% of orders, with 58.9% precision, 10.6% recall, and hypothetical savings
of ₹744 per 1,000 orders; every flagged order is Clothing. At approximately 10% review
capacity, precision falls to 50.5%, estimated savings are nearly zero, and the queue is
still entirely Clothing.

The first non-Clothing order enters only after 27.7% of all orders are flagged. Clothing
still represents 99.6% of that queue and estimated savings fall to ₹-3,795 per 1,000.
No threshold met the illustrative combined guardrails of positive savings, at most 30%
review capacity, and at most 75% of flags from one category. Category quotas are not used,
because intentionally flagging lower-scored orders would create avoidable false positives.

Before final evaluation, threshold 0.426 was retained as a possible soft-review policy.
The held-out economic failure then activated a stricter operational override: every score
is now shadow-monitoring only, even when it exceeds that frozen threshold.

## Final held-out test result

Release `return-risk-635c518b5861` was evaluated once on 1,021 chronological test orders
dated 2024-12-15 through 2025-09-03. No model, feature, calibration, or threshold change
was made after viewing these results.

| Metric | Validation | Final test |
|---|---:|---:|
| ROC-AUC | 0.606 | 0.600 |
| Average precision | 0.413 | 0.356 |
| Precision at frozen threshold | 58.9% | 30.9% |
| Recall at frozen threshold | 10.6% | 5.8% |
| Estimated savings / 1,000 orders | ₹744 | ₹-1,543 |

Held-out uncertainty is substantial: ROC-AUC has a 95% bootstrap interval of
0.563–0.635, average precision 0.314–0.409, threshold precision 19.2%–43.8%, and recall
3.4%–8.7%. More importantly, estimated savings are negative across the full interval
(₹-2,644 to ₹-514 per 1,000 orders). The threshold flags 55 orders: 17 true returns and
38 false positives. Every flag is still Clothing.

This frozen threshold fails the economic test and must not be deployed for customer-facing
intervention. The artifact remains useful as a defense-only demonstration of leakage-safe
evaluation, explainable scoring, false-positive accounting, and an honest negative
held-out result. A real deployment requires merchant outcome data, observed intervention
effects, and a newly locked external evaluation set—not retuning on this test set.

## Run the working application

Install the API, demo, and test dependencies:

```powershell
python -m pip install -e ".[modeling,api,demo,dev]"
```

Start the FastAPI service:

```powershell
python -m uvicorn return_risk.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive API. Available endpoints are
`GET /health`, `GET /model-card`, `POST /v1/score`, `POST /v1/score/batch`,
`POST /v1/outcomes`, `GET /v1/monitoring/summary`, and
`GET /v1/monitoring/recent`.
Example single-order PowerShell request:

```powershell
$order = @{
  product_category = "Clothing"
  product_price = 1200
  order_quantity = 1
  discount_applied = 35
  shipping_method = "Express"
  payment_method = "Credit Card"
  order_date = "2025-08-15"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/v1/score" `
  -Method Post `
  -ContentType "application/json" `
  -Body $order
```

Start the judge-facing Streamlit demo in a second terminal:

```powershell
python -m streamlit run demo/app.py --server.headless true --server.port 8501
```

Then open `http://127.0.0.1:8501`. The application verifies the frozen model hash at
startup. A score above 0.426 is reported only as “would flag under the frozen policy”; the
actual action always remains `monitor_only`.

Single-order results deliberately separate decision support from operational permission.
An ordinary order displays **No review signal**, an in-range score above the frozen
threshold displays **Human-review candidate (simulation only)**, and an out-of-range order
displays **Abstain — reliability warning**. These labels help a reviewer understand the
model response, while the separately displayed permitted action remains **Monitor only**
for every order in this economically unsafe frozen release.

The **Why this score?** panel combines local SHAP direction with aggregate return-rate
context calculated only from the training and validation partitions. For example, it can
show a category's development return rate beside the 29.1% development baseline and its
sample size. Dates outside the 2022–2024 development period are described as time-drift
extrapolation rather than evidence that a future year causes returns. These comparisons
explain learned associations, not causal effects, and the held-out test contributes no
explanation aggregates.

### Shadow outcomes and live metrics

Every successful score is assigned an anonymous `prediction_id` and stored without the
raw order reference, customer attributes, or model input fields. After the real return
window closes, submit the outcome using the dashboard's **Live monitoring** tab or
`POST /v1/outcomes`. Outcomes are immutable: a duplicate for the same prediction is
rejected rather than silently overwriting audit history.

The monitoring summary calculates precision, recall, F1, confusion counts, and a
counterfactual cost estimate only from predictions whose outcomes have arrived. These are
live shadow metrics—not held-out test claims—and the number of actual customer
interventions remains zero. `observed_return_cost` is optional, post-outcome reporting
data and is never passed back into the frozen model.

### Batch risk review and drift monitoring

The dedicated **Batch risk review** tab can run a deterministic 40-order reviewer demonstration
with one click or accept up to 1,000 merchant orders from CSV. The built-in batch is
deliberately varied, so an elevated drift warning is expected and demonstrates the safety
guardrail. Every row is validated and the required upload columns are:

```text
product_category, product_price, order_quantity, discount_applied,
shipping_method, payment_method, order_date
```

`order_reference` is optional and is never used by the model. Drift monitoring compares
numeric and categorical distributions with a versioned train-plus-validation reference
using population stability index (PSI). Batches below 30 valid orders are labelled
insufficient for a drift decision. PSI thresholds are illustrative, and shadow mode stays
mandatory regardless of the drift result.

The operational safety decision is stored separately in `models/operational_policy.json`;
the development drift reference is stored in `models/drift_reference.json`. This preserves
the original frozen model while applying a stricter post-evaluation safety gate.

### Policy Lab and expected-loss prioritization

The dashboard opens with a judge-oriented 60-second overview: the loss class, held-out
evidence, false-positive count, operational decision, architecture flow, and recommended
live walkthrough are visible in one place. The single-order detector also includes
reproducible routine, elevated-risk, and reliability-warning presets so a live demo does
not depend on typing inputs correctly under time pressure. Batch risk review is promoted to its
own top-level tab rather than hidden inside outcome monitoring.

A layered light visual system separates prediction, monitoring, policy, and evidence with
blue, mint, violet, and slate accents. Each major feature includes a one-line purpose hint,
while individual order fields use compact tooltips so non-technical reviewers can follow
the workflow without turning the form into a wall of instructions.

The dashboard's **Policy Lab** changes hypothetical verification cost, false-positive
friction, missed-return loss, intervention effectiveness, and maximum review capacity. It
then selects the lowest-cost point from an aggregate validation-only threshold frontier.
The runtime artifact contains confusion counts rather than row-level validation data, and
the held-out test never participates in threshold reselection. The operational result
always remains `monitor_only`, regardless of a favorable validation scenario.

The same tab compares the logistic baseline, frozen CatBoost champion, tuned candidate,
and rejected calibration experiment using validation metrics. Batch scoring can also rank
orders by expected-loss exposure: predicted return probability multiplied by a transparent
merchant estimate of reverse-logistics and merchandise loss. This ranking remains a
counterfactual review aid and never triggers an order intervention.

Scored batches include risk-band and product-category distribution charts, invalid-row
details, and a prominent drift status. Single-order responses visibly warn when otherwise
valid numeric inputs or dates fall outside the development reference range; the score is
still returned for shadow monitoring, but its reliability is explicitly qualified.

After batch scoring, merchants can download both row-level shadow scores and a standalone
HTML risk report. The report summarizes expected-loss exposure, frozen-threshold review
candidates, risk and category distributions, largest drift indicators, assumptions, model
release, and the mandatory monitor-only notice. It opens offline and can be printed to PDF.

### Post-test v2 research (isolated from the release)

After the official test result was opened, a separate research script compared three
candidates using only three expanding forward-in-time folds inside the original training
partition. It never loads `test.csv`, and its artifact is not used by the API or dashboard.

| Research candidate | Mean CV average precision | Mean CV ROC-AUC |
|---|---:|---:|
| Original checkout features + CatBoost | **0.3813** | 0.5847 |
| Engineered checkout features + CatBoost | 0.3787 | **0.5897** |
| Engineered checkout features + logistic regression | 0.3443 | 0.5590 |

The engineered feature set added discounted unit price, absolute discount, value logs,
calendar indicators, and category interactions. It did not improve the selection metric:
average precision decreased by 0.0026 and varied more between time folds. The simpler v1
feature set therefore remains the research winner as well as the frozen official release.
This negative result is evidence against unnecessary complexity, not a reason to retune on
the known test set.

Reproduce the isolated experiment with:

```powershell
python scripts/train_v2_candidate.py
```

The generated `reports/v2_research_validation.json` labels the original validation period
as reused exploratory evidence. Promotion would require freezing a candidate first and
then collecting a newly untouched, chronologically later outcome set.

## Submission preflight

Before a demo, commit, or submission build, run one deterministic engineering gate from
the project folder:

```powershell
python scripts/preflight.py
```

The command verifies that all required evidence exists, recomputes the frozen model's
SHA-256 hash, checks the model and evaluation release IDs, enforces the checkout-time
feature allowlist and leakage blocklist, confirms the chronological test lock, and checks
that the negative held-out economics still map to the shadow-only operational policy. It
also inspects the non-root/read-only Docker contract, runs Ruff, and runs the complete
pytest suite. It does not load the held-out rows, retrain the model, start a server, or
start Docker.

The same gate runs automatically on every push and pull request through
`.github/workflows/ci.yml`. A submission candidate is ready only when every preflight
line reports `PASS` and the summary has no failures.

## Current stages

1. Data contract and leakage audit
2. Chronological train/validation/test split
3. Dummy and logistic-regression baselines
4. CatBoost challenger and probability calibration
5. Cost-weighted threshold selection
6. SHAP reason codes
7. FastAPI service and Streamlit demonstration

Install later-stage dependencies only when those stages begin:

```powershell
python -m pip install -e ".[modeling,api,demo,dev]"
```
