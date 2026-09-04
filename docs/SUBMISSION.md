# Return Risk Manager — submission brief

## One-line pitch

Return Risk Manager scores an e-commerce order before shipment, explains the strongest
model influences, and prevents unsafe intervention when the measured cost of false
positives is greater than the expected benefit.

## Track and loss class

- Razorpay Buildathon — Track 02: AI Risk Manager
- Loss class: e-commerce product returns
- Decision point: checkout or pre-shipment
- Current operational action: shadow monitoring only

## Problem

Returns create reverse-logistics, merchandise and customer-support costs. A merchant needs
to know which orders deserve attention before shipping, but incorrectly flagging a good
customer also has a cost. A useful detector must therefore measure both prediction quality
and the economics of false positives.

## Solution

The system accepts checkout-time order details and returns:

- a return-risk probability;
- whether the score crosses the frozen review threshold;
- plain-language SHAP-based influences with development-data context;
- input reliability and drift warnings;
- an anonymous audit ID for delayed outcome feedback; and
- the action permitted by the release-wide safety policy.

The same frozen scoring contract is available through a FastAPI endpoint, a single-order
dashboard, and batch CSV review. The Policy Lab lets a merchant explore counterfactual
cost assumptions without changing the model, threshold, or safety lock.

## Leakage and privacy controls

Only information available before shipment enters scoring: price, quantity, discount,
order value, date-derived fields, product category, shipping method and payment method.
Return reason, days to return, return cost, profit/loss, sustainability outcomes,
identifiers, age, gender and location are excluded. A feature allowlist, a leakage
blocklist and automated tests enforce this boundary.

## Model development

The project compares logistic regression, CatBoost, an Optuna-tuned CatBoost candidate and
a calibrated CatBoost candidate on chronological validation data. The untuned, no-location
CatBoost model won on validation average precision while avoiding an unauditable location
proxy. The model and threshold were frozen before the held-out test was opened.

## Honest held-out result

The final chronological test contains 1,021 orders dated 15 December 2024 through
3 September 2025.

| Metric | Held-out result |
|---|---:|
| ROC-AUC | 0.600 |
| Average precision | 0.356 |
| Precision at frozen threshold | 30.9% |
| Recall at frozen threshold | 5.8% |
| True positives | 17 |
| False positives | 38 |
| False negatives | 277 |
| Estimated savings per 1,000 orders | ₹-1,543 |

The 95% bootstrap interval for estimated savings is ₹-2,644 to ₹-514 per 1,000 orders.
Because the full interval is negative, the application automatically remains in shadow
mode: it may score, explain and collect outcomes, but cannot block or delay orders, add
customer verification, or restrict return rights.

## What makes the project more than a notebook

- One immutable model release and SHA-256 verification at startup
- Chronological train, validation and one-time held-out test protocol
- Explicit false-positive friction in cost analysis
- Single-order and batch scoring through API and dashboard
- Plain-language explanations and out-of-distribution warnings
- Batch drift checks and expected-loss prioritization
- Delayed outcome capture with live pilot metrics
- Counterfactual Policy Lab isolated from official test evidence
- Non-root, read-only Docker services
- Deterministic preflight and automated test suite

## Limitations and responsible-use statement

The dataset is synthetic, intervention effectiveness is hypothetical, the ranking signal
is modest, and every frozen-threshold flag in the held-out test is a Clothing order. These
limitations rule out customer-facing deployment. Production promotion would require real
merchant outcomes, a measured intervention experiment, category-level guardrails and a
newly locked future evaluation set.

## Run locally

```powershell
docker compose up --build -d
```

- Dashboard: <http://127.0.0.1:8501>
- API documentation: <http://127.0.0.1:8000/docs>
- API health: <http://127.0.0.1:8000/health>

Run `docker compose down` after the demonstration.

