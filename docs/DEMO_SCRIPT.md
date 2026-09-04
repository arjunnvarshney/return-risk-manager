# Judge demo script — about 2 minutes 30 seconds

## 0:00–0:20 — problem and promise

Open **Judge overview**.

> Returns eat merchant margin, but incorrectly challenging a legitimate customer also
> creates loss. Return Risk Manager predicts return risk before shipment, explains the
> score, and allows action only when measured economics say it is safe.

Point to the three-step Input → Detect → Respond strip and the shadow-mode explanation.

## 0:20–0:45 — evaluation discipline

Stay on **Judge overview** and point to the held-out evidence and architecture.

> We use only checkout-time fields. Post-return outcomes, return costs, identifiers, age,
> gender and location never enter scoring. We used chronological splits, selected the model
> and threshold on validation, froze the release, and then opened the 1,021-order test once.

## 0:45–1:20 — working detector

Open **Try the detector**, click **Load elevated-risk order**, then **Calculate return risk**.

> The result is a probability, not a verdict about a customer. The panel compares it with
> the frozen review line, shows input reliability, and explains the largest model influences
> using historical development-data context. SHAP describes model behavior; it does not
> claim that a feature causes returns.

Point to **Decision support** versus **Permitted operational action**.

> Even a score above the review line cannot trigger customer action because the release-wide
> safety policy overrides the order-level signal.

## 1:20–1:45 — batch workflow

Open **Batch risk review** and click **Run the 40-order reviewer demo**.

> A merchant can score a CSV, inspect invalid rows and data drift, and prioritize orders by
> expected loss exposure. This ranking combines model risk with transparent merchant cost
> assumptions; it does not change the underlying probability. Results and an offline HTML
> report can be downloaded for audit.

## 1:45–2:05 — false-positive economics

Open **Policy Lab** and change one cost assumption.

> The interactive curve shows how review rate, precision, recall, false positives and
> estimated savings change under validation-only assumptions. It is deliberately separated
> from the frozen held-out test and cannot unlock intervention.

## 2:05–2:30 — honest result and close

Open **Model evidence**.

> On the held-out test, ROC-AUC was 0.600, precision was 30.9%, and recall was 5.8%: 17
> returns caught, 38 good orders falsely flagged, and 277 returns missed. Estimated savings
> were negative at ₹1,543 lost per 1,000 orders, with the entire 95% interval below zero.
> So the product does the responsible thing: it falls back to shadow monitoring. The result
> is not hidden—the safety decision is part of the working system.

Finish with:

> The core contribution is a complete risk-management loop: detect, explain, measure harm,
> monitor outcomes, and refuse unsafe action.

## Likely judge questions

**Why is the model weak?**  
The source is a small synthetic dataset with limited pre-shipment signal. The project reports
that limitation instead of inflating performance or using post-return leakage.

**Why not tune further after seeing the test result?**  
That would overfit the held-out test. The release remains frozen; future improvement requires
a newly untouched, chronologically later dataset.

**Why use CatBoost?**  
It handles mixed numeric and categorical checkout data directly. It beat logistic regression
on validation average precision. Optuna tuning and probability calibration were both tested
and rejected when they did not improve the predeclared criteria.

**Why keep the system if it cannot intervene?**  
Shadow mode is the correct first production stage: it collects real outcomes, measures drift
and validates economics without imposing customer harm.

**What would unlock human review?**  
Real merchant outcomes, measured review effectiveness and friction costs, acceptable category
concentration, positive economics on a newly frozen future test, and an explicit policy release.

