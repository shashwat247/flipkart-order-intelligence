# Part 1 — Model Report (Tasks 3-6, 9)

Regenerate with `python3 -m part1.train_return_risk`. Every number below is
printed by that run; nothing here is hand-written.

Split: stratified 80/20, `random_state=42`.
Train = 4800 rows (return rate 0.2275),
Test = 1200 rows (return rate 0.2275).

## Task 3 — Leakage-free preprocessing

A single `ColumnTransformer` inside a `Pipeline`:

* numeric (1+ columns) — `SimpleImputer(strategy="median")` then `StandardScaler()`
* categorical (`product_category`, `payment_method`) — `SimpleImputer(strategy="most_frequent")`
  then `OneHotEncoder(handle_unknown="ignore")`

`.fit()` is only ever called on `X_train`. Because the imputer medians/modes and
the scaler's mean/std are learned inside the pipeline, the test split is only
ever `.transform()`-ed — no test statistic can leak backwards into training.
`handle_unknown="ignore"` means an unseen category at inference time yields an
all-zero one-hot block instead of raising, which matters for Part 3 where the
agent feeds arbitrary order dictionaries into this same pipeline.

## Task 4 — Baseline (`DummyClassifier(strategy="most_frequent")`)

| metric | value |
|---|---:|
| accuracy | 0.7725 |
| precision (class 1) | 0.0000 |
| recall (class 1) | 0.0000 |
| **F1 (class 1)** | **0.0000** |

**Why 77.25% accuracy is a misleading number here.** The dataset
returns only 22.75% of orders, so a model that predicts
"not returned" for every single row is right 77.25% of the time
while never once identifying a return. The specific failure mode is
**high accuracy, zero recall**: recall for class 1 is 0.0000 and F1 for
class 1 is 0.0000. The business asked for the returns to be flagged, and
this model flags none of them — its accuracy is measuring how well it predicts
the majority class, which nobody needed predicted. This is why the project is
graded on class-1 recall/precision/F1 and ROC-AUC against this baseline rather
than on accuracy: **comparing against a baseline** and **using metrics aligned
to the real business problem** are the two honest-evaluation rules this task
is built on.

## Task 5 — Logistic Regression (`class_weight="balanced"`)

At the default 0.5 threshold, on the held-out test split:

| metric | value |
|---|---:|
| accuracy | 0.5917 |
| precision (class 1) | 0.2964 |
| recall (class 1) | 0.5788 |
| F1 (class 1) | 0.3921 |
| **ROC-AUC** | **0.6253** |

### Threshold sweep (0.10 -> 0.90, step 0.02)

Full grid: [`part1_threshold_sweep_logreg.csv`](part1_threshold_sweep_logreg.csv).
Every other row shown:

| threshold | precision | recall | F1 | orders flagged |
|---:|---:|---:|---:|---:|
| 0.10 | 0.2275 | 1.0000 | 0.3707 | 1200 |
| 0.14 | 0.2275 | 1.0000 | 0.3707 | 1200 |
| 0.18 | 0.2275 | 1.0000 | 0.3707 | 1200 |
| 0.22 | 0.2269 | 0.9963 | 0.3696 | 1199 |
| 0.26 | 0.2278 | 0.9853 | 0.3700 | 1181 |
| 0.30 | 0.2339 | 0.9597 | 0.3762 | 1120 |
| 0.34 | 0.2417 | 0.9121 | 0.3822 | 1030 |
| 0.38 | 0.2560 | 0.8645 | 0.3950 | 922 |
| 0.42 | 0.2692 | 0.7949 | 0.4022 | 806 |
| 0.46 | 0.2834 | 0.6923 | 0.4021 | 667 |
| 0.50 | 0.2964 | 0.5788 | 0.3921 | 533 |
| 0.54 | 0.3196 | 0.4542 | 0.3752 | 388 |
| 0.58 | 0.3345 | 0.3553 | 0.3446 | 290 |
| 0.62 | 0.3906 | 0.2747 | 0.3226 | 192 |
| 0.66 | 0.4159 | 0.1722 | 0.2435 | 113 |
| 0.70 | 0.3962 | 0.0769 | 0.1288 | 53 |
| 0.74 | 0.3846 | 0.0183 | 0.0350 | 13 |
| 0.78 | 0.2500 | 0.0037 | 0.0072 | 4 |
| 0.82 | 0.0000 | 0.0000 | 0.0000 | 2 |
| 0.86 | 0.0000 | 0.0000 | 0.0000 | 2 |
| 0.90 | 0.0000 | 0.0000 | 0.0000 | 0 |

**F1-maximising threshold t\*_logistic = 0.44**

| | default 0.50 | t\*_logistic = 0.44 | change |
|---|---:|---:|---:|
| precision | 0.2964 | 0.2801 | -1.63 pp |
| recall | 0.5788 | 0.7582 | +17.95 pp |
| F1 | 0.3921 | 0.4091 | +1.70 pp |

**The business trade-off this threshold change represents.** Moving the cut
point from 0.50 to 0.44 changes recall by
+17.95 percentage points and precision by
-1.63 percentage points. Concretely, the two error types cost
different things. A **false negative** is an order that quietly gets returned
with no intervention: Flipkart eats the reverse-pickup logistics, the
restocking, the refund processing and a customer who is already unhappy — the
expensive error. A **false positive** is a support agent proactively contacting
a customer about an order that was never going to come back: the cost is one
agent-minute plus a small risk of annoying a satisfied customer — the cheap
error. Because the expensive error is the one we are trying to avoid, we
deliberately accept **more false positives** in exchange for **fewer false
negatives**. The ceiling on that trade is capacity: at threshold
0.44 the model flags 739 of the
1200 test orders (61.6%),
and flagging much more than that stops being a triage signal and starts being
a queue nobody can work through.

## Task 6 — Random Forest + GridSearchCV

Same preprocessing pipeline, `RandomForestClassifier(class_weight="balanced",
random_state=42)`, grid `n_estimators in [100, 200] x max_depth in [6, 10, None]`,
scored on `roc_auc` with 5-fold `StratifiedKFold`.

| n_estimators | max_depth | mean CV ROC-AUC | std |
|---:|---:|---:|---:|
| 100 | 6 | 0.6186 | 0.0172 |
| 200 | 6 | 0.6193 | 0.0186 |
| 100 | 10 | 0.6059 | 0.0207 |
| 200 | 10 | 0.6071 | 0.0196 |
| 100 | None | 0.5954 | 0.0218 |
| 200 | None | 0.5930 | 0.0219 |

| result | value |
|---|---|
| **best parameters** | `{'clf__max_depth': 6, 'clf__n_estimators': 200}` |
| **best cross-validated ROC-AUC** | **0.6193** |
| **held-out test ROC-AUC** | **0.6203** |
| absolute gap (CV vs test) | 0.0011 |

The test ROC-AUC sits 0.0011 away from the cross-validated estimate. A
small gap is the evidence against severe overfitting: the cross-validation
score was an honest forecast of unseen-data performance rather than an
optimistic one.

At the default 0.5 threshold this Random Forest scores accuracy 0.6367,
precision 0.3240, recall 0.5495, F1 0.4076 for class 1.

## Task 9 — Saved artifact and t\*_rf

`models/return_risk_model.pkl` holds `search.best_estimator_` — the *whole*
fitted `Pipeline` (ColumnTransformer + tuned RandomForest), not the Logistic
Regression and not a bare estimator. It was reloaded with `joblib.load` and its
`predict_proba` verified identical to the in-memory model before the threshold
below was computed.

The Task 5 sweep was then re-run **on the saved Random Forest's own
`predict_proba` output** over the same test split. Full grid:
[`part1_threshold_sweep_rf.csv`](part1_threshold_sweep_rf.csv).

| threshold | precision | recall | F1 | orders flagged |
|---:|---:|---:|---:|---:|
| 0.10 | 0.2275 | 1.0000 | 0.3707 | 1200 |
| 0.14 | 0.2275 | 1.0000 | 0.3707 | 1200 |
| 0.18 | 0.2275 | 1.0000 | 0.3707 | 1200 |
| 0.22 | 0.2275 | 1.0000 | 0.3707 | 1200 |
| 0.26 | 0.2275 | 1.0000 | 0.3707 | 1200 |
| 0.30 | 0.2295 | 0.9963 | 0.3731 | 1185 |
| 0.34 | 0.2359 | 0.9487 | 0.3778 | 1098 |
| 0.38 | 0.2477 | 0.8828 | 0.3868 | 973 |
| 0.42 | 0.2706 | 0.8059 | 0.4052 | 813 |
| 0.46 | 0.2876 | 0.6520 | 0.3991 | 619 |
| 0.50 | 0.3240 | 0.5495 | 0.4076 | 463 |
| 0.54 | 0.3316 | 0.4579 | 0.3846 | 377 |
| 0.58 | 0.3333 | 0.3004 | 0.3160 | 246 |
| 0.62 | 0.3333 | 0.1026 | 0.1569 | 84 |
| 0.66 | 0.1818 | 0.0147 | 0.0271 | 22 |
| 0.70 | 0.0000 | 0.0000 | 0.0000 | 0 |
| 0.74 | 0.0000 | 0.0000 | 0.0000 | 0 |
| 0.78 | 0.0000 | 0.0000 | 0.0000 | 0 |
| 0.82 | 0.0000 | 0.0000 | 0.0000 | 0 |
| 0.86 | 0.0000 | 0.0000 | 0.0000 | 0 |
| 0.90 | 0.0000 | 0.0000 | 0.0000 | 0 |

| | default 0.50 | **t\*_rf = 0.50** |
|---|---:|---:|
| precision | 0.3240 | 0.3240 |
| recall | 0.5495 | 0.5495 |
| F1 | 0.4076 | **0.4076** |

**t\*_rf = 0.50**, persisted to `models/return_risk_metadata.json`.

Reusing t\*_logistic = 0.44 here would have been a scale error:
the two models produce different probability distributions, so a cut point
tuned on one model's scores says nothing about where the other model's scores
separate. Part 3's risk buckets are anchored to t\*_rf for exactly this reason.
