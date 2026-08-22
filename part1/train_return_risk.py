"""Part 1 Tasks 3-6 and 9 — train, tune, threshold and persist the return-risk model.

Pipeline of work:
  Task 3  leakage-free ColumnTransformer + Pipeline (fitted on train only)
  Task 4  DummyClassifier baseline
  Task 5  Logistic Regression + threshold sweep -> t*_logistic
  Task 6  RandomForest + GridSearchCV (5-fold StratifiedKFold, roc_auc)
  Task 9  persist the tuned RF pipeline, then re-run the sweep on the RF's OWN
          predict_proba to obtain t*_rf

    python3 -m part1.train_return_risk

Writes models/return_risk_model.pkl, models/return_risk_metadata.json,
reports/part1_model_report.md and the two threshold-sweep CSVs.
"""

import json

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from part1.common import (
    METADATA_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    RANDOM_STATE,
    best_threshold,
    build_preprocessor,
    ensure_dirs,
    load_dataset,
    metrics_at,
    split_data,
    sweep_thresholds,
)

REPORT_PATH = REPORTS_DIR / "part1_model_report.md"
LR_SWEEP_PATH = REPORTS_DIR / "part1_threshold_sweep_logreg.csv"
RF_SWEEP_PATH = REPORTS_DIR / "part1_threshold_sweep_rf.csv"


def class1_metrics(y_true, y_pred) -> tuple[float, float, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    return float(precision), float(recall), float(f1)


def sweep_to_markdown(sweep, every: int = 2) -> str:
    """Tabulate the sweep. `every=2` prints every other row (0.04 steps) to keep
    the report readable; the full 0.02-step grid is written to CSV alongside."""
    lines = ["| threshold | precision | recall | F1 | orders flagged |",
             "|---:|---:|---:|---:|---:|"]
    for _, r in sweep.iloc[::every].iterrows():
        lines.append(
            f"| {r['threshold']:.2f} | {r['precision']:.4f} | {r['recall']:.4f} "
            f"| {r['f1']:.4f} | {int(r['n_flagged'])} |"
        )
    return "\n".join(lines)


def main() -> None:
    ensure_dirs()
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_data(df)

    print(f"train={len(X_train)} rows  test={len(X_test)} rows")
    print(f"train return rate={y_train.mean():.4f}  test return rate={y_test.mean():.4f}")

    # ---------------------------------------------------------------- Task 4
    # Baseline. Fitted through the same preprocessing pipeline so the comparison
    # against the real models is apples-to-apples.
    dummy = Pipeline(
        [("prep", build_preprocessor()),
         ("clf", DummyClassifier(strategy="most_frequent"))]
    )
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    dummy_acc = accuracy_score(y_test, dummy_pred)
    dummy_p, dummy_r, dummy_f1 = class1_metrics(y_test, dummy_pred)
    print(f"\n[Task 4] Dummy  accuracy={dummy_acc:.4f}  class-1 F1={dummy_f1:.4f}")

    # ---------------------------------------------------------------- Task 5
    logreg = Pipeline(
        [("prep", build_preprocessor()),
         ("clf", LogisticRegression(
             class_weight="balanced", solver="lbfgs", max_iter=2000,
             random_state=RANDOM_STATE))]
    )
    logreg.fit(X_train, y_train)
    lr_proba = logreg.predict_proba(X_test)[:, 1]
    lr_pred_default = (lr_proba >= 0.5).astype(int)

    lr_acc = accuracy_score(y_test, lr_pred_default)
    lr_p, lr_r, lr_f1 = class1_metrics(y_test, lr_pred_default)
    lr_auc = roc_auc_score(y_test, lr_proba)
    print(f"[Task 5] LogReg @0.50  acc={lr_acc:.4f}  P={lr_p:.4f}  R={lr_r:.4f}  "
          f"F1={lr_f1:.4f}  ROC-AUC={lr_auc:.4f}")

    lr_sweep = sweep_thresholds(y_test, lr_proba)
    lr_sweep.to_csv(LR_SWEEP_PATH, index=False)
    lr_best = best_threshold(lr_sweep)
    t_star_logistic = float(lr_best["threshold"])
    recall_gain_pp = (float(lr_best["recall"]) - lr_r) * 100
    precision_drop_pp = (lr_p - float(lr_best["precision"])) * 100
    print(f"[Task 5] t*_logistic={t_star_logistic:.2f}  F1={lr_best['f1']:.4f}  "
          f"R={lr_best['recall']:.4f} (+{recall_gain_pp:.2f}pp)  "
          f"P={lr_best['precision']:.4f} (-{precision_drop_pp:.2f}pp)")

    # ---------------------------------------------------------------- Task 6
    rf_pipeline = Pipeline(
        [("prep", build_preprocessor()),
         ("clf", RandomForestClassifier(
             class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))]
    )
    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [6, 10, None],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        rf_pipeline, param_grid=param_grid, scoring="roc_auc", cv=cv, n_jobs=-1,
        refit=True, return_train_score=False,
    )
    print("\n[Task 6] running GridSearchCV (6 combos x 5 folds)...")
    search.fit(X_train, y_train)

    best_params = search.best_params_
    best_cv_auc = float(search.best_score_)
    best_model = search.best_estimator_          # preprocessing + tuned RF, one Pipeline

    rf_proba = best_model.predict_proba(X_test)[:, 1]
    rf_test_auc = float(roc_auc_score(y_test, rf_proba))
    rf_pred_default = (rf_proba >= 0.5).astype(int)
    rf_acc = accuracy_score(y_test, rf_pred_default)
    rf_p, rf_r, rf_f1 = class1_metrics(y_test, rf_pred_default)
    auc_gap = abs(best_cv_auc - rf_test_auc)
    print(f"[Task 6] best params={best_params}")
    print(f"[Task 6] best CV ROC-AUC={best_cv_auc:.4f}  test ROC-AUC={rf_test_auc:.4f}  "
          f"|gap|={auc_gap:.4f}")

    cv_table_rows = []
    for params, mean, std in zip(
        search.cv_results_["params"],
        search.cv_results_["mean_test_score"],
        search.cv_results_["std_test_score"],
    ):
        cv_table_rows.append(
            f"| {params['clf__n_estimators']} | {params['clf__max_depth']} "
            f"| {mean:.4f} | {std:.4f} |"
        )

    # ---------------------------------------------------------------- Task 9
    # Persist FIRST, then compute t*_rf from the artifact that was actually
    # saved, so the threshold provably belongs to the model Part 3 will load.
    joblib.dump(best_model, MODEL_PATH)
    print(f"\n[Task 9] saved {MODEL_PATH.relative_to(REPORTS_DIR.parent)}")

    saved_model = joblib.load(MODEL_PATH)
    saved_proba = saved_model.predict_proba(X_test)[:, 1]
    assert np.allclose(saved_proba, rf_proba), "reloaded model disagrees with fitted model"

    rf_sweep = sweep_thresholds(y_test, saved_proba)
    rf_sweep.to_csv(RF_SWEEP_PATH, index=False)
    rf_best = best_threshold(rf_sweep)
    t_star_rf = float(rf_best["threshold"])
    rf_default = metrics_at(y_test, saved_proba, 0.5)
    rf_recall_gain_pp = (float(rf_best["recall"]) - rf_default["recall"]) * 100
    rf_precision_drop_pp = (rf_default["precision"] - float(rf_best["precision"])) * 100
    print(f"[Task 9] t*_rf={t_star_rf:.2f}  F1={rf_best['f1']:.4f}  "
          f"P={rf_best['precision']:.4f}  R={rf_best['recall']:.4f}")

    metadata = {
        "threshold_rf": t_star_rf,
        "model_type": "RandomForestClassifier",
        "threshold_selection_metric": "F1",
        "threshold_selection_split": "held_out_test_split",
        "threshold_grid": {"start": 0.10, "stop": 0.90, "step": 0.02},
        "best_params": {k.replace("clf__", ""): v for k, v in best_params.items()},
        "best_cv_roc_auc": best_cv_auc,
        "test_roc_auc": rf_test_auc,
        "f1_at_threshold_rf": float(rf_best["f1"]),
        "precision_at_threshold_rf": float(rf_best["precision"]),
        "recall_at_threshold_rf": float(rf_best["recall"]),
        "risk_buckets": {
            "Low": "probability < threshold_rf",
            "Medium": "threshold_rf <= probability < threshold_rf + 0.15",
            "High": "probability >= threshold_rf + 0.15",
        },
        "medium_high_cut": round(t_star_rf + 0.15, 4),
        "class_names": {"0": "not_returned", "1": "returned"},
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[Task 9] saved {METADATA_PATH.relative_to(REPORTS_DIR.parent)}")

    # ------------------------------------------------------------------ report
    report = f"""# Part 1 — Model Report (Tasks 3-6, 9)

Regenerate with `python3 -m part1.train_return_risk`. Every number below is
printed by that run; nothing here is hand-written.

Split: stratified 80/20, `random_state=42`.
Train = {len(X_train)} rows (return rate {y_train.mean():.4f}),
Test = {len(X_test)} rows (return rate {y_test.mean():.4f}).

## Task 3 — Leakage-free preprocessing

A single `ColumnTransformer` inside a `Pipeline`:

* numeric ({len(["price_inr"])}+ columns) — `SimpleImputer(strategy="median")` then `StandardScaler()`
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
| accuracy | {dummy_acc:.4f} |
| precision (class 1) | {dummy_p:.4f} |
| recall (class 1) | {dummy_r:.4f} |
| **F1 (class 1)** | **{dummy_f1:.4f}** |

**Why {dummy_acc * 100:.2f}% accuracy is a misleading number here.** The dataset
returns only {df["returned"].mean() * 100:.2f}% of orders, so a model that predicts
"not returned" for every single row is right {dummy_acc * 100:.2f}% of the time
while never once identifying a return. The specific failure mode is
**high accuracy, zero recall**: recall for class 1 is {dummy_r:.4f} and F1 for
class 1 is {dummy_f1:.4f}. The business asked for the returns to be flagged, and
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
| accuracy | {lr_acc:.4f} |
| precision (class 1) | {lr_p:.4f} |
| recall (class 1) | {lr_r:.4f} |
| F1 (class 1) | {lr_f1:.4f} |
| **ROC-AUC** | **{lr_auc:.4f}** |

### Threshold sweep (0.10 -> 0.90, step 0.02)

Full grid: [`part1_threshold_sweep_logreg.csv`](part1_threshold_sweep_logreg.csv).
Every other row shown:

{sweep_to_markdown(lr_sweep)}

**F1-maximising threshold t\\*_logistic = {t_star_logistic:.2f}**

| | default 0.50 | t\\*_logistic = {t_star_logistic:.2f} | change |
|---|---:|---:|---:|
| precision | {lr_p:.4f} | {lr_best['precision']:.4f} | {-precision_drop_pp:+.2f} pp |
| recall | {lr_r:.4f} | {lr_best['recall']:.4f} | {recall_gain_pp:+.2f} pp |
| F1 | {lr_f1:.4f} | {lr_best['f1']:.4f} | {(float(lr_best['f1']) - lr_f1) * 100:+.2f} pp |

**The business trade-off this threshold change represents.** Moving the cut
point from 0.50 to {t_star_logistic:.2f} changes recall by
{recall_gain_pp:+.2f} percentage points and precision by
{-precision_drop_pp:+.2f} percentage points. Concretely, the two error types cost
different things. A **false negative** is an order that quietly gets returned
with no intervention: Flipkart eats the reverse-pickup logistics, the
restocking, the refund processing and a customer who is already unhappy — the
expensive error. A **false positive** is a support agent proactively contacting
a customer about an order that was never going to come back: the cost is one
agent-minute plus a small risk of annoying a satisfied customer — the cheap
error. Because the expensive error is the one we are trying to avoid, we
deliberately accept **more false positives** in exchange for **fewer false
negatives**. The ceiling on that trade is capacity: at threshold
{t_star_logistic:.2f} the model flags {int(lr_best['n_flagged'])} of the
{len(y_test)} test orders ({float(lr_best['n_flagged']) / len(y_test) * 100:.1f}%),
and flagging much more than that stops being a triage signal and starts being
a queue nobody can work through.

## Task 6 — Random Forest + GridSearchCV

Same preprocessing pipeline, `RandomForestClassifier(class_weight="balanced",
random_state=42)`, grid `n_estimators in [100, 200] x max_depth in [6, 10, None]`,
scored on `roc_auc` with 5-fold `StratifiedKFold`.

| n_estimators | max_depth | mean CV ROC-AUC | std |
|---:|---:|---:|---:|
{chr(10).join(cv_table_rows)}

| result | value |
|---|---|
| **best parameters** | `{best_params}` |
| **best cross-validated ROC-AUC** | **{best_cv_auc:.4f}** |
| **held-out test ROC-AUC** | **{rf_test_auc:.4f}** |
| absolute gap (CV vs test) | {auc_gap:.4f} |

The test ROC-AUC sits {auc_gap:.4f} away from the cross-validated estimate. A
small gap is the evidence against severe overfitting: the cross-validation
score was an honest forecast of unseen-data performance rather than an
optimistic one.

At the default 0.5 threshold this Random Forest scores accuracy {rf_acc:.4f},
precision {rf_p:.4f}, recall {rf_r:.4f}, F1 {rf_f1:.4f} for class 1.

## Task 9 — Saved artifact and t\\*_rf

`models/return_risk_model.pkl` holds `search.best_estimator_` — the *whole*
fitted `Pipeline` (ColumnTransformer + tuned RandomForest), not the Logistic
Regression and not a bare estimator. It was reloaded with `joblib.load` and its
`predict_proba` verified identical to the in-memory model before the threshold
below was computed.

The Task 5 sweep was then re-run **on the saved Random Forest's own
`predict_proba` output** over the same test split. Full grid:
[`part1_threshold_sweep_rf.csv`](part1_threshold_sweep_rf.csv).

{sweep_to_markdown(rf_sweep)}

| | default 0.50 | **t\\*_rf = {t_star_rf:.2f}** |
|---|---:|---:|
| precision | {rf_default['precision']:.4f} | {rf_best['precision']:.4f} |
| recall | {rf_default['recall']:.4f} | {rf_best['recall']:.4f} |
| F1 | {rf_default['f1']:.4f} | **{rf_best['f1']:.4f}** |

**t\\*_rf = {t_star_rf:.2f}**, persisted to `models/return_risk_metadata.json`.

Reusing t\\*_logistic = {t_star_logistic:.2f} here would have been a scale error:
the two models produce different probability distributions, so a cut point
tuned on one model's scores says nothing about where the other model's scores
separate. Part 3's risk buckets are anchored to t\\*_rf for exactly this reason.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nWrote {REPORT_PATH.relative_to(REPORTS_DIR.parent)}")


if __name__ == "__main__":
    main()
