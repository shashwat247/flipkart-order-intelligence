"""Part 1 — reload the saved artifact and re-verify it end to end.

This is the script that proves the committed `.pkl` is usable after a fresh
clone: it loads only from disk, re-derives the same deterministic test split,
and re-checks every Part 1 acceptance criterion against real numbers.

    python3 -m part1.evaluate_return_risk

Exit code 0 = every check passed.
"""

import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from part1.common import (
    METADATA_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    best_threshold,
    ensure_dirs,
    load_dataset,
    metrics_at,
    split_data,
    sweep_thresholds,
)

REPORT_PATH = REPORTS_DIR / "part1_artifact_verification.md"


def check(results: list, name: str, passed: bool, detail: str) -> None:
    results.append({"check": name, "passed": bool(passed), "detail": detail})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")


def main() -> int:
    ensure_dirs()
    results: list = []

    df = load_dataset()
    check(results, "dataset rows", len(df) == 6000, f"{len(df)} rows (expected 6000)")
    check(results, "dataset columns", df.shape[1] == 13,
          f"{df.shape[1]} columns (expected 13)")

    return_rate = df["returned"].mean()
    check(results, "return rate in [0.18, 0.27]", 0.18 <= return_rate <= 0.27,
          f"{return_rate:.4f}")

    missing = df["rating_given"].isna().mean()
    check(results, "missing rating_given in [0.08, 0.18]", 0.08 <= missing <= 0.18,
          f"{missing:.4f}")

    # ---------------------------------------------------------- saved artifact
    model = joblib.load(MODEL_PATH)
    check(results, "saved model loads", True, f"{type(model).__name__} from {MODEL_PATH.name}")

    clf_name = type(model.named_steps["clf"]).__name__
    check(results, "saved model is the Random Forest pipeline",
          clf_name == "RandomForestClassifier",
          f"final estimator is {clf_name} (must NOT be LogisticRegression)")

    check(results, "saved model exposes predict/predict_proba",
          hasattr(model, "predict") and hasattr(model, "predict_proba"),
          "both present")

    steps = list(model.named_steps)
    check(results, "preprocessing bundled into the same Pipeline",
          "prep" in steps, f"steps = {steps}")

    metadata = json.loads(METADATA_PATH.read_text())
    t_rf = metadata["threshold_rf"]
    check(results, "threshold metadata exists and is numeric",
          isinstance(t_rf, (int, float)), f"threshold_rf = {t_rf}")

    # ------------------------------------------------------- re-measure on test
    X_train, X_test, y_train, y_test = split_data(df)
    proba = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, proba)

    check(results, "test ROC-AUC >= 0.58", test_auc >= 0.58, f"{test_auc:.4f}")
    cv_auc = metadata["best_cv_roc_auc"]
    check(results, "best CV ROC-AUC >= 0.58", cv_auc >= 0.58, f"{cv_auc:.4f}")
    gap = abs(cv_auc - test_auc)
    check(results, "|CV ROC-AUC - test ROC-AUC| <= 0.05 (no severe overfitting)",
          gap <= 0.05, f"gap = {gap:.4f}")

    # t*_rf must be reproducible from the SAVED model's own predict_proba.
    recomputed = float(best_threshold(sweep_thresholds(y_test, proba))["threshold"])
    check(results, "t*_rf reproducible from the saved model's own predict_proba",
          abs(recomputed - t_rf) < 1e-9,
          f"recomputed {recomputed:.2f} == stored {t_rf:.2f}")

    at_t = metrics_at(y_test, proba, t_rf)
    print(f"\nAt t*_rf = {t_rf:.2f}: precision={at_t['precision']:.4f} "
          f"recall={at_t['recall']:.4f} F1={at_t['f1']:.4f}")
    print(f"Accuracy at t*_rf: {accuracy_score(y_test, (proba >= t_rf).astype(int)):.4f}")
    print("\nFull classification report at t*_rf:")
    report_txt = classification_report(
        y_test, (proba >= t_rf).astype(int),
        target_names=["not_returned", "returned"], digits=4,
    )
    print(report_txt)

    # ---------------------------------------- a single order, the way Part 3 calls it
    sample = X_test.iloc[[0]]
    single_proba = float(model.predict_proba(sample)[0][1])
    check(results, "single-row prediction works (Part 3 calling convention)",
          0.0 <= single_proba <= 1.0, f"P(returned) = {single_proba:.4f}")

    n_pass = sum(r["passed"] for r in results)
    n_total = len(results)
    print(f"\n{n_pass}/{n_total} Part 1 checks passed")

    rows = "\n".join(
        f"| {'PASS' if r['passed'] else 'FAIL'} | {r['check']} | {r['detail']} |"
        for r in results
    )
    REPORT_PATH.write_text(
        f"""# Part 1 — Saved Artifact Verification

Regenerate with `python3 -m part1.evaluate_return_risk`. This script loads
`models/return_risk_model.pkl` and `models/return_risk_metadata.json` **from
disk only** and re-derives the deterministic test split, so it verifies the
committed artifacts rather than anything held in memory from training.

**{n_pass}/{n_total} checks passed.**

| result | check | detail |
|---|---|---|
{rows}

## Held-out classification report at t\\*_rf = {t_rf:.2f}

```
{report_txt}
```

Test ROC-AUC **{test_auc:.4f}** vs cross-validated **{cv_auc:.4f}** (gap {gap:.4f}).
""",
        encoding="utf-8",
    )
    print(f"Wrote {REPORT_PATH.relative_to(REPORTS_DIR.parent)}")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
