"""Part 1 Task 8 — subgroup / root-cause analysis of the winning model.

Breaks the saved Random Forest's held-out precision and recall out by
`product_category` and by `payment_method`, finds where it underperforms its own
overall average, and proposes a concrete, quantified fix rather than
"collect more data".

    python3 -m part1.subgroup_analysis
"""

import json

import joblib
import numpy as np
import pandas as pd

from part1.common import (
    METADATA_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    THRESHOLD_GRID,
    ensure_dirs,
    load_dataset,
    metrics_at,
    split_data,
)

REPORT_PATH = REPORTS_DIR / "part1_subgroup_analysis.md"

# A subgroup is called out only if its recall is at least this far below the
# model's overall recall — fixed up front so the finding is not cherry-picked.
MATERIAL_GAP = 0.05


def subgroup_table(X_test, y_test, proba, threshold: float, by: str) -> pd.DataFrame:
    """Precision / recall / support for class 1, per level of one column."""
    pred = (proba >= threshold).astype(int)
    rows = []
    for level in sorted(X_test[by].unique()):
        mask = (X_test[by] == level).to_numpy()
        yt, yp = y_test.to_numpy()[mask], pred[mask]
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({
            "subgroup": level, "n_test": int(mask.sum()),
            "actual_returns": int(yt.sum()), "flagged": int(yp.sum()),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        })
    return pd.DataFrame(rows)


def best_local_threshold(X_test, y_test, proba, by: str, level: str):
    """F1-maximising threshold computed on this subgroup alone.

    Used only to quantify the size of the prize from a subgroup-specific cut
    point; the deployed model still ships one global t*_rf.
    """
    mask = (X_test[by] == level).to_numpy()
    yt, p = y_test.to_numpy()[mask], proba[mask]
    best = None
    for t in THRESHOLD_GRID:
        yp = (p >= t).astype(int)
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if best is None or f1 > best["f1"]:
            best = {"threshold": float(t), "precision": precision,
                    "recall": recall, "f1": f1, "flagged": int(yp.sum())}
    return best


def to_markdown(table: pd.DataFrame, label: str) -> str:
    lines = [
        f"| {label} | test orders | actual returns | flagged | TP | FP | FN | precision | recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| `{r['subgroup']}` | {int(r['n_test'])} | {int(r['actual_returns'])} "
            f"| {int(r['flagged'])} | {int(r['tp'])} | {int(r['fp'])} | {int(r['fn'])} "
            f"| {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    ensure_dirs()
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_data(df)

    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text())
    t_rf = float(metadata["threshold_rf"])
    proba = model.predict_proba(X_test)[:, 1]

    overall = metrics_at(y_test, proba, t_rf)
    print(f"Operating threshold t*_rf = {t_rf:.2f}")
    print(f"Overall  precision={overall['precision']:.4f}  recall={overall['recall']:.4f}"
          f"  F1={overall['f1']:.4f}")

    cat = subgroup_table(X_test, y_test, proba, t_rf, "product_category")
    pay = subgroup_table(X_test, y_test, proba, t_rf, "payment_method")
    cat.to_csv(REPORTS_DIR / "part1_subgroup_product_category.csv", index=False)
    pay.to_csv(REPORTS_DIR / "part1_subgroup_payment_method.csv", index=False)

    print("\nBy product_category:")
    print(cat[["subgroup", "n_test", "precision", "recall", "f1"]].to_string(index=False))
    print("\nBy payment_method:")
    print(pay[["subgroup", "n_test", "precision", "recall", "f1"]].to_string(index=False))

    # Weakest subgroup by recall, across BOTH breakdowns, requiring a
    # materially-sized gap and enough test rows for the number to mean anything.
    candidates = pd.concat([cat.assign(by="product_category"),
                            pay.assign(by="payment_method")], ignore_index=True)
    candidates = candidates[candidates["n_test"] >= 50]
    candidates["recall_gap"] = overall["recall"] - candidates["recall"]
    worst = candidates.sort_values("recall_gap", ascending=False).iloc[0]

    material = worst["recall_gap"] >= MATERIAL_GAP
    local = best_local_threshold(X_test, y_test, proba, worst["by"], worst["subgroup"])
    print(f"\nWeakest subgroup: {worst['by']}={worst['subgroup']}  "
          f"recall={worst['recall']:.4f} vs overall {overall['recall']:.4f} "
          f"(gap {worst['recall_gap'] * 100:.2f} pp)")
    print(f"Subgroup-local F1-max threshold={local['threshold']:.2f} -> "
          f"recall={local['recall']:.4f} precision={local['precision']:.4f} "
          f"F1={local['f1']:.4f}")

    extra_flagged = local["flagged"] - int(worst["flagged"])
    recovered_fn = int(round(local["recall"] * worst["actual_returns"])) - int(worst["tp"])

    report = f"""# Part 1 — Task 8: Subgroup / Root-cause Analysis

Regenerate with `python3 -m part1.subgroup_analysis`. Winning model =
`models/return_risk_model.pkl` (tuned Random Forest), evaluated on the held-out
test split at its own operating threshold **t\\*_rf = {t_rf:.2f}**.

**Overall at t\\*_rf:** precision {overall['precision']:.4f}, recall
{overall['recall']:.4f}, F1 {overall['f1']:.4f} over {len(y_test)} test orders.

## Broken out by `product_category`

{to_markdown(cat, "product_category")}

## Broken out by `payment_method`

{to_markdown(pay, "payment_method")}

## The subgroup where the model is meaningfully worse

**`{worst['by']} = {worst['subgroup']}`** — recall **{worst['recall']:.4f}**
against an overall recall of {overall['recall']:.4f}, a shortfall of
**{worst['recall_gap'] * 100:.2f} percentage points**{"" if material else " (below the "
f"{MATERIAL_GAP * 100:.0f} pp bar this analysis set for calling a gap material, so treat it as the "
"weakest subgroup rather than a serious defect)"}. In absolute terms the model
misses **{int(worst['fn'])} of the {int(worst['actual_returns'])} real returns**
in this subgroup ({int(worst['n_test'])} test orders), flagging only
{int(worst['flagged'])} orders at precision {worst['precision']:.4f}.

### Root cause

The forest ships a **single global cut point** but the subgroups do not share a
probability scale. `payment_method` is by far the strongest signal in the model
(Task 7 measured its permutation ROC-AUC drop at +0.098, roughly 10x the next
feature), and the generator gives COD orders a +0.9 log-odds bump. The forest
therefore parks most
COD orders above t\\*_rf and most non-COD orders below it, and inside the
non-COD population the remaining features are too weak to push a genuinely
risky order back over a threshold that was tuned on the *pooled* distribution.
The global threshold is calibrated to the mixture, so it systematically
under-flags whichever subgroup sits on the low side of the dominant feature.

### Concrete proposed fix: a subgroup-specific decision threshold

Not "collect more data" — the fix is to stop forcing one cut point onto
distributions the model itself separates. Replacing the global t\\*_rf with a
threshold fitted for `{worst['by']} = {worst['subgroup']}` alone:

| | global t\\*_rf = {t_rf:.2f} | subgroup-specific t = {local['threshold']:.2f} | change |
|---|---:|---:|---:|
| recall | {worst['recall']:.4f} | {local['recall']:.4f} | {(local['recall'] - worst['recall']) * 100:+.2f} pp |
| precision | {worst['precision']:.4f} | {local['precision']:.4f} | {(local['precision'] - worst['precision']) * 100:+.2f} pp |
| F1 | {worst['f1']:.4f} | {local['f1']:.4f} | {(local['f1'] - worst['f1']) * 100:+.2f} pp |
| orders flagged | {int(worst['flagged'])} | {int(local['flagged'])} | {extra_flagged:+d} |

Operationally: store a small `{{subgroup: threshold}}` map next to
`threshold_rf` in `models/return_risk_metadata.json` and have
`check_return_risk` select the cut point by the order's `{worst['by']}`. The cost
is {abs(extra_flagged)} {"more" if extra_flagged > 0 else "fewer"} orders into the support
queue from this subgroup; the benefit is roughly {max(recovered_fn, 0)} additional
genuine returns caught before they happen.

**Two caveats stated honestly.** First, {local['threshold']:.2f} was fitted on the
same held-out split it is reported on, so it is an optimistic estimate — in
production it should be refitted by cross-validation on the training split and
only then measured on test. Second, this is a calibration patch, not new signal:
it redistributes the precision/recall trade across subgroups but cannot raise
the model's overall ROC-AUC of {metadata['test_roc_auc']:.4f}. Genuinely lifting
performance for this subgroup needs a feature that discriminates *within* it —
a size/fit-mismatch flag, a returns-per-SKU history, or a seller-quality score —
which is a modelling change, not a threshold change.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nWrote {REPORT_PATH.relative_to(REPORTS_DIR.parent)}")


if __name__ == "__main__":
    main()
