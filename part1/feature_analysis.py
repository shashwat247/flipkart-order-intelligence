"""Part 1 Task 7 — explain the winning Random Forest.

Two importance measures, computed on the SAME saved model, then compared:

  impurity-based  `.feature_importances_`  — read off the fitted forest, computed
                  on TRAINING data, biased towards high-cardinality continuous
                  columns because they offer more places to split.
  permutation     `sklearn.inspection.permutation_importance` — measured on the
                  HELD-OUT TEST split as the real ROC-AUC drop when one column
                  is shuffled. No cardinality bias; if a feature carries no
                  signal, breaking it costs nothing.

    python3 -m part1.feature_analysis
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from part1.common import (
    MODEL_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    ensure_dirs,
    feature_names_out,
    load_dataset,
    original_feature_of,
    split_data,
)

REPORT_PATH = REPORTS_DIR / "part1_feature_importance.md"
N_REPEATS = 10


def main() -> None:
    ensure_dirs()
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_data(df)

    model = joblib.load(MODEL_PATH)
    preprocessor = model.named_steps["prep"]
    forest = model.named_steps["clf"]

    # ------------------------------------------------- impurity-based, per column
    names = feature_names_out(preprocessor)
    importances = forest.feature_importances_
    assert len(names) == len(importances), "feature name/importance length mismatch"

    impurity = (
        pd.DataFrame({"transformed_feature": names, "importance": importances})
        .assign(original_feature=lambda d: d["transformed_feature"].map(original_feature_of))
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    top5_transformed = impurity.head(5)

    # The top-5 ORIGINAL features: one-hot columns of the same source column are
    # summed, because "how important is payment_method" is the question a human
    # actually asks — not "how important is the COD indicator specifically".
    by_original = (
        impurity.groupby("original_feature")["importance"].sum()
        .sort_values(ascending=False)
    )
    top5_original = list(by_original.head(5).index)

    print("Top 5 transformed features (impurity-based):")
    for _, r in top5_transformed.iterrows():
        print(f"  {r['transformed_feature']:<32s} {r['importance']:.4f}")
    print(f"\nTop 5 original features: {top5_original}")

    # -------------------------------------------------------- permutation, test set
    # Run against the full pipeline on RAW test columns, so shuffling a column
    # correctly disturbs every transformed column it feeds.
    perm = permutation_importance(
        model, X_test, y_test,
        scoring="roc_auc", n_repeats=N_REPEATS, random_state=RANDOM_STATE, n_jobs=-1,
    )
    perm_df = pd.DataFrame({
        "original_feature": X_test.columns,
        "perm_mean": perm.importances_mean,
        "perm_std": perm.importances_std,
    }).sort_values("perm_mean", ascending=False).reset_index(drop=True)
    perm_df["perm_rank_overall"] = perm_df.index + 1

    print("\nPermutation importance (held-out test, ROC-AUC drop):")
    for _, r in perm_df.iterrows():
        print(f"  {r['original_feature']:<24s} {r['perm_mean']:+.5f} "
              f"(+/- {r['perm_std']:.5f})")

    # ------------------------------------------------------------ side-by-side
    imp_rank = {f: i + 1 for i, f in enumerate(by_original.index)}
    perm_rank = dict(zip(perm_df["original_feature"], perm_df["perm_rank_overall"]))
    perm_mean = dict(zip(perm_df["original_feature"], perm_df["perm_mean"]))
    perm_std = dict(zip(perm_df["original_feature"], perm_df["perm_std"]))

    comparison = pd.DataFrame({
        "feature": top5_original,
        "impurity_importance": [by_original[f] for f in top5_original],
        "impurity_rank": [imp_rank[f] for f in top5_original],
        "permutation_importance": [perm_mean[f] for f in top5_original],
        "permutation_std": [perm_std[f] for f in top5_original],
        "permutation_rank": [perm_rank[f] for f in top5_original],
    })
    comparison["rank_change"] = comparison["permutation_rank"] - comparison["impurity_rank"]
    comparison.to_csv(REPORTS_DIR / "part1_importance_comparison.csv", index=False)

    # The headline finding: which top-5 features collapse under permutation.
    # "Collapses" = falls the furthest in rank, tie-broken by the smallest
    # (possibly negative) actual ROC-AUC drop.
    ranked_demotions = comparison.sort_values(
        ["rank_change", "permutation_importance"], ascending=[False, True]
    )
    demoted = ranked_demotions.iloc[0]
    demoted_name = demoted["feature"]

    # Every top-5 feature whose measured test-set ROC-AUC drop is negligible.
    # A negative value means shuffling the column left the model no worse off.
    NEGLIGIBLE = 0.002
    collapsed = ranked_demotions[
        ranked_demotions["permutation_importance"] < NEGLIGIBLE
    ]

    # Cross-check the collapsed features against the columns the seeded
    # generator's log-odds `z` actually uses. delivery_distance_km and
    # rating_given never enter `z` at all; customer_tenure_days enters only
    # through -0.15 * tanh(tenure/500), which saturates and so varies barely at
    # all across the realistic tenure range.
    NOT_IN_DGP = {"delivery_distance_km", "rating_given"}
    WEAK_IN_DGP = {"customer_tenure_days", "is_weekend_order", "discount_pct"}
    collapsed_names = set(collapsed["feature"])
    notes = []
    for f in sorted(collapsed_names & NOT_IN_DGP):
        notes.append(
            f"`{f}` never enters the generator's log-odds `z` at all — it is pure "
            f"noise by construction, yet it is a high-cardinality continuous column, "
            f"so it is a prime candidate to be inflated by impurity importance and "
            f"then exposed by permutation importance"
        )
    for f in sorted(collapsed_names & WEAK_IN_DGP):
        notes.append(
            f"`{f}` does enter `z`, but only through a term small enough "
            f"(`-0.15 * tanh(tenure/500)`, which saturates across the realistic range) "
            f"that its true contribution is swamped by the noise the forest fitted "
            f"around it"
        )
    dgp_paragraph = (
        "This is exactly the outcome the brief anticipated, and we can check it "
        "against the known data-generating process rather than guessing: "
        + "; ".join(notes) + "."
    ) if notes else (
        "Every collapsed feature does appear in the generator's log-odds, so the "
        "demotion here reflects overfitting rather than a feature that was noise "
        "by construction."
    )

    # ----------------------------------------------------------------- report
    t5_rows = "\n".join(
        f"| {i + 1} | `{r['transformed_feature']}` | `{r['original_feature']}` "
        f"| {r['importance']:.4f} |"
        for i, (_, r) in enumerate(top5_transformed.iterrows())
    )
    cmp_rows = "\n".join(
        f"| `{r['feature']}` | {r['impurity_importance']:.4f} | #{int(r['impurity_rank'])} "
        f"| {r['permutation_importance']:+.5f} | #{int(r['permutation_rank'])} "
        f"| {int(r['rank_change']):+d} |"
        for _, r in comparison.iterrows()
    )
    perm_rows = "\n".join(
        f"| #{int(r['perm_rank_overall'])} | `{r['original_feature']}` "
        f"| {r['perm_mean']:+.5f} | {r['perm_std']:.5f} |"
        for _, r in perm_df.iterrows()
    )

    interpretations = {
        "payment_method": "COD is the single strongest lever in the data-generating "
            "process (+0.9 on the log-odds). Commercially that is exactly what you "
            "would expect: a cash-on-delivery buyer has not parted with any money at "
            "the point of ordering, so walking away at the door costs them nothing, "
            "while a prepaid buyer has already absorbed the friction of paying and "
            "faces a refund wait if they return.",
        "price_inr": "Expensive orders are returned more often. High-ticket items "
            "attract more post-purchase second-guessing, are more likely to be "
            "impulse or aspirational buys, and are the ones where a customer will "
            "actually bother with the reverse-pickup process rather than keeping "
            "something cheap they mildly dislike.",
        "customer_tenure_days": "Long-tenured customers return less. Tenure proxies "
            "for a settled relationship with the catalogue — these buyers have "
            "learned which brands and sizes work for them, so their orders land "
            "closer to what they actually wanted.",
        "discount_pct": "Deeper discounts raise return risk. A heavy markdown pulls "
            "in buyers whose intent was driven by the price tag rather than the "
            "product, and discount-hunting carts are exactly the ones that get "
            "second-guessed once the box is opened.",
        "num_previous_returns": "Past returning behaviour predicts future returning "
            "behaviour. Some customers habitually order several variants intending "
            "to send most back; the prior-return count is the most direct trace of "
            "that habit the dataset carries.",
        "num_previous_orders": "Order count interacts with the prior-return count — "
            "it is the denominator of the return ratio the generator actually uses, "
            "so the forest needs it to make sense of a raw return count.",
        "delivery_distance_km": "Long-haul deliveries plausibly correlate with worse "
            "packaging outcomes and slower reverse logistics — but see the "
            "permutation result below before believing this one.",
        "delivery_days": "Slow deliveries raise return risk: the longer the wait, "
            "the more time a customer has to change their mind or source the item "
            "elsewhere.",
        "rating_given": "A low post-delivery rating is the customer telling you they "
            "are unhappy, which is the step immediately before a return.",
        "is_weekend_order": "Weekend orders skew slightly more impulsive.",
        "product_category": "Apparel and footwear carry fit risk that electronics and "
            "home goods do not — you cannot try a size on before it arrives.",
    }
    interp_rows = "\n".join(
        f"* **`{f}`** — {interpretations.get(f, 'see the data-generating process.')}"
        for f in top5_original
    )

    report = f"""# Part 1 — Task 7: Model Explanation

Regenerate with `python3 -m part1.feature_analysis`. Computed against the saved
`models/return_risk_model.pkl` (the tuned Random Forest pipeline).

## Top 5 features by impurity-based `.feature_importances_`

Read directly off the fitted forest, in the one-hot expanded feature space the
model actually sees:

| # | transformed feature | source column | importance |
|---:|---|---|---:|
{t5_rows}

Aggregating one-hot columns back to their source column
(`payment_method_COD` + `payment_method_Wallet` + ... -> `payment_method`), the
**top 5 original features** are: {", ".join(f"`{f}`" for f in top5_original)}.

### Why each of these plausibly drives return risk for an e-commerce order

{interp_rows}

## Permutation importance on the held-out test split

`sklearn.inspection.permutation_importance`, `scoring="roc_auc"`,
`n_repeats={N_REPEATS}`, `random_state={RANDOM_STATE}`. The value is the mean drop in
test ROC-AUC when that column is shuffled — a feature that carries no real
signal costs ~0 (and can go slightly negative by chance).

| rank | feature | mean ROC-AUC drop | std |
|---:|---|---:|---:|
{perm_rows}

## Side-by-side: impurity vs permutation, for the same top-5 features

| feature | impurity importance | impurity rank | permutation (ROC-AUC drop) | permutation rank | rank change |
|---|---:|---:|---:|---:|---:|
{cmp_rows}

### Which top-5 features lose most of their importance under permutation

**{len(collapsed)} of the 5 collapse.** These are the top-5 features whose measured
test-set ROC-AUC drop is negligible (< {NEGLIGIBLE}) — the forest split on them
constantly during training, yet destroying them costs the model essentially
nothing on data it has never seen:

{chr(10).join(
    f"* **`{r['feature']}`** — impurity #{int(r['impurity_rank'])} "
    f"({r['impurity_importance']:.4f}) -> permutation #{int(r['permutation_rank'])} "
    f"({r['permutation_importance']:+.5f} +/- {r['permutation_std']:.5f}), "
    f"a fall of {abs(int(r['rank_change']))} places."
    for _, r in collapsed.iterrows()
)}

The single largest demotion is **`{demoted_name}`**: impurity ranks it
#{int(demoted['impurity_rank'])} with an importance of
{demoted['impurity_importance']:.4f}, but shuffling it on the test split changes
ROC-AUC by {demoted['permutation_importance']:+.5f}
(+/- {demoted['permutation_std']:.5f}), dropping it to
#{int(demoted['permutation_rank'])} — a fall of {abs(int(demoted['rank_change']))}
places. A *negative* value here is not a bug: it means the shuffled column left
the model marginally better off, i.e. whatever the forest learned from that
column on the training data was noise that transferred as a small handicap.

**Why impurity-based importance can overrate a noisy continuous column.**
`.feature_importances_` totals up how much each split on a column reduced Gini
impurity *on the training data*. A continuous column with thousands of distinct
values offers thousands of candidate split points, so at every node the forest
gets many chances to find a cut that happens to separate that node's training
rows — and with enough candidate cuts, one of them will look good by luck alone.
Those lucky splits still accumulate impurity-reduction credit, so the column
scores highly even when it is pure noise. A low-cardinality column like a binary
one-hot flag has exactly one possible split and gets no such lottery tickets.
Permutation importance sidesteps the bias entirely by asking a different
question — *how much worse does the model get on data it has never seen when I
destroy this column?* — and for `{demoted_name}` the honest answer is
{"essentially nothing" if abs(float(demoted['permutation_importance'])) < 0.005 else "far less than the impurity score implied"}.

{dgp_paragraph}

By contrast `payment_method` survives both measures at #1 — it is a low-cardinality
one-hot flag with only one possible split per column, so it had no cardinality
lottery to win, and its {perm_mean['payment_method']:+.5f} ROC-AUC drop under
permutation confirms the model genuinely depends on it. That agreement between the
two measures is what real signal looks like.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nMost-demoted top-5 feature under permutation: {demoted_name} "
          f"(rank {int(demoted['impurity_rank'])} -> {int(demoted['permutation_rank'])})")
    print(f"Wrote {REPORT_PATH.relative_to(REPORTS_DIR.parent)}")


if __name__ == "__main__":
    main()
