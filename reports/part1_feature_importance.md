# Part 1 — Task 7: Model Explanation

Regenerate with `python3 -m part1.feature_analysis`. Computed against the saved
`models/return_risk_model.pkl` (the tuned Random Forest pipeline).

## Top 5 features by impurity-based `.feature_importances_`

Read directly off the fitted forest, in the one-hot expanded feature space the
model actually sees:

| # | transformed feature | source column | importance |
|---:|---|---|---:|
| 1 | `cat__payment_method_COD` | `payment_method` | 0.1788 |
| 2 | `num__price_inr` | `price_inr` | 0.1323 |
| 3 | `num__delivery_distance_km` | `delivery_distance_km` | 0.0957 |
| 4 | `num__customer_tenure_days` | `customer_tenure_days` | 0.0900 |
| 5 | `num__delivery_days` | `delivery_days` | 0.0884 |

Aggregating one-hot columns back to their source column
(`payment_method_COD` + `payment_method_Wallet` + ... -> `payment_method`), the
**top 5 original features** are: `payment_method`, `price_inr`, `delivery_distance_km`, `customer_tenure_days`, `delivery_days`.

### Why each of these plausibly drives return risk for an e-commerce order

* **`payment_method`** — COD is the single strongest lever in the data-generating process (+0.9 on the log-odds). Commercially that is exactly what you would expect: a cash-on-delivery buyer has not parted with any money at the point of ordering, so walking away at the door costs them nothing, while a prepaid buyer has already absorbed the friction of paying and faces a refund wait if they return.
* **`price_inr`** — Expensive orders are returned more often. High-ticket items attract more post-purchase second-guessing, are more likely to be impulse or aspirational buys, and are the ones where a customer will actually bother with the reverse-pickup process rather than keeping something cheap they mildly dislike.
* **`delivery_distance_km`** — Long-haul deliveries plausibly correlate with worse packaging outcomes and slower reverse logistics — but see the permutation result below before believing this one.
* **`customer_tenure_days`** — Long-tenured customers return less. Tenure proxies for a settled relationship with the catalogue — these buyers have learned which brands and sizes work for them, so their orders land closer to what they actually wanted.
* **`delivery_days`** — Slow deliveries raise return risk: the longer the wait, the more time a customer has to change their mind or source the item elsewhere.

## Permutation importance on the held-out test split

`sklearn.inspection.permutation_importance`, `scoring="roc_auc"`,
`n_repeats=10`, `random_state=42`. The value is the mean drop in
test ROC-AUC when that column is shuffled — a feature that carries no real
signal costs ~0 (and can go slightly negative by chance).

| rank | feature | mean ROC-AUC drop | std |
|---:|---|---:|---:|
| #1 | `payment_method` | +0.09802 | 0.00981 |
| #2 | `price_inr` | +0.01020 | 0.00422 |
| #3 | `num_previous_returns` | +0.00846 | 0.00239 |
| #4 | `product_category` | +0.00603 | 0.00582 |
| #5 | `delivery_days` | +0.00257 | 0.00383 |
| #6 | `is_weekend_order` | +0.00120 | 0.00068 |
| #7 | `delivery_distance_km` | -0.00021 | 0.00156 |
| #8 | `discount_pct` | -0.00023 | 0.00261 |
| #9 | `rating_given` | -0.00188 | 0.00132 |
| #10 | `num_previous_orders` | -0.00240 | 0.00151 |
| #11 | `customer_tenure_days` | -0.00549 | 0.00182 |

## Side-by-side: impurity vs permutation, for the same top-5 features

| feature | impurity importance | impurity rank | permutation (ROC-AUC drop) | permutation rank | rank change |
|---|---:|---:|---:|---:|---:|
| `payment_method` | 0.2692 | #1 | +0.09802 | #1 | +0 |
| `price_inr` | 0.1323 | #2 | +0.01020 | #2 | +0 |
| `delivery_distance_km` | 0.0957 | #3 | -0.00021 | #7 | +4 |
| `customer_tenure_days` | 0.0900 | #4 | -0.00549 | #11 | +7 |
| `delivery_days` | 0.0884 | #5 | +0.00257 | #5 | +0 |

### Which top-5 features lose most of their importance under permutation

**2 of the 5 collapse.** These are the top-5 features whose measured
test-set ROC-AUC drop is negligible (< 0.002) — the forest split on them
constantly during training, yet destroying them costs the model essentially
nothing on data it has never seen:

* **`customer_tenure_days`** — impurity #4 (0.0900) -> permutation #11 (-0.00549 +/- 0.00182), a fall of 7 places.
* **`delivery_distance_km`** — impurity #3 (0.0957) -> permutation #7 (-0.00021 +/- 0.00156), a fall of 4 places.

The single largest demotion is **`customer_tenure_days`**: impurity ranks it
#4 with an importance of
0.0900, but shuffling it on the test split changes
ROC-AUC by -0.00549
(+/- 0.00182), dropping it to
#11 — a fall of 7
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
destroy this column?* — and for `customer_tenure_days` the honest answer is
far less than the impurity score implied.

This is exactly the outcome the brief anticipated, and we can check it against the known data-generating process rather than guessing: `delivery_distance_km` never enters the generator's log-odds `z` at all — it is pure noise by construction, yet it is a high-cardinality continuous column, so it is a prime candidate to be inflated by impurity importance and then exposed by permutation importance; `customer_tenure_days` does enter `z`, but only through a term small enough (`-0.15 * tanh(tenure/500)`, which saturates across the realistic range) that its true contribution is swamped by the noise the forest fitted around it.

By contrast `payment_method` survives both measures at #1 — it is a low-cardinality
one-hot flag with only one possible split per column, so it had no cardinality
lottery to win, and its +0.09802 ROC-AUC drop under
permutation confirms the model genuinely depends on it. That agreement between the
two measures is what real signal looks like.
