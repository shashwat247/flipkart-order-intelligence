# Part 1 — Task 8: Subgroup / Root-cause Analysis

Regenerate with `python3 -m part1.subgroup_analysis`. Winning model =
`models/return_risk_model.pkl` (tuned Random Forest), evaluated on the held-out
test split at its own operating threshold **t\*_rf = 0.50**.

**Overall at t\*_rf:** precision 0.3240, recall
0.5495, F1 0.4076 over 1200 test orders.

## Broken out by `product_category`

| product_category | test orders | actual returns | flagged | TP | FP | FN | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Apparel` | 385 | 100 | 164 | 52 | 112 | 48 | 0.3171 | 0.5200 | 0.3939 |
| `Beauty` | 116 | 31 | 40 | 19 | 21 | 12 | 0.4750 | 0.6129 | 0.5352 |
| `Electronics` | 261 | 52 | 70 | 23 | 47 | 29 | 0.3286 | 0.4423 | 0.3770 |
| `Footwear` | 217 | 56 | 91 | 33 | 58 | 23 | 0.3626 | 0.5893 | 0.4490 |
| `Home` | 221 | 34 | 98 | 23 | 75 | 11 | 0.2347 | 0.6765 | 0.3485 |

## Broken out by `payment_method`

| payment_method | test orders | actual returns | flagged | TP | FP | FN | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `COD` | 503 | 155 | 443 | 145 | 298 | 10 | 0.3273 | 0.9355 | 0.4849 |
| `Prepaid_Card` | 283 | 49 | 5 | 1 | 4 | 48 | 0.2000 | 0.0204 | 0.0370 |
| `Prepaid_UPI` | 294 | 48 | 6 | 2 | 4 | 46 | 0.3333 | 0.0417 | 0.0741 |
| `Wallet` | 120 | 21 | 9 | 2 | 7 | 19 | 0.2222 | 0.0952 | 0.1333 |

## The subgroup where the model is meaningfully worse

**`payment_method = Prepaid_Card`** — recall **0.0204**
against an overall recall of 0.5495, a shortfall of
**52.90 percentage points**. In absolute terms the model
misses **48 of the 49 real returns**
in this subgroup (283 test orders), flagging only
5 orders at precision 0.2000.

### Root cause

The forest ships a **single global cut point** but the subgroups do not share a
probability scale. `payment_method` is by far the strongest signal in the model
(Task 7: permutation ROC-AUC drop ~0.098, roughly 10x the next feature), and the
generator gives COD orders a +0.9 log-odds bump. The forest therefore parks most
COD orders above t\*_rf and most non-COD orders below it, and inside the
non-COD population the remaining features are too weak to push a genuinely
risky order back over a threshold that was tuned on the *pooled* distribution.
The global threshold is calibrated to the mixture, so it systematically
under-flags whichever subgroup sits on the low side of the dominant feature.

### Concrete proposed fix: a subgroup-specific decision threshold

Not "collect more data" — the fix is to stop forcing one cut point onto
distributions the model itself separates. Replacing the global t\*_rf with a
threshold fitted for `payment_method = Prepaid_Card` alone:

| | global t\*_rf = 0.50 | subgroup-specific t = 0.42 | change |
|---|---:|---:|---:|
| recall | 0.0204 | 0.6939 | +67.35 pp |
| precision | 0.2000 | 0.2810 | +8.10 pp |
| F1 | 0.0370 | 0.4000 | +36.30 pp |
| orders flagged | 5 | 121 | +116 |

Operationally: store a small `{subgroup: threshold}` map next to
`threshold_rf` in `models/return_risk_metadata.json` and have
`check_return_risk` select the cut point by the order's `payment_method`. The cost
is 116 more orders into the support
queue from this subgroup; the benefit is roughly 33 additional
genuine returns caught before they happen.

**Two caveats stated honestly.** First, 0.42 was fitted on the
same held-out split it is reported on, so it is an optimistic estimate — in
production it should be refitted by cross-validation on the training split and
only then measured on test. Second, this is a calibration patch, not new signal:
it redistributes the precision/recall trade across subgroups but cannot raise
the model's overall ROC-AUC of 0.6203. Genuinely lifting
performance for this subgroup needs a feature that discriminates *within* it —
a size/fit-mismatch flag, a returns-per-SKU history, or a seller-quality score —
which is a modelling change, not a threshold change.
