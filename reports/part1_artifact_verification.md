# Part 1 — Saved Artifact Verification

Regenerate with `python3 -m part1.evaluate_return_risk`. This script loads
`models/return_risk_model.pkl` and `models/return_risk_metadata.json` **from
disk only** and re-derives the deterministic test split, so it verifies the
committed artifacts rather than anything held in memory from training.

**14/14 checks passed.**

| result | check | detail |
|---|---|---|
| PASS | dataset rows | 6000 rows (expected 6000) |
| PASS | dataset columns | 13 columns (expected 13) |
| PASS | return rate in [0.18, 0.27] | 0.2275 |
| PASS | missing rating_given in [0.08, 0.18] | 0.1305 |
| PASS | saved model loads | Pipeline from return_risk_model.pkl |
| PASS | saved model is the Random Forest pipeline | final estimator is RandomForestClassifier (must NOT be LogisticRegression) |
| PASS | saved model exposes predict/predict_proba | both present |
| PASS | preprocessing bundled into the same Pipeline | steps = ['prep', 'clf'] |
| PASS | threshold metadata exists and is numeric | threshold_rf = 0.5 |
| PASS | test ROC-AUC >= 0.58 | 0.6203 |
| PASS | best CV ROC-AUC >= 0.58 | 0.6193 |
| PASS | |CV ROC-AUC - test ROC-AUC| <= 0.05 (no severe overfitting) | gap = 0.0011 |
| PASS | t*_rf reproducible from the saved model's own predict_proba | recomputed 0.50 == stored 0.50 |
| PASS | single-row prediction works (Part 3 calling convention) | P(returned) = 0.5378 |

## Held-out classification report at t\*_rf = 0.50

```
              precision    recall  f1-score   support

not_returned     0.8331    0.6624    0.7380       927
    returned     0.3240    0.5495    0.4076       273

    accuracy                         0.6367      1200
   macro avg     0.5785    0.6059    0.5728      1200
weighted avg     0.7173    0.6367    0.6628      1200

```

Test ROC-AUC **0.6203** vs cross-validated **0.6193** (gap 0.0011).
