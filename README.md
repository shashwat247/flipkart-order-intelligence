# Flipkart Order Intelligence & Support Assistant

One connected system, not three scripts. A return-risk model trained on order
history, a product-image categoriser trained by transfer learning, and a
LangGraph support agent that calls **both saved artifacts as real tools** on top
of its own retrieval-augmented policy knowledge base.

```
  Part 1  Return-risk model          models/return_risk_model.pkl  ─┐
          (tuned Random Forest)      + t*_rf in metadata.json       │
                                                                    ├──►  Part 3
  Part 2  Product-image classifier   models/product_classifier.pt  ─┤     Support
          (ResNet-18 transfer)       + 10 real test-split PNGs      │     agent
                                                                    │
  Part 3  Policy knowledge base ──► sentence chunks ──► FAISS ──────┘
          (15 documents)             (45 chunks)        index

                    LangGraph:  guard → intent → ⟨retrieval | tools⟩ → response
```

Parts 1 and 2 are not thrown away after grading. Part 3's `check_return_risk`
loads Part 1's pickle and calls its `predict_proba`; Part 3's
`classify_product_image` loads Part 2's `.pt` and runs it over the pixels of the
PNGs Part 2 exported. Nothing in Part 3 is a hardcoded stand-in.

**Everything runs locally and free.** Part 3's default `MOCK_LLM` mode needs
**zero API keys and makes zero outbound network calls** — there is a test that
proves it by poisoning every socket and re-running the agent.

---

## Table of contents

- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Part 1 — Return-risk scoring pipeline](#part-1--return-risk-scoring-pipeline)
- [Part 2 — Product image categoriser](#part-2--product-image-categoriser)
- [Part 3 — Flipkart support agent](#part-3--flipkart-support-agent)
- [Streamlit App — Flipkart Intelligence & Support Center](#streamlit-app--flipkart-intelligence--support-center)
- [Order Intelligence Console](#order-intelligence-console)
- [Example transcript](#example-transcript)
- [Multi-turn state vs a fresh conversation](#multi-turn-state-vs-a-fresh-conversation)
- [Retrieval evaluation](#retrieval-evaluation)
- [Model artifacts](#model-artifacts)
- [Tests and validation](#tests-and-validation)
- [Git workflow](#git-workflow)
- [Limitations](#limitations)

---

## Quick start

### Install

```bash
git clone <this-repo-url>
cd flipkart-order-intelligence

python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
pip install --upgrade pip
pip install -r requirements.txt
```

On **Windows** activate the virtual environment with:

```powershell
python -m venv .venv
.venv\Scripts\activate             # PowerShell
REM  .venv\Scripts\activate.bat    (cmd.exe)
pip install -r requirements.txt
```

Requires **Python 3.10+** (built and tested on 3.13.7). No account, no API key,
no paid service at any point.

### Reproduce everything, in order

```bash
# --- Part 1: dataset + return-risk model -------------------------------
python3 generate_orders.py                    # writes orders_dataset.csv
python3 -m part1.verify_dataset               # Task 2: shape, rates, MAR analysis
python3 -m part1.train_return_risk            # Tasks 3-6, 9: trains and saves the model
python3 -m part1.feature_analysis             # Task 7: importance + permutation importance
python3 -m part1.subgroup_analysis            # Task 8: subgroup / root-cause analysis
python3 -m part1.evaluate_return_risk         # reloads the saved artifact and re-verifies

# --- Part 2: product-image categoriser ---------------------------------
python3 -m part2.cache_features               # downloads Fashion-MNIST, caches backbone features
python3 -m part2.train_product_classifier     # trains the head (conditionally fine-tunes)
python3 -m part2.evaluate_product_classifier  # final test-set evaluation + confusion matrix
python3 -m part2.export_samples               # writes 10 real test-split PNGs

# --- Part 3: support agent ---------------------------------------------
python3 -m part3.build_index                  # embeds the KB, builds the FAISS index
python3 -m part3.calibrate_threshold          # measures the groundedness threshold
python3 -m part3.evaluate_retrieval           # Task 10: Precision@3 / Recall@3
python3 -m part3.run_transcripts              # writes the 10 transcripts

# --- verify the whole thing --------------------------------------------
pytest
python3 validate_project.py

# --- the app: backend API + React console ------------------------------
python3 -m backend.api                  # terminal 1 -> http://127.0.0.1:8000
cd frontend && npm install && npm run dev   # terminal 2 -> http://localhost:5173

# --- Streamlit app (secondary UI) --------------------------------------
streamlit run streamlit_app/app.py      # http://localhost:8501
```

### Run the agent (default MOCK_LLM mode — no API key)

```bash
python3 -m part3.agent --demo                 # one example of every intent
python3 -m part3.agent                        # interactive REPL
python3 -m part3.agent --ask "How many days do I have to return a mobile phone?"
python3 -m part3.agent --ask "What category is this?" \
                       --image data/sample_images/07_sneaker.png
```

**No API key is required.** `MOCK_LLM` is the default and the only mode any
transcript in this repo was produced in.

Approximate one-off costs on a laptop (measured on an Apple M3, 8 GB):
Part 1 ~40 s, Part 2 feature caching ~4 min (then ~7 s of head training),
Part 2 evaluation ~1 min, Part 3 index build ~5 s.

---

## Repository layout

```
.
├── README.md                       ← you are here
├── requirements.txt                pinned, tested versions
├── generate_orders.py              Part 1 Task 1 — the exact seeded generator
├── orders_dataset.csv              6,000 x 13, committed
├── validate_project.py             end-to-end acceptance checks
├── conftest.py
│
├── part1/                          return-risk pipeline
│   ├── common.py                   features, split, preprocessing, threshold sweep
│   ├── verify_dataset.py           Task 2
│   ├── train_return_risk.py        Tasks 3-6, 9 (+ t*_rf)
│   ├── feature_analysis.py         Task 7
│   ├── subgroup_analysis.py        Task 8
│   └── evaluate_return_risk.py     artifact reload + verification
│
├── part2/                          image categoriser
│   ├── config.py                   preprocessing contract, split sizes, hyperparameters
│   ├── data.py                     Fashion-MNIST, stratified val split, transforms
│   ├── model.py                    architecture + classify_product_image()
│   ├── cache_features.py           frozen-backbone feature caching
│   ├── train_product_classifier.py head training + conditional fine-tuning
│   ├── evaluate_product_classifier.py  test-set evaluation + confusion analysis
│   └── export_samples.py           writes real test-split PNGs
│
├── part3/                          support agent
│   ├── knowledge_base/             15 policy documents (POL01..POL15) + README
│   ├── config.py                   thresholds, paths, MOCK_LLM switch
│   ├── chunking.py                 sentence-wise chunking with parent-doc mapping
│   ├── embeddings.py               local all-MiniLM-L6-v2
│   ├── retrieval.py                FAISS search, doc rollup, groundedness check
│   ├── build_index.py              builds and persists the index
│   ├── tools.py                    check_return_risk + classify_product_image
│   ├── guardrails.py               prompt-injection detection
│   ├── prompts.py                  4S system prompt + few-shot examples
│   ├── mock_llm.py                 deterministic response generator
│   ├── live_llm.py                 OPTIONAL live extension (never required)
│   ├── state.py                    LangGraph conversation state
│   ├── graph.py                    the graph + Conversation
│   ├── agent.py                    CLI entry point
│   ├── eval_queries.py             retrieval answer key
│   ├── calibrate_threshold.py      groundedness threshold calibration
│   ├── evaluate_retrieval.py       Task 10
│   └── run_transcripts.py          Task 9
│
├── backend/                        HTTP API the console talks to
│   └── api.py                      FastAPI: /api/chat, /api/return-risk, /api/classify, ...
├── frontend/                       Order Intelligence Console (React/Vite/TS) — PRIMARY UI
│   └── src/screens/assistant/      the chat interface (default route)
├── streamlit_app/                  secondary Streamlit UI over the same artifacts
│   └── app.py                      Dashboard, chat, risk/image tools, KB explorer, insights
│
├── models/
│   ├── return_risk_model.pkl       tuned RF pipeline (Part 1)
│   ├── return_risk_metadata.json   t*_rf and bucket definitions
│   ├── product_classifier.pt       ResNet-18 state_dict (Part 2)
│   └── product_classifier_metadata.json
│
├── data/
│   ├── sample_images/              10 real Fashion-MNIST test PNGs + manifest
│   └── policy_index/               FAISS index + chunk metadata
│
├── reports/                        every generated analysis (all text/CSV)
├── transcripts/                    10 agent transcripts + INDEX.md
├── scripts/
│   └── export_reports.py          writes frontend/public/reports/*.json from the real artifacts
└── tests/                          136 pytest tests
```

---

## Part 1 — Return-risk scoring pipeline

### Dataset generation (Task 1)

`generate_orders.py` is the brief's generator, used verbatim — same
`np.random.default_rng(42)`, same category/payment lists and probabilities, same
return-generating formula, same order of RNG consumption. A test
(`test_generator_is_deterministic`) re-runs it in a temp directory and asserts
the output is byte-identical to the committed `orders_dataset.csv`.

| property | measured |
|---|---:|
| rows | **6,000** |
| columns | **13** |
| overall return rate | **0.2275** (22.75%) — inside the required 18-27% |
| missing `rating_given` | **13.05%** — inside the required 8-18% |

### Verification and the missingness mechanism (Task 2)

Full report: [`reports/part1_data_verification.md`](reports/part1_data_verification.md)

Return rate by category and by payment method:

| product_category | orders | return rate | | payment_method | orders | return rate |
|---|---:|---:|---|---|---:|---:|
| Apparel | 1979 | 26.43% | | COD | 2501 | **30.75%** |
| Footwear | 1071 | 25.96% | | Wallet | 594 | 17.85% |
| Beauty | 579 | 20.03% | | Prepaid_UPI | 1448 | 16.92% |
| Home | 1055 | 19.15% | | Prepaid_Card | 1457 | 16.82% |
| Electronics | 1316 | 18.69% | | | | |

**`rating_given` is MAR — missing at random.**

| group | missing rate |
|---|---:|
| COD orders | **22.83%** |
| non-COD orders | **6.06%** |
| **measured gap** | **16.77 percentage points** |

* **Not MCAR**: the missing rate is not uniform — COD orders drop their rating
  3.77x as often as non-COD, a measured 16.77 pp gap. There is a real dependency.
* **MAR**: that dependency is entirely on `payment_method`, a column we
  **observe** on every row. Condition on it and the missingness carries no
  further information.
* **Not MNAR**: the mask is drawn independently of the rating value itself, so
  nothing depends on the unobserved `rating_given`.

### Leakage-free preprocessing (Task 3)

One `ColumnTransformer` inside a `Pipeline`:

* **numeric** (9 columns) — `SimpleImputer(strategy="median")` → `StandardScaler()`
* **categorical** (`product_category`, `payment_method`) —
  `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore")`

`order_id` is deliberately **excluded** — it is a row identifier, not a signal.
`.fit()` is only ever called on the training split; the test split is only ever
`.transform()`-ed, so no test statistic (median, mode, mean/std, category level)
can leak backwards. Split: stratified 80/20, `random_state=42` → 4,800 train /
1,200 test, both at a 0.2275 return rate.

### Baseline (Task 4)

| `DummyClassifier(strategy="most_frequent")` | value |
|---|---:|
| accuracy | **0.7725** |
| **F1 (class 1)** | **0.0000** |
| recall (class 1) | 0.0000 |

**Why 77.25% accuracy is misleading.** Only 22.75% of orders are returned, so a
model that predicts "not returned" every single time is right 77.25% of the
time while never once identifying a return. The failure mode is
**high accuracy, zero recall**. The business asked for returns to be flagged and
this model flags none of them. That is why this project is graded on class-1
precision/recall/F1 and ROC-AUC against a baseline — *comparing against a
baseline* and *using metrics aligned to the real business problem* are the two
honest-evaluation rules this task is built on.

### Logistic Regression and the threshold sweep (Task 5)

`LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=2000)`.
Full report: [`reports/part1_model_report.md`](reports/part1_model_report.md),
full sweep: [`reports/part1_threshold_sweep_logreg.csv`](reports/part1_threshold_sweep_logreg.csv).

| at default threshold 0.50 | value |
|---|---:|
| accuracy | 0.5917 |
| precision (class 1) | 0.2964 |
| recall (class 1) | 0.5788 |
| F1 (class 1) | **0.3921** (≥ 0.30 required) |
| **ROC-AUC** | **0.6253** (≥ 0.58 required) |

Sweeping 0.10 → 0.90 in steps of 0.02:

| | default 0.50 | **t\*_logistic = 0.44** | change |
|---|---:|---:|---:|
| precision | 0.2964 | 0.2801 | **−1.63 pp** |
| recall | 0.5788 | **0.7582** | **+17.95 pp** |
| F1 | 0.3921 | **0.4091** | +1.70 pp |

The F1-maximising threshold lifts recall by **17.95 percentage points** for a
precision cost of only 1.63 pp.

**The business trade-off.** The two errors cost very different things. A **false
negative** is an order that quietly gets returned with no intervention —
Flipkart absorbs reverse-pickup logistics, restocking, refund processing and an
already-unhappy customer. A **false positive** is an agent proactively
contacting a customer about an order that was never coming back — one
agent-minute plus a small annoyance risk. Because the expensive error is the one
worth avoiding, we accept **more false positives** to get **fewer false
negatives**. The ceiling is capacity: at 0.44 the model flags 739 of 1,200 test
orders (61.6%), and flagging much more than that stops being triage and becomes
a queue nobody can work.

### Random Forest + GridSearchCV (Task 6)

Same preprocessing pipeline,
`RandomForestClassifier(class_weight="balanced", random_state=42)`,
grid `n_estimators ∈ [100, 200] × max_depth ∈ [6, 10, None]`, scored on
`roc_auc` with 5-fold `StratifiedKFold`.

| result | value |
|---|---|
| **best parameters** | `max_depth=6, n_estimators=200` |
| **best cross-validated ROC-AUC** | **0.6193** |
| **held-out test ROC-AUC** | **0.6203** |
| absolute gap | **0.0011** |

A 0.0011 gap is the evidence against severe overfitting: cross-validation was an
honest forecast of unseen-data performance, not an optimistic one.

### Model explanation (Task 7)

Full report: [`reports/part1_feature_importance.md`](reports/part1_feature_importance.md)

**Top 5 by impurity-based `.feature_importances_`** (one-hot expanded space):

| # | transformed feature | source column | importance |
|---:|---|---|---:|
| 1 | `cat__payment_method_COD` | `payment_method` | 0.1788 |
| 2 | `num__price_inr` | `price_inr` | 0.1323 |
| 3 | `num__delivery_distance_km` | `delivery_distance_km` | 0.0957 |
| 4 | `num__customer_tenure_days` | `customer_tenure_days` | 0.0900 |
| 5 | `num__delivery_days` | `delivery_days` | 0.0884 |

Why each plausibly drives return risk: a **COD** buyer has parted with no money
so walking away at the door is free; **expensive** orders attract more
post-purchase second-guessing and are worth the reverse-pickup hassle;
**long-tenured** customers have learned which brands and sizes work for them;
**slow deliveries** give a customer more time to change their mind or source the
item elsewhere.

**Permutation importance on the held-out test split** (`scoring="roc_auc"`,
`n_repeats=10`) — mean drop in test ROC-AUC when the column is shuffled:

| rank | feature | ROC-AUC drop | | rank | feature | ROC-AUC drop |
|---:|---|---:|---|---:|---|---:|
| 1 | `payment_method` | **+0.09802** | | 7 | `delivery_distance_km` | −0.00021 |
| 2 | `price_inr` | +0.01020 | | 8 | `discount_pct` | −0.00023 |
| 3 | `num_previous_returns` | +0.00846 | | 9 | `rating_given` | −0.00188 |
| 4 | `product_category` | +0.00603 | | 10 | `num_previous_orders` | −0.00240 |
| 5 | `delivery_days` | +0.00257 | | 11 | `customer_tenure_days` | −0.00549 |
| 6 | `is_weekend_order` | +0.00120 | | | | |

**Side by side — three of the impurity top-5 collapse:**

| feature | impurity | impurity rank | permutation | permutation rank | change |
|---|---:|---:|---:|---:|---:|
| `payment_method` | 0.1788 | #1 | +0.09802 | #1 | 0 |
| `price_inr` | 0.1323 | #2 | +0.01020 | #2 | 0 |
| `delivery_distance_km` | 0.0957 | #3 | **−0.00021** | #7 | **−4** |
| `customer_tenure_days` | 0.0900 | #4 | **−0.00549** | #11 | **−7** |
| `delivery_days` | 0.0884 | #5 | +0.00257 | #5 | 0 |

**Which top-5 feature loses most of its importance under permutation:**
**`customer_tenure_days`** falls furthest (#4 → #11), and
**`delivery_distance_km`** is the cleanest illustration — impurity rank #3 with
importance 0.0957, but shuffling it changes test ROC-AUC by **−0.00021**, i.e.
nothing.

**Why impurity-based importance can overrate a noisy continuous column.**
`.feature_importances_` totals how much each split on a column reduced Gini
impurity *on the training data*. A continuous column with thousands of distinct
values offers thousands of candidate split points, so at every node the forest
gets many chances to find a cut that separates that node's training rows by
luck — and those lucky splits still accumulate impurity-reduction credit. A
binary one-hot flag has exactly one possible split and gets no such lottery
tickets. Permutation importance asks a different question — *how much worse does
the model get on unseen data when I destroy this column?* — and is immune to the
bias.

We can check this against the known data-generating process rather than
guessing: **`delivery_distance_km` never enters the generator's log-odds `z` at
all**, so it is noise by construction, while `customer_tenure_days` enters only
through `−0.15 · tanh(tenure/500)`, which saturates and so varies barely at all.
By contrast `payment_method` — the one feature that agrees across both measures —
is exactly the +0.9 log-odds term the generator actually uses.

### Subgroup / root-cause analysis (Task 8)

Full report: [`reports/part1_subgroup_analysis.md`](reports/part1_subgroup_analysis.md).
Winning model at its operating threshold **t\*_rf = 0.50**. Overall: precision
0.3240, recall 0.5495, F1 0.4076.

**By `product_category`:**

| category | test orders | precision | recall | F1 |
|---|---:|---:|---:|---:|
| Beauty | 116 | 0.4750 | 0.6129 | 0.5352 |
| Footwear | 217 | 0.3626 | 0.5893 | 0.4490 |
| Apparel | 385 | 0.3171 | 0.5200 | 0.3939 |
| Electronics | 261 | 0.3286 | 0.4423 | 0.3770 |
| Home | 221 | 0.2347 | 0.6765 | 0.3485 |

**By `payment_method`:**

| payment method | test orders | precision | recall | F1 |
|---|---:|---:|---:|---:|
| COD | 503 | 0.3273 | **0.9355** | 0.4849 |
| Wallet | 120 | 0.2222 | 0.0952 | 0.1333 |
| Prepaid_UPI | 294 | 0.3333 | 0.0417 | 0.0741 |
| **Prepaid_Card** | 283 | 0.2000 | **0.0204** | 0.0370 |

**The genuinely weaker subgroup: `payment_method = Prepaid_Card`.** Recall
**0.0204** against an overall 0.5495 — a shortfall of **52.90 percentage
points**. It catches **1 of 49** real returns in that subgroup.

**Root cause.** The forest ships a *single global cut point* but the subgroups
do not share a probability scale. `payment_method` is by far the strongest
signal (permutation ROC-AUC drop +0.098, ~10x the next feature) and the
generator gives COD a +0.9 log-odds bump. The forest parks most COD orders above
t\*_rf and most prepaid orders below it, and within the prepaid population the
remaining features are too weak to push a genuinely risky order back over a
threshold tuned on the *pooled* distribution.

**Concrete proposed fix — a subgroup-specific decision threshold** (not "collect
more data"). Fitting a threshold for `Prepaid_Card` alone:

| | global t\*_rf = 0.50 | **Prepaid_Card t = 0.42** | change |
|---|---:|---:|---:|
| recall | 0.0204 | **0.6939** | **+67.35 pp** |
| precision | 0.2000 | 0.2810 | +8.10 pp |
| F1 | 0.0370 | **0.4000** | +36.30 pp |
| orders flagged | 5 | 121 | +116 |

Operationally: store a `{subgroup: threshold}` map alongside `threshold_rf` in
`models/return_risk_metadata.json` and have `check_return_risk` select the cut
point by the order's `payment_method`.

**Two caveats, stated honestly.** (1) 0.42 was fitted on the same held-out split
it is reported on, so it is optimistic — in production it should be refit by
cross-validation on the training split and only then measured on test. (2) This
is a calibration patch, not new signal: it redistributes the precision/recall
trade across subgroups but cannot raise the overall test ROC-AUC of 0.6203.
Genuinely lifting this subgroup needs a feature that discriminates *within* it —
a fit-mismatch flag, a returns-per-SKU history, or a seller-quality score.

### The saved artifact and t\*_rf (Task 9)

`models/return_risk_model.pkl` holds `search.best_estimator_` — the **whole**
fitted `Pipeline` (ColumnTransformer + tuned RandomForest), not the Logistic
Regression and not a bare estimator. It is persisted **first**, then reloaded
with `joblib.load`, its `predict_proba` verified identical to the in-memory
model, and only then is the threshold computed — so t\*_rf provably belongs to
the artifact Part 3 loads.

The Task 5 sweep was re-run on **the saved Random Forest's own `predict_proba`**
over the same test split:

| | default 0.50 | **t\*_rf = 0.50** |
|---|---:|---:|
| precision | 0.3240 | 0.3240 |
| recall | 0.5495 | 0.5495 |
| F1 | 0.4076 | **0.4076** |

> **t\*_rf = 0.50**, stored in
> [`models/return_risk_metadata.json`](models/return_risk_metadata.json).

Here t\*_rf coincides with the default 0.5 — that is simply where this forest's
F1 peaks, and it is a *measured* result, not a default that was left in place.
The sweep is re-run rather than reused because **reusing t\*_logistic = 0.44
would have been a scale error**: the two models produce different probability
distributions, so a cut point tuned on one says nothing about where the other
separates.

---

## Part 2 — Product image categoriser

Full report: [`reports/part2_evaluation.md`](reports/part2_evaluation.md) ·
training log: [`reports/part2_training_log.json`](reports/part2_training_log.json)

### Dataset

**Fashion-MNIST** (Zalando Research), fetched with zero configuration via
`torchvision.datasets.FashionMNIST(root="data", download=True)` — no login, no
API key. 70,000 greyscale 28x28 images across 10 apparel/footwear/accessory
categories, an exact match for Flipkart's own catalogue departments.

| split | images | note |
|---|---:|---|
| train | **50,000** | the 60k train half minus the validation carve-out |
| validation | **10,000** | **stratified**, exactly 1,000 per class, carved from the train half |
| test | **10,000** | the canonical test split — **untouched** until the final number |

### Preprocessing for a pretrained backbone

| step | value |
|---|---|
| resize | **224 × 224** (ResNet-18's native ImageNet resolution) |
| channels | 1 grey channel **replicated to 3** |
| normalisation | ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |

### Transfer learning

Pretrained **ResNet-18** (`ResNet18_Weights.IMAGENET1K_V1`). `conv1`, `bn1` and
`layer1`–`layer4` are **frozen**; `fc` is replaced with a fresh
`Linear(512, 10)` and is the only thing trained.

| hyperparameter | value |
|---|---|
| optimizer | **Adam** |
| learning rate | 1e-3 |
| batch size | 256 |
| epochs | 15 |
| device | `mps` (Apple Silicon; CUDA and CPU both work unchanged) |

### Feature caching — the speed-up that matters

Because the backbone is frozen, its output for a given image is **constant
across epochs**. `part2/cache_features.py` runs the backbone over every image
**once** and caches the 512-d vectors; the head then trains on those cached
tensors. This is mathematically identical to re-running the frozen backbone
every epoch:

| approach | measured |
|---|---|
| naive (15 epochs × 70,000 forward passes) | ~60 minutes |
| **cached** (1 extraction pass + head fit) | **~4 min extraction + ~7 s training** |

### Fine-tuning was conditional — and not needed

| stage | validation accuracy |
|---|---:|
| after feature extraction (frozen backbone, head only) | **0.8925** |
| after fine-tuning | **not run** — 0.8925 already cleared the 0.80 bar |

Stated explicitly: **feature extraction alone was sufficient**, so `layer4` was
never unfrozen and the before/after numbers are identical because no second
stage ran. The fine-tuning path is implemented and would trigger automatically
if validation accuracy came in below 0.80.

### Final test-set accuracy

> **0.8872 (88.72%)** on the 10,000 held-out test images.

### Confusion matrix (10 × 10, real predictions)

Rows = true, columns = predicted. Also at
[`reports/part2_confusion_matrix.csv`](reports/part2_confusion_matrix.csv).

| true \ pred | T-shirt/top | Trouser | Pullover | Dress | Coat | Sandal | Shirt | Sneaker | Bag | Ankle boot |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **T-shirt/top** | **822** | 5 | 20 | 21 | 3 | 2 | 119 | 0 | 7 | 1 |
| **Trouser** | 1 | **973** | 3 | 15 | 2 | 1 | 4 | 0 | 1 | 0 |
| **Pullover** | 13 | 0 | **861** | 3 | 56 | 0 | 66 | 0 | 1 | 0 |
| **Dress** | 22 | 7 | 19 | **849** | 35 | 0 | 67 | 0 | 1 | 0 |
| **Coat** | 1 | 0 | 64 | 24 | **801** | 0 | 107 | 0 | 3 | 0 |
| **Sandal** | 0 | 0 | 0 | 0 | 0 | **950** | 1 | 36 | 2 | 11 |
| **Shirt** | 104 | 0 | 42 | 27 | 88 | 1 | **730** | 0 | 8 | 0 |
| **Sneaker** | 0 | 0 | 0 | 0 | 0 | 16 | 0 | **962** | 1 | 21 |
| **Bag** | 1 | 0 | 2 | 3 | 1 | 2 | 9 | 0 | **981** | 1 |
| **Ankle boot** | 0 | 0 | 0 | 0 | 1 | 13 | 0 | 42 | 1 | **943** |

Every row sums to 1,000 (the per-class test support) and the diagonal sums to
8,872 — the 88.72% accuracy above.

### Per-class precision / recall / F1

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| T-shirt/top | 0.8527 | 0.8220 | 0.8371 | 1000 |
| Trouser | 0.9878 | 0.9730 | 0.9804 | 1000 |
| Pullover | 0.8516 | 0.8610 | 0.8563 | 1000 |
| Dress | 0.9013 | 0.8490 | 0.8744 | 1000 |
| Coat | 0.8116 | 0.8010 | 0.8062 | 1000 |
| Sandal | 0.9645 | 0.9500 | 0.9572 | 1000 |
| **Shirt** | **0.6618** | **0.7300** | **0.6942** | 1000 |
| Sneaker | 0.9250 | 0.9620 | 0.9431 | 1000 |
| Bag | 0.9751 | 0.9810 | 0.9781 | 1000 |
| Ankle boot | 0.9652 | 0.9430 | 0.9540 | 1000 |

### Confusion patterns — read off the matrix, not guessed

Largest off-diagonal cells: `T-shirt/top → Shirt` 119, `Coat → Shirt` 107,
`Shirt → T-shirt/top` 104, `Shirt → Coat` 88, `Dress → Shirt` 67,
`Pullover → Shirt` 66.

**Pair 1: `Shirt` ↔ `T-shirt/top` — 223 misclassifications (119 + 104).**
Both are short-to-medium-sleeved upper-body garments photographed flat against
the same background, and at 28x28 the only thing distinguishing them is a button
placket or a collar — features that occupy three or four pixels at this
resolution and are frequently smoothed away entirely by the downsampling. The
silhouettes are near-identical: same shoulder width, same torso taper, same
sleeve stubs. Even a human labelling the raw thumbnails disagrees with the
ground truth on this pair regularly, which is why it is the canonical hard pair
in Fashion-MNIST. Upsampling to 224x224 cannot recover detail that was never
captured — it interpolates the existing 784 pixels, so the collar that would
settle the question simply is not in the signal.

**Pair 2: `Shirt` ↔ `Coat` — 195 misclassifications (107 + 88).**
A coat and a long-sleeved shirt share the same basic outline: a rectangular
torso with two sleeves extending to roughly the same length. The real-world
difference is thickness, layering and fastening hardware, all of which read as
subtle intensity gradients rather than shape changes. Because the images are
greyscale, the model loses the colour and texture cues (wool vs cotton weave) a
shopper would use instantly, and is left comparing two very similar binary
silhouettes.

**What this means for the catalogue use case.** The errors cluster inside two
visually coherent families — upper-body garments and footwear — and almost never
cross between them. For "is this photo filed under roughly the right
department?", that is the useful failure mode: a mis-tagged shirt still lands in
apparel, so a support agent using Part 3's tool gets the right department even on
the model's bad days.

### Exported sample images

`part2/export_samples.py` writes **10 real PNGs** — one per class, the first
test-split occurrence of each, chosen deterministically — to
`data/sample_images/`, along with a `manifest.json` recording the exact
test-split index each came from. All 10 are classified correctly by the saved
model.

**The filename is never read.** A test (`test_prediction_ignores_the_filename`)
copies `07_sneaker.png` to `99_definitely_a_handbag.png` and asserts the
prediction and confidence are unchanged.

---

## Part 3 — Flipkart support agent

### Knowledge base

**15 policy documents** in [`part3/knowledge_base/`](part3/knowledge_base/)
(`POL01`–`POL15`), 2–4 sentences each, clearly labelled as synthetic project
content. They cover: apparel / footwear / electronics / home return windows, COD
and prepaid refund timelines, delivery SLA, delayed delivery, reverse-pickup
eligibility and process, damaged product, wrong product, cancellation, exchange
and non-returnable items.

**Chunking is sentence-wise**: 15 documents → **45 chunks**, each carrying
`chunk_id`, `document_id`, `document_title` and `chunk_text`. Every chunk keeps a
pointer to its parent document, which is what makes document-level retrieval
evaluation possible.

*Why sentence-wise?* A policy sentence is the natural unit of a policy answer —
"apparel may be returned within 10 days of delivery" is a complete, quotable
rule. Fixed-size or overlapping windows cut mid-rule; multi-sentence chunks
bundle an unrelated rule into every hit and inflate apparent recall. Each
sentence is written to be self-contained rather than relying on the previous one.

### Embeddings and index

**`all-MiniLM-L6-v2`** via sentence-transformers, running entirely locally
(~90 MB, downloaded once and cached; free, no account, no API key). Embeddings
are L2-normalised, so the **FAISS `IndexFlatIP`** inner product *is* cosine
similarity. Index and chunk metadata persist to `data/policy_index/`.

### The graph

Five nodes and one real conditional edge:

```
START ──► guard_node ──► intent_node ──► [conditional: route_by_intent]
                                          ├── blocked ──────────────► response_node
                                          ├── policy ──► retrieval_node ──► response_node
                                          ├── return_risk ─────► tool_node ──► response_node
                                          └── product_category ─► tool_node ──► response_node
                                                                    response_node ──► END
```

The branch is genuinely load-bearing: a policy question never touches the tool
node, a return-risk question never runs retrieval, and a blocked input reaches
neither. Every transcript prints the actual node path.

### Intent classification — and how the few-shot examples do real work

`part3/prompts.py` holds **16 few-shot examples** across the three intents. The
intent node **embeds those exemplars and routes the user's message to the intent
of the nearest one** — so the examples *are* the classifier, not decoration.
Editing the list changes routing.

Every transcript prints which exemplar matched and at what similarity, e.g.:

```
   nearest few-shot example : "Score the return risk for order 1024."
   that example's intent    : return_risk
   cosine similarity        : 0.7821
   FINAL INTENT             : return_risk
```

**This was tuned against measurements, not vibes.** An earlier draft used three
exemplars per intent and phrased the `return_risk` ones around the bare word
"returned" (*"Is order 1024 likely to be returned?"*). Measured on the
evaluation queries that misrouted **2 of 7 genuine policy questions** into the
return-risk lane — *"Can I return a used lipstick?"* is lexically closer to
"likely to be returned" than to any policy exemplar. The fix was to make the
`return_risk` exemplars turn on what actually distinguishes that intent (a
*specific order* being *scored*) rather than on the word "return", which every
lane shares. All 7 now route correctly.

**Routing floor (0.25).** When a message is far from *every* exemplar the argmax
is noise, so below 0.25 the agent falls back to the **policy** lane deliberately
— policy is the only branch with an evidence check behind it, so an unroutable
question ends up honestly refused rather than confidently handed to a model tool.
Measured: genuine questions bottom out at 0.2819 exemplar similarity, while
*"Who won the cricket match last night?"* sits at 0.1492.

### The 4S prompt principles

| principle | how the system prompt implements it |
|---|---|
| **Specific** | Enumerates the three permitted intents by name and states everything else is out of scope, instead of a vague "be helpful". |
| **Short** | A hard three-sentence answer ceiling with "lead with the rule or the number"; the prompt itself carries only constraints that change behaviour. |
| **Surround** | The evidence constraint is stated *before* the task ("use ONLY the retrieved chunks") and restated immediately *after* it ("no evidence, no answer"), so the binding rule brackets the response behaviour rather than trailing off at the end where it is easiest to drift past. |
| **Single** | Each node gets one objective and is explicitly forbidden the others — the retrieval node must not answer, the response node must not fetch. |
| **Role prompting** | Opens with *"You are Flipkart's order-support assistant."*, fixing persona, domain and register before any instruction is read. |

### The two tools — both call real artifacts

**`check_return_risk(order_features: dict) -> dict`** loads
`models/return_risk_model.pkl` and calls its `predict_proba`. A test asserts the
agent's number equals the saved model called directly:

```python
via_tool = check_return_risk(lookup_order(1790))["return_probability"]
direct   = joblib.load(MODEL_PATH).predict_proba(...)[0][1]
assert via_tool == round(direct, 4)     # test_part3.py
```

**Risk buckets are anchored to t\*_rf = 0.50** — the F1-maximising threshold
computed on the *saved Random Forest's own* `predict_proba`, loaded from
`models/return_risk_metadata.json`:

| bucket | rule | with t\*_rf = 0.50 |
|---|---|---|
| **Low** | `probability < t*_rf` | `< 0.50` |
| **Medium** | `t*_rf ≤ probability < t*_rf + 0.15` | `0.50 – 0.65` |
| **High** | `probability ≥ t*_rf + 0.15` | `≥ 0.65` |

> **The bucket cut points are anchored to the saved Random Forest's own
> F1-maximising probability threshold t\*_rf = 0.50 (not a fixed 0.3/0.6 split
> and not the Logistic Regression's 0.44), because two equally valid forests can
> produce probability distributions concentrated in completely different ranges,
> and a fixed split can silently collapse almost every order into one bucket.**

**`classify_product_image(image_path: str) -> dict`** is Part 2's function
imported directly — not a copy. It loads `models/product_classifier.pt`,
reconstructs the ResNet-18 architecture, applies the training preprocessing and
returns the model's argmax label plus its softmax confidence. It is pointed at
the real `.png` files in `data/sample_images/`, never at raw IDX data, and never
reads the filename.

### Guardrails

**Input side — prompt-injection detection.** The *application* scans raw user
text against **13 instruction-override patterns** before any node does work
(`ignore previous instructions`, `ignore all rules`, `pretend you are`,
`forget your instructions`, `reveal your system prompt`, `developer mode`,
`jailbreak`, …). This is not delegated to the prompt, because a model that has
been successfully injected is exactly the component you can no longer trust to
report the injection. Patterns require the surrounding override structure, so
ordinary questions like *"Can I ignore the delivery SMS?"* are not falsely
blocked (there is a test for that).

When blocked, the graph path is `guard_node → intent_node → response_node` —
**no retrieval and no tool call happen at all**.

**Output side — groundedness.** A policy question is refused if no retrieved
chunk clears **cosine similarity 0.45**, and the refusal prints the score it
fell short by.

**How 0.45 was chosen** (`python3 -m part3.calibrate_threshold`, full report at
[`reports/part3_threshold_calibration.md`](reports/part3_threshold_calibration.md)):
by measuring where in-domain and out-of-domain questions actually land, using
query sets declared up front in `part3/eval_queries.py` — *not* by looking at a
transcript and picking a flattering number.

| | value |
|---|---:|
| lowest in-domain score (*"Can I return a used lipstick?"*) | **0.4584** |
| highest out-of-domain score (*"What is Flipkart's GST registration number?"*) | **0.4379** |
| separation gap | **+0.0205** |
| **threshold chosen** | **0.45** — inside the gap |

Every question the corpus can answer passes; every question it cannot is
refused. The margin is genuinely narrow, and that is a stated limitation below.

### MOCK_LLM

`part3/mock_llm.py` is the default and the only mode any transcript here was
produced in. Given the retrieved chunks or the tool output, it composes the final
answer with rules and templates — **zero API keys, zero network calls, fully
deterministic**. Because it can only quote what it was handed, it *cannot*
fabricate a policy.

Every response conforms to a fixed JSON schema:

```json
{ "answer": "...", "source": "policy_kb", "confidence": 0.6429 }
```

`source` is always one of `policy_kb`, `return_risk_tool`,
`image_classifier_tool`. `confidence` is always a real measured quantity: the
best retrieval similarity for policy answers, `max(p, 1−p)` for return risk, the
softmax probability for image classification.

**`test_answer_path_makes_zero_network_calls`** proves the no-network claim: it
warms up the encoder and both models from local disk, then poisons
`socket.socket.connect` and `socket.create_connection` and re-runs all three
answer paths.

**Optional live LLM.** `part3/live_llm.py` exists behind `USE_LIVE_LLM=1` and is
**never required and never used by any transcript**. Even when enabled it cannot
change facts: the deterministic answer is computed first, only its *phrasing* is
sent for rewriting, `source` and `confidence` are copied from the mock response,
and any failure at all returns the mock response unchanged. With the flag unset
— the default — the module is never even imported. Removing the key leaves every
acceptance criterion satisfied.

---

## Streamlit App — Flipkart Intelligence & Support Center

```bash
streamlit run streamlit_app/app.py      # http://localhost:8501
```

A Streamlit UI over the real Part 1/2/3 artifacts — every page calls the same
functions the CLI and the test suite call (`check_return_risk`,
`classify_product_image`, `part3.graph.run_once` / `Conversation`,
`part3.retrieval.search_documents`). It holds **no** copy of the
return-risk formula, the Random Forest's threshold, or a policy answer; it
only renders what those functions return. `streamlit_app/app.py` keeps every
backend call in small, Streamlit-free functions at the top of the file
(`analyze_risk`, `classify_uploaded_bytes`, `ask_policy`, `search_kb`,
`get_system_status`, …) precisely so they can be — and are —
imported and tested directly (`tests/test_streamlit_app.py`), not just clicked
through by hand.

| page | what it does |
|---|---|
| **Dashboard** | A "Good morning" overview with five feature cards (Return Risk, Product AI, Policy AI, AI Assistant, System), each showing real status/metrics read from the saved artifacts and a button that jumps straight to that page. |
| **AI Support** | One chat for policy / return-risk / product-category questions, routed by the real LangGraph agent. Shows source, confidence and the tool used per turn, with node trace, groundedness and retrieved evidence collapsed in a technical-details expander. A **New Conversation** button explicitly rebuilds the `Conversation` object, clearing LangGraph state. |
| **Return Risk** | A form over the model's real 11 features (with an optional "load a real order by id" shortcut). *Analyze Return Risk* calls `check_return_risk` directly; shows the probability, the LOW/MEDIUM/HIGH bucket, a threshold-anchored probability gauge, and `t*_rf` in the technical-details expander. Recent analyses export to CSV. |
| **Product Classifier** | Upload/drop a PNG/JPG/JPEG or pick a real Fashion-MNIST sample; calls `classify_product_image` on a safely-scoped temp file (never the filename) and shows the predicted class, a confidence bar, and top-3. |
| **Policy Knowledge** | Policy documents as searchable, category-tagged cards (category derived deterministically from each document's own id/title), each with an **Ask AI about this policy** button that jumps to AI Support and asks the real agent. Semantic search over `search_documents` and a live Task 10 Precision@3/Recall@3 re-run are one click away in expanders. |
| **Model Insights** | Charts and tables read from `models/*_metadata.json` and `reports/*.csv` — ROC-AUC, F1/precision/recall at `t*_rf`, impurity-vs-permutation feature importance, the threshold sweep, the 10x10 confusion matrix, and per-class precision/recall. All figures are generated from saved evaluation results, not recomputed live. |
| **System Status** | Every component's live Ready/Unavailable status (KB, FAISS index, return-risk model, image model, LangGraph, AI mode) with its real detail message and a last-checked timestamp — the detailed counterpart to the Dashboard's summary cards. |

**Offline by default.** The sidebar always shows the real AI-mode badge —
`Offline Demo Mode (MOCK_LLM)` unless `USE_LIVE_LLM=1` is set — and every page
works with zero API keys and zero outbound network calls, same as the CLI agent.

**Light/dark theme.** `.streamlit/config.toml` declares both a light and a
dark palette, so Streamlit's native Settings menu (⋮ top right) can switch
between them — the browser remembers the choice. The app's own CSS reads
`st.context.theme.type` and swaps its own color tokens to match, so the
custom cards/badges/gauges stay legible in both modes.

**Error handling.** Every backend call on every page is wrapped so a missing
artifact, an unreadable image, an empty question or a retrieval failure
surfaces as a plain `st.error(...)` message, never a raw traceback. The
Dashboard reflects missing artifacts as `NOT READY` rather than silently
degrading.

This Streamlit app is the **secondary** UI. The primary interface is the React
console at `frontend/`, whose Support Assistant screen is a live chat backed by
`backend/api.py` — see the next section.

---

## Backend API

`backend/api.py` is the HTTP layer the console talks to. It is transport only:
every route delegates to the real agent or a real saved model, and there is no
answer table anywhere in the package.

```bash
python3 -m backend.api          # http://127.0.0.1:8000  (OpenAPI docs at /docs)
```

| Route | Method | What it does |
| --- | --- | --- |
| `/api/chat` | POST | One turn through the LangGraph agent. Returns the answer plus the intent, node trace, groundedness scores, retrieved documents and tool result. |
| `/api/conversations/reset` | POST | A new conversation id with genuinely empty state. |
| `/api/conversations/{id}/state` | GET | What that conversation currently remembers. |
| `/api/return-risk` | POST | Part 1's saved Random Forest, by `order_id` or explicit `order_features`. |
| `/api/classify` | POST | Part 2's saved ResNet-18 on one of the committed sample PNGs. |
| `/api/policies` | GET | The real knowledge base and its chunk count. |
| `/api/status` | GET | Live readiness of all six components. |
| `/api/health` | GET | Liveness plus the active mode. |

That delegation is enforced, not asserted: `validate_project.py` calls
`/api/chat`, `/api/return-risk` and `/api/classify` and asserts each result is
**identical** to a direct in-process call to the same agent and models, and
`tests/test_backend_api.py` does the same across 21 tests. The API binds to
127.0.0.1, keeps conversations in process memory, and needs no API key.

---

## Order Intelligence Console

The **primary user interface**: a React/Vite/TypeScript app at `frontend/`.

It has two halves.

**Support Assistant (`/assistant`, the default screen) — live.** A chat
interface that POSTs whatever you type to `POST /api/chat` on the backend, which
runs the real LangGraph agent. There is no question list, no keyword matching
and no canned answer anywhere in the frontend: arbitrary natural language goes
to the agent and whatever the agent returns is rendered. Each answer shows the
source, the confidence, the graph path that produced it, the retrieved policy
documents with their similarity scores, and — for a tool call — a return-risk
card anchored to `t*_rf` or a product-category card with the classifier's top-3.
Typing indicator, message animations, conversation state, new-conversation
reset, empty and error states, keyboard send (Enter; Shift+Enter for a newline)
and a layout that works from 375 px up.

**The inspection screens — static.** Every other screen makes an artifact,
metric, decision boundary or agent trace inspectable, reading a pre-exported
JSON file. If a report is missing, the screen names the exact file and the
command that produces it rather than showing a placeholder.

```bash
# 1. export every report the console reads (from the real, committed
#    artifacts — orders_dataset.csv, models/*.json, reports/*.csv,
#    transcripts/*.txt, the policy knowledge base)
python3 scripts/export_reports.py

# 2. install once
cd frontend
npm install

# 3. run it
npm run dev              # http://localhost:5173

# ...or build the static bundle
npm run build
```

Re-run `python3 scripts/export_reports.py` any time the underlying reports
change (a retrain, a new transcript run) — every output file is overwritten
from scratch, and nothing in `frontend/` is a second source of truth.

**Design system** — "night warehouse": a sorting facility at 2am, not a
generic dark SaaS dashboard. Bricolage Grotesque for the one hero number per
screen, Public Sans for prose, IBM Plex Mono (tabular figures) for every
metric, threshold, file path and code fragment — prose is human, numbers are
machine. The signature element is the **Graph Rail**, a persistent vertical
circuit diagram of the LangGraph agent (`guard → intent → [retrieve | tool |
direct] → generate`) that lights the traversed path in amber per turn and
clamps red at a blocked guardrail.

**30 screens** across four groups (Project; Part 1, 9 screens; Part 2, 8
screens; Part 3, 10 screens), a 12-component shared library
(`MetricTile`, `DataTable`, `Heatmap`, `RankedBars`, `ThresholdChart`,
`RiskBadge`, `ChunkCard`, `JsonCard`, `GraphRail`, `EmptyState`, `CodeBlock`,
`ProvenanceStrip`), hash-based routing and an in-house fuzzy command palette
(⌘K) — no router or fuzzy-search dependency beyond `recharts` and
`lucide-react`.

Two things the console deliberately does NOT pretend to do, because it is
static with no backend: the **Agent Console** and **Graph Inspector** replay
the real, committed `transcripts/*.txt` conversations rather than running a
live model call; the **Retrieval Explorer** is restricted to the pre-scored
evaluation queries in `part3/eval_queries.py` rather than offering free-text
embedding search it cannot back statically. Both are stated as such on
screen, not silently narrowed.

`scripts/export_reports.py` is covered by `tests/test_export_reports.py`
(part of the `pytest` run below), and `python3 validate_project.py` checks
that every report parses and that `npm run build` succeeds.

---

## Example transcript

One complete run, verbatim from
[`transcripts/01_policy_electronics_return_window.txt`](transcripts/01_policy_electronics_return_window.txt).
All 10 transcripts are indexed in
[`transcripts/INDEX.md`](transcripts/INDEX.md).

```
==============================================================================
FLIPKART ORDER INTELLIGENCE & SUPPORT ASSISTANT — Part 3 transcript
==============================================================================
Transcript      : Policy question answered via RAG (1 of 2)
Demonstrates    : Task 9(a) — a policy question routed to retrieval and answered from the knowledge base
LLM mode        : MOCK_LLM (deterministic; zero API keys; zero outbound network calls)
Embedding model : all-MiniLM-L6-v2 (local)
Retrieval       : FAISS IndexFlatIP, top_k=3 chunks
Groundedness    : refuse a policy answer below cosine 0.45
Intent routing  : nearest few-shot exemplar, floor 0.25
==============================================================================

------------------------------------------------------------------------------
TURN 1
------------------------------------------------------------------------------
USER: How many days do I have to return a mobile phone?

-- INTENT NODE (few-shot exemplars drive this routing) --
   nearest few-shot example : "How long does a refund take for a prepaid order?"
   that example's intent    : policy
   cosine similarity        : 0.4361
   routing floor            : 0.25 (below floor: False)
   FINAL INTENT             : policy
   runner-up                : "What is the delivery time to a non-metro address?" (policy) @ 0.3443
   runner-up                : "Can I cancel my order after it has shipped?" (policy) @ 0.3327
   order id                 : None (not available)

-- INPUT GUARDRAIL (prompt-injection scan, runs before any tool) --
   patterns checked : 13
   BLOCKED          : False

-- RETRIEVAL NODE (top-k chunks, cosine similarity) --
   [1] score=0.6429  doc=POL03_electronics_return_window
       "Electronics such as mobile phones, laptops, headphones and smart
       watches can be returned within 7 days of delivery."
   [2] score=0.4667  doc=POL04_home_return_window
       "Home and kitchen products such as cookware, bedsheets, curtains and
       small storage items can be returned within 7 days of delivery."
   [3] score=0.4536  doc=POL10_reverse_pickup_process
       "The returned item reaches the quality-inspection facility within 3 to
       5 business days of collection, and the refund clock starts only after
       that inspection passes."

   chunks rolled up to unique parent documents:
     POL03_electronics_return_window  (best chunk POL03_electronics_return_window::s00, score 0.6429)
     POL04_home_return_window  (best chunk POL04_home_return_window::s00, score 0.4667)
     POL10_reverse_pickup_process  (best chunk POL10_reverse_pickup_process::s02, score 0.4536)

-- OUTPUT GUARDRAIL (groundedness check) --
   best retrieved similarity : 0.6429
   required minimum          : 0.45
   GROUNDED                  : True
   verdict                   : answer permitted

-- GRAPH PATH --
   guard_node -> intent_node -> retrieval_node -> response_node

-- FINAL STRUCTURED RESPONSE --
   {
     "answer": "Electronics such as mobile phones, laptops, headphones and smart watches can be returned within 7 days of delivery. [Source: Electronics Return Window (POL03_electronics_return_window)] [Related policies: Home Products Return Window (POL04_home_return_window), Reverse Pickup Process and Timeline (POL10_reverse_pickup_process)]",
     "source": "policy_kb",
     "confidence": 0.6429
   }

-- CONVERSATION STATE AFTER THIS TURN --
   turn_index     : 1
   order_id       : None
   order_features : none
   history        : 2 messages
```

---

## Multi-turn state vs a fresh conversation

Short-term state lives on a `Conversation` object and is threaded explicitly
into `graph.invoke()` each turn. **There is no module-level dict, no global and
no persistent store** — so a new `Conversation` genuinely starts empty.

### Multi-turn — state carried

Full transcript:
[`transcripts/07_multiturn_state_carried.txt`](transcripts/07_multiturn_state_carried.txt)

```
TURN 1
USER: Check order 2314 for me.
   order id : 2314 (mentioned in this turn)
   → "Order 2314 has a 64.62% predicted probability of being returned,
      which is a MEDIUM risk. ..."

TURN 2 (same conversation)
USER: What is its return risk?
   order id : 2314 (carried from earlier in this conversation)
   → "Order 2314 has a 64.62% predicted probability of being returned,
      which is a MEDIUM risk. ..."

CONVERSATION STATE AFTER TURN 2
   turn_index : 2      order_id : 2314      order_features : remembered
```

Turn 2's text contains **no order id at all**. The agent names order 2314
because the LangGraph state carried it forward from turn 1.

### Fresh conversation — state correctly absent

Full transcript:
[`transcripts/08_fresh_conversation_state_reset.txt`](transcripts/08_fresh_conversation_state_reset.txt)

```
TURN 1 (brand-new conversation)
USER: What is its return risk?          ← identical to turn 2 above
   order id : None (not available)
   → "I need an order to score before I can answer that. Tell me an order id
      (for example 'order 4021') or supply the order's features. I won't
      estimate it, because a number I made up would look exactly like a number
      the model produced."

CONVERSATION STATE AFTER THIS TURN
   turn_index : 1      order_id : None      order_features : none
```

Run back to back in the same process, these two transcripts show state that is
genuinely conversation-scoped: carried correctly within one conversation, absent
in a fresh one. A test also asserts two `Conversation` objects cannot see each
other's order ids.

---

## Retrieval evaluation

Task 10, full report with per-query arithmetic:
[`reports/part3_retrieval_evaluation.md`](reports/part3_retrieval_evaluation.md)

**Scored at the document level.** Each retrieved chunk is mapped back to its
`document_id` and **deduplicated** before being compared against the answer key
in `part3/eval_queries.py` (written before any retrieval was run). The chunk
search runs over a wider pool (12 chunks) before the rollup, so "top-3 documents"
really is three distinct documents rather than a denominator that was never
filled.

### Per-query arithmetic

| # | query | relevant docs | top-3 retrieved docs | Precision@3 | Recall@3 |
|---:|---|---|---|---:|---:|
| 1 | How many days do I have to return a mobile phone? | {POL03} | {POL03, POL04, POL10} | 1/3 = 0.3333 | 1/1 = **1.0000** |
| 2 | When will I get my money back for a cash on delivery order? | {POL05, POL10} | {POL05, POL13, POL06} | 1/3 = 0.3333 | 1/2 = 0.5000 |
| 3 | Will a courier come to my house to collect the item I am returning? | {POL09, POL10} | {POL10, POL09, POL04} | 2/3 = **0.6667** | 2/2 = **1.0000** |
| 4 | How long does delivery take to a remote address? | {POL07} | {POL07, POL08, POL12} | 1/3 = 0.3333 | 1/1 = **1.0000** |
| 5 | Can I return a used lipstick? | {POL15} | {POL04, POL15, POL14} | 1/3 = 0.3333 | 1/1 = **1.0000** |
| 6 | My order arrived broken. What should I do? | {POL11} | {POL11, POL12, POL10} | 1/3 = 0.3333 | 1/1 = **1.0000** |
| 7 | Can I swap my shoes for a bigger size? | {POL02, POL14} | {POL02, POL14, POL04} | 2/3 = **0.6667** | 2/2 = **1.0000** |

### Averages

```
Average Precision@3 = (0.3333 + 0.3333 + 0.6667 + 0.3333 + 0.3333 + 0.3333 + 0.6667) / 7 = 0.4286
Average Recall@3    = (1.0000 + 0.5000 + 1.0000 + 1.0000 + 1.0000 + 1.0000 + 1.0000) / 7 = 0.9286
```

| metric | value |
|---|---:|
| **Average Precision@3** | **0.4286** |
| **Average Recall@3** | **0.9286** |
| queries evaluated | 7 |

**Reading these honestly.** Recall@3 is the metric that matters here — it asks
whether the document containing the answer reached the response generator at
all, and it does for 6 of 7 queries. Precision@3 is **structurally capped**:
five of the seven queries have only one relevant document, so retrieving three
documents means the best achievable Precision@3 for those is 1/3 = 0.3333.
Reporting 0.4286 without saying that would be misleading. The one genuine miss
is query 2, where `POL10` (the "refund clock starts after inspection"
precondition) was displaced by `POL13` and `POL06`.

---

## Model artifacts

| file | what it is | consumed by |
|---|---|---|
| [`models/return_risk_model.pkl`](models/) | The tuned Random Forest **as one fitted sklearn `Pipeline`** — `ColumnTransformer` + `RandomForestClassifier(max_depth=6, n_estimators=200, class_weight="balanced")`. Loaded with `joblib.load`. | Part 3's `check_return_risk` |
| [`models/return_risk_metadata.json`](models/return_risk_metadata.json) | `threshold_rf = 0.5`, the bucket definitions, best params, CV and test ROC-AUC. | Part 3's risk buckets |
| [`models/product_classifier.pt`](models/) | `torch.save(model.state_dict(), ...)` of the ResNet-18 with a `Linear(512, 10)` head. | Part 3's `classify_product_image` |
| [`models/product_classifier_metadata.json`](models/product_classifier_metadata.json) | Architecture, class names, input size, normalisation, load snippet. | model reconstruction |

**Documented loading snippet** (this is exactly what Part 3's tool calls):

```python
import torch
from part2.model import build_model

model = build_model(pretrained=False)                 # ResNet-18 + Linear(512, 10)
model.load_state_dict(torch.load("models/product_classifier.pt", map_location="cpu"))
model.eval()

# ...or just use the packaged one-function API:
from part2.model import classify_product_image
classify_product_image("data/sample_images/07_sneaker.png")
# {'predicted_class': 'Sneaker', 'confidence': 0.9987, ...}
```

---

## Tests and validation

```bash
pytest                      # 136 tests (103 Part 1-3 + 28 Streamlit app + 5 export_reports.py)
python3 validate_project.py # 93 end-to-end acceptance checks
```

**All 136 tests pass.** Coverage includes: generator determinism (byte-identical
re-run), the MAR missingness gap, artifact loading, `t*_rf` reproducibility from
the saved model's own `predict_proba`, 10-class output, PNG provenance,
filename-independence of image predictions, intent routing for all three lanes,
the routing-floor fallback, the fixed JSON schema, tool-vs-direct-model
agreement, bucket anchoring to `t*_rf`, injection detection (and
*non*-detection of benign text), the groundedness refusal, multi-turn state,
fresh-conversation reset, conversation isolation, per-turn scratch isolation,
MOCK_LLM determinism, and the zero-network proof.

`tests/test_streamlit_app.py` (28 tests) covers the Streamlit app separately: the app
boots and every page renders without an exception (`streamlit.testing.v1.AppTest`),
the Return Risk page's actual form widgets submit to the real saved model,
the Product Classifier's sample-image flow reaches the real classifier, Policy
Knowledge's search and its "Ask AI about this policy" card button both reach
real retrieval and the real agent, the Dashboard's feature-card buttons
navigate to their real pages, chat state is carried and the *New Conversation*
button really clears it, and a broken/missing artifact produces a friendly
status row instead of raising.

`validate_project.py` additionally checks the committed repository — no
virtualenv, no raw IDX data, no feature cache, but model artifacts and sample
PNGs present — and inspects the git history.

---

## Git workflow

```
*   merge  Merge feature/flipkart-assistant into main
|\
| * Part 3: LangGraph support agent, RAG, guardrails, transcripts
| * Part 2: ResNet-18 transfer-learning product categoriser
| * Part 1: return-risk scoring pipeline
|/
* Scaffold project: gitignore, pinned requirements, README placeholder
```

A feature branch `feature/flipkart-assistant` was created off `main`, committed
to three times (one per Part, each a real development stage), and merged back
with `git merge --no-ff`. Verify with:

```bash
git log --graph --oneline --all
```

---

## Limitations

**Part 1 — the ceiling is the data, not the model.** Test ROC-AUC is 0.6203.
That is modest in absolute terms and it is close to the ceiling this dataset
allows: the generator's log-odds are dominated by one term (`payment_method`),
and permutation importance confirms the model has extracted essentially all of
it. Adding model capacity will not help — six of eleven features have a
permutation importance at or below zero. The honest read is that the label is
mostly noise plus one strong signal.

**Part 1 — the subgroup fix is unvalidated.** The `Prepaid_Card` threshold of
0.42 was fitted on the same held-out split it is reported on, so its +67 pp
recall gain is an optimistic in-sample estimate. It needs cross-validated
refitting before anyone deploys it.

**Part 1 — this is synthetic data.** Every number is a property of a seeded
generator, not of real Flipkart orders. The pipeline transfers; the coefficients
do not.

**Part 2 — 28x28 is the hard limit.** 88.72% test accuracy is bounded by source
resolution, not architecture. The `Shirt` class (F1 0.6942) fails because the
collar and placket that define it were never captured in a 784-pixel greyscale
thumbnail. Upsampling to 224x224 interpolates; it cannot invent detail. A better
backbone would not fix this — higher-resolution source images would.

**Part 2 — Fashion-MNIST is not a Flipkart catalogue.** Real catalogue photos are
colour, higher resolution, inconsistently lit, often on models rather than flat,
and drawn from far more than 10 categories. The transfer-learning *method* would
carry over; these specific weights would not.

**Part 3 — the groundedness margin is narrow.** The gap between the lowest
in-domain score (0.4584) and the highest out-of-domain score (0.4379) is only
0.0205. The threshold is well-supported for *this* 15-document corpus but would
need recalibrating the moment documents are added — a "Careers" or "Tax and
invoicing" policy would legitimately move the boundary. The right long-term fix
is to grow the corpus, not to keep nudging the number.

**Part 3 — intent routing is nearest-neighbour over 16 exemplars.** It is
deterministic, keyless and explainable, but it is not a trained classifier. It
routed 2 of 7 policy questions wrongly until the exemplars were rewritten, which
shows how sensitive it is to exemplar phrasing. A larger labelled set and a
proper classifier would be more robust.

**Part 3 — MOCK_LLM cannot reason.** It quotes retrieved text and fills
templates. It cannot synthesise across two policies, resolve a conflict between
them, or handle a compound question ("can I return these shoes *and* how long
will the refund take?") as anything other than the single best-matching rule
plus a list of related documents. That is a deliberate trade: it is why the
system provably cannot hallucinate a policy, and it is also why it cannot answer
a genuinely multi-hop question.

**Part 3 — the order lookup uses the Part 1 dataset as the order book.** That is
honest for a demo (those really are the orders the model was trained and
evaluated on, and the transcripts use held-out test-split orders), but a real
deployment would query an order service, and the feature values would arrive with
production data-quality problems this dataset does not have.

**Scope.** There is a local HTTP API (`backend/api.py`), a React console
(`frontend/`) and a Streamlit UI (`streamlit_app/app.py`), but no
authentication, no rate limiting, no persistent multi-user session store, no
monitoring and no model-drift detection. The API binds to 127.0.0.1 and keeps
conversations in process memory. This is a reproducible academic pipeline plus a local agent and UI,
not a production service — the Streamlit app is single-process and holds one
conversation per browser session in memory, not in a database.
