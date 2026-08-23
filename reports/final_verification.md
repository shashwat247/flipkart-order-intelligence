# Final Verification — Flipkart Order Intelligence & Support Assistant

This is the closing verification pass: every number below was produced by
running the project in this audit session (`pytest`, `python3
validate_project.py`, direct Python calls against the saved artifacts, and a
live-launched instance of the Streamlit frontend), not copied from earlier
documentation. See `reports/final_audit.md` for the requirement-by-requirement
PASS/FAIL table with file-level citations.

---

## Part 1 — Return-risk pipeline

| metric | value |
|---|---:|
| dataset size | 6,000 rows x 13 columns |
| overall return rate | 0.2275 (22.75%) |
| missing `rating_given` | 0.1305 (13.05%) |
| COD missing-rating rate / non-COD | 22.83% / 6.06% (gap 16.77 pp) |
| Baseline (`DummyClassifier`) accuracy / class-1 F1 / class-1 recall | 0.7725 / 0.0000 / 0.0000 |
| Logistic Regression @0.50: accuracy / precision / recall / F1 / ROC-AUC | 0.5917 / 0.2964 / 0.5788 / 0.3921 / 0.6253 |
| t\*_logistic (F1-maximising) | 0.44 |
| Random Forest best CV ROC-AUC | 0.6193 |
| Random Forest held-out test ROC-AUC | 0.6203 (gap to CV: 0.0011) |
| **t\*_rf** (re-derived live from the saved model's own `predict_proba`) | **0.50** |
| F1 / precision / recall at t\*_rf | 0.4076 / 0.3240 / 0.5495 |
| Top-5 features (impurity) | `payment_method` .1788, `price_inr` .1323, `delivery_distance_km` .0957, `customer_tenure_days` .0900, `delivery_days` .0884 |
| Top permutation feature | `payment_method` (+0.0980 ROC-AUC drop — ~10x the next feature) |
| Weakest subgroup | `payment_method = Prepaid_Card`: recall 0.0204 vs. overall 0.5495 (−52.9 pp), n=283 |

## Part 2 — Product image classifier

| metric | value |
|---|---:|
| train / validation / test sizes | 50,000 / 10,000 (stratified) / 10,000 |
| validation accuracy (feature extraction) | 0.8925 |
| fine-tuning | not triggered (0.8925 already clears the 0.80 bar); path is implemented and conditional |
| final validation accuracy | 0.8925 (unchanged — fine-tuning never ran) |
| **final test accuracy** | **0.8872** (recomputed live from the confusion-matrix diagonal: 8,872 / 10,000) |
| worst per-class F1 | Shirt: precision 0.6618, recall 0.7300, F1 0.6942 |
| high-confusion pairs (read off the real matrix) | Shirt <-> T-shirt/top (223 misclassifications), Shirt <-> Coat (195) |

## Part 3 — Support agent

| metric | value |
|---|---:|
| policy documents | 15 (POL01-POL15) |
| chunks (sentence-wise) | 45 |
| embedding model | `all-MiniLM-L6-v2` (local, sentence-transformers) |
| vector index | FAISS `IndexFlatIP` at `data/policy_index/policy.faiss` |
| **Precision@3** (live re-run, this session) | **0.4286** |
| **Recall@3** (live re-run, this session) | **0.9286** |
| queries evaluated | 7 |
| groundedness threshold | 0.45 (in-domain floor 0.4584, out-of-domain ceiling 0.4379) |
| transcripts | 10 (>= 8 required); all headers confirm `LLM mode: MOCK_LLM` |
| guardrail tests | 6 injection patterns + 4 benign non-triggers, all passing |
| state tests | multi-turn carry, fresh-conversation reset, cross-conversation isolation, per-turn scratch isolation — all passing |
| tool tests | `check_return_risk` verified equal to a direct `model.predict_proba` call; `classify_product_image` verified against direct call — both passing |

---

## Frontend — Flipkart Intelligence & Support Center

- **Framework:** Streamlit 1.62.0 (chosen per the task's stated preference).
  A lightweight, dependency-free `webapp.py` (stdlib `http.server`) remains
  available as a zero-install fallback.
- **Pages (7):** Dashboard, Support Assistant, Policy Assistant, Return-Risk
  Analyzer, Product Image Analyzer, Knowledge Base Explorer, Model Insights.
- **Backend integrations:** every page calls a real Part 1/2/3 function —
  `check_return_risk`, `classify_product_image`, `part3.graph.run_once` /
  `Conversation`, `part3.retrieval.search_documents`,
  `part3.evaluate_retrieval.score_query`. Verified live (this session):
  `analyze_risk()` output is byte-identical to `check_return_risk()` called
  directly; `classify_uploaded_bytes()` output matches
  `classify_product_image()` called on the same file directly;
  `ask_policy()` output matches `run_once()` called directly.
- **Offline / MOCK_LLM support:** the sidebar always shows the live `AI Mode`
  badge (`Local / MOCK_LLM` unless `USE_LIVE_LLM=1`); every workflow was
  exercised in this session with `USE_LIVE_LLM` unset and zero API keys.
- **Live launch verification (this session):**
  ```
  streamlit run streamlit_app/app.py --server.headless true --server.port 8501
  curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:8501   ->  HTTP 200
  ```
  The running instance was then driven with a real headless Chrome (via the
  Chrome DevTools Protocol) and screenshotted. The Dashboard rendered as a
  fully styled dark-theme page: sidebar navigation with branding, six
  green **READY** status cards (matching `get_system_status()` exactly —
  Policy Knowledge Base, Vector Index, Return Risk Engine, Product Image AI,
  Support Agent, AI Mode), headline metrics (ROC-AUC 0.6203, t\*_rf 0.50,
  image accuracy 0.8925, 15 documents indexed), and the architecture diagram.
  The Return-Risk Analyzer rendered its full real-feature form (category,
  payment method, price, discount slider, tenure, previous orders/returns,
  delivery distance/days, weekend checkbox, rating slider) with no
  overlapping text or unstyled elements. No visual defects, blank pages, or
  on-screen error messages were observed in any captured page.
- **Error handling:** every backend call is wrapped so a missing artifact,
  unreadable image, empty question, or retrieval failure surfaces as a plain
  `st.error(...)`, verified live by `test_status_checks_report_not_ready_instead_of_raising`,
  `test_classify_uploaded_bytes_raises_a_catchable_error_for_garbage_input`,
  and `test_get_system_status_never_raises_even_if_a_component_is_broken`.
- **Security:** upload restricted to `png/jpg/jpeg`; uploaded bytes are
  written to an OS-managed `tempfile.mkstemp` file with a suffix forced into
  the allow-list regardless of what the client claims, and the temp file is
  always deleted (`finally`) whether classification succeeds or raises. No
  environment variables, API keys, or filesystem paths beyond what a normal
  user needs are ever rendered.

### Performance (measured this session, Apple Silicon, local process)

| operation | cold (first call) | warm (steady-state avg) |
|---|---:|---:|
| Part 1/2/3 module import | 2.43 s | — |
| LangGraph compile | 0.35 s | — |
| Return-risk inference | 0.135 s | 0.017 s |
| Image classifier inference | 0.570 s | 0.007 s |
| Policy retrieval (embeds + FAISS search) | 8.385 s (one-time sentence-transformer load) | 0.010 s |
| Streamlit server boot to first HTTP 200 | ~6 s | — |

The 8.4 s cold retrieval figure is the one-time `SentenceTransformer` weight
load; both `part3/embeddings.get_encoder()` and every model loader in
`part1`/`part2`/`part3` are `@lru_cache`d, so every workflow after the first
one in a running process is fast (all warm figures above are single-digit
milliseconds to tens of milliseconds). No additional caching was needed.

---

## Testing

| suite | result |
|---|---|
| `pytest` (114 tests: 88 Part 1-3 + 26 frontend) | **114 passed** |
| `python3 validate_project.py` (81 acceptance checks: 71 Part 1-3/repo + 10 frontend) | **81/81 passed** |
| Frontend: live launch + headless-Chrome screenshot walkthrough | passed — see Frontend section above |

---

## Remaining issues (stated honestly)

These are pre-existing, already-documented limitations of the pipeline
itself, not defects introduced or left unfixed by this audit:

- **Part 1 ceiling is the data, not the model.** Test ROC-AUC is 0.6203;
  permutation importance shows most features carry little signal beyond
  `payment_method`. More model capacity will not move this number.
- **Part 1 subgroup fix is unvalidated.** The proposed `Prepaid_Card`
  threshold (0.42) was fitted and reported on the same held-out split — an
  optimistic in-sample estimate that needs cross-validated refitting before
  any real deployment.
- **Part 1/2 are synthetic/benchmark data.** `orders_dataset.csv` is a seeded
  generator, not real order history; Fashion-MNIST is not a real Flipkart
  product catalogue (28x28 greyscale vs. real colour catalogue photography).
- **Part 2's 88.72% accuracy is bounded by 28x28 source resolution**, not by
  architecture — the Shirt class's collar/placket detail was never captured
  at that resolution, and upsampling to 224x224 cannot invent it back.
- **Part 3's groundedness margin is narrow** (0.0205 between the lowest
  in-domain and highest out-of-domain score) and would need recalibrating if
  the knowledge base grows.
- **Part 3's intent router is nearest-neighbour over 16 hand-written
  exemplars**, not a trained classifier — deterministic and explainable, but
  sensitive to how the exemplars are phrased.
- **MOCK_LLM cannot reason across policies** — it quotes retrieved text and
  fills templates, so a genuinely multi-hop question gets the single
  best-matching rule plus related-document pointers, not a synthesised
  answer. This is a deliberate trade against hallucination, not an oversight.
- **Frontend scope.** Single-process, single-conversation-per-browser-session
  Streamlit app with in-memory state only — no database, no auth, no
  multi-user session isolation beyond Streamlit's own per-browser
  `session_state`. That matches the task's explicit instruction not to add
  unnecessary databases/authentication, and is appropriate for a local
  academic demo, not a production deployment.

---

## Requirement table

| Requirement | Status | Evidence |
|---|---|---|
| Part 1 dataset | PASS | 6,000 rows x 13 cols, seed 42, exact category/payment lists and probabilities verified against `generate_orders.py`; byte-identical regeneration test passes |
| Part 1 preprocessing | PASS | Split-then-fit `ColumnTransformer` inside a `Pipeline`, `order_id` excluded, no leakage |
| Baseline | PASS | `DummyClassifier(strategy="most_frequent")`: accuracy 0.7725, class-1 F1/recall 0.0000, high-accuracy/zero-recall trap explained |
| Logistic Regression | PASS | `class_weight="balanced"`, full 0.10-0.90 step-0.02 sweep, t\*_logistic=0.44 genuinely F1-maximising, business trade-off explained |
| Random Forest | PASS | `class_weight="balanced", random_state=42`, `GridSearchCV` over the exact required grid, 5-fold `StratifiedKFold`, `scoring="roc_auc"`; best CV 0.6193, test 0.6203 |
| t\*_rf | PASS | Independently re-derived live from the SAVED model's own `predict_proba`; equals 0.50; provably not the LogReg threshold, not a fixed 0.3/0.6, not hand-picked |
| Feature importance | PASS | Impurity top-5 + permutation importance side-by-side, bias explanation grounded in the generator's own formula |
| Subgroup analysis | PASS | Precision/recall by category and payment method with sample counts; `Prepaid_Card` identified as weakest with a concrete, caveated intervention |
| Part 2 dataset | PASS | Real Fashion-MNIST, 50k/10k stratified-val/10k test, test set untouched until final eval |
| Transfer learning | PASS | Frozen ResNet-18 backbone, new `Linear(512,10)` head, 3-channel replication, 224x224, ImageNet normalization, Adam, cached features, conditional fine-tuning implemented and correctly not triggered |
| Confusion matrix | PASS | Real 10x10 matrix from real predictions, rows sum to 1,000, 2 high-confusion pairs read directly off it |
| Saved image model | PASS | `.pt` state_dict reloads via the documented snippet and produces correct real-time inference |
| Part 3 knowledge base | PASS | 15 documents covering all required topics, sentence-wise chunking, parent-doc pointers on every chunk |
| RAG | PASS | Local `all-MiniLM-L6-v2` + FAISS `IndexFlatIP`, real query->embed->search->dedup->response flow traced live |
| LangGraph | PASS | 5 nodes, genuine conditional routing across all 3 intents + blocked path, every branch executed live |
| Tools | PASS | `check_return_risk`/`classify_product_image` verified to match saved artifacts called directly, no hardcoding |
| MOCK_LLM | PASS | Default and only mode used by any transcript; zero-network-call test passes; deterministic |
| Guardrails | PASS | Input-side injection blocking (13 patterns, tested against attacks and benign text) and output-side groundedness refusal (similarity + threshold printed) both verified live |
| Conversation state | PASS | Multi-turn carry and fresh-conversation reset both verified live in this session |
| Retrieval evaluation | PASS | 7 query/doc pairs, document-level P@3/R@3, live re-run reproduces README's 0.4286/0.9286 exactly |
| Transcripts | PASS | 10 transcripts (>= 8), all required categories present, all MOCK_LLM |
| Git requirements | PASS | Feature branch with 4 commits merged into main via `--no-ff`, re-verified live via `git rev-list` |
| Frontend | PASS | Streamlit app, 7 pages, all real backend integrations, live-launched and screenshot-verified, offline/MOCK_LLM by default |
| Automated tests | PASS | 114/114 pytest, 81/81 `validate_project.py` acceptance checks |
