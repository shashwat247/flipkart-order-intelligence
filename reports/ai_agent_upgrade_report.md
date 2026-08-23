# AI agent upgrade report — Part 3 flagship support agent

Written 2026-08-23. Every number below was re-measured against the live code
at the time of writing (`python3 -m part3.eval_agent_upgrade`,
`python3 validate_project.py`, `pytest`) — none of it is asserted from memory.

## 1. Previous architecture

`part3/graph.py` already used an embedding-based nearest-few-shot-exemplar
classifier (`classify_intent`), not a keyword or exact-string match, routing
into exactly three lanes: `policy` (→ FAISS retrieval over 15 policy
documents), `return_risk` (→ the saved tuned Random Forest), and
`product_category` (→ the saved ResNet-18 classifier). The Streamlit chat
(`streamlit_app/app.py`) was already a free-text `st.chat_input`, not a fixed-menu
picker.

**What was real about the "feels predefined" complaint:** with only three
intents, a greeting or a "what can you do?" fell below the intent-routing
floor, was forced into the `policy` lane by the fallback rule, found nothing
in the policy corpus, and was refused with the same wording used for a
genuinely out-of-scope question — reading exactly like a rigid
if/else refusal even though the routing underneath it was already semantic.
There was also no product-level knowledge base, no slot memory beyond an
`order_id`, and no pronoun/follow-up resolution.

## 2. New architecture

```
USER MESSAGE
   |
guard_node (prompt-injection regex, unchanged)
   |
intent_node
   |-- classify_fine_intent(query): nearest-exemplar match over 50 few-shot
   |     exemplars covering 14 fine-grained intents, mapped down through
   |     LANE_FOR_INTENT onto one of 4 graph lanes
   |-- extract_order_id (unchanged) + extract_order_slots (NEW: price,
   |     payment method, category, delivery delay from free text, merged
   |     into the carried order_features dict across turns)
   |-- resolve_query_context (NEW: pronoun/follow-up expansion of the
   |     RETRIEVAL query only, against last_topic)
   |
[conditional edge on the LANE]
   |-- blocked          -> response_node
   |-- conversational    (NEW) -> response_node   (no retrieval, no tool)
   |-- policy            -> retrieval_node -> response_node
   |-- return_risk       -> tool_node      -> response_node
   |-- product_category  -> tool_node      -> response_node
   |
response_node: MOCK_LLM composes the final {"answer","source","confidence"}
   from whatever evidence the earlier nodes actually produced
```

The three original lanes and their tests are byte-for-byte unchanged in
behaviour for the original test queries — `classify_intent(query)` still
returns exactly one of `policy` / `return_risk` / `product_category` for the
original few-shot exemplars, and a below-routing-floor question still falls
back to `policy`, not the new `conversational` lane. What's new is layered on
top, not swapped in underneath.

## 3. Fine-grained intent detection

14 fine intents (`policy`, `return_risk`, `product_category`, `greeting`,
`general_help`, `unsupported`, `refund`, `exchange`, `delivery`,
`damaged_product`, `cancellation`, `comparison`, `explanation`,
`product_lookup`) are driven by 50 embedded few-shot exemplars
(`part3/prompts.py::FEW_SHOT_EXAMPLES`) — nearest-neighbour cosine similarity
over local `all-MiniLM-L6-v2` embeddings, exactly the same mechanism the
original three intents used, just with more exemplars. `LANE_FOR_INTENT`
(`part3/graph.py`) maps every fine intent onto one of the four graph lanes.
There is no keyword/if-contains routing anywhere in this layer.

## 4. Product catalog (new knowledge layer)

`part3/knowledge_base/products.json`: 54 hand-authored, realistic synthetic
products across Apparel (11), Footwear (10), Electronics (15), Home (10) and
Beauty (8) — the exact five categorical levels Part 1's Random Forest was
trained on. Each record carries `product_id, product_name, category,
subcategory, price, return_window, exchange_available, cod_available,
delivery_sla, non_returnable, warranty, description`.

`part3/products.py` provides two retrieval paths, both exercised by
`retrieval_node` when the fine intent is `product_lookup`:
- **Structured filter** (`filter_products` / `parse_filter_criteria`) for
  "which products..." questions (COD-eligible, non-returnable, exchange,
  a return-window cutoff, a category) — a filter question isn't answered
  well by cosine similarity (no single product's *description* is "about"
  the filter), so this path is tried first.
- **Semantic search** (`search_products`, brute-force cosine over the same
  embedding model used everywhere else) for a question naming a specific
  product ("are headphones returnable?").

## 5. Conversation memory

- `order_features` (already existed for a looked-up order id) now also
  accumulates free-text slots turn over turn via `part3/slots.py`: price,
  payment method (mapped to the exact Part-1 categorical levels COD /
  Prepaid_Card / Prepaid_UPI / Wallet), delivery delay in days, and product
  category (matched against the real category/subcategory vocabulary from
  the catalog). `check_return_risk`'s existing `missing_features` reporting
  is reused, not reimplemented, so the agent asks by name for exactly what's
  still missing (see `compose_missing_input`'s `FEATURE_PROMPTS`).
- `last_topic` (new carried state key) remembers the title of the last
  grounded policy/product hit. `resolve_query_context` expands a short,
  pronoun-leaning follow-up ("what about if they're damaged?") with it —
  for the retrieval query only, never for routing or for the displayed
  answer.

## 6. MOCK_LLM upgrades

Still fully offline, zero API keys, zero network calls, deterministic (same
query -> same answer, verified by `test_mock_llm_is_deterministic`). New
compose functions: `compose_greeting_answer`, `compose_general_help_answer`
(reports real, live-counted numbers — policy doc count, catalog size — never
invented), `compose_unsupported_refusal` (a warm, specific decline, not the
policy-corpus refusal wording), `compose_product_answer`,
`compose_comparison_answer` (multi-document, only used when retrieval
actually surfaced 2+ distinct documents), and a deterministic (CRC32-of-query,
not the randomised builtin `hash()`) lead-in rotation for
`compose_policy_answer` so answers don't all start with the same sentence.

## 7. Live LLM

Unchanged (`part3/live_llm.py`): still optional, still only rephrases the
already-computed MOCK_LLM answer's wording (never its `source`/`confidence`),
still falls back to the deterministic answer on any failure or missing key.

## 8. Test results (measured, not asserted)

```
$ python3 -m pytest -q
131 passed
```
48 original Part 3 tests (`tests/test_part3.py`) + 15 new flexibility tests
(`tests/test_chatbot_flexibility.py`, unseen-question/typo/greeting/
product-catalog/comparison/slot-accumulation coverage) + the original Part
1/2/frontend suites, all green.

```
$ python3 validate_project.py
TOTAL  88/88 passed
```
(81 original checks + 7 new: catalog size >= 50, catalog categories are a
subset of Part 1's real levels, a greeting is answered conversationally
rather than refused, an unseen natural-language question is still grounded,
a product-catalog question is answered from catalog records, and both new
reports exist.)

## 9. Unseen-question evaluation

`reports/chatbot_upgrade_evaluation.md` — 22 single-turn queries plus 3
multi-turn scenarios, none of them present in the few-shot exemplars or the
original demo/transcript set, run live and categorized from the actual
result fields. 0 queries categorized `incorrect`. One category worth reading
in full: `safe_refusal` — every refusal in that run is the honest,
similarity-and-threshold-printing kind, never a fabricated policy.

## 10. Known, honestly-reported limitation

Retrieval quality (sentence-level chunking + `all-MiniLM-L6-v2`) is a
pre-existing property this upgrade did not touch. A handful of short queries
retrieve a *neighbouring* category's sentence marginally ahead of the correct
one (e.g. "return window for shoes" can retrieve the electronics-window
sentence about 0.002 cosine similarity ahead of the footwear one). The agent
still only ever quotes and cites what it actually retrieved, so the failure
mode is "occasionally cites the wrong adjacent rule," never "invents one" —
documented here rather than silently left for the reader to discover.

Also deliberately out of scope for this pass (agent-architecture upgrade, not
UI polish): a fake token-streaming animation, copy/regenerate buttons, and a
dedicated product-catalog browser page. The chat already surfaces retrieved
product records and policy evidence in the existing "Technical details"
expander.

## 11. Assignment-requirement compliance

Nothing enumerated in the original brief was removed: the 6000-row Part 1
dataset and its exact formulas/thresholds, `DummyClassifier`/Logistic
Regression/Random Forest + GridSearchCV + `t*_rf`, Part 2's Fashion-MNIST
transfer-learning CNN and confusion matrix, the sentence-chunked policy KB
with FAISS + Precision@3/Recall@3, LangGraph's five nodes and real
conditional edge, MOCK_LLM, both guardrails, the transcripts and tests. The
product catalog and fine-grained intent layer are additive knowledge/routing
layers on top of that same graph, not a replacement for it.

## 12. How to run everything

```bash
# Launch the app
streamlit run streamlit_app/app.py

# Or the CLI agent directly
python3 -m part3.agent --demo
python3 -m part3.agent --ask "Which products support cash on delivery?"

# Tests
pytest -q

# Full acceptance gate (81 original + 4 new checks, runs pytest internally too)
python3 validate_project.py

# Regenerate the unseen-question evaluation report
python3 -m part3.eval_agent_upgrade
```

## 13. Viva demonstration guidance

1. Open the **AI Support** page and type something never shown in any demo
   list — e.g. "my order came late what can i do" or "which products are
   non-returnable" — and point out the "🧭 Understood as..." line: it names
   the real fine-grained intent and what actually ran, not a canned label.
2. Ask a greeting ("hi") immediately after, to show the same session refuses
   nothing and doesn't reuse the policy-refusal wording.
3. Describe an order across two turns without ever giving an order id
   ("I bought running shoes for ₹4,500 using COD", then "it arrived 6 days
   late, what's the risk?") and open "Show raw state" to show the
   accumulated `order_features` dict.
4. Ask a comparison question ("compare footwear and electronics returns") and
   expand "Technical details" to show 2+ distinct cited documents.
5. Run the injection example from the sample chips to show the guardrail
   still blocks before any retrieval/tool call, unchanged.
6. Point to `reports/chatbot_upgrade_evaluation.md` and `validate_project.py`
   output as the evidence trail — every claim above is re-derivable live.
