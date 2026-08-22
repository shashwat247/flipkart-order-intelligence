# Part 3 — Agent Transcripts

All 10 transcripts were produced by
`python3 -m part3.run_transcripts` running the real LangGraph agent in
**MOCK_LLM** mode: zero API keys, zero outbound network calls, deterministic
output. Re-running the command reproduces them byte for byte.

| transcript | demonstrates | source | confidence |
|---|---|---|---:|
| [`01_policy_electronics_return_window.txt`](01_policy_electronics_return_window.txt) | Task 9(a) policy via RAG | `policy_kb` | 0.6429 |
| [`02_policy_cod_refund_timeline.txt`](02_policy_cod_refund_timeline.txt) | Task 9(a) policy via RAG | `policy_kb` | 0.7094 |
| [`03_return_risk_high.txt`](03_return_risk_high.txt) | Task 9(b) return-risk tool | `return_risk_tool` | 0.6877 |
| [`04_return_risk_low.txt`](04_return_risk_low.txt) | Extra — return-risk tool | `return_risk_tool` | 0.7228 |
| [`05_product_category_sneaker.txt`](05_product_category_sneaker.txt) | Task 9(c) image classifier | `image_classifier_tool` | 0.9987 |
| [`06_product_category_shirt.txt`](06_product_category_shirt.txt) | Extra — image classifier | `image_classifier_tool` | 0.7941 |
| [`07_multiturn_state_carried.txt`](07_multiturn_state_carried.txt) | Task 9(d) multi-turn state | `return_risk_tool` | 0.6462 |
| [`08_fresh_conversation_state_reset.txt`](08_fresh_conversation_state_reset.txt) | Task 9(d) fresh-conversation reset | `return_risk_tool` | 0.0 |
| [`09_prompt_injection_blocked.txt`](09_prompt_injection_blocked.txt) | Task 9(e) injection blocked | `return_risk_tool` | 1.0 |
| [`10_ungrounded_policy_refused.txt`](10_ungrounded_policy_refused.txt) | Task 9(f) ungrounded refusal | `policy_kb` | 0.4379 |

## Required coverage (Task 9)

| requirement | transcript |
|---|---|
| (a) two different policy questions answered via RAG | 01, 02 |
| (b) return-risk question calling `check_return_risk` | 03 (and 04) |
| (c) product-category question calling `classify_product_image` | 05 (and 06) |
| (d) multi-turn conversation with state carried | 07 |
| (d) matching fresh conversation with state correctly absent | 08 |
| (e) prompt-injection attempt, visibly blocked | 09 |
| (f) ungrounded question, refused with score vs threshold printed | 10 |

## The 4S prompt principles these runs exercise

| **Specific** | The prompt enumerates the three permitted intents by name and states that everything else is out of scope, instead of a vague 'be helpful'. |
| **Short** | A hard three-sentence ceiling with 'lead with the rule or the number', and the prompt itself is kept to the constraints that change behaviour. |
| **Surround** | The evidence constraint is stated before the task ('use ONLY the retrieved chunks') and restated immediately after it ('no evidence, no answer'), so the binding rule brackets the response behaviour rather than trailing off at the end where it is easiest to drift past. |
| **Single** | Each graph node is given one objective and explicitly forbidden the others - the retrieval node must not answer, the response node must not fetch. One job per stage, never a compound instruction. |
| **Role prompting** | Opens with 'You are Flipkart's order-support assistant.', fixing the persona, the domain and the implied register before any instruction is read. |

Every transcript's INTENT NODE block names the few-shot exemplar that drove
routing and the cosine similarity it matched at — which is how you can see the
few-shot examples doing real work rather than sitting decoratively in the
prompt text.
