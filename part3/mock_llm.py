"""Part 3 Task 7 — MOCK_LLM: the deterministic response generator.

This is the default and the only mode any graded transcript uses. It needs zero
API keys and makes zero network calls: given the retrieved chunks or the tool
output, it composes the final structured answer with rules and templates.

Because it can only quote what it was handed, it **cannot** fabricate a policy.
That is the point — the groundedness guardrail decides whether there is
anything to quote, and this module never invents a rule to fill a gap.

Every function returns the fixed schema:
    {"answer": str, "source": str, "confidence": float}
where `source` is one of policy_kb / return_risk_tool / image_classifier_tool.
"""

import zlib

from part3.config import VALID_SOURCES

MODE = "MOCK_LLM"


def _response(answer: str, source: str, confidence: float) -> dict:
    """Build and validate one response object against the fixed schema."""
    assert source in VALID_SOURCES, f"invalid source {source!r}"
    return {
        "answer": " ".join(answer.split()),      # collapse whitespace, keep it one block
        "source": source,
        "confidence": round(float(confidence), 4),
    }


def _pick(query: str, pool: list[str]) -> str:
    """Deterministically rotate through a phrasing pool.

    `zlib.crc32` (not the builtin `hash()`, which is randomised per process
    unless PYTHONHASHSEED is fixed) so the same query always selects the same
    phrasing, which is what keeps MOCK_LLM's determinism guarantee intact
    while still varying the wording across different questions.
    """
    return pool[zlib.crc32(query.encode("utf-8")) % len(pool)]


# A second retrieved document is only quoted as part of the answer when it is
# nearly as good a match as the best one. Measured on the evaluation queries,
# the best-to-second gap is 0.018-0.182: a small gap means the two documents are
# genuinely both about the question, while a large one means the runner-up just
# happens to be the next-closest text in a small corpus. Quoting a distant
# runner-up produced visibly wrong answers in testing — asking about a mobile
# phone's return window pulled in the *home products* window as if it were part
# of the same rule.
CO_QUOTE_MAX_GAP = 0.05

# Varied lead-ins so answers don't all start identically. Selected
# deterministically from the query (see `_pick`), so MOCK_LLM stays
# reproducible: the same question always gets the same lead-in.
_LEAD_INS_DEFAULT = [
    "Here's what applies:",
    "Based on the policy documents I found:",
    "Here's the relevant rule:",
    "For your situation:",
    "Yes — here's what the policy says:",
]
_LEAD_INS_EXPLANATION = [
    "In simple terms:",
    "Here's the short version:",
    "Put simply:",
    "Breaking it down:",
]


def compose_policy_answer(query: str, doc_hits: list[dict], groundedness: dict,
                          fine_intent: str | None = None) -> dict:
    """Answer a policy question by quoting the retrieved chunks.

    Confidence = the best retrieved chunk's cosine similarity. That is a real
    measured quantity (how well the corpus matched the question), not a
    self-assessed number. `fine_intent` only ever changes the lead-in phrasing
    (e.g. a plainer tone for "explanation"-style questions) -- it never changes
    which text is quoted or cited.
    """
    best = doc_hits[0]
    quoted = [best]

    # Co-quote a second document only if it is both above the evidence floor and
    # close behind the best match.
    if len(doc_hits) > 1:
        runner_up = doc_hits[1]
        close_enough = (best["score"] - runner_up["score"]) <= CO_QUOTE_MAX_GAP
        if close_enough and runner_up["score"] >= groundedness["threshold"]:
            quoted.append(runner_up)

    pool = _LEAD_INS_EXPLANATION if fine_intent == "explanation" else _LEAD_INS_DEFAULT
    lead_in = _pick(query, pool)
    answer = lead_in + " " + " ".join(hit["best_chunk_text"] for hit in quoted)
    citation = ", ".join(f"{h['document_title']} ({h['document_id']})" for h in quoted)
    answer += f" [Source: {citation}]"

    # Any other retrieved document that cleared the floor is surfaced as a
    # pointer, never asserted as part of the rule.
    related = [h for h in doc_hits[len(quoted):]
               if h["score"] >= groundedness["threshold"]]
    if related:
        answer += (" [Related policies: "
                   + ", ".join(f"{h['document_title']} ({h['document_id']})"
                               for h in related) + "]")

    return _response(answer, "policy_kb", groundedness["best_score"])


def compose_comparison_answer(query: str, doc_hits: list[dict], groundedness: dict) -> dict:
    """Answer a comparison question by quoting every distinct retrieved
    document side by side, rather than the single-best-match quote used for a
    plain policy question.

    Falls back to `compose_policy_answer` when retrieval only actually
    surfaced one distinct document -- there is nothing to compare, so the
    normal single-quote answer is the honest one.
    """
    if len(doc_hits) < 2:
        return compose_policy_answer(query, doc_hits, groundedness)

    parts = [f"{hit['document_title']} — {hit['best_chunk_text']}" for hit in doc_hits[:4]]
    citation = ", ".join(f"{h['document_title']} ({h['document_id']})" for h in doc_hits[:4])
    answer = "Comparing what's documented: " + "  |  ".join(parts) + f" [Source: {citation}]"
    return _response(answer, "policy_kb", groundedness["best_score"])


def compose_product_answer(product_hits: list[dict]) -> dict:
    """Answer a product-catalog question from the retrieved catalog records.

    Confidence = the best product hit's cosine similarity, the same honest
    convention `compose_policy_answer` uses for policy chunks.
    """
    items = []
    for hit in product_hits[:5]:
        product = hit["product"]
        returnability = ("non-returnable" if product["non_returnable"]
                         else f"{product['return_window']}-day return window")
        items.append(
            f"{product['product_name']} (₹{product['price']}, {returnability}, "
            f"{'exchange available' if product['exchange_available'] else 'no exchange'}, "
            f"{'COD available' if product['cod_available'] else 'no COD'})"
        )
    answer = ("From the product catalog: " + "; ".join(items)
              + f" [Source: product catalog, {len(product_hits)} matching record(s)]")
    return _response(answer, "product_catalog", product_hits[0]["score"])


def compose_ungrounded_refusal(query: str, groundedness: dict) -> dict:
    """Refuse a policy question nothing in the corpus supports.

    Prints the measured similarity against the threshold so the refusal is
    verifiable rather than a bare "I don't know".
    """
    answer = (
        f"I can't answer that from Flipkart's policy knowledge base. The closest "
        f"policy text scored {groundedness['best_score']:.4f} cosine similarity "
        f"against a required minimum of {groundedness['threshold']:.2f}, so no "
        f"document actually covers this question. Rather than guess at a policy, "
        f"I'd rather tell you it isn't documented — please escalate to a human "
        f"agent."
    )
    return _response(answer, "policy_kb", groundedness["best_score"])


def compose_return_risk_answer(tool_result: dict, order_id: int | None) -> dict:
    """Report the Random Forest's prediction and its bucket.

    Confidence = max(p, 1 - p): how decisive the model's own probability is
    about the binary call it just made. It is derived from the model output, not
    invented.
    """
    probability = tool_result["return_probability"]
    bucket = tool_result["risk_bucket"]
    threshold = tool_result["threshold_rf"]
    subject = f"Order {order_id}" if order_id is not None else "This order"

    answer = (
        f"{subject} has a {tool_result['return_probability_percent']:.2f}% "
        f"predicted probability of being returned, which is a "
        f"{bucket.upper()} risk. The buckets are anchored to the model's own "
        f"F1-maximising threshold t*_rf = {threshold:.2f}: Low below "
        f"{threshold:.2f}, Medium from {threshold:.2f} to "
        f"{threshold + 0.15:.2f}, High at or above {threshold + 0.15:.2f}. "
        f"Predicted by the saved tuned Random Forest."
    )
    confidence = max(probability, 1.0 - probability)
    return _response(answer, "return_risk_tool", confidence)


def compose_image_answer(tool_result: dict) -> dict:
    """Report the classifier's category and its softmax confidence."""
    runner_up = tool_result["top3"][1] if len(tool_result["top3"]) > 1 else None
    tail = (
        f" Next most likely was {runner_up['label']} at "
        f"{runner_up['probability'] * 100:.2f}%."
        if runner_up else ""
    )
    answer = (
        f"This product image is classified as "
        f"**{tool_result['predicted_class']}** with "
        f"{tool_result['confidence_percent']:.2f}% confidence.{tail} "
        f"Classified by the saved ResNet-18 transfer-learning model reading the "
        f"pixels of {tool_result['image_path']}."
    )
    return _response(answer, "image_classifier_tool", tool_result["confidence"])


def compose_blocked(injection: dict, lane_source: str) -> dict:
    """Refuse an input that tripped the prompt-injection guardrail.

    Confidence is 1.0 because this is not a judgement call: a regex either
    matched or it did not, and the matched substring is named in the answer.
    """
    names = ", ".join(m["pattern"] for m in injection["matches"])
    quoted = "; ".join(f'"{m["matched_text"]}"' for m in injection["matches"])
    answer = (
        f"I can't act on that. Your message contains an instruction-override "
        f"pattern ({names}) — specifically {quoted} — which my input guardrail "
        f"blocks before any retrieval or model call happens. I'm still here for "
        f"return policies, order return-risk and product-photo categories."
    )
    return _response(answer, lane_source, 1.0)


# Machine feature name -> the plain-English phrase used to ask for it. Keeps
# `compose_missing_input` from asking for "num_previous_returns" verbatim.
FEATURE_PROMPTS = {
    "price_inr": "the order's price",
    "discount_pct": "the discount percentage applied",
    "customer_tenure_days": "how long the customer has had an account (in days)",
    "num_previous_orders": "how many previous orders the customer has placed",
    "num_previous_returns": "how many previous returns the customer has made",
    "delivery_distance_km": "the delivery distance in kilometres",
    "delivery_days": "how many days delivery took",
    "is_weekend_order": "whether the order was placed on a weekend",
    "rating_given": "the rating the customer gave",
    "product_category": "the product category",
    "payment_method": "the payment method used",
}


def _list_naturally(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def compose_missing_input(result: dict, lane_source: str) -> dict:
    """Ask for the input a tool needs instead of guessing at it.

    `result` is a tool_result dict with `status == "missing_input"`. Two
    shapes reach here: `check_return_risk`'s partial-feature-dict report
    (`missing_features`, named individually so the user is only asked for
    what's genuinely still missing -- see part3/slots.py for what free text
    already fills in) and the generic "no order/image at all yet" case
    (`what`/`detail`).
    """
    missing_features = result.get("missing_features")
    if missing_features:
        named = [FEATURE_PROMPTS.get(f, f) for f in missing_features]
        answer = (
            f"I can score this order's return risk once I also know "
            f"{_list_naturally(named)}. I won't estimate any of these myself, "
            f"because a guessed number would look exactly like a number the "
            f"model produced."
        )
    else:
        what = result.get("what", "more information")
        detail = result.get("detail", "")
        answer = (
            f"I need {what} before I can answer that. {detail} I won't estimate it, "
            f"because a number I made up would look exactly like a number the model "
            f"produced."
        )
    return _response(answer, lane_source, 0.0)


# ------------------------------------------------------- conversational lane
# No retrieval, no tool call: greeting/general_help/unsupported never touch
# the policy corpus or a model, so confidence here is the intent classifier's
# own measured exemplar-similarity, never a manufactured number.
def compose_greeting_answer(confidence: float) -> dict:
    answer = (
        "Hi! I'm Flipkart's order-support assistant. I can help with return "
        "policies, refunds, exchanges, delivery, order return-risk, product-photo "
        "categories, and questions about the products in the catalog. What can I "
        "help you with?"
    )
    return _response(answer, "conversational", confidence)


def compose_general_help_answer(confidence: float) -> dict:
    from part3.chunking import load_documents
    from part3.products import load_catalog

    n_docs = len(load_documents())
    n_products = len(load_catalog())
    answer = (
        f"I can look up return, refund, exchange, delivery and cancellation rules "
        f"from {n_docs} policy documents; answer questions about the {n_products} "
        f"products in the catalog (return window, COD eligibility, exchange "
        f"availability); score how likely a specific order is to be returned with "
        f"the saved Random Forest model; and classify a product photo with the "
        f"saved image classifier. Ask in your own words — you don't need a "
        f"particular phrasing."
    )
    return _response(answer, "conversational", confidence)


def compose_unsupported_refusal(query: str, confidence: float) -> dict:
    answer = (
        "I can help with e-commerce support: product policies, delivery, returns, "
        "refunds, exchanges, return-risk analysis and product-photo classification. "
        "I don't have reliable information about that topic, so I'd rather say so "
        "than guess."
    )
    return _response(answer, "conversational", confidence)
