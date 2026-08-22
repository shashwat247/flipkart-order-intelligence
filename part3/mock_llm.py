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


# A second retrieved document is only quoted as part of the answer when it is
# nearly as good a match as the best one. Measured on the evaluation queries,
# the best-to-second gap is 0.018-0.182: a small gap means the two documents are
# genuinely both about the question, while a large one means the runner-up just
# happens to be the next-closest text in a small corpus. Quoting a distant
# runner-up produced visibly wrong answers in testing — asking about a mobile
# phone's return window pulled in the *home products* window as if it were part
# of the same rule.
CO_QUOTE_MAX_GAP = 0.05


def compose_policy_answer(query: str, doc_hits: list[dict], groundedness: dict) -> dict:
    """Answer a policy question by quoting the retrieved chunks.

    Confidence = the best retrieved chunk's cosine similarity. That is a real
    measured quantity (how well the corpus matched the question), not a
    self-assessed number.
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

    answer = " ".join(hit["best_chunk_text"] for hit in quoted)
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


def compose_missing_input(what: str, detail: str, lane_source: str) -> dict:
    """Ask for the input a tool needs instead of guessing at it."""
    answer = (
        f"I need {what} before I can answer that. {detail} I won't estimate it, "
        f"because a number I made up would look exactly like a number the model "
        f"produced."
    )
    return _response(answer, lane_source, 0.0)
