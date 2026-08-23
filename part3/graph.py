"""Part 3 Task 5 — the LangGraph agent.

Five nodes and one real conditional edge:

    START -> guard_node -> intent_node -> [conditional route_by_intent]
                                            |-- blocked ---------> response_node
                                            |-- policy ---------->  retrieval_node -> response_node
                                            |-- return_risk -----> tool_node ------> response_node
                                            |-- product_category-> tool_node ------> response_node
                                            |-- conversational --> response_node
                                          response_node -> END

The route is genuinely branch-dependent: a policy question never touches the
tool node, a return-risk question never runs retrieval, a blocked input
reaches neither, and ordinary conversation (a greeting, "what can you do?")
never touches retrieval or a tool either.

2026-08-23 flagship-agent upgrade: `classify_intent` still returns exactly one
of the four graph lanes above (the original three plus "conversational") --
every existing caller and test keeps working unchanged. What's new is a
FINE-GRAINED intent underneath it (`classify_fine_intent` / `fine_intent`):
greeting, general_help, unsupported, refund, exchange, delivery,
damaged_product, cancellation, comparison, explanation and product_lookup all
route through the *same* lanes as before (mostly "policy"), but the fine tag
lets retrieval widen for a comparison, search the product catalog for a
product_lookup, and the response generator vary its tone -- without adding a
single "if question contains X" branch. See `LANE_FOR_INTENT` below and
`part3/prompts.py::FEW_SHOT_EXAMPLES` for the exemplars that drive it.
"""

import re
from functools import lru_cache

import numpy as np

from part3.config import (
    INTENT_ROUTING_FLOOR,
    SIMILARITY_THRESHOLD,
    TOP_K,
    USE_LIVE_LLM,
)
from part3.embeddings import embed, embed_one
from part3.guardrails import check_groundedness, detect_injection
from part3.mock_llm import (
    compose_blocked,
    compose_comparison_answer,
    compose_general_help_answer,
    compose_greeting_answer,
    compose_image_answer,
    compose_missing_input,
    compose_policy_answer,
    compose_product_answer,
    compose_return_risk_answer,
    compose_ungrounded_refusal,
    compose_unsupported_refusal,
)
from part3.prompts import FEW_SHOT_EXAMPLES
from part3.slots import extract_order_slots
from part3.state import CARRIED_KEYS, AgentState, new_state
from part3.tools import check_return_risk, classify_product_image, lookup_order

# Which structured source each LANE's answer is attributed to (the
# conversational lane's own compose_* functions set "conversational" directly;
# this is only the fallback used for compose_blocked's source).
INTENT_SOURCE = {
    "policy": "policy_kb",
    "return_risk": "return_risk_tool",
    "product_category": "image_classifier_tool",
    "conversational": "conversational",
}

# Fine-grained conversational intent -> the graph lane it is handled by. Every
# fine intent maps to one of exactly four lanes; nothing downstream ever sees
# a lane string the graph's conditional edge doesn't know how to route.
LANE_FOR_INTENT = {
    "policy": "policy",
    "return_risk": "return_risk",
    "product_category": "product_category",
    "refund": "policy",
    "exchange": "policy",
    "delivery": "policy",
    "damaged_product": "policy",
    "cancellation": "policy",
    "comparison": "policy",
    "explanation": "policy",
    "product_lookup": "policy",
    "greeting": "conversational",
    "general_help": "conversational",
    "unsupported": "conversational",
}

# A short follow-up that leans on the previous turn's topic instead of naming
# a new one ("what about if they're damaged?", "and how long do I have?").
_PRONOUN_RE = re.compile(r"\b(it|its|they|them|their|that|these|those|this one)\b",
                         re.IGNORECASE)


# --------------------------------------------------------------------- intent
@lru_cache(maxsize=1)
def _exemplar_vectors():
    """Embed the few-shot examples once.

    This is what makes the few-shot block load-bearing rather than decorative:
    the exemplars ARE the intent classifier's training set.
    """
    return embed([ex["user"] for ex in FEW_SHOT_EXAMPLES])


def _nearest_exemplar(query: str):
    """Cosine-nearest few-shot exemplar, plus the runner-up ranking."""
    similarities = (_exemplar_vectors() @ embed_one(query).T).ravel()
    order = np.argsort(similarities)[::-1]
    best = int(order[0])
    best_similarity = float(similarities[best])
    ranked = [
        {
            "example": FEW_SHOT_EXAMPLES[int(i)]["user"],
            "fine_intent": FEW_SHOT_EXAMPLES[int(i)]["fine"],
            "similarity": round(float(similarities[int(i)]), 4),
        }
        for i in order[:3]
    ]
    return best, best_similarity, ranked


def classify_fine_intent(query: str) -> tuple[str, str, dict]:
    """Route to the FINE intent of the nearest few-shot exemplar, then map it
    down to the graph lane that actually handles it.

    Deterministic (embedding of a fixed string is fixed) and keyless. Returns
    `(fine_intent, lane, evidence)`.
    """
    best, best_similarity, ranked = _nearest_exemplar(query)
    nearest_fine = FEW_SHOT_EXAMPLES[best]["fine"]

    # Below the floor the nearest exemplar is noise, not evidence. Fall back to
    # the policy lane exactly as before the fine-grained layer existed: it is
    # the only lane with a groundedness check that can honestly refuse, so an
    # unroutable question is refused with a printed score rather than handed
    # confidently to a tool or (now) to the ungrounded conversational lane.
    below_floor = best_similarity < INTENT_ROUTING_FLOOR
    fine_intent = "policy" if below_floor else nearest_fine
    lane = LANE_FOR_INTENT.get(fine_intent, "policy")

    evidence = {
        "method": "nearest few-shot exemplar (cosine similarity over local embeddings)",
        "matched_example": FEW_SHOT_EXAMPLES[best]["user"],
        "matched_intent": nearest_fine,
        "similarity": round(best_similarity, 4),
        "routing_floor": INTENT_ROUTING_FLOOR,
        "below_routing_floor": below_floor,
        "fine_intent": fine_intent,
        "final_intent": lane,
        "runners_up": ranked[1:],
    }
    if below_floor:
        evidence["fallback_reason"] = (
            f"best exemplar similarity {best_similarity:.4f} < floor "
            f"{INTENT_ROUTING_FLOOR}; routed to the guarded policy lane instead of "
            f"'{nearest_fine}'"
        )
    return fine_intent, lane, evidence


def classify_intent(query: str) -> tuple[str, dict]:
    """Backward-compatible entry point: returns the graph LANE (one of
    policy/return_risk/product_category/conversational) plus the routing
    evidence. This is the function the graph's conditional edge and every
    existing caller/test rely on; see `classify_fine_intent` for the richer,
    fine-grained tag this now sits on top of.
    """
    _fine_intent, lane, evidence = classify_fine_intent(query)
    return lane, evidence


def extract_order_id(text: str) -> int | None:
    """Pull an order id out of free text: 'order 4021', 'order #4021', '#4021'."""
    match = re.search(r"\border\s*#?\s*(\d{1,6})\b", text or "", re.IGNORECASE)
    if not match:
        match = re.search(r"#(\d{1,6})\b", text or "")
    return int(match.group(1)) if match else None


def resolve_query_context(query: str, state: AgentState) -> str:
    """Expand a short pronoun-leaning follow-up with the last grounded topic.

    Used for the RETRIEVAL query only -- never for routing and never shown to
    the user -- so it can only widen what RAG searches for, never change what
    the agent claims to have understood or answer with unretrieved text.
    """
    last_topic = state.get("last_topic")
    if not last_topic:
        return query
    is_short_followup = len(query.split()) <= 12 and bool(_PRONOUN_RE.search(query))
    if not is_short_followup:
        return query
    return f"{query} ({last_topic})"


# ---------------------------------------------------------------------- nodes
def guard_node(state: AgentState) -> dict:
    """INPUT GUARDRAIL. Runs first, before retrieval or any model call."""
    injection = detect_injection(state.get("query", ""))
    return {"injection": injection, "trace": state.get("trace", []) + ["guard_node"]}


def intent_node(state: AgentState) -> dict:
    """Classify the intent and resolve conversational references.

    This is also where short-term state does its work: if the message names an
    order, that id is remembered; if it does not, the id remembered from an
    earlier turn in THIS conversation is reused, which is what lets
    "what is its return risk?" resolve on turn 2. Likewise, any order-risk
    features mentioned in free text (price, payment method, category, a
    delivery delay) are merged into the carried partial feature dict, so a
    return-risk score can be built up over several turns without ever naming a
    dataset order id.
    """
    query = state.get("query", "")
    fine_intent, lane, evidence = classify_fine_intent(query)

    mentioned = extract_order_id(query)
    carried = state.get("order_id")
    order_id = mentioned if mentioned is not None else carried
    evidence["order_id_source"] = (
        "mentioned in this turn" if mentioned is not None
        else ("carried from earlier in this conversation" if carried is not None
              else "not available")
    )

    # Look the order up only when the id changed, so a follow-up turn reuses the
    # features already in state rather than re-reading the order history.
    order_features = state.get("order_features")
    if order_id is not None and (mentioned is not None or order_features is None):
        looked_up = lookup_order(order_id)
        if looked_up is not None:
            order_features = looked_up
            evidence["order_lookup"] = f"order {order_id} found in order history"
        else:
            evidence["order_lookup"] = f"order {order_id} not found in order history"

    # Merge any free-text order-feature slots into the carried partial dict.
    # New keys are added; a key already carried is only overwritten when the
    # user actually re-mentions it this turn.
    slots = extract_order_slots(query)
    if slots:
        order_features = {**(order_features or {}), **slots}
        evidence["slots_extracted"] = slots

    # A reply to a clarifying question belongs to the lane that asked it. Without
    # this, "20% off, 300 days, 5 previous orders, rated 4" is classified on its
    # own words — which look nothing like a risk question — and the conversation
    # the agent itself started becomes a dead end. Only an actual answer counts:
    # supplying a feature, or naming an order.
    pending = state.get("pending_tool")
    if pending == "return_risk" and (slots or mentioned is not None):
        if lane != "return_risk":
            evidence["lane_override"] = (
                f"stayed in the return_risk lane (classified as '{lane}') because the "
                "previous turn asked for missing order features and this turn supplies "
                f"{sorted(slots) if slots else 'an order id'}"
            )
        lane, fine_intent = "return_risk", "return_risk"

    retrieval_query = resolve_query_context(query, state)

    return {
        "intent": lane,
        "fine_intent": fine_intent,
        "intent_evidence": evidence,
        "order_id": order_id,
        "order_features": order_features,
        "retrieval_query": retrieval_query,
        "trace": state.get("trace", []) + ["intent_node"],
    }


def route_by_intent(state: AgentState) -> str:
    """THE CONDITIONAL EDGE. Decides which branch of the graph actually runs."""
    if state.get("injection", {}).get("blocked"):
        return "blocked"
    return state.get("intent", "policy")


def retrieval_node(state: AgentState) -> dict:
    """RAG retrieval for policy questions, plus the output-side groundedness
    check. Also the product-catalog search for a `product_lookup` question,
    and a wider document pool for a `comparison` question -- both purely
    additive to the same retrieval call, never a separate graph branch.
    """
    from part3.retrieval import search_chunks, to_documents

    fine_intent = state.get("fine_intent", "policy")
    query = state.get("retrieval_query") or state.get("query", "")

    # A comparison question needs more than the default top-3 chunks to have a
    # real chance of covering more than one topic; every other fine intent
    # behaves exactly as before (same function, same TOP_K).
    chunk_hits = search_chunks(query, top_k=8 if fine_intent == "comparison" else TOP_K)
    doc_hits = to_documents(chunk_hits)
    groundedness = check_groundedness(chunk_hits, SIMILARITY_THRESHOLD)

    result = {
        "chunk_hits": chunk_hits,
        "doc_hits": doc_hits,
        "groundedness": groundedness,
        "trace": state.get("trace", []) + ["retrieval_node"],
    }

    if fine_intent == "product_lookup":
        from part3 import products as products_module

        criteria = products_module.parse_filter_criteria(query)
        filtered = products_module.filter_products(**criteria) if criteria else []
        if filtered:
            # A structured filter match ("which products support COD") is an
            # exact criteria match, not a similarity guess -- score 1.0 is
            # honest here in the same sense compose_blocked's 1.0 is: this is
            # not a judgement call.
            product_hits = [{"kind": "product_filter", "score": 1.0,
                             "document_id": p["product_id"],
                             "document_title": p["product_name"], "product": p}
                            for p in filtered]
        else:
            product_hits = [h for h in products_module.search_products(query, top_k=5)
                           if h["score"] >= SIMILARITY_THRESHOLD]
        if product_hits:
            result["product_hits"] = product_hits

    # Remember what this turn was grounded in, so a short pronoun-leaning
    # follow-up next turn has something real to resolve against.
    if result.get("product_hits"):
        result["last_topic"] = result["product_hits"][0]["document_title"]
    elif groundedness["grounded"] and doc_hits:
        result["last_topic"] = doc_hits[0]["document_title"]

    return result


def tool_node(state: AgentState) -> dict:
    """Call the real saved model for the routed intent."""
    intent = state.get("intent")
    trace = state.get("trace", []) + ["tool_node"]

    if intent == "return_risk":
        features = state.get("order_features")
        if not features:
            return {
                "tool_result": {
                    "status": "missing_input",
                    "what": "an order to score",
                    "detail": "Tell me an order id (for example 'order 4021') or "
                              "describe the order — its category, price, payment "
                              "method and so on.",
                },
                "pending_tool": "return_risk",
                "trace": trace,
            }
        tool_result = check_return_risk(features)
        # Stay "pending" while the model still lacks columns, so the customer's
        # next message — which will be a bare list of the values we just asked
        # for — is read as the answer to this question and not re-classified
        # from scratch.
        pending = "return_risk" if tool_result.get("status") == "missing_input" else None
        return {"tool_result": tool_result, "pending_tool": pending, "trace": trace}

    image_path = state.get("image_path")
    if not image_path:
        return {
            "tool_result": {
                "status": "missing_input",
                "what": "a product image to classify",
                "detail": "Point me at a PNG file, for example "
                          "data/sample_images/07_sneaker.png.",
            },
            "trace": trace,
        }
    try:
        result = classify_product_image(image_path)
        result["status"] = "ok"
    except FileNotFoundError as exc:
        result = {"status": "missing_input", "what": "a readable image file",
                  "detail": str(exc)}
    return {"tool_result": result, "trace": trace}


def response_node(state: AgentState) -> dict:
    """Compose the final structured JSON answer.

    In MOCK_LLM mode (the default, and the mode every graded transcript uses)
    this is pure rule-based templating over what the earlier nodes produced —
    deterministic, keyless, and incapable of inventing a policy.
    """
    intent = state.get("intent", "policy")
    fine_intent = state.get("fine_intent", intent)
    lane_source = INTENT_SOURCE.get(intent, "policy_kb")

    if state.get("injection", {}).get("blocked"):
        response = compose_blocked(state["injection"], lane_source)

    elif intent == "conversational":
        # Ordinary conversation never touches retrieval or a tool. Confidence
        # is the intent classifier's own measured exemplar similarity — a real
        # number, never a manufactured one.
        similarity = state.get("intent_evidence", {}).get("similarity", 0.0)
        if fine_intent == "greeting":
            response = compose_greeting_answer(similarity)
        elif fine_intent == "general_help":
            response = compose_general_help_answer(similarity)
        else:
            response = compose_unsupported_refusal(state.get("query", ""), similarity)

    elif intent == "policy":
        groundedness = state.get("groundedness", {"grounded": False, "best_score": 0.0,
                                                  "threshold": SIMILARITY_THRESHOLD})
        product_hits = state.get("product_hits")
        doc_hits = state.get("doc_hits")
        if product_hits:
            response = compose_product_answer(product_hits)
        elif groundedness["grounded"] and doc_hits:
            if fine_intent == "comparison":
                response = compose_comparison_answer(state["query"], doc_hits, groundedness)
            else:
                response = compose_policy_answer(state["query"], doc_hits, groundedness,
                                                 fine_intent=fine_intent)
        else:
            response = compose_ungrounded_refusal(state["query"], groundedness)

    else:
        result = state.get("tool_result", {})
        if result.get("status") == "missing_input":
            response = compose_missing_input(result, lane_source)
        elif intent == "return_risk":
            response = compose_return_risk_answer(result, state.get("order_id"))
        else:
            response = compose_image_answer(result)

    if USE_LIVE_LLM:
        # Optional, never required, never used by a graded transcript. The mock
        # answer above is always computed first and is the fallback.
        from part3.live_llm import polish

        response = polish(state, response)

    history = state.get("history", []) + [
        {"role": "user", "content": state.get("query", "")},
        {"role": "assistant", "content": response["answer"]},
    ]
    return {
        "response": response,
        "history": history,
        "turn_index": state.get("turn_index", 0) + 1,
        "trace": state.get("trace", []) + ["response_node"],
    }


# ---------------------------------------------------------------------- graph
@lru_cache(maxsize=1)
def build_graph():
    """Compile the agent graph (once per process)."""
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(AgentState)
    builder.add_node("guard_node", guard_node)
    builder.add_node("intent_node", intent_node)
    builder.add_node("retrieval_node", retrieval_node)
    builder.add_node("tool_node", tool_node)
    builder.add_node("response_node", response_node)

    builder.add_edge(START, "guard_node")
    builder.add_edge("guard_node", "intent_node")

    # The conditional edge: intent (or a blocked input) selects the branch.
    builder.add_conditional_edges(
        "intent_node",
        route_by_intent,
        {
            "blocked": "response_node",
            "policy": "retrieval_node",
            "return_risk": "tool_node",
            "product_category": "tool_node",
            "conversational": "response_node",
        },
    )
    builder.add_edge("retrieval_node", "response_node")
    builder.add_edge("tool_node", "response_node")
    builder.add_edge("response_node", END)
    return builder.compile()


class Conversation:
    """One conversation. Holds its own state; shares nothing with any other.

    Short-term state lives on this instance and is threaded into each
    `graph.invoke()` call. Two Conversation objects cannot see each other's
    order ids, and a newly constructed one starts empty — no globals, no
    checkpoint store, no cross-talk.
    """

    def __init__(self, name: str = "conversation"):
        self.name = name
        self.graph = build_graph()
        self.state: AgentState = new_state()

    def ask(self, query: str, image_path: str | None = None) -> AgentState:
        # Only the carried keys survive; per-turn scratch (retrieval hits, tool
        # results, traces) starts clean so nothing stale can leak forward.
        turn_input: AgentState = {k: self.state.get(k) for k in CARRIED_KEYS}
        turn_input["query"] = query
        turn_input["image_path"] = image_path
        turn_input["trace"] = []

        result = self.graph.invoke(turn_input)
        self.state = result
        return result

    def answer(self, query: str, image_path: str | None = None) -> dict:
        """Just the structured JSON response."""
        return self.ask(query, image_path=image_path)["response"]


def run_once(query: str, image_path: str | None = None) -> AgentState:
    """Single-turn convenience wrapper — a fresh conversation every call."""
    return Conversation("single-turn").ask(query, image_path=image_path)
