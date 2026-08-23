"""Part 3 Task 5 — the LangGraph conversation state.

This is short-term state that belongs to **one conversation**. It is threaded
explicitly through `graph.invoke()` by the `Conversation` object in
`part3/graph.py`; there is no module-level dict, no global, and no persistent
store anywhere in Part 3. Starting a new `Conversation` therefore starts from a
genuinely empty state — which is what the fresh-conversation transcript
demonstrates.
"""

from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    # --- current turn's input ---
    query: str
    image_path: str | None

    # --- carried across turns within one conversation ---
    turn_index: int
    history: list[dict]              # [{"role": "user"|"assistant", "content": str}]
    order_id: int | None             # remembered so "its return risk" resolves
    order_features: dict | None      # remembered alongside the id, merged with
                                      # free-text slots turn over turn (see
                                      # part3/slots.py)
    last_topic: str | None           # title of the last grounded policy/product
                                      # hit, so "what about if they're damaged?"
                                      # can be resolved against it
    pending_tool: str | None         # a tool asked for missing input last turn
                                      # and is still waiting for it, so a reply
                                      # that supplies it stays in that lane

    # --- produced by the guard node ---
    injection: dict

    # --- produced by the intent node ---
    intent: str                      # the graph LANE: policy/return_risk/
                                      # product_category/conversational
    fine_intent: str                 # finer-grained conversational tag (e.g.
                                      # refund, greeting, comparison) -- not
                                      # carried across turns, recomputed fresh
    intent_evidence: dict            # which few-shot exemplar matched, and how well
    retrieval_query: str             # query used for retrieval, after resolving
                                      # a pronoun/follow-up against last_topic

    # --- produced by the retrieval node ---
    chunk_hits: list[dict]
    doc_hits: list[dict]
    product_hits: list[dict]         # catalog hits, only set for product_lookup
    groundedness: dict

    # --- produced by the tool node ---
    tool_result: dict

    # --- produced by the response node ---
    response: dict                   # {"answer", "source", "confidence"}

    # --- diagnostics ---
    trace: list[str]                 # node execution order, for the transcripts


# Keys that survive from one turn to the next inside a conversation. Everything
# else is per-turn scratch and is cleared, so a stale retrieval or tool result
# can never leak into the next answer.
CARRIED_KEYS = ("turn_index", "history", "order_id", "order_features", "last_topic",
                "pending_tool")


def new_state() -> AgentState:
    """A fresh, empty conversation state."""
    return {
        "turn_index": 0,
        "history": [],
        "order_id": None,
        "order_features": None,
        "last_topic": None,
        "pending_tool": None,
        "trace": [],
    }
