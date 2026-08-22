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
    order_features: dict | None      # remembered alongside the id

    # --- produced by the guard node ---
    injection: dict

    # --- produced by the intent node ---
    intent: str
    intent_evidence: dict            # which few-shot exemplar matched, and how well

    # --- produced by the retrieval node ---
    chunk_hits: list[dict]
    doc_hits: list[dict]
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
CARRIED_KEYS = ("turn_index", "history", "order_id", "order_features")


def new_state() -> AgentState:
    """A fresh, empty conversation state."""
    return {
        "turn_index": 0,
        "history": [],
        "order_id": None,
        "order_features": None,
        "trace": [],
    }
