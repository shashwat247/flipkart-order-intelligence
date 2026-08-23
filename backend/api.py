"""FastAPI application exposing the real Part 3 agent over HTTP.

The React console in `frontend/` is the primary user interface and talks to
this API. Every route below calls the actual LangGraph agent or the actual
saved model tools; none of them contains a lookup table of expected questions.

    python3 -m backend.api            # http://127.0.0.1:8000
    uvicorn backend.api:app --reload  # same thing, with auto-reload

MOCK_LLM stays the default: no API key is read, and no outbound network call is
made while answering.
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from part3.config import SIMILARITY_THRESHOLD, USE_LIVE_LLM
from part3.graph import Conversation

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data" / "sample_images"

# One Conversation per conversation id. Short-term state lives on the
# Conversation object itself (see part3/graph.py), so two ids can never see
# each other's remembered order — the same isolation the CLI relies on.
_conversations: dict[str, Conversation] = {}
_lock = threading.Lock()

app = FastAPI(
    title="Flipkart Order Intelligence API",
    description="HTTP access to the real Part 1/2/3 artifacts and the LangGraph support agent.",
    version="1.0.0",
)

# The Vite dev server runs on a different origin during development. The build
# is served from the same origin, so this only matters for `npm run dev`.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------- request models
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None
    image: str | None = None  # basename of a file in data/sample_images/


class RiskRequest(BaseModel):
    order_id: int | None = None
    order_features: dict[str, Any] | None = None


class ClassifyRequest(BaseModel):
    image: str  # basename of a file in data/sample_images/


# ---------------------------------------------------------------------- helpers
def _resolve_sample(name: str) -> Path:
    """Basename-only resolution: a request can never escape the sample dir."""
    path = SAMPLE_DIR / Path(name).name
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"sample image not found: {name}")
    # Hand the agent the repo-relative path the CLI and the transcripts use, so
    # the answer it composes cites "data/sample_images/..." rather than this
    # machine's home directory.
    return path.relative_to(ROOT)


def _conversation(conversation_id: str | None) -> tuple[str, Conversation]:
    with _lock:
        if conversation_id and conversation_id in _conversations:
            return conversation_id, _conversations[conversation_id]
        new_id = conversation_id or uuid.uuid4().hex[:12]
        _conversations[new_id] = Conversation(f"api-{new_id}")
        return new_id, _conversations[new_id]


def _serialise(result: dict, conversation_id: str) -> dict:
    """The full turn, exactly as the graph produced it.

    Every field is read off the agent's own state — the trace, the groundedness
    scores, the retrieved documents and the tool result are the real ones, so
    the UI can show evidence rather than assert it.
    """
    return {
        "conversation_id": conversation_id,
        "response": result.get("response"),
        "intent": result.get("intent"),
        "fine_intent": result.get("fine_intent"),
        "intent_evidence": result.get("intent_evidence"),
        "trace": result.get("trace", []),
        "groundedness": result.get("groundedness"),
        "doc_hits": result.get("doc_hits", []),
        "chunk_hits": result.get("chunk_hits", []),
        "product_hits": result.get("product_hits", []),
        "tool_result": result.get("tool_result"),
        "injection": result.get("injection"),
        "order_id": result.get("order_id"),
        "turn_index": result.get("turn_index", 0),
        "similarity_threshold": SIMILARITY_THRESHOLD,
    }


# ----------------------------------------------------------------------- routes
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "mode": "USE_LIVE_LLM" if USE_LIVE_LLM else "MOCK_LLM"}


@app.get("/api/status")
def status() -> dict:
    """Live component readiness, computed by the same code the Streamlit app uses."""
    from streamlit_app.app import get_system_status

    rows = get_system_status()
    return {
        "components": rows,
        "ready": sum(1 for r in rows if r["ready"]),
        "total": len(rows),
        "mode": "USE_LIVE_LLM" if USE_LIVE_LLM else "MOCK_LLM",
    }


@app.get("/api/samples")
def samples() -> dict:
    return {"images": sorted(p.name for p in SAMPLE_DIR.glob("*.png"))}


@app.get("/api/policies")
def policies() -> dict:
    """The real knowledge base, straight from disk."""
    from part3.chunking import build_chunks, load_documents

    documents = load_documents()
    chunks = build_chunks(documents)
    return {
        "documents": [{"id": d["id"], "title": d["title"], "text": d["text"]} for d in documents],
        "n_documents": len(documents),
        "n_chunks": len(chunks),
    }


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    """One turn through the real LangGraph agent.

    Arbitrary natural language in; whatever the graph decides comes out. The
    route does no intent guessing, no keyword matching and no answer lookup of
    its own — it only chooses which Conversation object receives the turn.
    """
    conversation_id, conversation = _conversation(request.conversation_id)
    image_path = str(_resolve_sample(request.image)) if request.image else None

    with _lock:
        result = conversation.ask(request.message, image_path=image_path)

    return _serialise(result, conversation_id)


@app.post("/api/conversations/reset")
def reset(request: ChatRequest | None = None) -> dict:
    """Start a genuinely fresh conversation — new object, empty state."""
    new_id = uuid.uuid4().hex[:12]
    with _lock:
        _conversations[new_id] = Conversation(f"api-{new_id}")
    return {"conversation_id": new_id, "turn_index": 0, "order_id": None}


@app.get("/api/conversations/{conversation_id}/state")
def conversation_state(conversation_id: str) -> dict:
    with _lock:
        conversation = _conversations.get(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="unknown conversation")
    state = conversation.state
    return {
        "conversation_id": conversation_id,
        "turn_index": state.get("turn_index", 0),
        "order_id": state.get("order_id"),
        "order_features": state.get("order_features"),
        "last_topic": state.get("last_topic"),
    }


@app.post("/api/return-risk")
def return_risk(request: RiskRequest) -> dict:
    """Direct access to the real Part 1 Random Forest via the Part 3 tool."""
    from part3.tools import check_return_risk, lookup_order

    features = request.order_features
    if features is None:
        if request.order_id is None:
            raise HTTPException(status_code=400, detail="order_id or order_features required")
        features = lookup_order(request.order_id)
        if features is None:
            raise HTTPException(status_code=404, detail=f"order {request.order_id} not found")

    return check_return_risk(features)


@app.post("/api/classify")
def classify(request: ClassifyRequest) -> dict:
    """Direct access to the real Part 2 ResNet-18 via the Part 3 tool."""
    from part3.tools import classify_product_image

    return classify_product_image(str(_resolve_sample(request.image)))


def main() -> int:
    import uvicorn

    print(f"[mode] {'USE_LIVE_LLM' if USE_LIVE_LLM else 'MOCK_LLM — deterministic, zero API keys'}")
    print("  API on http://127.0.0.1:8000  (docs at /docs)\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
