"""Part 3 Tasks 2-3 — FAISS index build/load, semantic search, document-level
rollup, and the output-side groundedness check.

The index is `IndexFlatIP` over L2-normalised vectors, so the inner product it
returns *is* cosine similarity — which is what the groundedness threshold is
expressed in.
"""

import json
from functools import lru_cache

import numpy as np

from part3.chunking import build_chunks, load_documents
from part3.config import (
    CHUNKS_PATH,
    INDEX_DIR,
    INDEX_PATH,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from part3.embeddings import embed, embed_one


def build_index(verbose: bool = True):
    """Embed every chunk and persist the FAISS index + chunk metadata."""
    import faiss

    documents = load_documents()
    chunks = build_chunks(documents)
    if verbose:
        print(f"{len(documents)} documents -> {len(chunks)} sentence-wise chunks")

    vectors = embed([c["chunk_text"] for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    CHUNKS_PATH.write_text(json.dumps(chunks, indent=2) + "\n", encoding="utf-8")

    if verbose:
        print(f"embedded to {vectors.shape[1]}-d vectors")
        print(f"wrote {INDEX_PATH.name} and {CHUNKS_PATH.name} to {INDEX_DIR}")
    return index, chunks


@lru_cache(maxsize=1)
def load_index():
    """Load the persisted index and chunk metadata (once per process)."""
    import faiss

    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Policy index missing at {INDEX_DIR}. Build it with:\n"
            f"  python3 -m part3.build_index"
        )
    index = faiss.read_index(str(INDEX_PATH))
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return index, chunks


def search_chunks(query: str, top_k: int = TOP_K) -> list[dict]:
    """Top-k chunks by cosine similarity, highest first."""
    index, chunks = load_index()
    scores, positions = index.search(embed_one(query), top_k)

    results = []
    for score, position in zip(scores[0], positions[0]):
        if position < 0:                      # FAISS pads with -1 if k > ntotal
            continue
        hit = dict(chunks[int(position)])
        hit["score"] = round(float(score), 4)
        results.append(hit)
    return results


def to_documents(chunk_hits: list[dict]) -> list[dict]:
    """Roll chunk hits up to unique parent documents, keeping the best score.

    This is the step Task 10's document-level scoring depends on: two chunks
    from the same policy are one retrieved *document*, not two, so they must be
    deduplicated before precision and recall are computed.
    """
    best: dict[str, dict] = {}
    for hit in chunk_hits:
        doc_id = hit["document_id"]
        if doc_id not in best or hit["score"] > best[doc_id]["score"]:
            best[doc_id] = {
                "document_id": doc_id,
                "document_title": hit["document_title"],
                "score": hit["score"],
                "best_chunk_id": hit["chunk_id"],
                "best_chunk_text": hit["chunk_text"],
            }
    return sorted(best.values(), key=lambda d: d["score"], reverse=True)


def search_documents(query: str, n_documents: int = 3,
                     chunk_pool: int = 12) -> list[dict]:
    """Top-N unique parent *documents* for a query.

    Searches a wider pool of chunks first, then deduplicates to documents, so
    "top 3 documents" really is three distinct documents. Without the wider
    pool, three chunks from the same policy would collapse to one document and
    a document-level Precision@3 would be scored against a denominator that was
    never actually filled.
    """
    return to_documents(search_chunks(query, top_k=chunk_pool))[:n_documents]


def check_groundedness(chunk_hits: list[dict],
                       threshold: float = SIMILARITY_THRESHOLD) -> dict:
    """Output-side guardrail: is the best retrieved chunk close enough to answer?

    If nothing clears the floor, the agent must refuse rather than let the
    response generator compose a confident-sounding policy out of loosely
    related text. Returns the numbers so a transcript can show *why* it refused.
    """
    best_score = max((float(h["score"]) for h in chunk_hits), default=0.0)
    return {
        "grounded": best_score >= threshold,
        "best_score": round(best_score, 4),
        "threshold": threshold,
        "n_chunks_considered": len(chunk_hits),
    }
