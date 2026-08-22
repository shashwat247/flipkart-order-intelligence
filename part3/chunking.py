"""Part 3 Task 1 — load the policy knowledge base and chunk it sentence-wise.

Every chunk keeps `document_id`, which is the whole point: Task 10 scores
retrieval at the **document** level, so a chunk that cannot name its parent
cannot be scored.
"""

import re

from part3.config import KB_DIR


def load_documents() -> list[dict]:
    """Read every `POL*.md` policy file into {id, title, text}.

    `README.md` is skipped — it documents the knowledge base rather than being
    part of it.
    """
    documents = []
    for path in sorted(KB_DIR.glob("POL*.md")):
        raw = path.read_text(encoding="utf-8").strip()
        lines = raw.splitlines()
        if not lines or not lines[0].startswith("# "):
            raise ValueError(f"{path.name}: first line must be '# Title'")
        title = lines[0][2:].strip()
        body = " ".join(line.strip() for line in lines[1:] if line.strip())
        documents.append({"id": path.stem, "title": title, "text": body})
    if len(documents) < 12:
        raise ValueError(f"knowledge base has only {len(documents)} documents; need >= 12")
    return documents


def split_sentences(text: str) -> list[str]:
    """Split on sentence-final punctuation followed by whitespace.

    Deliberately simple: these are hand-written policy sentences with no
    abbreviations or decimals to trip the regex, so a dependency-free split is
    both sufficient and easy to explain.
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_document(document: dict) -> list[dict]:
    """One chunk per sentence, each carrying a pointer to its parent document."""
    return [
        {
            "chunk_id": f"{document['id']}::s{i:02d}",
            "document_id": document["id"],
            "document_title": document["title"],
            "chunk_text": sentence,
        }
        for i, sentence in enumerate(split_sentences(document["text"]))
    ]


def build_chunks(documents: list[dict] | None = None) -> list[dict]:
    documents = documents if documents is not None else load_documents()
    chunks = []
    for document in documents:
        chunks.extend(chunk_document(document))
    return chunks


if __name__ == "__main__":
    docs = load_documents()
    chunks = build_chunks(docs)
    print(f"{len(docs)} documents -> {len(chunks)} sentence-wise chunks")
    for doc in docs:
        n = sum(1 for c in chunks if c["document_id"] == doc["id"])
        print(f"  {doc['id']:<34s} {n} chunks  '{doc['title']}'")
