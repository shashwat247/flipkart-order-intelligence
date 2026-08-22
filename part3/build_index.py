"""Build and persist the FAISS policy index.

    python3 -m part3.build_index

Free, local, keyless: embeddings come from all-MiniLM-L6-v2 running on this
machine and the index is a local FAISS file. No account, no vector-database
service.
"""

from part3.retrieval import build_index, search_chunks, to_documents


def main() -> None:
    build_index(verbose=True)

    # Smoke-test the freshly written index so a broken build fails loudly here
    # rather than silently inside the agent.
    probe = "How long do I have to return a pair of shoes?"
    hits = search_chunks(probe)
    print(f"\nsmoke test: {probe!r}")
    for hit in hits:
        print(f"  {hit['score']:.4f}  [{hit['document_id']}] {hit['chunk_text'][:78]}...")
    print(f"  -> {len(to_documents(hits))} unique parent documents")


if __name__ == "__main__":
    main()
