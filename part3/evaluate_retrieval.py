"""Part 3 Task 10 — document-level retrieval evaluation.

Precision@3 and Recall@3 against the answer key in `part3/eval_queries.py`,
scored at the **document** level: every retrieved chunk is mapped back to its
parent document and deduplicated before scoring, because the answer key names
documents, not chunks.

Per-query arithmetic is printed and written out — not just the two averages.

    python3 -m part3.evaluate_retrieval
"""

from part3.config import REPORTS_DIR, TOP_K
from part3.eval_queries import EVAL_QUERIES
from part3.retrieval import search_chunks, search_documents, to_documents

REPORT_PATH = REPORTS_DIR / "part3_retrieval_evaluation.md"
K = 3


def short(doc_id: str) -> str:
    """'POL03_electronics_return_window' -> 'POL03' for compact set arithmetic."""
    return doc_id.split("_", 1)[0]


def score_query(query: str, relevant: set[str], retrieved: list[str]) -> dict:
    hits = [d for d in retrieved if d in relevant]
    precision = len(hits) / K
    recall = len(hits) / len(relevant) if relevant else 0.0
    return {
        "query": query,
        "relevant": sorted(relevant),
        "retrieved": retrieved,
        "hits": hits,
        "n_hits": len(hits),
        "precision_at_3": precision,
        "recall_at_3": recall,
    }


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for query, relevant, rationale in EVAL_QUERIES:
        docs = search_documents(query, n_documents=K)
        retrieved = [d["document_id"] for d in docs]
        row = score_query(query, relevant, retrieved)
        row["rationale"] = rationale
        row["scores"] = [d["score"] for d in docs]
        rows.append(row)

    mean_precision = sum(r["precision_at_3"] for r in rows) / len(rows)
    mean_recall = sum(r["recall_at_3"] for r in rows) / len(rows)

    print(f"Document-level retrieval evaluation over {len(rows)} queries\n")
    for i, r in enumerate(rows, start=1):
        rel = "{" + ", ".join(short(d) for d in r["relevant"]) + "}"
        ret = "{" + ", ".join(short(d) for d in r["retrieved"]) + "}"
        print(f"Query {i}: {r['query']}")
        print(f"  Relevant documents      = {rel}")
        print(f"  Top-3 retrieved         = {ret}")
        print(f"  Precision@3 = {r['n_hits']}/{K} = {r['precision_at_3']:.4f}")
        print(f"  Recall@3    = {r['n_hits']}/{len(r['relevant'])} "
              f"= {r['recall_at_3']:.4f}\n")
    print(f"Average Precision@3 = {mean_precision:.4f}")
    print(f"Average Recall@3    = {mean_recall:.4f}")

    # Secondary view: what the agent's own answer path (top-3 CHUNKS, then
    # deduplicated) retrieves. Reported for transparency, since fewer than 3
    # distinct documents can come back that way.
    agent_rows = []
    for query, relevant, _ in EVAL_QUERIES:
        docs = to_documents(search_chunks(query, top_k=TOP_K))
        retrieved = [d["document_id"] for d in docs]
        hits = [d for d in retrieved if d in relevant]
        agent_rows.append({
            "query": query, "n_docs": len(retrieved), "n_hits": len(hits),
            "recall": len(hits) / len(relevant) if relevant else 0.0,
        })
    agent_mean_recall = sum(r["recall"] for r in agent_rows) / len(agent_rows)

    per_query_sections = []
    for i, r in enumerate(rows, start=1):
        rel = "{" + ", ".join(short(d) for d in r["relevant"]) + "}"
        ret = "{" + ", ".join(short(d) for d in r["retrieved"]) + "}"
        hit = "{" + ", ".join(short(d) for d in r["hits"]) + "}" if r["hits"] else "{}"
        score_list = ", ".join(f"{short(d)}={s:.4f}"
                               for d, s in zip(r["retrieved"], r["scores"]))
        per_query_sections.append(f"""### Query {i}: "{r['query']}"

*Why those documents are the key:* {r['rationale']}

| | |
|---|---|
| Relevant documents | {rel} (n = {len(r['relevant'])}) |
| Top-3 retrieved documents | {ret} |
| Similarity scores | {score_list} |
| Correctly retrieved | {hit} (n = {r['n_hits']}) |

```
Precision@3 = |relevant ∩ retrieved| / 3          = {r['n_hits']}/3 = {r['precision_at_3']:.4f}
Recall@3    = |relevant ∩ retrieved| / |relevant| = {r['n_hits']}/{len(r['relevant'])} = {r['recall_at_3']:.4f}
```
""")

    summary_rows = "\n".join(
        f"| {i} | {r['query']} | {r['n_hits']}/3 = {r['precision_at_3']:.4f} "
        f"| {r['n_hits']}/{len(r['relevant'])} = {r['recall_at_3']:.4f} |"
        for i, r in enumerate(rows, start=1)
    )
    agent_summary = "\n".join(
        f"| {r['query'][:52]} | {r['n_docs']} | {r['n_hits']} | {r['recall']:.4f} |"
        for r in agent_rows
    )

    precision_sum = " + ".join(f"{r['precision_at_3']:.4f}" for r in rows)
    recall_sum = " + ".join(f"{r['recall_at_3']:.4f}" for r in rows)

    REPORT_PATH.write_text(f"""# Part 3 — Task 10: Retrieval Evaluation

Regenerate with `python3 -m part3.evaluate_retrieval`.

## How this is scored

Scoring is at the **document** level, not the chunk level. The pipeline is:

1. embed the query with `all-MiniLM-L6-v2`;
2. search the FAISS index over sentence-wise chunks;
3. map every retrieved chunk back to its `document_id`;
4. **deduplicate** — two chunks from the same policy are one retrieved document;
5. take the top 3 distinct documents and compare against the answer key.

The answer key lives in `part3/eval_queries.py` and was written before any
retrieval was run, so it cannot have been fitted to these results.

Step 4 is why the chunk search runs over a wider pool (12 chunks) before the
rollup: with only 3 chunks retrieved, three sentences from the same policy would
collapse to a single document and Precision@3 would be scored against a
denominator that was never actually filled.

## Per-query arithmetic

{chr(10).join(per_query_sections)}

## Summary

| # | query | Precision@3 | Recall@3 |
|---:|---|---:|---:|
{summary_rows}

```
Average Precision@3 = ({precision_sum}) / {len(rows)} = {mean_precision:.4f}
Average Recall@3    = ({recall_sum}) / {len(rows)} = {mean_recall:.4f}
```

| metric | value |
|---|---:|
| **Average Precision@3** | **{mean_precision:.4f}** |
| **Average Recall@3** | **{mean_recall:.4f}** |
| queries evaluated | {len(rows)} |

### Reading these numbers honestly

Recall@3 is the metric that matters for this system: it asks whether the
document containing the answer made it in front of the response generator at
all. Precision@3 is structurally capped here — most queries have only one
relevant document, so retrieving 3 documents means the best achievable
Precision@3 for those queries is 1/3 = 0.3333. A low precision number is
therefore mostly an artifact of the denominator, not evidence of a bad
retriever, and reporting the average without saying so would be misleading.

## Secondary view: the agent's own retrieval configuration

The answer path uses `TOP_K = {TOP_K}` **chunks** (not documents), which after
deduplication can yield fewer than 3 documents. Reported for transparency:

| query | distinct documents returned | relevant among them | recall |
|---|---:|---:|---:|
{agent_summary}

Average recall in the agent's own configuration: **{agent_mean_recall:.4f}**.
""", encoding="utf-8")
    print(f"\nWrote {REPORT_PATH.relative_to(REPORTS_DIR.parent)}")


if __name__ == "__main__":
    main()
