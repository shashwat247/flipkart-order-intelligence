# Policy Knowledge Base

**These are synthetic policy documents written for this academic project.** They
are *not* real Flipkart policies and must not be used as such. They exist to give
Part 3's RAG pipeline a small, grounded corpus to retrieve from.

## Format

One document per `.md` file. The document id is the filename stem
(e.g. `POL01_apparel_return_window`). The first line is `# Title`; everything
after it is the document body.

Each document is 2-4 sentences. `part3/chunking.py` splits every document
**sentence-wise**, and each resulting chunk keeps a pointer back to its parent
document id — which is what makes Task 10's **document-level** Precision@3 /
Recall@3 evaluation possible.

## Why sentence-wise chunking

A policy sentence is the natural unit of a policy answer: "apparel may be
returned within 10 days of delivery" is a complete, quotable rule. Fixed-size or
overlapping windows would cut mid-rule and force the retriever to return two
half-rules; multi-sentence chunks would bundle an unrelated rule into every hit
and inflate apparent recall. Each sentence below is therefore written to be
self-contained — it restates its subject rather than relying on the previous
sentence for context.
