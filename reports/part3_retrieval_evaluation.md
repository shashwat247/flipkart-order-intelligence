# Part 3 — Task 10: Retrieval Evaluation

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

### Query 1: "How many days do I have to return a mobile phone?"

*Why those documents are the key:* A mobile phone is electronics; only the electronics document states the 7-day window.

| | |
|---|---|
| Relevant documents | {POL03} (n = 1) |
| Top-3 retrieved documents | {POL03, POL04, POL10} |
| Similarity scores | POL03=0.6429, POL04=0.4667, POL10=0.4536 |
| Correctly retrieved | {POL03} (n = 1) |

```
Precision@3 = |relevant ∩ retrieved| / 3          = 1/3 = 0.3333
Recall@3    = |relevant ∩ retrieved| / |relevant| = 1/1 = 1.0000
```

### Query 2: "When will I get my money back for a cash on delivery order?"

*Why those documents are the key:* The COD document gives the 7-10 business day timeline; the reverse-pickup process document states that the refund clock only starts after quality inspection, which is needed to answer 'when' correctly.

| | |
|---|---|
| Relevant documents | {POL05, POL10} (n = 2) |
| Top-3 retrieved documents | {POL05, POL13, POL06} |
| Similarity scores | POL05=0.7094, POL13=0.6457, POL06=0.5682 |
| Correctly retrieved | {POL05} (n = 1) |

```
Precision@3 = |relevant ∩ retrieved| / 3          = 1/3 = 0.3333
Recall@3    = |relevant ∩ retrieved| / |relevant| = 1/2 = 0.5000
```

### Query 3: "Will a courier come to my house to collect the item I am returning?"

*Why those documents are the key:* Eligibility says whether free collection is available at all; the process document says when it is scheduled and how many attempts are made.

| | |
|---|---|
| Relevant documents | {POL09, POL10} (n = 2) |
| Top-3 retrieved documents | {POL10, POL09, POL04} |
| Similarity scores | POL10=0.6730, POL09=0.5300, POL04=0.4674 |
| Correctly retrieved | {POL10, POL09} (n = 2) |

```
Precision@3 = |relevant ∩ retrieved| / 3          = 2/3 = 0.6667
Recall@3    = |relevant ∩ retrieved| / |relevant| = 2/2 = 1.0000
```

### Query 4: "How long does delivery take to a remote address?"

*Why those documents are the key:* The delivery SLA document is the only one that distinguishes metro from non-metro/remote commitments.

| | |
|---|---|
| Relevant documents | {POL07} (n = 1) |
| Top-3 retrieved documents | {POL07, POL08, POL12} |
| Similarity scores | POL07=0.7741, POL08=0.5924, POL12=0.4465 |
| Correctly retrieved | {POL07} (n = 1) |

```
Precision@3 = |relevant ∩ retrieved| / 3          = 1/3 = 0.3333
Recall@3    = |relevant ∩ retrieved| / |relevant| = 1/1 = 1.0000
```

### Query 5: "Can I return a used lipstick?"

*Why those documents are the key:* Cosmetics are named in the non-returnable categories for hygiene reasons.

| | |
|---|---|
| Relevant documents | {POL15} (n = 1) |
| Top-3 retrieved documents | {POL04, POL15, POL14} |
| Similarity scores | POL04=0.4584, POL15=0.4015, POL14=0.3848 |
| Correctly retrieved | {POL15} (n = 1) |

```
Precision@3 = |relevant ∩ retrieved| / 3          = 1/3 = 0.3333
Recall@3    = |relevant ∩ retrieved| / |relevant| = 1/1 = 1.0000
```

### Query 6: "My order arrived broken. What should I do?"

*Why those documents are the key:* The damaged/defective document covers the 48-hour reporting window, the photograph requirement and the replacement-or-refund remedy.

| | |
|---|---|
| Relevant documents | {POL11} (n = 1) |
| Top-3 retrieved documents | {POL11, POL12, POL10} |
| Similarity scores | POL11=0.4780, POL12=0.4598, POL10=0.4277 |
| Correctly retrieved | {POL11} (n = 1) |

```
Precision@3 = |relevant ∩ retrieved| / 3          = 1/3 = 0.3333
Recall@3    = |relevant ∩ retrieved| / |relevant| = 1/1 = 1.0000
```

### Query 7: "Can I swap my shoes for a bigger size?"

*Why those documents are the key:* The exchange document defines size swaps; the footwear document confirms footwear is eligible and gives the matching 10-day window.

| | |
|---|---|
| Relevant documents | {POL02, POL14} (n = 2) |
| Top-3 retrieved documents | {POL02, POL14, POL04} |
| Similarity scores | POL02=0.5370, POL14=0.3878, POL04=0.3323 |
| Correctly retrieved | {POL02, POL14} (n = 2) |

```
Precision@3 = |relevant ∩ retrieved| / 3          = 2/3 = 0.6667
Recall@3    = |relevant ∩ retrieved| / |relevant| = 2/2 = 1.0000
```


## Summary

| # | query | Precision@3 | Recall@3 |
|---:|---|---:|---:|
| 1 | How many days do I have to return a mobile phone? | 1/3 = 0.3333 | 1/1 = 1.0000 |
| 2 | When will I get my money back for a cash on delivery order? | 1/3 = 0.3333 | 1/2 = 0.5000 |
| 3 | Will a courier come to my house to collect the item I am returning? | 2/3 = 0.6667 | 2/2 = 1.0000 |
| 4 | How long does delivery take to a remote address? | 1/3 = 0.3333 | 1/1 = 1.0000 |
| 5 | Can I return a used lipstick? | 1/3 = 0.3333 | 1/1 = 1.0000 |
| 6 | My order arrived broken. What should I do? | 1/3 = 0.3333 | 1/1 = 1.0000 |
| 7 | Can I swap my shoes for a bigger size? | 2/3 = 0.6667 | 2/2 = 1.0000 |

```
Average Precision@3 = (0.3333 + 0.3333 + 0.6667 + 0.3333 + 0.3333 + 0.3333 + 0.6667) / 7 = 0.4286
Average Recall@3    = (1.0000 + 0.5000 + 1.0000 + 1.0000 + 1.0000 + 1.0000 + 1.0000) / 7 = 0.9286
```

| metric | value |
|---|---:|
| **Average Precision@3** | **0.4286** |
| **Average Recall@3** | **0.9286** |
| queries evaluated | 7 |

### Reading these numbers honestly

Recall@3 is the metric that matters for this system: it asks whether the
document containing the answer made it in front of the response generator at
all. Precision@3 is structurally capped here — most queries have only one
relevant document, so retrieving 3 documents means the best achievable
Precision@3 for those queries is 1/3 = 0.3333. A low precision number is
therefore mostly an artifact of the denominator, not evidence of a bad
retriever, and reporting the average without saying so would be misleading.

## Secondary view: the agent's own retrieval configuration

The answer path uses `TOP_K = 3` **chunks** (not documents), which after
deduplication can yield fewer than 3 documents. Reported for transparency:

| query | distinct documents returned | relevant among them | recall |
|---|---:|---:|---:|
| How many days do I have to return a mobile phone? | 3 | 1 | 1.0000 |
| When will I get my money back for a cash on delivery | 2 | 1 | 0.5000 |
| Will a courier come to my house to collect the item  | 2 | 2 | 1.0000 |
| How long does delivery take to a remote address? | 2 | 1 | 1.0000 |
| Can I return a used lipstick? | 3 | 1 | 1.0000 |
| My order arrived broken. What should I do? | 3 | 1 | 1.0000 |
| Can I swap my shoes for a bigger size? | 2 | 2 | 1.0000 |

Average recall in the agent's own configuration: **0.9286**.
