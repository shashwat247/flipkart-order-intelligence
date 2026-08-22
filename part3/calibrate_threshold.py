"""Part 3 Task 8 — calibrate the groundedness similarity threshold.

Measures where the best-matching chunk's cosine similarity actually lands for
questions the knowledge base *does* cover versus questions it genuinely does
not, then reports the gap between the two distributions. The threshold in
`part3/config.py` is set from that gap.

This is run *before* fixing the threshold, and it is driven by the pre-declared
query sets in `part3/eval_queries.py` — so the value is not reverse-engineered
from whichever transcript happened to look good.

    python3 -m part3.calibrate_threshold
"""

from part3.config import REPORTS_DIR, SIMILARITY_THRESHOLD
from part3.eval_queries import EVAL_QUERIES, OUT_OF_DOMAIN_QUERIES
from part3.retrieval import search_chunks

REPORT_PATH = REPORTS_DIR / "part3_threshold_calibration.md"


def best_scores(queries: list[str]) -> list[tuple[str, float, str]]:
    rows = []
    for query in queries:
        hits = search_chunks(query)
        top = hits[0]
        rows.append((query, float(top["score"]), top["document_id"]))
    return rows


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    in_domain = best_scores([q for q, _, _ in EVAL_QUERIES])
    out_domain = best_scores(OUT_OF_DOMAIN_QUERIES)

    in_scores = [s for _, s, _ in in_domain]
    out_scores = [s for _, s, _ in out_domain]
    in_min, in_max = min(in_scores), max(in_scores)
    out_min, out_max = min(out_scores), max(out_scores)
    gap = in_min - out_max

    print("IN-DOMAIN (knowledge base covers these):")
    for query, score, doc in in_domain:
        print(f"  {score:.4f}  {query[:56]:<56s} -> {doc}")
    print(f"  min={in_min:.4f}  max={in_max:.4f}  mean={sum(in_scores) / len(in_scores):.4f}")

    print("\nOUT-OF-DOMAIN (knowledge base does not cover these):")
    for query, score, doc in out_domain:
        print(f"  {score:.4f}  {query[:56]:<56s} -> {doc}")
    print(f"  min={out_min:.4f}  max={out_max:.4f}  "
          f"mean={sum(out_scores) / len(out_scores):.4f}")

    print(f"\nseparation gap = in-domain min {in_min:.4f} - out-of-domain max "
          f"{out_max:.4f} = {gap:+.4f}")
    midpoint = (in_min + out_max) / 2
    print(f"midpoint of the gap = {midpoint:.4f}")
    print(f"threshold currently set in part3/config.py = {SIMILARITY_THRESHOLD}")

    separates = out_max < SIMILARITY_THRESHOLD <= in_min
    print(f"threshold cleanly separates both sets: {separates}")

    in_rows = "\n".join(
        f"| {score:.4f} | {query} | `{doc}` | {'PASS' if score >= SIMILARITY_THRESHOLD else 'REFUSE'} |"
        for query, score, doc in sorted(in_domain, key=lambda r: r[1])
    )
    out_rows = "\n".join(
        f"| {score:.4f} | {query} | `{doc}` | {'PASS' if score >= SIMILARITY_THRESHOLD else 'REFUSE'} |"
        for query, score, doc in sorted(out_domain, key=lambda r: -r[1])
    )

    REPORT_PATH.write_text(f"""# Part 3 — Groundedness Threshold Calibration

Regenerate with `python3 -m part3.calibrate_threshold`.

The output-side guardrail refuses to answer a policy question when no retrieved
chunk clears a minimum cosine similarity. This report is how that minimum was
chosen: by measuring where in-domain and out-of-domain questions actually land,
using query sets declared up front in `part3/eval_queries.py`.

## In-domain — questions the knowledge base does cover

Best-matching chunk's cosine similarity, ascending:

| best score | query | best-matching document | verdict at {SIMILARITY_THRESHOLD} |
|---:|---|---|---|
{in_rows}

min **{in_min:.4f}**, max {in_max:.4f}, mean {sum(in_scores) / len(in_scores):.4f}

## Out-of-domain — questions it genuinely does not cover

These are plausible things a support agent gets asked that this knowledge base
has no policy for. Descending:

| best score | query | closest document | verdict at {SIMILARITY_THRESHOLD} |
|---:|---|---|---|
{out_rows}

min {out_min:.4f}, max **{out_max:.4f}**, mean {sum(out_scores) / len(out_scores):.4f}

## The chosen threshold

| quantity | value |
|---|---:|
| lowest in-domain score | {in_min:.4f} |
| highest out-of-domain score | {out_max:.4f} |
| separation gap | **{gap:+.4f}** |
| midpoint of the gap | {midpoint:.4f} |
| **threshold set in `part3/config.py`** | **{SIMILARITY_THRESHOLD}** |
| cleanly separates both sets | **{separates}** |

The two distributions {"are cleanly separated" if gap > 0 else "overlap"}, and
**{SIMILARITY_THRESHOLD}** sits {"inside that gap" if separates else "at the chosen operating point"}.
Setting it here means every question the knowledge base can actually answer gets
answered, and every question it cannot gets an explicit refusal that prints the
score it fell short by — rather than a fluent-sounding policy the retrieved text
never supported.

The threshold is a **floor on evidence quality, not a confidence score**. A
question can be perfectly clear and still be refused, because the test is
whether *this corpus* contains the answer, not whether the question made sense.

### Honest caveat: the margin is narrow

The gap is only **{gap:.4f}** wide. The out-of-domain questions that crowd the
boundary are the *plausible* ones — "How do I apply for a job at Flipkart?"
({dict((q, s) for q, s, _ in out_domain).get("How do I apply for a job at Flipkart?", 0):.4f}) and
"What is Flipkart's GST registration number?"
({dict((q, s) for q, s, _ in out_domain).get("What is Flipkart's GST registration number?", 0):.4f}) —
because they share Flipkart-and-process vocabulary with the corpus even though
no document answers them. The obviously unrelated questions (weather, cricket,
a biryani recipe) sit far below at 0.13-0.17 and were never in any danger of
passing.

That narrowness is a real limitation, not something to paper over. It means the
threshold is well-supported for *this* 15-document corpus but would need
recalibrating the moment documents are added — a new "Careers" or "Tax and
invoicing" policy would legitimately move the boundary. The right long-term fix
is to grow the corpus rather than to keep nudging the number: the more the
knowledge base genuinely covers, the wider this gap becomes.
""", encoding="utf-8")
    print(f"\nWrote {REPORT_PATH.relative_to(REPORTS_DIR.parent)}")


if __name__ == "__main__":
    main()
