# Part 3 — Groundedness Threshold Calibration

Regenerate with `python3 -m part3.calibrate_threshold`.

The output-side guardrail refuses to answer a policy question when no retrieved
chunk clears a minimum cosine similarity. This report is how that minimum was
chosen: by measuring where in-domain and out-of-domain questions actually land,
using query sets declared up front in `part3/eval_queries.py`.

## In-domain — questions the knowledge base does cover

Best-matching chunk's cosine similarity, ascending:

| best score | query | best-matching document | verdict at 0.45 |
|---:|---|---|---|
| 0.4584 | Can I return a used lipstick? | `POL04_home_return_window` | PASS |
| 0.4780 | My order arrived broken. What should I do? | `POL11_damaged_product` | PASS |
| 0.5370 | Can I swap my shoes for a bigger size? | `POL02_footwear_return_window` | PASS |
| 0.6429 | How many days do I have to return a mobile phone? | `POL03_electronics_return_window` | PASS |
| 0.6730 | Will a courier come to my house to collect the item I am returning? | `POL10_reverse_pickup_process` | PASS |
| 0.7094 | When will I get my money back for a cash on delivery order? | `POL05_cod_refund_timeline` | PASS |
| 0.7741 | How long does delivery take to a remote address? | `POL07_delivery_sla` | PASS |

min **0.4584**, max 0.7741, mean 0.6104

## Out-of-domain — questions it genuinely does not cover

These are plausible things a support agent gets asked that this knowledge base
has no policy for. Descending:

| best score | query | closest document | verdict at 0.45 |
|---:|---|---|---|
| 0.4379 | What is Flipkart's GST registration number? | `POL06_prepaid_refund_timeline` | REFUSE |
| 0.4193 | How do I apply for a job at Flipkart? | `POL06_prepaid_refund_timeline` | REFUSE |
| 0.3616 | How do I permanently close my Flipkart account? | `POL06_prepaid_refund_timeline` | REFUSE |
| 0.3227 | What is the share price of Flipkart today? | `POL06_prepaid_refund_timeline` | REFUSE |
| 0.2787 | What is the extended warranty price for a washing machine? | `POL01_apparel_return_window` | REFUSE |
| 0.1743 | What is the weather in Bangalore tomorrow? | `POL08_delayed_delivery` | REFUSE |
| 0.1328 | Who won the cricket match last night? | `POL06_prepaid_refund_timeline` | REFUSE |
| 0.1302 | Can you recommend a good biryani recipe? | `POL04_home_return_window` | REFUSE |

min 0.1302, max **0.4379**, mean 0.2822

## The chosen threshold

| quantity | value |
|---|---:|
| lowest in-domain score | 0.4584 |
| highest out-of-domain score | 0.4379 |
| separation gap | **+0.0205** |
| midpoint of the gap | 0.4481 |
| **threshold set in `part3/config.py`** | **0.45** |
| cleanly separates both sets | **True** |

The two distributions are cleanly separated, and
**0.45** sits inside that gap.
Setting it here means every question the knowledge base can actually answer gets
answered, and every question it cannot gets an explicit refusal that prints the
score it fell short by — rather than a fluent-sounding policy the retrieved text
never supported.

The threshold is a **floor on evidence quality, not a confidence score**. A
question can be perfectly clear and still be refused, because the test is
whether *this corpus* contains the answer, not whether the question made sense.

### Honest caveat: the margin is narrow

The gap is only **0.0205** wide. The out-of-domain questions that crowd the
boundary are the *plausible* ones — "How do I apply for a job at Flipkart?"
(0.4193) and
"What is Flipkart's GST registration number?"
(0.4379) —
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
