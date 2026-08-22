"""Part 3 Task 1/Task 10 — the retrieval answer key.

For each realistic support query, the one or two **documents** a human would
consider relevant. The key is deliberately written at document level (not chunk
level) because that is how Task 10 scores Precision@3 and Recall@3, and it was
fixed before any retrieval was run so it cannot be reverse-engineered from the
results.

Also holds a set of out-of-domain questions used only to calibrate the
groundedness threshold (see part3/calibrate_threshold.py).
"""

# (query, relevant document ids, why those documents)
EVAL_QUERIES: list[tuple[str, set[str], str]] = [
    (
        "How many days do I have to return a mobile phone?",
        {"POL03_electronics_return_window"},
        "A mobile phone is electronics; only the electronics document states the "
        "7-day window.",
    ),
    (
        "When will I get my money back for a cash on delivery order?",
        {"POL05_cod_refund_timeline", "POL10_reverse_pickup_process"},
        "The COD document gives the 7-10 business day timeline; the reverse-pickup "
        "process document states that the refund clock only starts after quality "
        "inspection, which is needed to answer 'when' correctly.",
    ),
    (
        "Will a courier come to my house to collect the item I am returning?",
        {"POL09_reverse_pickup_eligibility", "POL10_reverse_pickup_process"},
        "Eligibility says whether free collection is available at all; the process "
        "document says when it is scheduled and how many attempts are made.",
    ),
    (
        "How long does delivery take to a remote address?",
        {"POL07_delivery_sla"},
        "The delivery SLA document is the only one that distinguishes metro from "
        "non-metro/remote commitments.",
    ),
    (
        "Can I return a used lipstick?",
        {"POL15_non_returnable_items"},
        "Cosmetics are named in the non-returnable categories for hygiene reasons.",
    ),
    (
        "My order arrived broken. What should I do?",
        {"POL11_damaged_product"},
        "The damaged/defective document covers the 48-hour reporting window, the "
        "photograph requirement and the replacement-or-refund remedy.",
    ),
    (
        "Can I swap my shoes for a bigger size?",
        {"POL14_exchange_policy", "POL02_footwear_return_window"},
        "The exchange document defines size swaps; the footwear document confirms "
        "footwear is eligible and gives the matching 10-day window.",
    ),
]

# Questions a Flipkart support agent could plausibly be asked but that this
# knowledge base genuinely does not cover. Used ONLY to measure where
# out-of-domain similarity scores land, so the groundedness threshold can be set
# from the gap between the two distributions rather than from one transcript.
OUT_OF_DOMAIN_QUERIES: list[str] = [
    "What is the weather in Bangalore tomorrow?",
    "Who won the cricket match last night?",
    "How do I apply for a job at Flipkart?",
    "What is Flipkart's GST registration number?",
    "How do I permanently close my Flipkart account?",
    "What is the share price of Flipkart today?",
    "Can you recommend a good biryani recipe?",
    "What is the extended warranty price for a washing machine?",
]
