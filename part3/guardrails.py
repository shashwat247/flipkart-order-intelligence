"""Part 3 Task 8 — the two guardrails.

**Input side (this file):** the *application* inspects the raw user text for
prompt-injection patterns before any node runs. It does not delegate the check
to the prompt, because a model that has been successfully injected is exactly
the component you can no longer trust to report the injection.

**Output side:** `check_groundedness` lives in `part3/retrieval.py` next to the
similarity scores it judges; it is re-exported here so both guardrails have one
import site.
"""

import re

from part3.retrieval import check_groundedness  # noqa: F401  (re-export)

# Each pattern targets an *instruction-override* phrasing rather than a single
# suspicious word. Matching on a bare "ignore" or "pretend" would block ordinary
# support questions ("can I ignore the delivery SMS?"), so every pattern below
# requires the surrounding override structure.
INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|earlier|above)\s+"
     r"(?:instruction|instructions|prompt|prompts|message|messages|context)",
     "ignore-previous-instructions"),
    (r"ignore\s+(?:all\s+|any\s+|the\s+)?(?:rule|rules|guideline|guidelines|"
     r"restriction|restrictions|policy\s+constraints)",
     "ignore-all-rules"),
    (r"ignore\s+(?:your|the)\s+system\s+prompt", "ignore-system-prompt"),
    (r"disregard\s+(?:all\s+|any\s+|the\s+|your\s+)?(?:previous|prior|earlier|above)?"
     r"\s*(?:instruction|instructions|rule|rules|prompt)",
     "disregard-instructions"),
    (r"forget\s+(?:all\s+|your\s+|the\s+)?(?:previous\s+)?"
     r"(?:instruction|instructions|rule|rules|training|prompt)",
     "forget-your-instructions"),
    (r"pretend\s+(?:you\s+are|to\s+be|you're)", "pretend-you-are"),
    (r"act\s+as\s+(?:if\s+you|though\s+you|a\s+|an\s+)", "act-as"),
    (r"you\s+are\s+now\s+(?:a|an|no\s+longer)", "you-are-now"),
    (r"(?:reveal|show|print|repeat|output|display)\s+(?:me\s+)?"
     r"(?:your|the)\s+(?:system\s+prompt|initial\s+prompt|instructions|"
     r"hidden\s+prompt|prompt)",
     "reveal-system-prompt"),
    (r"(?:developer|debug|god|admin)\s+mode", "developer-mode"),
    (r"\bjailbreak\b", "jailbreak"),
    (r"(?:bypass|override|disable)\s+(?:your\s+|the\s+|all\s+)?"
     r"(?:guardrail|guardrails|safety|filter|filters|restriction|restrictions)",
     "bypass-guardrails"),
    (r"do\s+not\s+(?:follow|obey)\s+(?:your|the)\s+(?:instruction|instructions|rules)",
     "do-not-follow-instructions"),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), name) for p, name in INJECTION_PATTERNS]


def detect_injection(text: str) -> dict:
    """Scan raw user input for instruction-override attempts.

    Returns the matched pattern names and the exact substrings that matched, so
    a transcript can show precisely what tripped the guardrail rather than
    asserting that something did.
    """
    matches = []
    for pattern, name in _COMPILED:
        found = pattern.search(text or "")
        if found:
            matches.append({"pattern": name, "matched_text": found.group(0)})
    return {
        "blocked": bool(matches),
        "matches": matches,
        "n_patterns_checked": len(_COMPILED),
    }
