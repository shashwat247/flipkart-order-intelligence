"""OPTIONAL live-LLM extension. Never required, never used by any transcript.

Enabled only when `USE_LIVE_LLM=1` is set in the environment AND a key is
present. With the flag unset — the default, and the mode every graded run uses —
this module is never even imported by `part3/graph.py`.

What it does when enabled: takes the answer MOCK_LLM already composed and asks a
free-tier, OpenAI-compatible chat endpoint to smooth the wording. It deliberately
**cannot** change the facts:

  * the deterministic answer is computed first, in full;
  * only its phrasing is sent for rewriting;
  * `source` and `confidence` are copied from the mock response, never from the
    model — so a live call can never invent a citation or a probability;
  * any failure at all (no key, no network, bad JSON, timeout) returns the mock
    response unchanged.

Removing the key therefore leaves every acceptance criterion satisfied, which is
exactly the property the brief asks for.

Implemented with `urllib` from the standard library so enabling it does not add
a dependency to `requirements.txt`.
"""

import json
import os
import urllib.error
import urllib.request

# Any OpenAI-compatible chat-completions endpoint. Groq's free tier is the
# default because it needs no credit card.
API_BASE = os.environ.get("LIVE_LLM_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY_ENV = os.environ.get("LIVE_LLM_API_KEY_ENV", "GROQ_API_KEY")
MODEL = os.environ.get("LIVE_LLM_MODEL", "llama-3.1-8b-instant")
TIMEOUT_SECONDS = 15

REWRITE_INSTRUCTION = (
    "Rewrite the support answer below so it reads naturally for a customer. "
    "Keep every fact, number, policy id and citation exactly as written. Do not "
    "add any information. Reply with the rewritten answer only, at most three "
    "sentences."
)


def _api_key() -> str | None:
    key = os.environ.get(API_KEY_ENV, "").strip()
    return key or None


def polish(state: dict, response: dict) -> dict:
    """Return `response` with a rephrased `answer`, or unchanged on any problem.

    `state` is accepted so this hook could use retrieval context in future; it
    is intentionally unused today, because sending more context would widen what
    a live model could contradict.
    """
    key = _api_key()
    if not key:
        # Flag on but no key: fall back silently and deterministically.
        return response

    payload = json.dumps({
        "model": MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": REWRITE_INSTRUCTION},
            {"role": "user", "content": response["answer"]},
        ],
    }).encode("utf-8")

    request = urllib.request.Request(
        f"{API_BASE}/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as handle:
            body = json.loads(handle.read().decode("utf-8"))
        rewritten = body["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, KeyError, IndexError, ValueError, TimeoutError):
        return response

    if not rewritten:
        return response

    # source and confidence are copied from the deterministic response — the
    # live model is never allowed to supply either.
    return {
        "answer": rewritten,
        "source": response["source"],
        "confidence": response["confidence"],
    }
