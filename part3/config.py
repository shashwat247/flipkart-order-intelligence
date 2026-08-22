"""Part 3 configuration.

Nothing here needs an API key, an account, or a network call at answer time.
The only download that ever happens is the one-off fetch of the local
sentence-transformer weights the first time an index is built.
"""

import os
from pathlib import Path

# Reuse Part 2's certifi fix so a fresh clone on macOS can fetch the
# sentence-transformer weights without a CERTIFICATE_VERIFY_FAILED.
from part2.config import _ensure_ssl_certificates  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = Path(__file__).resolve().parent / "knowledge_base"
INDEX_DIR = ROOT / "data" / "policy_index"
TRANSCRIPTS_DIR = ROOT / "transcripts"
REPORTS_DIR = ROOT / "reports"

INDEX_PATH = INDEX_DIR / "policy.faiss"
CHUNKS_PATH = INDEX_DIR / "chunks.json"

# Free, local, keyless embedding model (~90 MB, downloaded once and cached).
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 3

# --- output-side groundedness guardrail -------------------------------------
# Cosine similarity floor a policy question's best-matching chunk must clear
# before the agent is allowed to answer it at all.
#
# Chosen from measured behaviour, not from staring at one transcript: run
#   python3 -m part3.calibrate_threshold
# which scores the pre-declared in-domain evaluation queries against a set of
# pre-declared out-of-domain questions and prints both distributions.
#
# Measured on this corpus: in-domain best-chunk similarity bottoms out at 0.4584
# ("Can I return a used lipstick?"), while out-of-domain tops out at 0.4379
# ("What is Flipkart's GST registration number?"). 0.45 sits inside that gap, so
# every question the corpus can answer passes and every question it cannot is
# refused. reports/part3_threshold_calibration.md records the run.
SIMILARITY_THRESHOLD = 0.45

# --- LLM mode ---------------------------------------------------------------
# MOCK_LLM is the default and the only mode any graded transcript uses. The live
# path is optional, never required, and never exercised unless the user opts in.
USE_LIVE_LLM = os.environ.get("USE_LIVE_LLM", "").strip() not in ("", "0", "false", "False")

VALID_SOURCES = ("policy_kb", "return_risk_tool", "image_classifier_tool")
INTENTS = ("policy", "return_risk", "product_category")

# --- intent routing floor ---------------------------------------------------
# The intent node routes to the nearest few-shot exemplar. When a message is far
# from EVERY exemplar the argmax is just noise, so below this similarity we fall
# back to the `policy` lane deliberately: policy is the only lane with an
# evidence check behind it, so an unroutable question ends up refused with a
# printed score rather than confidently handed to a model tool.
#
# Measured (see part3/calibrate_threshold.py output and the README): genuine
# in-domain questions bottom out at 0.2819 exemplar similarity, while unrelated
# questions like "Who won the cricket match last night?" sit at 0.1492 and
# "What is the share price of Flipkart today?" at 0.1877. 0.25 sits below every
# real question and above that noise band.
INTENT_ROUTING_FLOOR = 0.25
