"""Part 3 Task 2 — local, keyless sentence embeddings.

`all-MiniLM-L6-v2` runs entirely on the machine. The weights are fetched once on
first use and cached by sentence-transformers; every call after that is offline.
Embeddings are L2-normalised so a FAISS inner-product search returns cosine
similarity directly.
"""

from functools import lru_cache

import numpy as np

from part3.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_encoder():
    """Load the sentence-transformer once per process."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    """Encode to L2-normalised float32 vectors, shape (n, dim)."""
    vectors = get_encoder().encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype="float32")


def embed_one(text: str) -> np.ndarray:
    """Encode a single string to shape (1, dim)."""
    return embed([text])
