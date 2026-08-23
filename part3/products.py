"""Part 3 — the synthetic product catalog: a second knowledge source alongside
the policy documents.

`category` on every record uses the exact categorical levels Part 1's Random
Forest was trained on (Apparel, Footwear, Electronics, Home, Beauty), so a
category read off a product record is always valid input to the return-risk
tool. Search is brute-force cosine similarity over the same local
sentence-transformer used everywhere else in Part 3 (`part3.embeddings`) --
54 rows does not need a persisted FAISS index; the same in-memory pattern
`part3/graph.py::_exemplar_vectors` already uses for 27 few-shot exemplars is
sufficient here too.
"""

import json
from functools import lru_cache

from part3.config import PRODUCTS_PATH
from part3.embeddings import embed, embed_one


@lru_cache(maxsize=1)
def load_catalog() -> list[dict]:
    """Read the synthetic product catalog once per process."""
    if not PRODUCTS_PATH.exists():
        raise FileNotFoundError(f"{PRODUCTS_PATH} not found.")
    return json.loads(PRODUCTS_PATH.read_text(encoding="utf-8"))


def _searchable_text(product: dict) -> str:
    return (f"{product['product_name']} -- {product['subcategory']}, "
            f"{product['category']}. {product['description']}")


@lru_cache(maxsize=1)
def _catalog_vectors():
    """Embed every product's searchable text once."""
    catalog = load_catalog()
    return embed([_searchable_text(p) for p in catalog])


def search_products(query: str, top_k: int = 5) -> list[dict]:
    """Top-k catalog products by cosine similarity to the query.

    Returns the same shape a policy `doc_hits` entry uses (`score`,
    `document_id`, `document_title`) plus the full product record, so the
    response generator can treat a product hit and a policy hit uniformly
    where useful, while still being able to tell them apart via `kind`.
    """
    import numpy as np

    catalog = load_catalog()
    vectors = _catalog_vectors()
    similarities = (vectors @ embed_one(query).T).ravel()
    order = np.argsort(similarities)[::-1][:top_k]

    results = []
    for i in order:
        product = catalog[int(i)]
        results.append({
            "kind": "product",
            "score": round(float(similarities[int(i)]), 4),
            "document_id": product["product_id"],
            "document_title": product["product_name"],
            "product": product,
        })
    return results


def filter_products(category: str | None = None,
                    cod_available: bool | None = None,
                    exchange_available: bool | None = None,
                    non_returnable: bool | None = None,
                    max_return_window: int | None = None) -> list[dict]:
    """Structured (non-semantic) catalog queries: "which products support COD",
    "which products are non-returnable", "products with a 10-day window", etc.
    """
    catalog = load_catalog()
    results = catalog
    if category is not None:
        results = [p for p in results if p["category"].lower() == category.lower()]
    if cod_available is not None:
        results = [p for p in results if p["cod_available"] == cod_available]
    if exchange_available is not None:
        results = [p for p in results if p["exchange_available"] == exchange_available]
    if non_returnable is not None:
        results = [p for p in results if p["non_returnable"] == non_returnable]
    if max_return_window is not None:
        results = [p for p in results if p["return_window"] <= max_return_window]
    return results


def catalog_categories() -> list[str]:
    return sorted({p["category"] for p in load_catalog()})


# A "which products ... ?" question is a structured filter, not a similarity
# search -- no single product's description is semantically "about" the
# question, so cosine search alone can't answer it. This is narrow,
# high-precision phrase matching on the filter itself (COD/exchange/
# non-returnable/a day count), not a whole-response keyword router: it only
# ever feeds `filter_products`, and any query it doesn't recognise falls
# through to semantic `search_products` untouched.
import re as _re

_COD_RE = _re.compile(r"\bcod\b|cash[\s-]?on[\s-]?delivery", _re.IGNORECASE)
_NON_RETURNABLE_RE = _re.compile(r"non[\s-]?returnable|not\s+returnable|cannot\s+be\s+returned",
                                 _re.IGNORECASE)
_EXCHANGE_RE = _re.compile(r"\bexchange\b", _re.IGNORECASE)
_RETURN_WINDOW_RE = _re.compile(r"(\d+)[\s-]*day", _re.IGNORECASE)


def parse_filter_criteria(query: str) -> dict:
    """Read `filter_products` keyword arguments off a "which products..."
    style question. Returns {} when nothing structured is recognised.
    """
    criteria: dict = {}
    if _NON_RETURNABLE_RE.search(query):
        criteria["non_returnable"] = True
    elif _COD_RE.search(query):
        criteria["cod_available"] = True
    elif _EXCHANGE_RE.search(query):
        criteria["exchange_available"] = True

    window_match = _RETURN_WINDOW_RE.search(query)
    if window_match and ("window" in query.lower() or "return" in query.lower()):
        criteria["max_return_window"] = int(window_match.group(1))

    for category in ("Apparel", "Footwear", "Electronics", "Home", "Beauty"):
        if category.lower() in query.lower():
            criteria["category"] = category
            break

    return criteria
