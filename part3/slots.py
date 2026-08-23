"""Part 3 — free-text order-feature extraction for the return-risk tool.

This is structured entity extraction, not intent routing: there is no way to
read "₹4,500" or "paid using COD" out of a sentence except a numeric/lexical
pattern, so this module is deliberately narrow and only extracts fields it can
get right with high confidence.

Two tiers. The first reads what a customer volunteers unprompted -- price,
payment method, product category, delivery time. The second reads the remaining
Part-1 features (discount, customer tenure, order/return history, distance,
weekend flag, rating) but only when they arrive **explicitly labelled**
("20% off", "300 days", "5 previous orders", "rated 4"), because that is the
shape they come back in after `check_return_risk` reports them as
`missing_features` and the agent asks for them by name (see
`part3/mock_llm.py::compose_missing_input`). Anything not labelled stays
unextracted and is asked for again rather than guessed -- an invented feature
value would produce a confident probability from data nobody supplied.
"""

import re

# Exact Part-1 categorical levels (part1/common.py) -- COD/prepaid keyword ->
# the literal level the trained model expects.
_PAYMENT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bcod\b|cash[\s-]?on[\s-]?delivery", re.IGNORECASE), "COD"),
    (re.compile(r"\bupi\b|gpay|google\s*pay|phonepe|paytm\s*upi", re.IGNORECASE), "Prepaid_UPI"),
    (re.compile(r"\bwallet\b", re.IGNORECASE), "Wallet"),
    (re.compile(r"credit\s*card|debit\s*card|\bcard\b", re.IGNORECASE), "Prepaid_Card"),
]

# A price needs an explicit currency marker, before the number ("₹4,500",
# "Rs 4500") or after it ("4500 rupees"). A bare integer is left alone on
# purpose: "a 12000 phone" and "12000 orders" are indistinguishable to a regex,
# and a wrong price would produce a confident probability from data the
# customer never gave.
_PRICE_RE = re.compile(
    r"(?:₹|rs\.?\s|inr\s)\s*([\d][\d,]*(?:\.\d+)?)"
    r"|([\d][\d,]*(?:\.\d+)?)\s*(?:rupees|rs\.?\b|inr\b)",
    re.IGNORECASE,
)
_DELIVERY_DELAY_RE = re.compile(r"(\d+)\s*days?\s*(?:late|delayed|of\s*delay)", re.IGNORECASE)
_DELIVERY_TOOK_RE = re.compile(r"(?:took|arrived in|delivered in)\s*(\d+)\s*days?", re.IGNORECASE)

# Tier two: every pattern needs its own label word, so a bare number in a
# sentence can never be silently bound to a feature.
_LABELLED_PATTERNS: list[tuple[str, re.Pattern, type]] = [
    ("discount_pct", re.compile(
        r"(\d+(?:\.\d+)?)\s*%\s*(?:off|discount)"
        r"|discount(?:\s+(?:of|was|is))?\s*:?\s*(\d+(?:\.\d+)?)\s*%?", re.IGNORECASE), float),
    ("customer_tenure_days", re.compile(
        r"(?:tenure|customer|account|member|signed up|registered)\D{0,20}?(\d+)\s*days?",
        re.IGNORECASE), int),
    # The qualifier is mandatory in both directions. Without it "order 1790"
    # reads as "1790 previous orders" and silently corrupts a looked-up order's
    # real features -- the id is not a count.
    ("num_previous_returns", re.compile(
        r"(\d+)\s*(?:previous|prior|past|earlier)\s+returns?\b"
        r"|(?:previous|prior|past|earlier)\s+returns?\s*:?\s*(\d+)", re.IGNORECASE), int),
    ("num_previous_orders", re.compile(
        r"(\d+)\s*(?:previous|prior|past|earlier)\s+orders?\b"
        r"|(?:previous|prior|past|earlier)\s+orders?\s*:?\s*(\d+)", re.IGNORECASE), int),
    ("delivery_distance_km", re.compile(
        r"(\d+(?:\.\d+)?)\s*(?:km\b|kilometre|kilometer)", re.IGNORECASE), float),
    ("rating_given", re.compile(
        r"rated\s*(?:it\s*)?(\d(?:\.\d)?)"
        r"|(\d(?:\.\d)?)\s*(?:star|/\s*5)"
        r"|rating\s*(?:of|was|is)?\s*:?\s*(\d(?:\.\d)?)", re.IGNORECASE), float),
]
_WEEKEND_RE = re.compile(r"\bweekend\b", re.IGNORECASE)
_WEEKDAY_RE = re.compile(r"\bweek\s?day\b", re.IGNORECASE)


def _extract_labelled(text: str) -> dict:
    """Read the explicitly-labelled Part-1 features out of a reply."""
    found: dict = {}
    for name, pattern, cast in _LABELLED_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        raw = next((g for g in match.groups() if g is not None), None)
        if raw is None:
            continue
        try:
            found[name] = cast(raw)
        except ValueError:
            continue
    if _WEEKEND_RE.search(text):
        found["is_weekend_order"] = 1
    elif _WEEKDAY_RE.search(text):
        found["is_weekend_order"] = 0
    return found


def _extract_price(text: str) -> float | None:
    match = _PRICE_RE.search(text)
    if not match:
        return None
    raw = match.group(1) or match.group(2)
    try:
        return float(raw.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _extract_payment_method(text: str) -> str | None:
    for pattern, level in _PAYMENT_PATTERNS:
        if pattern.search(text):
            return level
    return None


def _extract_delivery_days(text: str) -> int | None:
    match = _DELIVERY_DELAY_RE.search(text) or _DELIVERY_TOOK_RE.search(text)
    return int(match.group(1)) if match else None


def _extract_category(text: str) -> str | None:
    """Match against the real Part-1 category levels and the product
    catalog's subcategories (e.g. "running shoes" -> Footwear) -- driven by
    actual project data, not an arbitrary keyword list.
    """
    from part3.products import load_catalog

    lowered = text.lower()
    for category in ("Apparel", "Footwear", "Electronics", "Home", "Beauty"):
        if category.lower() in lowered:
            return category

    catalog = load_catalog()
    subcategories = sorted({p["subcategory"] for p in catalog}, key=len, reverse=True)
    for subcategory in subcategories:
        singular = subcategory.lower().rstrip("s")
        if subcategory.lower() in lowered or singular in lowered:
            return next(p["category"] for p in catalog if p["subcategory"] == subcategory)

    # Customers use the short form of a catalogue word: "phone" for Smartphones,
    # "shoes" for Running shoes. Match a word of the message *inside* a
    # subcategory name, still driven entirely by products.json rather than a
    # hand-written synonym list. Five characters minimum, so short common words
    # ("care", "wear") cannot drag an unrelated sentence into a category.
    words = {w for w in re.findall(r"[a-z]+", lowered) if len(w) >= 5}
    for subcategory in subcategories:
        haystack = subcategory.lower()
        if any(word in haystack for word in words):
            return next(p["category"] for p in catalog if p["subcategory"] == subcategory)
    return None


def extract_order_slots(text: str) -> dict:
    """Pull whatever order-risk features can be read reliably out of free text.

    Returns only the keys that were actually found -- an empty dict means
    nothing new was mentioned in this turn, so the caller's merge is a no-op.
    """
    slots: dict = {}
    price = _extract_price(text)
    if price is not None:
        slots["price_inr"] = price
    payment_method = _extract_payment_method(text)
    if payment_method is not None:
        slots["payment_method"] = payment_method
    delivery_days = _extract_delivery_days(text)
    if delivery_days is not None:
        slots["delivery_days"] = delivery_days
    category = _extract_category(text)
    if category is not None:
        slots["product_category"] = category
    slots.update(_extract_labelled(text))
    return slots
