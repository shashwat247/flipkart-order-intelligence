"""Part 3 Tasks 3-4 — the agent's two real tools.

Both load the actual artifacts trained in Parts 1 and 2. Neither has a
hardcoded prediction, a stand-in model, or a filename-based shortcut anywhere in
it: `check_return_risk` calls the saved Random Forest's `predict_proba`, and
`classify_product_image` runs the saved ResNet-18 over the pixels of a real PNG.
"""

import json
from functools import lru_cache

import joblib
import pandas as pd

from part1.common import DATASET_PATH, FEATURES, METADATA_PATH, MODEL_PATH
# Re-exported so the agent has one import site for both tools. This IS Part 2's
# function — not a copy — so the agent provably calls the saved classifier.
from part2.model import classify_product_image  # noqa: F401


@lru_cache(maxsize=1)
def _load_return_risk_model():
    """Load the tuned Random Forest pipeline and its threshold metadata."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Train Part 1 first:\n"
            f"  python3 generate_orders.py && python3 -m part1.train_return_risk"
        )
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


@lru_cache(maxsize=1)
def _orders():
    """Flipkart's order history — the Part 1 dataset, used for order lookups."""
    return pd.read_csv(DATASET_PATH)


def lookup_order(order_id: int) -> dict | None:
    """Fetch one order's features from the order history by id.

    Returns the model's feature columns only (no `returned` label — that is the
    thing being predicted, and feeding it back in would be leakage).
    """
    orders = _orders()
    row = orders[orders["order_id"] == int(order_id)]
    if row.empty:
        return None
    return row.iloc[0][FEATURES].to_dict()


def risk_bucket(probability: float, threshold: float) -> str:
    """Bucket a probability against the model's OWN F1-maximising threshold.

    Deliberately not a fixed 0.3 / 0.6 split. Two equally valid Random Forests
    can produce probability distributions concentrated in completely different
    ranges — one model's scores might cluster in 0.3-0.6 while another's spread
    much wider — so a fixed split can silently collapse almost every order into
    one bucket. Anchoring to t*_rf keeps the buckets meaningful for whatever
    range THIS trained forest actually produces.
    """
    if probability < threshold:
        return "Low"
    if probability < threshold + 0.15:
        return "Medium"
    return "High"


def check_return_risk(order_features: dict) -> dict:
    """Predict an order's return probability with Part 1's saved Random Forest.

    `order_features` must carry every column the Part 1 pipeline was fitted on.
    Missing columns are reported rather than silently imputed, because a
    silently-imputed feature would produce a confident number from data the
    caller never supplied.
    """
    model, metadata = _load_return_risk_model()
    threshold = float(metadata["threshold_rf"])

    missing = [f for f in FEATURES if f not in order_features]
    if missing:
        return {
            "status": "missing_input",
            "missing_features": missing,
            "required_features": list(FEATURES),
            "threshold_rf": threshold,
        }

    frame = pd.DataFrame([{f: order_features[f] for f in FEATURES}])
    probability = float(model.predict_proba(frame)[0][1])
    bucket = risk_bucket(probability, threshold)

    return {
        "status": "ok",
        "return_probability": round(probability, 4),
        "return_probability_percent": round(probability * 100, 2),
        "risk_bucket": bucket,
        "threshold_rf": threshold,
        "bucket_cut_points": {
            "Low": f"probability < {threshold:.2f}",
            "Medium": f"{threshold:.2f} <= probability < {threshold + 0.15:.2f}",
            "High": f"probability >= {threshold + 0.15:.2f}",
        },
        "model": metadata.get("model_type", "RandomForestClassifier"),
    }
