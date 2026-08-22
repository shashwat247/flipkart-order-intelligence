"""Shared Part 1 plumbing: feature lists, the leakage-free split, the
preprocessing pipeline and the threshold sweep.

Everything that Part 1's scripts must agree on lives here, so the train script,
the evaluation script and the analysis scripts can never drift apart on which
columns are features or how the train/test split is drawn.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Repo root, resolved relative to this file so the project works after cloning
# from any directory.
ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "orders_dataset.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODEL_PATH = MODELS_DIR / "return_risk_model.pkl"
METADATA_PATH = MODELS_DIR / "return_risk_metadata.json"

TARGET = "returned"

# order_id is excluded on purpose: it is a row identifier, not a signal. Leaving
# it in would let a tree memorise arbitrary index ranges.
NUMERIC_FEATURES = [
    "price_inr",
    "discount_pct",
    "customer_tenure_days",
    "num_previous_orders",
    "num_previous_returns",
    "delivery_distance_km",
    "delivery_days",
    "is_weekend_order",
    "rating_given",
]
CATEGORICAL_FEATURES = ["product_category", "payment_method"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Threshold sweep grid required by Task 5: 0.10 -> 0.90, step 0.02.
THRESHOLD_GRID = np.round(np.arange(0.10, 0.9001, 0.02), 2)


def load_dataset() -> pd.DataFrame:
    """Load orders_dataset.csv, with a clear error if Task 1 has not been run."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} not found. Run `python3 generate_orders.py` first."
        )
    return pd.read_csv(DATASET_PATH)


def split_data(df: pd.DataFrame):
    """Stratified 80/20 split. This is the ONLY place the split is defined."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def build_preprocessor() -> ColumnTransformer:
    """Median-impute + scale numerics, mode-impute + one-hot the categoricals.

    Returned unfitted: the caller puts it inside a Pipeline and calls fit() on
    the TRAINING split only, which is what keeps test statistics (medians,
    modes, means/stds, category levels) out of the transform.
    """
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def sweep_thresholds(y_true, proba, grid=THRESHOLD_GRID) -> pd.DataFrame:
    """Precision / recall / F1 for class 1 at every threshold in the grid.

    `proba` is P(returned=1). Used twice in this project: once on the Logistic
    Regression (Task 5) and once on the Random Forest's own predict_proba
    (Task 9) — the two live on different probability scales, which is exactly
    why the sweep is re-run rather than the threshold being reused.
    """
    rows = []
    for t in grid:
        pred = (proba >= t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", pos_label=1, zero_division=0
        )
        rows.append(
            {
                "threshold": float(t),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "n_flagged": int(pred.sum()),
            }
        )
    return pd.DataFrame(rows)


def best_threshold(sweep: pd.DataFrame) -> pd.Series:
    """Row of the sweep with the highest F1 (lowest threshold wins ties)."""
    return sweep.loc[sweep["f1"].idxmax()]


def metrics_at(y_true, proba, threshold: float) -> dict:
    """Precision / recall / F1 for class 1 at one specific threshold."""
    pred = (proba >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", pos_label=1, zero_division=0
    )
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def feature_names_out(preprocessor: ColumnTransformer) -> list[str]:
    """Transformed column names, in the order the model sees them.

    Needed to line `.feature_importances_` up with human-readable names once
    the one-hot encoder has expanded the two categorical columns.
    """
    return list(preprocessor.get_feature_names_out())


def original_feature_of(transformed_name: str) -> str:
    """Map a transformed column back to the raw dataset column it came from.

    'num__price_inr'                 -> 'price_inr'
    'cat__payment_method_COD'        -> 'payment_method'
    """
    stripped = transformed_name.split("__", 1)[-1]
    if stripped in NUMERIC_FEATURES:
        return stripped
    for cat in CATEGORICAL_FEATURES:
        if stripped.startswith(cat + "_"):
            return cat
    return stripped


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
