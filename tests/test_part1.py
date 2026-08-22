"""Part 1 tests — dataset, preprocessing, saved artifact and t*_rf."""

import json
import shutil
import subprocess
import sys

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from part1.common import (
    CATEGORICAL_FEATURES,
    DATASET_PATH,
    FEATURES,
    METADATA_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    ROOT,
    best_threshold,
    load_dataset,
    split_data,
    sweep_thresholds,
)

EXPECTED_COLUMNS = [
    "order_id", "product_category", "price_inr", "discount_pct", "payment_method",
    "customer_tenure_days", "num_previous_orders", "num_previous_returns",
    "delivery_distance_km", "delivery_days", "is_weekend_order", "rating_given",
    "returned",
]


@pytest.fixture(scope="module")
def df():
    return load_dataset()


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def metadata():
    return json.loads(METADATA_PATH.read_text())


# --------------------------------------------------------------- the dataset
def test_generator_produces_6000_rows(df):
    assert len(df) == 6000


def test_dataset_has_13_columns_in_the_expected_order(df):
    assert list(df.columns) == EXPECTED_COLUMNS


def test_required_feature_columns_exist(df):
    for column in FEATURES:
        assert column in df.columns


def test_order_id_is_an_identifier_not_a_feature():
    # order_id must never be handed to the model as a predictor.
    assert "order_id" not in NUMERIC_FEATURES
    assert "order_id" not in CATEGORICAL_FEATURES
    assert "order_id" not in FEATURES


def test_return_rate_in_expected_range(df):
    assert 0.18 <= df["returned"].mean() <= 0.27


def test_missing_rating_rate_in_expected_range(df):
    assert 0.08 <= df["rating_given"].isna().mean() <= 0.18


def test_missingness_is_mar_on_payment_method(df):
    """Missingness must depend on the OBSERVED payment_method column."""
    cod = df["payment_method"] == "COD"
    cod_rate = df.loc[cod, "rating_given"].isna().mean()
    other_rate = df.loc[~cod, "rating_given"].isna().mean()
    assert cod_rate > other_rate * 2, "expected a strong COD/non-COD missingness gap"


def test_generator_is_deterministic(tmp_path):
    """Re-running the generator must reproduce the committed CSV byte for byte."""
    script = tmp_path / "generate_orders.py"
    shutil.copy(ROOT / "generate_orders.py", script)
    subprocess.run([sys.executable, str(script)], cwd=tmp_path, check=True,
                   capture_output=True)
    regenerated = (tmp_path / "orders_dataset.csv").read_bytes()
    assert regenerated == DATASET_PATH.read_bytes()


# --------------------------------------------------------- the saved artifact
def test_saved_model_loads(model):
    assert model is not None


def test_saved_model_is_a_pipeline_with_preprocessing(model):
    assert isinstance(model, Pipeline)
    assert "prep" in model.named_steps


def test_saved_model_is_the_random_forest_not_the_logistic_regression(model):
    assert isinstance(model.named_steps["clf"], RandomForestClassifier)


def test_saved_model_has_predict_and_predict_proba(model):
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_saved_model_prediction_works_on_a_single_row(df, model):
    row = df[FEATURES].iloc[[0]]
    proba = model.predict_proba(row)
    assert proba.shape == (1, 2)
    assert 0.0 <= float(proba[0][1]) <= 1.0


def test_saved_model_handles_a_missing_rating(df, model):
    """The imputer must absorb a NaN rather than the pipeline raising."""
    row = df[FEATURES].iloc[[0]].copy()
    row.loc[:, "rating_given"] = float("nan")
    assert 0.0 <= float(model.predict_proba(row)[0][1]) <= 1.0


# ------------------------------------------------------------------- t*_rf
def test_threshold_metadata_exists_and_is_numeric(metadata):
    assert isinstance(metadata["threshold_rf"], (int, float))
    assert 0.10 <= metadata["threshold_rf"] <= 0.90


def test_metadata_records_the_random_forest_as_the_model(metadata):
    assert metadata["model_type"] == "RandomForestClassifier"
    assert metadata["threshold_selection_metric"] == "F1"
    assert metadata["threshold_selection_split"] == "held_out_test_split"


def test_threshold_rf_is_reproducible_from_the_saved_models_own_proba(df, model, metadata):
    """t*_rf must come from THIS model's predict_proba, not the LogReg's."""
    _X_train, X_test, _y_train, y_test = split_data(df)
    proba = model.predict_proba(X_test)[:, 1]
    recomputed = float(best_threshold(sweep_thresholds(y_test, proba))["threshold"])
    assert recomputed == pytest.approx(metadata["threshold_rf"], abs=1e-9)


def test_metadata_roc_auc_values_are_reported(metadata):
    assert metadata["best_cv_roc_auc"] >= 0.58
    assert metadata["test_roc_auc"] >= 0.58
    assert abs(metadata["best_cv_roc_auc"] - metadata["test_roc_auc"]) <= 0.05
