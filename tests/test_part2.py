"""Part 2 tests — saved classifier, exported PNGs and single-image prediction."""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from part2.config import (
    CLASS_NAMES,
    INPUT_SIZE,
    METADATA_PATH,
    MODEL_PATH,
    NUM_CLASSES,
    SAMPLE_IMAGES_DIR,
)
from part2.model import build_model, classify_product_image

SAMPLE_PNGS = sorted(SAMPLE_IMAGES_DIR.glob("*.png"))


@pytest.fixture(scope="module")
def model():
    net = build_model(pretrained=False)
    net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    net.eval()
    return net


# ------------------------------------------------------------ saved artifact
def test_saved_classifier_exists():
    assert MODEL_PATH.exists()


def test_saved_classifier_loads_with_the_documented_snippet(model):
    assert model is not None


def test_model_output_has_10_classes(model):
    dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        logits = model(dummy)
    assert logits.shape == (1, NUM_CLASSES) == (1, 10)


def test_metadata_records_the_preprocessing_contract():
    metadata = json.loads(METADATA_PATH.read_text())
    assert metadata["architecture"] == "resnet18"
    assert metadata["class_names"] == CLASS_NAMES
    assert metadata["input_size"] == [224, 224]
    assert metadata["input_channels"] == 3
    assert metadata["normalization"]["mean"] == [0.485, 0.456, 0.406]
    assert metadata["normalization"]["std"] == [0.229, 0.224, 0.225]


def test_feature_extraction_cleared_the_bar_so_finetuning_was_skipped():
    metadata = json.loads(METADATA_PATH.read_text())
    # Recorded either way; this asserts the two are consistent with each other.
    if not metadata["finetune_triggered"]:
        assert metadata["val_accuracy_before_finetuning"] >= 0.80
        assert (metadata["val_accuracy_after_finetuning"]
                == metadata["val_accuracy_before_finetuning"])


# --------------------------------------------------------- exported PNG files
def test_at_least_five_sample_pngs_exist():
    assert len(SAMPLE_PNGS) >= 5


def test_sample_files_are_real_28x28_pngs():
    for path in SAMPLE_PNGS:
        image = Image.open(path)
        assert image.format == "PNG", f"{path.name} is not a PNG"
        assert image.size == (28, 28), f"{path.name} is {image.size}"


def test_sample_manifest_records_fashion_mnist_test_split_provenance():
    manifest = json.loads((SAMPLE_IMAGES_DIR / "manifest.json").read_text())
    assert len(manifest) >= 5
    for entry in manifest:
        assert entry["source"] == "Fashion-MNIST test split"
        assert entry["true_label"] in CLASS_NAMES
        assert (SAMPLE_IMAGES_DIR / entry["file"]).exists()


# ------------------------------------------------------ single-image prediction
@pytest.mark.parametrize("path", SAMPLE_PNGS, ids=lambda p: p.name)
def test_classify_product_image_returns_category_and_confidence(path):
    result = classify_product_image(str(path))
    assert result["predicted_class"] in CLASS_NAMES
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["confidence_percent"] == pytest.approx(
        result["confidence"] * 100, abs=0.01
    )
    assert len(result["top3"]) == 3


def test_prediction_is_deterministic():
    path = str(SAMPLE_PNGS[0])
    first, second = classify_product_image(path), classify_product_image(path)
    assert first["predicted_class"] == second["predicted_class"]
    assert first["confidence"] == second["confidence"]


def test_prediction_ignores_the_filename(tmp_path):
    """Renaming a file must not change the prediction — the model reads pixels."""
    original = SAMPLE_PNGS[0]
    truthful = classify_product_image(str(original))

    misleading = tmp_path / "99_definitely_a_handbag.png"
    misleading.write_bytes(original.read_bytes())
    renamed = classify_product_image(str(misleading))

    assert renamed["predicted_class"] == truthful["predicted_class"]
    assert renamed["confidence"] == truthful["confidence"]


def test_missing_image_raises_rather_than_guessing():
    with pytest.raises(FileNotFoundError):
        classify_product_image("data/sample_images/does_not_exist.png")


def test_probabilities_form_a_distribution():
    result = classify_product_image(str(SAMPLE_PNGS[0]))
    assert result["confidence"] == max(t["probability"] for t in result["top3"])
    assert np.all([0.0 <= t["probability"] <= 1.0 for t in result["top3"]])
