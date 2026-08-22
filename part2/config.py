"""Part 2 configuration — every constant Part 2 and Part 3 must agree on.

The preprocessing constants here are the contract between training and
inference: `classify_product_image` in Part 3 rebuilds this exact transform, so
a change here without a retrain would silently corrupt predictions.
"""

import os
import ssl
from pathlib import Path

import torch


def _ensure_ssl_certificates() -> None:
    """Point urllib at certifi's CA bundle on machines that lack a system one.

    The python.org macOS builds ship without the root certificates wired into
    OpenSSL, so torchvision's one-time downloads (Fashion-MNIST and the ResNet-18
    ImageNet weights) fail with CERTIFICATE_VERIFY_FAILED on a fresh clone. This
    sets the standard env vars from certifi if the user has not set them already.
    It only affects certificate *lookup* — verification stays on.
    """
    if os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import certifi
    except ImportError:
        return
    bundle = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", bundle)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)
    ssl._create_default_https_context = lambda *a, **k: ssl.create_default_context(
        cafile=bundle
    )


_ensure_ssl_certificates()

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"
FEATURE_CACHE_DIR = DATA_DIR / "feature_cache"

MODEL_PATH = MODELS_DIR / "product_classifier.pt"
METADATA_PATH = MODELS_DIR / "product_classifier_metadata.json"

# Fashion-MNIST label order, fixed by Zalando Research. Index i in the model's
# output vector means CLASS_NAMES[i] — never re-order this list.
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
NUM_CLASSES = len(CLASS_NAMES)

# --- preprocessing contract -------------------------------------------------
# Fashion-MNIST ships 28x28 single-channel images. ResNet-18 was pretrained on
# 3-channel 224x224 ImageNet crops, so we replicate the grey channel 3x and
# resize up to the backbone's native resolution, then normalise with the
# ImageNet statistics the pretrained weights expect.
INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BACKBONE = "resnet18"
FEATURE_DIM = 512          # ResNet-18 penultimate (post-avgpool) width

# --- data splits ------------------------------------------------------------
# Standard Fashion-MNIST splits: 60,000 train / 10,000 test. We carve a
# stratified validation set out of the TRAIN half only; the 10,000 test images
# stay untouched until the single final evaluation.
VAL_SIZE = 10_000          # 1,000 per class — comfortably above the 5,000 floor
SPLIT_SEED = 42

# --- head training ----------------------------------------------------------
HEAD_EPOCHS = 15
HEAD_LR = 1e-3
HEAD_BATCH_SIZE = 256
FEATURE_EXTRACT_BATCH_SIZE = 256

# --- conditional fine-tuning ------------------------------------------------
# Only triggered if feature-extraction validation accuracy lands below this.
FINETUNE_TRIGGER_ACC = 0.80
FINETUNE_EPOCHS = 2
FINETUNE_LR = 1e-4         # an order of magnitude below the head LR
FINETUNE_BATCH_SIZE = 128


def get_device() -> torch.device:
    """MPS on Apple Silicon, CUDA where available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
