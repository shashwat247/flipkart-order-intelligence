"""Part 2 — model construction, artifact loading, and single-image prediction.

The saved artifact is a plain `state_dict` of a torchvision ResNet-18 whose
`fc` has been replaced by a 10-class head. That keeps reconstruction trivial
and unambiguous: build the same architecture, load the weights, done.

`classify_product_image(path)` is the function Part 3's image tool calls. It
does real inference — it never looks at the filename.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

from part2.config import (
    BACKBONE,
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    METADATA_PATH,
    MODEL_PATH,
    NUM_CLASSES,
    get_device,
)
from part2.data import build_transform


def build_model(pretrained: bool = True) -> nn.Module:
    """ResNet-18 with a fresh 10-class head.

    `pretrained=True` pulls the ImageNet weights (training); `False` gives the
    bare architecture to load our own saved weights into (inference).
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def freeze_backbone(model: nn.Module) -> nn.Module:
    """Freeze everything except the new classifier head.

    This is the feature-extraction stage: conv1/bn1 and layer1-layer4 keep
    their ImageNet weights and are never updated, so their output for a given
    image is constant and can be cached (see part2/cache_features.py).
    """
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    return model


def unfreeze_late_layers(model: nn.Module) -> nn.Module:
    """Gradual unfreezing: release `layer4` (+ head), keep early/middle frozen.

    Only used if feature extraction alone misses the validation-accuracy bar.
    """
    freeze_backbone(model)
    for param in model.layer4.parameters():
        param.requires_grad = True
    return model


def feature_extractor_from(model: nn.Module) -> nn.Module:
    """Everything up to and including avgpool — the frozen part we cache.

    Output is (N, 512, 1, 1); flattening gives the 512-d vector the head sees.
    Training a Linear(512, 10) on these cached vectors is mathematically
    identical to re-running the frozen backbone every epoch, but turns an
    hours-long CPU loop into one forward pass plus a fast head fit.
    """
    return nn.Sequential(*list(model.children())[:-1])


def trainable_parameters(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------- inference
@lru_cache(maxsize=1)
def _load_for_inference():
    """Load the saved classifier once and reuse it across calls.

    Cached because Part 3's agent may classify several images in one session and
    rebuilding a ResNet-18 per call would dominate the runtime.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Train it first:\n"
            f"  python3 -m part2.cache_features\n"
            f"  python3 -m part2.train_product_classifier"
        )
    device = get_device()
    model = build_model(pretrained=False)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    metadata = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
    class_names = metadata.get("class_names", CLASS_NAMES)
    return model, device, class_names, build_transform()


def classify_product_image(image_path: str) -> dict:
    """Classify one real image file with the saved Part 2 model.

    The filename is never inspected — it exists only so a human reading
    `data/sample_images/` can tell what the true label was. The returned label
    comes from `argmax` over the model's softmax output.

    Returns: predicted_class, predicted_index, confidence (0-1),
             confidence_percent, top3, image_path.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    model, device, class_names, transform = _load_for_inference()

    # convert("L") makes this robust to PNGs saved with an alpha or palette
    # channel; the transform then replicates that single channel to 3.
    image = Image.open(path).convert("L")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    index = int(np.argmax(probabilities))
    order = np.argsort(probabilities)[::-1][:3]
    return {
        "predicted_class": class_names[index],
        "predicted_index": index,
        "confidence": round(float(probabilities[index]), 4),
        "confidence_percent": round(float(probabilities[index]) * 100, 2),
        "top3": [
            {"label": class_names[int(i)], "probability": round(float(probabilities[i]), 4)}
            for i in order
        ],
        "image_path": str(path),
    }


def save_metadata(extra: dict | None = None) -> None:
    """Persist enough to rebuild the model reliably from the .pt alone."""
    metadata = {
        "architecture": BACKBONE,
        "head": f"Linear({512}, {NUM_CLASSES})",
        "class_names": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
        "input_size": [INPUT_SIZE, INPUT_SIZE],
        "input_channels": 3,
        "source_channels": 1,
        "channel_handling": "grayscale replicated to 3 channels",
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "state_dict_file": MODEL_PATH.name,
        "load_snippet": (
            "from part2.model import build_model; import torch; "
            "m = build_model(pretrained=False); "
            "m.load_state_dict(torch.load('models/product_classifier.pt', "
            "map_location='cpu')); m.eval()"
        ),
    }
    if extra:
        metadata.update(extra)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
