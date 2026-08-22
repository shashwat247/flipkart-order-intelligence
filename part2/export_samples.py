"""Part 2 Task 8 — export real test-split images as actual PNG files.

Fashion-MNIST ships as raw IDX binary, not as a folder of images, so Part 3's
`classify_product_image(image_path)` tool has nothing to point at until we
write some out. This exports one genuine image per class from the **test**
split at its native 28x28 resolution (the upscaling to 224x224 happens inside
the preprocessing transform at inference time, exactly as in training).

    python3 -m part2.export_samples

Writes data/sample_images/NN_<label>.png plus a manifest recording which test
index each file came from, so any grader can verify the provenance.
"""

import json
import re

import numpy as np
from PIL import Image
from torchvision.datasets import FashionMNIST

from part2.config import CLASS_NAMES, DATA_DIR, SAMPLE_IMAGES_DIR

MANIFEST_PATH = SAMPLE_IMAGES_DIR / "manifest.json"


def slugify(label: str) -> str:
    """'T-shirt/top' -> 'tshirt_top', 'Ankle boot' -> 'ankle_boot'."""
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def main() -> None:
    SAMPLE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # transform=None so we get the raw PIL image, not a normalised tensor.
    test = FashionMNIST(root=str(DATA_DIR), train=False, download=True, transform=None)
    targets = test.targets.numpy()

    manifest = []
    for class_index, label in enumerate(CLASS_NAMES):
        # First test-split occurrence of this class — deterministic, no cherry-picking.
        test_index = int(np.flatnonzero(targets == class_index)[0])
        image, actual_label = test[test_index]
        assert actual_label == class_index, "label mismatch while exporting"

        filename = f"{class_index:02d}_{slugify(label)}.png"
        path = SAMPLE_IMAGES_DIR / filename
        image.save(path, format="PNG")

        # Read the file back and confirm it is a real, correct PNG on disk.
        reloaded = Image.open(path)
        assert reloaded.format == "PNG", f"{filename} is not a PNG"
        assert reloaded.size == (28, 28), f"{filename} has size {reloaded.size}"
        assert np.array_equal(np.array(reloaded.convert("L")), np.array(image)), \
            f"{filename} does not round-trip losslessly"

        manifest.append({
            "file": filename,
            "true_label": label,
            "true_index": class_index,
            "source": "Fashion-MNIST test split",
            "test_split_index": test_index,
        })
        print(f"  wrote {filename:<22s} true label='{label}' "
              f"(test split index {test_index})")

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nExported {len(manifest)} real PNGs to "
          f"{SAMPLE_IMAGES_DIR.relative_to(DATA_DIR.parent)}/")
    print("Filenames carry the true label for human convenience only — Part 3's "
          "image tool never reads them.")


if __name__ == "__main__":
    main()
