"""Part 2 — cache the frozen ResNet-18 backbone's features once.

The backbone is frozen during feature extraction, so its output for a given
image never changes between epochs. Running it once and caching the 512-d
vectors turns 15 epochs x 70,000 forward passes into 1 x 70,000, then trains
the head on cached tensors in seconds. Mathematically identical, roughly an
order of magnitude less wall-clock.

    python3 -m part2.cache_features

Writes data/feature_cache/{train,val,test}.pt (gitignored — regenerate on clone).
"""

import time

import torch

from part2.config import (
    FEATURE_CACHE_DIR,
    FEATURE_EXTRACT_BATCH_SIZE,
    get_device,
)
from part2.data import build_splits, make_loader, subset_targets
from part2.model import build_model, feature_extractor_from, freeze_backbone


@torch.no_grad()
def extract(extractor, loader, device, label: str) -> tuple[torch.Tensor, torch.Tensor]:
    feats, labels = [], []
    total = len(loader.dataset)
    seen = 0
    start = time.time()
    for batch_no, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        out = extractor(images)                    # (N, 512, 1, 1)
        feats.append(out.flatten(1).cpu())         # (N, 512)
        labels.append(targets)
        seen += images.size(0)
        # Progress every 20 batches keeps logs readable when piped to a file.
        if batch_no % 20 == 0 or seen == total:
            rate = seen / (time.time() - start)
            print(f"  {label}: {seen}/{total} images ({rate:.0f} img/s)", flush=True)
    return torch.cat(feats), torch.cat(labels)


def main() -> None:
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"device: {device}")

    train_ds, val_ds, test_ds, train_idx, val_idx = build_splits()
    print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # Sanity: the validation split must be stratified and drawn from train only.
    val_targets = subset_targets(val_ds)
    counts = torch.bincount(val_targets, minlength=10).tolist()
    print(f"val class counts: {counts}")
    assert len(set(counts)) == 1, "validation split is not balanced across classes"
    assert not (set(train_idx.tolist()) & set(val_idx.tolist())), "train/val overlap"

    model = freeze_backbone(build_model(pretrained=True))
    extractor = feature_extractor_from(model).to(device).eval()

    for name, dataset in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        loader = make_loader(dataset, FEATURE_EXTRACT_BATCH_SIZE, shuffle=False)
        started = time.time()
        feats, labels = extract(extractor, loader, device, name)
        torch.save({"features": feats, "labels": labels},
                   FEATURE_CACHE_DIR / f"{name}.pt")
        print(f"  saved {name}.pt  features={tuple(feats.shape)}  "
              f"labels={tuple(labels.shape)}  ({time.time() - started:.1f}s)")

    torch.save({"train_idx": torch.as_tensor(train_idx),
                "val_idx": torch.as_tensor(val_idx)},
               FEATURE_CACHE_DIR / "split_indices.pt")
    print("\nFeature cache complete.")


if __name__ == "__main__":
    main()
