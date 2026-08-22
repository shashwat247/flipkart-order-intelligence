"""Part 2 — Fashion-MNIST loading, the stratified validation split, and the
single preprocessing transform shared by training and inference.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from part2.config import (
    DATA_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    SPLIT_SEED,
    VAL_SIZE,
)


def build_transform() -> transforms.Compose:
    """The preprocessing contract, in one place.

    PIL 'L' (28x28 grey) -> resize to 224x224 -> replicate to 3 channels ->
    tensor -> ImageNet normalisation. Deterministic: no augmentation, because
    the frozen-backbone features are cached exactly once and reused every epoch.
    """
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_raw_datasets():
    """The canonical 60,000 / 10,000 Fashion-MNIST splits from Zalando Research.

    Downloaded by torchvision on first run (no login, no API key) from the
    pinned canonical source behind `torchvision.datasets.FashionMNIST`.
    """
    tf = build_transform()
    train_full = FashionMNIST(root=str(DATA_DIR), train=True, download=True, transform=tf)
    test = FashionMNIST(root=str(DATA_DIR), train=False, download=True, transform=tf)
    return train_full, test


def stratified_train_val_indices(targets, val_size: int = VAL_SIZE, seed: int = SPLIT_SEED):
    """Split train indices into train/val with equal representation per class.

    Stratified by construction: an equal quota is drawn from each class rather
    than relying on a random split happening to be balanced.
    """
    targets = np.asarray(targets)
    classes = np.unique(targets)
    per_class = val_size // len(classes)
    rng = np.random.default_rng(seed)

    val_idx = []
    for c in classes:
        idx = np.flatnonzero(targets == c)
        val_idx.append(rng.choice(idx, size=per_class, replace=False))
    val_idx = np.sort(np.concatenate(val_idx))

    mask = np.ones(len(targets), dtype=bool)
    mask[val_idx] = False
    train_idx = np.flatnonzero(mask)
    return train_idx, val_idx


def build_splits():
    """Return (train_subset, val_subset, test_dataset, train_idx, val_idx)."""
    train_full, test = load_raw_datasets()
    targets = train_full.targets.numpy()
    train_idx, val_idx = stratified_train_val_indices(targets)
    return (
        Subset(train_full, train_idx.tolist()),
        Subset(train_full, val_idx.tolist()),
        test,
        train_idx,
        val_idx,
    )


def make_loader(dataset, batch_size: int, shuffle: bool = False, num_workers: int = 4):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=False, persistent_workers=num_workers > 0,
    )


def subset_targets(dataset) -> torch.Tensor:
    """Labels of a Subset (or plain dataset), in loader order."""
    if isinstance(dataset, Subset):
        return dataset.dataset.targets[torch.as_tensor(dataset.indices)]
    return dataset.targets
