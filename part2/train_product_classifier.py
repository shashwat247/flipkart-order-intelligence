"""Part 2 Tasks 3-4, 7 — train the transfer-learning classifier and save it.

Stage 1 (feature extraction): the backbone is frozen, so its features were
cached once by `part2.cache_features`. Only `fc` (Linear(512, 10)) is trained,
on those cached vectors.

Stage 2 (conditional fine-tuning): if stage-1 validation accuracy lands below
FINETUNE_TRIGGER_ACC, `layer4` is unfrozen (early/middle layers stay frozen) and
training continues at a 10x lower learning rate over the real images. If stage 1
already clears the bar, this stage is skipped and that is recorded.

    python3 -m part2.cache_features            # once
    python3 -m part2.train_product_classifier
"""

import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from part2.config import (
    FEATURE_CACHE_DIR,
    FINETUNE_BATCH_SIZE,
    FINETUNE_EPOCHS,
    FINETUNE_LR,
    FINETUNE_TRIGGER_ACC,
    HEAD_BATCH_SIZE,
    HEAD_EPOCHS,
    HEAD_LR,
    MODEL_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    SPLIT_SEED,
    get_device,
)
from part2.data import build_splits, make_loader
from part2.model import (
    build_model,
    count_parameters,
    save_metadata,
    trainable_parameters,
    unfreeze_late_layers,
)

TRAINING_LOG_PATH = REPORTS_DIR / "part2_training_log.json"


def load_cached(split: str):
    path = FEATURE_CACHE_DIR / f"{split}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run `python3 -m part2.cache_features` first."
        )
    blob = torch.load(path)
    return blob["features"], blob["labels"]


@torch.no_grad()
def evaluate_head(head, features, labels, device) -> float:
    head.eval()
    logits = head(features.to(device))
    return (logits.argmax(1).cpu() == labels).float().mean().item()


@torch.no_grad()
def evaluate_full(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for images, targets in loader:
        preds = model(images.to(device)).argmax(1).cpu()
        correct += (preds == targets).sum().item()
        total += targets.numel()
    return correct / total


def train_head(device) -> tuple[nn.Linear, list, float]:
    """Stage 1 — fit Linear(512, 10) on the cached frozen-backbone features."""
    Xtr, ytr = load_cached("train")
    Xva, yva = load_cached("val")
    print(f"cached features: train={tuple(Xtr.shape)} val={tuple(Xva.shape)}")

    torch.manual_seed(SPLIT_SEED)
    head = nn.Linear(Xtr.shape[1], 10).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=HEAD_LR)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=HEAD_BATCH_SIZE, shuffle=True)

    history = []
    best_acc, best_state = 0.0, None
    for epoch in range(1, HEAD_EPOCHS + 1):
        head.train()
        running, seen = 0.0, 0
        started = time.time()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(head(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            seen += xb.size(0)
        train_loss = running / seen
        val_acc = evaluate_head(head, Xva, yva, device)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 5),
                        "val_accuracy": round(val_acc, 5),
                        "seconds": round(time.time() - started, 2)})
        print(f"  epoch {epoch:2d}/{HEAD_EPOCHS}  train_loss={train_loss:.4f}  "
              f"val_acc={val_acc:.4f}  ({time.time() - started:.1f}s)")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state)          # keep the best epoch, not the last
    return head, history, best_acc


def fine_tune(model, device) -> tuple[list, float]:
    """Stage 2 — gradual unfreezing of layer4 at a lower LR, on real images."""
    train_ds, val_ds, _test_ds, _tr, _va = build_splits()
    train_loader = make_loader(train_ds, FINETUNE_BATCH_SIZE, shuffle=True)
    val_loader = make_loader(val_ds, FINETUNE_BATCH_SIZE, shuffle=False)

    unfreeze_late_layers(model)
    total_p, trainable_p = count_parameters(model)
    print(f"  fine-tuning: {trainable_p:,}/{total_p:,} parameters trainable "
          f"(layer4 + fc; conv1/bn1/layer1-3 stay frozen)")

    optimizer = torch.optim.Adam(trainable_parameters(model), lr=FINETUNE_LR)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_acc, best_state = 0.0, None
    for epoch in range(1, FINETUNE_EPOCHS + 1):
        model.train()
        running, seen = 0.0, 0
        started = time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            seen += xb.size(0)
        val_acc = evaluate_full(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": round(running / seen, 5),
                        "val_accuracy": round(val_acc, 5),
                        "seconds": round(time.time() - started, 2)})
        print(f"  fine-tune epoch {epoch}/{FINETUNE_EPOCHS}  "
              f"train_loss={running / seen:.4f}  val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return history, best_acc


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"device: {device}")

    # ------------------------------------------------- stage 1: feature extraction
    print("\n[Stage 1] Feature extraction — frozen backbone, head only")
    head, head_history, val_acc_before = train_head(device)
    print(f"[Stage 1] best validation accuracy = {val_acc_before:.4f}")

    # Fold the trained head into a full ResNet-18. The backbone weights here are
    # the same ImageNet weights that produced the cached features, so this model
    # is exactly the one the cached-feature training implied.
    model = build_model(pretrained=True)
    model.fc.load_state_dict(head.state_dict())
    model.to(device)

    # ------------------------------------------------ stage 2: conditional fine-tune
    fine_tuned = val_acc_before < FINETUNE_TRIGGER_ACC
    finetune_history, val_acc_after = [], val_acc_before
    if fine_tuned:
        print(f"\n[Stage 2] validation accuracy {val_acc_before:.4f} < "
              f"{FINETUNE_TRIGGER_ACC:.2f} -> fine-tuning late layers")
        finetune_history, val_acc_after = fine_tune(model, device)
        print(f"[Stage 2] validation accuracy after fine-tuning = {val_acc_after:.4f}")
    else:
        print(f"\n[Stage 2] SKIPPED — feature extraction alone reached "
              f"{val_acc_before:.4f} >= {FINETUNE_TRIGGER_ACC:.2f}, so unfreezing "
              f"late layers was not needed.")

    # ------------------------------------------------------------------- save
    model.cpu().eval()
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nSaved {MODEL_PATH.relative_to(MODELS_DIR.parent)}")

    total_p, _ = count_parameters(model)
    log = {
        "device": str(device),
        "backbone": "resnet18 (torchvision, ImageNet1K_V1 weights)",
        "strategy": "freeze conv1/bn1/layer1-4, train new Linear(512,10) head",
        "feature_caching": "frozen-backbone features cached once to data/feature_cache/",
        "optimizer": "Adam",
        "head_learning_rate": HEAD_LR,
        "head_batch_size": HEAD_BATCH_SIZE,
        "head_epochs": HEAD_EPOCHS,
        "split_sizes": {"train": 50000, "val": 10000, "test": 10000},
        "validation_split": "stratified, 1000 per class, carved from the 60k train half",
        "val_accuracy_before_finetuning": round(val_acc_before, 5),
        "finetune_triggered": fine_tuned,
        "finetune_trigger_threshold": FINETUNE_TRIGGER_ACC,
        "val_accuracy_after_finetuning": round(val_acc_after, 5),
        "finetune_learning_rate": FINETUNE_LR if fine_tuned else None,
        "finetune_epochs": FINETUNE_EPOCHS if fine_tuned else 0,
        "total_parameters": total_p,
        "head_history": head_history,
        "finetune_history": finetune_history,
    }
    TRAINING_LOG_PATH.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    save_metadata({
        "val_accuracy_before_finetuning": round(val_acc_before, 5),
        "val_accuracy_after_finetuning": round(val_acc_after, 5),
        "finetune_triggered": fine_tuned,
    })
    print(f"Saved {TRAINING_LOG_PATH.relative_to(MODELS_DIR.parent)}")


if __name__ == "__main__":
    main()
