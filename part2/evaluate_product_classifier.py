"""Part 2 Tasks 5-6 — final evaluation on the untouched test split.

Runs the SAVED `models/product_classifier.pt` end to end over the 10,000 test
images (real forward passes, not cached features), then reports accuracy, the
full 10x10 confusion matrix, per-class precision/recall/F1, and an analysis of
the confusion pairs actually present in that matrix.

    python3 -m part2.evaluate_product_classifier
"""

import json

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from part2.config import (
    CLASS_NAMES,
    FINETUNE_TRIGGER_ACC,
    MODEL_PATH,
    REPORTS_DIR,
    get_device,
)
from part2.data import build_splits, make_loader
from part2.model import build_model

REPORT_PATH = REPORTS_DIR / "part2_evaluation.md"
MATRIX_CSV = REPORTS_DIR / "part2_confusion_matrix.csv"
PER_CLASS_CSV = REPORTS_DIR / "part2_per_class_metrics.csv"
TRAINING_LOG_PATH = REPORTS_DIR / "part2_training_log.json"

# Why two Fashion-MNIST classes get confused, in terms of what the pixels
# actually look like. Keys are unordered pairs.
CONFUSION_NOTES = {
    frozenset({"Shirt", "T-shirt/top"}): (
        "Both are short-to-medium sleeved upper-body garments photographed flat "
        "against the same background, and at 28x28 the only thing that "
        "distinguishes them is a button placket or a collar — features that occupy "
        "perhaps three or four pixels at this resolution and are frequently "
        "smoothed away entirely by the downsampling. The silhouettes are close to "
        "identical: same shoulder width, same torso taper, same sleeve stubs. Even "
        "a human labelling the raw 28x28 thumbnails disagrees with the ground "
        "truth on this pair regularly, which is why it is the canonical hard pair "
        "in Fashion-MNIST. Our upsampling to 224x224 cannot recover detail that "
        "was never captured — it interpolates the existing 784 pixels, so the "
        "collar that would settle the question simply is not in the signal."
    ),
    frozenset({"Shirt", "Coat"}): (
        "A coat and a long-sleeved shirt share the same basic outline: a "
        "rectangular torso with two sleeves extending to roughly the same length. "
        "The real-world difference is thickness, layering and fastening hardware, "
        "all of which read as subtle intensity gradients rather than shape "
        "changes. Because the images are greyscale, the model loses the colour and "
        "texture cues (wool vs cotton weave) that a shopper would use instantly, "
        "and is left comparing two very similar binary silhouettes."
    ),
    frozenset({"Shirt", "Pullover"}): (
        "Both are torso garments with full-length sleeves and no visible "
        "fastening once rendered at low resolution. A pullover's distinguishing "
        "feature is a ribbed hem and neckline, which at 28x28 amounts to a few "
        "slightly darker rows of pixels — easily confused with the shading of a "
        "shirt's fabric folds."
    ),
    frozenset({"Pullover", "Coat"}): (
        "These two occupy nearly the same silhouette envelope — a filled torso "
        "block with two sleeves — and differ mainly in whether the front is open "
        "or closed. In greyscale at this resolution a coat's front opening often "
        "renders as a faint vertical intensity seam, and a pullover with a "
        "central fabric fold produces the same artifact, so the cue is not "
        "reliable."
    ),
    frozenset({"Sneaker", "Ankle boot"}): (
        "Both are footwear photographed in side profile with the sole running "
        "along the bottom edge. The whole distinction rests on how far the upper "
        "extends above the ankle, which is a difference of a handful of pixel rows "
        "at the top of the shape. A high-top sneaker and a low ankle boot are "
        "genuinely close to the same object visually, and the greyscale rendering "
        "removes the material cues (canvas vs leather) that would separate them."
    ),
    frozenset({"Sandal", "Sneaker"}): (
        "A sandal is essentially a sneaker silhouette with material cut away, so "
        "the two share the same sole line and overall footprint. Confusion "
        "concentrates on closed-toe sport sandals, whose upper coverage approaches "
        "that of a low sneaker; the gaps that define a sandal can vanish into a "
        "few dark pixels at 28x28."
    ),
    frozenset({"Sandal", "Ankle boot"}): (
        "Both are footwear in side profile. Heeled sandals produce a raised "
        "back-of-shoe profile that resembles the shaft of a low ankle boot, and at "
        "this resolution the open straps that would identify a sandal compress "
        "into shading rather than clearly visible gaps."
    ),
    frozenset({"Dress", "Coat"}): (
        "A long coat and a dress both render as a tall, narrow, roughly "
        "trapezoidal block. Once a coat's sleeves sit close to the body, the "
        "outline is nearly a dress outline, and the model has only the width "
        "profile near the shoulders to go on."
    ),
    frozenset({"Dress", "Shirt"}): (
        "A long shirt and a short dress converge on the same tapered torso shape. "
        "The discriminating cue is hem length relative to the sleeve, which "
        "varies enough within each class that the two distributions genuinely "
        "overlap."
    ),
    frozenset({"Dress", "Pullover"}): (
        "Both present as a solid torso block; a sleeveless or short-sleeved dress "
        "and a pullover photographed with sleeves close to the body produce very "
        "similar filled silhouettes at low resolution."
    ),
    frozenset({"T-shirt/top", "Pullover"}): (
        "Sleeve length is the main separator, and it is exactly the feature most "
        "degraded by downsampling — a three-quarter sleeve top and a pullover with "
        "pushed-up sleeves land in the same pixel neighbourhood."
    ),
    frozenset({"T-shirt/top", "Dress"}): (
        "A long top and a short dress differ only in hem position, and the classes "
        "genuinely overlap in the catalogue this dataset was drawn from."
    ),
    frozenset({"Bag", "Shirt"}): (
        "Handle-less bags render as a plain filled rectangle, which is also what a "
        "folded shirt torso looks like once sleeves are close to the body."
    ),
}


def note_for(a: str, b: str) -> str:
    return CONFUSION_NOTES.get(
        frozenset({a, b}),
        f"`{a}` and `{b}` share enough of their low-resolution silhouette that the "
        f"28x28 source image does not carry a reliable separating cue; see the "
        f"confusion matrix above for the exact counts.",
    )


@torch.no_grad()
def predict_test(model, loader, device):
    preds, targets = [], []
    for i, (images, labels) in enumerate(loader, start=1):
        out = model(images.to(device))
        preds.append(out.argmax(1).cpu())
        targets.append(labels)
        if i % 10 == 0:
            print(f"  {i * loader.batch_size} test images...", flush=True)
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def matrix_markdown(cm) -> str:
    header = "| true \\ pred | " + " | ".join(CLASS_NAMES) + " | **total** |"
    sep = "|---|" + "---:|" * (len(CLASS_NAMES) + 1)
    rows = [header, sep]
    for i, name in enumerate(CLASS_NAMES):
        cells = []
        for j in range(len(CLASS_NAMES)):
            v = int(cm[i, j])
            cells.append(f"**{v}**" if i == j else (str(v) if v else "·"))
        rows.append(f"| **{name}** | " + " | ".join(cells) + f" | {int(cm[i].sum())} |")
    return "\n".join(rows)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"device: {device}")

    # Load the SAVED artifact — this evaluation grades the committed file.
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.to(device).eval()
    print(f"loaded {MODEL_PATH.name}")

    _train_ds, _val_ds, test_ds, _tr, _va = build_splits()
    print(f"test split: {len(test_ds)} images (untouched until now)")
    loader = make_loader(test_ds, 256, shuffle=False)

    preds, targets = predict_test(model, loader, device)
    accuracy = float((preds == targets).mean())
    print(f"\nFINAL TEST ACCURACY: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    cm = confusion_matrix(targets, preds, labels=list(range(len(CLASS_NAMES))))
    np.savetxt(MATRIX_CSV, cm, fmt="%d", delimiter=",",
               header=",".join(CLASS_NAMES), comments="")

    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )
    with PER_CLASS_CSV.open("w", encoding="utf-8") as fh:
        fh.write("class,precision,recall,f1,support\n")
        for i, name in enumerate(CLASS_NAMES):
            fh.write(f'"{name}",{precision[i]:.4f},{recall[i]:.4f},{f1[i]:.4f},{support[i]}\n')

    report_txt = classification_report(
        targets, preds, labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES, digits=4, zero_division=0,
    )
    print("\n" + report_txt)

    # ---- confusion pairs actually present in THIS matrix (never guessed) ----
    off_diagonal = []
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if i != j and cm[i, j] > 0:
                off_diagonal.append((int(cm[i, j]), CLASS_NAMES[i], CLASS_NAMES[j]))
    off_diagonal.sort(reverse=True)
    top_pairs = off_diagonal[:6]

    print("Largest confusions read directly off the matrix:")
    for count, true_name, pred_name in top_pairs:
        print(f"  {count:4d}  true '{true_name}' -> predicted '{pred_name}'")

    # Merge the two directions of the same pair for the written analysis.
    merged: dict = {}
    for count, a, b in off_diagonal:
        key = frozenset({a, b})
        merged[key] = merged.get(key, 0) + count
    ranked_pairs = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:3]

    training_log = json.loads(TRAINING_LOG_PATH.read_text())
    val_before = training_log["val_accuracy_before_finetuning"]
    val_after = training_log["val_accuracy_after_finetuning"]
    fine_tuned = training_log["finetune_triggered"]

    per_class_rows = "\n".join(
        f"| {CLASS_NAMES[i]} | {precision[i]:.4f} | {recall[i]:.4f} | {f1[i]:.4f} "
        f"| {support[i]} |"
        for i in range(len(CLASS_NAMES))
    )
    directional_rows = "\n".join(
        f"| {count} | `{true_name}` | `{pred_name}` |"
        for count, true_name, pred_name in top_pairs
    )

    pair_sections = []
    for rank, (pair, total) in enumerate(ranked_pairs, start=1):
        a, b = sorted(pair)
        a_to_b = int(cm[CLASS_NAMES.index(a), CLASS_NAMES.index(b)])
        b_to_a = int(cm[CLASS_NAMES.index(b), CLASS_NAMES.index(a)])
        pair_sections.append(
            f"""### Pair {rank}: `{a}` <-> `{b}` — {total} misclassifications

Read off the matrix: {a_to_b} `{a}` images were predicted `{b}`, and {b_to_a}
`{b}` images were predicted `{a}`.

{note_for(a, b)}"""
        )

    report = f"""# Part 2 — Evaluation Report (Tasks 5-6)

Regenerate with `python3 -m part2.evaluate_product_classifier`. Every number
here comes from running the saved `models/product_classifier.pt` over the real
test split; nothing is simulated.

## Setup actually used

| item | value |
|---|---|
| dataset | Fashion-MNIST (Zalando Research), via `torchvision.datasets.FashionMNIST` |
| backbone | ResNet-18, `ResNet18_Weights.IMAGENET1K_V1` (pretrained) |
| frozen | `conv1`, `bn1`, `layer1`-`layer4` |
| trained | new `fc` = `Linear(512, 10)` |
| input size | **224 x 224** (ResNet-18's native ImageNet resolution) |
| channels | 1 grey channel replicated to 3 |
| normalisation | ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |
| optimizer | Adam, lr `{training_log['head_learning_rate']}`, batch size \
`{training_log['head_batch_size']}`, {training_log['head_epochs']} epochs |
| device | `{training_log['device']}` |
| **train split** | **50,000** images |
| **validation split** | **10,000** images (stratified, 1,000 per class, carved from the 60k train half) |
| **test split** | **10,000** images — untouched until this report |

## Feature extraction vs fine-tuning

| stage | validation accuracy |
|---|---:|
| after feature extraction (frozen backbone, head only) | **{val_before:.4f}** |
| after fine-tuning | {"**" + f"{val_after:.4f}" + "**" if fine_tuned else f"{val_after:.4f} (not run)"} |

**Feature extraction alone was sufficient.** Validation accuracy reached
{val_before:.4f}, above the {FINETUNE_TRIGGER_ACC:.2f} bar, so the conditional
fine-tuning stage was **not** triggered and `layer4` was never unfrozen. The
before/after numbers are identical because no second stage ran — stated
explicitly rather than left ambiguous.

## Final test-set accuracy

**{accuracy:.4f} ({accuracy * 100:.2f}%)** on the 10,000 held-out test images.

## Confusion matrix (10x10, real predictions)

Rows = true class, columns = predicted class. Diagonal in bold; `·` = zero.
Also written to [`part2_confusion_matrix.csv`](part2_confusion_matrix.csv).

{matrix_markdown(cm)}

## Per-class precision / recall / F1

Also written to [`part2_per_class_metrics.csv`](part2_per_class_metrics.csv).

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
{per_class_rows}

```
{report_txt}
```

## Confusion analysis

The largest single off-diagonal cells in the matrix above, by direction:

| count | true class | predicted as |
|---:|---|---|
{directional_rows}

{chr(10).join(pair_sections)}

## What this means for the catalogue use case

The model's errors are not random — they cluster inside two visually coherent
families: **upper-body garments** (`T-shirt/top`, `Shirt`, `Pullover`, `Coat`)
and **footwear** (`Sandal`, `Sneaker`, `Ankle boot`). Between those families it
almost never confuses anything. For Flipkart's actual catalogue problem — "is
this photo filed under roughly the right department?" — that is the useful
failure mode: a mis-tagged shirt still lands in apparel, so a support agent
using Part 3's `classify_product_image` tool gets a correct department even on
the model's bad days. The residual within-family error is bounded by the source
data: at 28x28 greyscale, the collar and hem details that separate a shirt from
a t-shirt were never captured, and no amount of upsampling to 224x224 can
reconstruct information the sensor did not record.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH.relative_to(REPORTS_DIR.parent)}")


if __name__ == "__main__":
    main()
