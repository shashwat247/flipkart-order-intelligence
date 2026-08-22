# Part 2 — Evaluation Report (Tasks 5-6)

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
| optimizer | Adam, lr `0.001`, batch size `256`, 15 epochs |
| device | `mps` |
| **train split** | **50,000** images |
| **validation split** | **10,000** images (stratified, 1,000 per class, carved from the 60k train half) |
| **test split** | **10,000** images — untouched until this report |

## Feature extraction vs fine-tuning

| stage | validation accuracy |
|---|---:|
| after feature extraction (frozen backbone, head only) | **0.8925** |
| after fine-tuning | 0.8925 (not run) |

**Feature extraction alone was sufficient.** Validation accuracy reached
0.8925, above the 0.80 bar, so the conditional
fine-tuning stage was **not** triggered and `layer4` was never unfrozen. The
before/after numbers are identical because no second stage ran — stated
explicitly rather than left ambiguous.

## Final test-set accuracy

**0.8872 (88.72%)** on the 10,000 held-out test images.

## Confusion matrix (10x10, real predictions)

Rows = true class, columns = predicted class. Diagonal in bold; `·` = zero.
Also written to [`part2_confusion_matrix.csv`](part2_confusion_matrix.csv).

| true \ pred | T-shirt/top | Trouser | Pullover | Dress | Coat | Sandal | Shirt | Sneaker | Bag | Ankle boot | **total** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **T-shirt/top** | **822** | 5 | 20 | 21 | 3 | 2 | 119 | · | 7 | 1 | 1000 |
| **Trouser** | 1 | **973** | 3 | 15 | 2 | 1 | 4 | · | 1 | · | 1000 |
| **Pullover** | 13 | · | **861** | 3 | 56 | · | 66 | · | 1 | · | 1000 |
| **Dress** | 22 | 7 | 19 | **849** | 35 | · | 67 | · | 1 | · | 1000 |
| **Coat** | 1 | · | 64 | 24 | **801** | · | 107 | · | 3 | · | 1000 |
| **Sandal** | · | · | · | · | · | **950** | 1 | 36 | 2 | 11 | 1000 |
| **Shirt** | 104 | · | 42 | 27 | 88 | 1 | **730** | · | 8 | · | 1000 |
| **Sneaker** | · | · | · | · | · | 16 | · | **962** | 1 | 21 | 1000 |
| **Bag** | 1 | · | 2 | 3 | 1 | 2 | 9 | · | **981** | 1 | 1000 |
| **Ankle boot** | · | · | · | · | 1 | 13 | · | 42 | 1 | **943** | 1000 |

## Per-class precision / recall / F1

Also written to [`part2_per_class_metrics.csv`](part2_per_class_metrics.csv).

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| T-shirt/top | 0.8527 | 0.8220 | 0.8371 | 1000 |
| Trouser | 0.9878 | 0.9730 | 0.9804 | 1000 |
| Pullover | 0.8516 | 0.8610 | 0.8563 | 1000 |
| Dress | 0.9013 | 0.8490 | 0.8744 | 1000 |
| Coat | 0.8116 | 0.8010 | 0.8062 | 1000 |
| Sandal | 0.9645 | 0.9500 | 0.9572 | 1000 |
| Shirt | 0.6618 | 0.7300 | 0.6942 | 1000 |
| Sneaker | 0.9250 | 0.9620 | 0.9431 | 1000 |
| Bag | 0.9751 | 0.9810 | 0.9781 | 1000 |
| Ankle boot | 0.9652 | 0.9430 | 0.9540 | 1000 |

```
              precision    recall  f1-score   support

 T-shirt/top     0.8527    0.8220    0.8371      1000
     Trouser     0.9878    0.9730    0.9804      1000
    Pullover     0.8516    0.8610    0.8563      1000
       Dress     0.9013    0.8490    0.8744      1000
        Coat     0.8116    0.8010    0.8062      1000
      Sandal     0.9645    0.9500    0.9572      1000
       Shirt     0.6618    0.7300    0.6942      1000
     Sneaker     0.9250    0.9620    0.9431      1000
         Bag     0.9751    0.9810    0.9781      1000
  Ankle boot     0.9652    0.9430    0.9540      1000

    accuracy                         0.8872     10000
   macro avg     0.8897    0.8872    0.8881     10000
weighted avg     0.8897    0.8872    0.8881     10000

```

## Confusion analysis

The largest single off-diagonal cells in the matrix above, by direction:

| count | true class | predicted as |
|---:|---|---|
| 119 | `T-shirt/top` | `Shirt` |
| 107 | `Coat` | `Shirt` |
| 104 | `Shirt` | `T-shirt/top` |
| 88 | `Shirt` | `Coat` |
| 67 | `Dress` | `Shirt` |
| 66 | `Pullover` | `Shirt` |

### Pair 1: `Shirt` <-> `T-shirt/top` — 223 misclassifications

Read off the matrix: 104 `Shirt` images were predicted `T-shirt/top`, and 119
`T-shirt/top` images were predicted `Shirt`.

Both are short-to-medium sleeved upper-body garments photographed flat against the same background, and at 28x28 the only thing that distinguishes them is a button placket or a collar — features that occupy perhaps three or four pixels at this resolution and are frequently smoothed away entirely by the downsampling. The silhouettes are close to identical: same shoulder width, same torso taper, same sleeve stubs. Even a human labelling the raw 28x28 thumbnails disagrees with the ground truth on this pair regularly, which is why it is the canonical hard pair in Fashion-MNIST. Our upsampling to 224x224 cannot recover detail that was never captured — it interpolates the existing 784 pixels, so the collar that would settle the question simply is not in the signal.
### Pair 2: `Coat` <-> `Shirt` — 195 misclassifications

Read off the matrix: 107 `Coat` images were predicted `Shirt`, and 88
`Shirt` images were predicted `Coat`.

A coat and a long-sleeved shirt share the same basic outline: a rectangular torso with two sleeves extending to roughly the same length. The real-world difference is thickness, layering and fastening hardware, all of which read as subtle intensity gradients rather than shape changes. Because the images are greyscale, the model loses the colour and texture cues (wool vs cotton weave) that a shopper would use instantly, and is left comparing two very similar binary silhouettes.
### Pair 3: `Coat` <-> `Pullover` — 120 misclassifications

Read off the matrix: 64 `Coat` images were predicted `Pullover`, and 56
`Pullover` images were predicted `Coat`.

These two occupy nearly the same silhouette envelope — a filled torso block with two sleeves — and differ mainly in whether the front is open or closed. In greyscale at this resolution a coat's front opening often renders as a faint vertical intensity seam, and a pullover with a central fabric fold produces the same artifact, so the cue is not reliable.

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
