# ResNet-50 experiments

## What was done

The project supports three experiments on the DeepWeeds dataset:

1. **Frozen pretrained features** — ImageNet-pretrained backbone frozen,
   only the new 9-class head trained; backbone BatchNorm statistics fixed.
2. **End-to-end fine-tuning** — all weights updated from ImageNet init.
3. **From scratch** (`--mode scratch`) — random initialisation, no ImageNet
   weights. Run as a pretraining ablation, not a candidate model.
   Use `--arch resnet18` here; ResNet-50 from random init overfits 13k images.

The implementation is in [`src/train_resnet.py`](src/train_resnet.py).
PyTorch and torchvision were added to [`requirements.txt`](requirements.txt).

## Dataset protocol

The existing frozen split manifest in [`splits/folds.csv`](splits/folds.csv)
is used:

- 17,509 total images.
- 13,274 non-test images.
- 4,235 held-out test images.
- Five validation folds within the non-test images.
- The grouped split keeps images from the same recording session together.

For a grouped fold, the process is:

```text
Training:   four grouped folds
Validation: the remaining grouped fold
Test:       the separate held-out test set
```

The held-out test set is evaluated only when `--eval-test` is passed
(default off), reserved for the single final evaluation once results
are frozen. 
The primary model-selection metrics are the validation metrics.

## Metrics

Each run reports:

- Accuracy.
- Macro F1.
- Weighted F1.
- Weed-to-negative classification rate.
- Validation summary across the selected folds.
- Held-out test reference metrics (only with `--eval-test`).

Validation summaries are appended to [`results.csv`](results.csv).
The best model checkpoint for each fold is saved under:

```text
outputs/checkpoints/
```

## How to run

Frozen-feature baseline, all five grouped folds:

```bash
python src/train_resnet.py --mode frozen --scheme grouped --epochs 10
```

End-to-end fine-tuning, one grouped fold:

```bash
python src/train_resnet.py --mode finetune --arch resnet50 --scheme grouped \
  --fold 0 --epochs 20 --batch-size 64 --workers 2 --device cuda
```

From-scratch pretraining ablation:

```bash
python src/train_resnet.py --mode scratch --arch resnet18 --scheme grouped \
  --fold 0 --epochs 40 --batch-size 64 --workers 2 --device cuda
```

One-fold smoke test:

```bash
python src/train_resnet.py --mode frozen --scheme grouped --fold 0 \
  --epochs 1 --batch-size 64 --workers 0 --device cpu
```

Omit `--fold` to run all five folds. Use `--scheme random` to compare against
the random stratified folds, and `--weighted-loss` for inverse-frequency class
weights (off in all reported results).

## Compute

Runs are executed on Google Colab (T4). Local CPU-only installations are
usable for smoke tests but not for full runs. Colab provides 2 CPUs, so
use `--workers 2`; higher values trigger a DataLoader warning and can
bottleneck loading.

## Headline results

| model | folds | grouped | random |
|---|---|---|---|
| majority class | 5 | 0.077 | 0.077 |
| HOG/LBP/HSV + random forest | 5 | 0.357 | — |
| ResNet-18 from scratch, 40 ep | 1 | 0.555 | — |
| ResNet-50 frozen + logreg | 5 | 0.604 | 0.774 |
| ResNet-50 fine-tuned, 20 ep | 1 | 0.804 | 0.947 |

Macro-F1; no weighted loss. Single-fold rows have no error estimate.
Fine-tuned ResNet-50 on the random split reaches 96.3% accuracy, matching the
95.7% reported by Olsen et al. under their protocol. A repeat of the grouped
fine-tune with an identical configuration gave 0.798 vs 0.804 macro-F1,
indicating run-to-run variation of roughly 0.006 — far smaller than the
grouped/random gap.

## Interpretation

The three modes isolate different contributions. **Scratch** measures what the
architecture alone achieves on 13k images. **Frozen** measures how well
ImageNet features transfer without adaptation. **Fine-tuning** measures what
adapting those features to weed imagery adds. The scratch-to-frozen gap
(0.555 → 0.604 grouped) is attributable to ImageNet pretraining; the
frozen-to-fine-tuned gap (0.604 → 0.804) to adaptation.

The ~0.15 macro-F1 gap between grouped and random splits persists at both the
frozen and fine-tuned rungs, so fine-tuning does not overcome the session
shortcut. Because DeepWeeds was collected continuously in the field, a random
stratified split places near-duplicate frames from the same session on both
sides of the partition. Our random-split numbers reproduce the published
result; our grouped-split numbers are what the model generalises to across
unseen sessions.

The 95.7% accuracy reported by Olsen et al. (2019) remains a reference point
rather than a target, since their exact preprocessing, augmentation and
training schedule differ from ours.