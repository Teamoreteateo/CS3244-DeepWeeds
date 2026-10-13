# ResNet-50 experiments

## What was done

The project now supports two transfer-learning experiments on the DeepWeeds
dataset:

1. **Frozen pretrained features**
   - Loads an ImageNet-pretrained ResNet-50.
   - Replaces the original 1,000-class ImageNet head with a 9-class DeepWeeds
     classifier.
   - Freezes the ResNet-50 backbone.
   - Trains only the new final linear layer on the DeepWeeds training images.
   - Keeps the backbone's BatchNorm running statistics fixed.

2. **End-to-end fine-tuning**
   - Starts with the same ImageNet-pretrained ResNet-50.
   - Replaces the final layer with a 9-class classifier.
   - Updates all ResNet-50 weights using the DeepWeeds training data.

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

The test set is not used to update model weights or select the best epoch.
The current script reports test performance for each fold model as a
reference. The primary model-selection metrics are the validation metrics.

## Metrics

Each run reports:

- Accuracy.
- Macro F1.
- Weighted F1.
- Weed-to-negative classification rate.
- Validation summary across the selected folds.
- Held-out test reference metrics.

Validation summaries are appended to [`results.csv`](results.csv).
The best model checkpoint for each fold is saved under:

```text
outputs/checkpoints/
```

## How to run

Run the frozen-feature baseline across all five grouped folds:

```powershell
python src/train_resnet.py `
  --mode frozen `
  --scheme grouped `
  --epochs 10
```

Run end-to-end fine-tuning across all five grouped folds:

```powershell
python src/train_resnet.py `
  --mode finetune `
  --scheme grouped `
  --epochs 20 `
  --lr 0.0001
```

Run a quick one-fold smoke test:

```powershell
python src/train_resnet.py `
  --mode frozen `
  --scheme grouped `
  --fold 0 `
  --epochs 1 `
  --batch-size 64 `
  --device cpu
```

Use `--scheme random` to compare against the random stratified folds. Use
`--weighted-loss` to train with inverse-frequency class weights.

## Current compute limitation

The current environment has a CPU-only PyTorch installation:

```text
torch 2.14.0+cpu
CUDA available: False
```

As a result, a full five-fold ResNet-50 run is slow. The frozen experiment
currently recomputes the frozen backbone features once per epoch. A future
optimization would cache those features once and train the linear classifier
on the cached vectors. End-to-end fine-tuning cannot use this optimization
because the backbone weights change during training.

For practical training times, run the experiments on a CUDA-enabled machine,
Google Colab, or another GPU environment.

## Interpretation

The frozen experiment measures how well ImageNet features transfer to
DeepWeeds without adapting the feature extractor. The fine-tuned experiment
measures whether adapting the complete ResNet-50 improves performance.

The 95.7% accuracy reported by Olsen et al. (2019) should be treated as a
reference point unless their exact dataset split, preprocessing, augmentation,
training schedule, and evaluation protocol are reproduced. Our grouped
session-level split is designed to provide a stricter leakage-safe evaluation.
