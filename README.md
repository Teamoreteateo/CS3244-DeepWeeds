# CS3244 Group 22 — Weed Species Classification on DeepWeeds

We compare deep learning models for classifying field images of eight Australian weed species
plus a Negative (background) class, using the DeepWeeds dataset (Olsen et al., 2019).

**Research questions**
1. Which species are hardest to tell apart?
2. How do conditions such as lighting affect performance?
3. How do different deep learning models compare?
4. When one class dominates the data, how should we measure success?

**Models:** random-forest baseline on hand-crafted colour/texture features, ResNet-50 (the paper's
model), ConvNeXt-Tiny, MobileNetV3-Large and a Vision Transformer (ViT-B/16 and a small scratch ViT).

**Final model:** MobileNetV3-Large, chosen on validation macro-F1.

---

## Evaluation protocol

- **Splits:** the official DeepWeeds five-fold splits (60% train / 20% validation / 20% test per fold),
  stored in `splits/folds_paper.csv` (columns `paper_fold0` … `paper_fold4` = `train` / `val` / `test`).
  Every image appears in exactly one test subset.
- **Primary metric:** macro-F1 (all nine classes weighted equally). Negative is ~52% of the data, so an
  always-Negative model scores 52% accuracy but only 0.076 macro-F1.
- **Other metrics:** accuracy, weighted-F1 and the weed→Negative rate (share of weed images predicted
  as Negative, i.e. weeds a sprayer would miss). All metrics come from `src/metrics.py`.
- **Model selection** uses validation macro-F1 only. The test subsets are used once, for the final model.
- `splits/folds.csv` holds our earlier random and session-grouped splits (`fold_random`, `fold_grouped`,
  `is_test`), kept for the session-leakage analysis.

## Results (paper splits)

Validation = mean over the five validation folds unless noted.

| Model | Macro-F1 | Accuracy | Source |
|---|---|---|---|
| Always predict Negative | 0.076 | 52.0% | `results.csv` → `majority_class` |
| Random forest, hand-crafted features | 0.522 ± 0.015 | 68.6% | `results.csv` → `rf_smoke` |
| ResNet-50, frozen + logistic regression | 0.755 ± 0.005 | 81.1% | `results2.csv` → `resnet50_frozen_logreg` |
| ResNet-18 from scratch (fold 0 only) | 0.812 | 85.4% | `results2.csv` → `resnet18_scratch` |
| ResNet-50, fine-tuned | 0.952 ± 0.002 | 96.2% | `results2.csv` → `resnet50_finetune` |
| ConvNeXt-Tiny, fine-tuned | 0.924 | 94.2% | `results2.csv` → `convnext_tiny_ce_val_best` |
| MobileNetV3-Large, fine-tuned (LR 1e-4) | 0.953 | 96.3% | `results_mobilenet_paper.csv` |
| ViT-B/16 pretrained, fine-tuned (fold 0 only) | 0.927 | 93.9% | `ViT_validation_comparison.csv` |
| ViT from scratch (fold 0 only) | 0.598 | 62.5% | `ViT_validation_comparison.csv` |

**Final test result (MobileNetV3-Large, mean over the five test subsets):**
macro-F1 0.948, accuracy 95.89%, weed→Negative rate 3.72%.

The original paper reports 95.7% accuracy for ResNet-50.

---

## Repository layout

```
src/
  data.py                  manifest building, validation, split loading (load_folds, load_folds_paper)
  data_pipeline.py         Dataset class, train/eval transforms (augmentation), DataLoaders, CutMix
  metrics.py               evaluate(), summarise(), log_result() — shared by every model
  splits.py                generates splits/folds.csv (random + session-grouped)
  baseline.py              always-Negative baseline
  features.py              hand-crafted colour (HSV histogram) and texture (LBP) features
  smoke_test.py            random forest on the hand-crafted features
  embed*.py, eval_*.py     frozen-backbone embeddings + linear heads (ResNet-50, ConvNeXt, MobileNet)
  train_resnet.py          ResNet frozen / fine-tune / scratch, one fold
  resnet_finetuning/run_all_folds.py   ResNet-50 fine-tuning over all five folds
  train_convnext.py        ConvNeXt-Tiny fine-tuning (CE, weighted CE, focal loss, CutMix, colour aug)
  train_mobilenet.py       MobileNetV3-Large fine-tuning (random/grouped splits)
  error_analysis.py        per-class metrics, confusion matrix, misclassified images (validation fold)
  test_error_analysis.py   same, on a test subset
  gradcam.py               Grad-CAM heatmaps for misclassified images (ResNet)
  convnext_*.py, show_errors_convnext.py   ConvNeXt error analysis
  image_stats.py           brightness / contrast / sharpness of correct vs misclassified images (ConvNeXt)
notebooks/
  01_eda.ipynb                         exploratory data analysis
  02_splits_augmentation.ipynb         split checks and augmentation design
  03_mobilenet_paper.ipynb             MobileNetV3 on paper splits: LR selection, test, error analysis
  V3_deepweeds_vit_scratch_vs_hf_vit_b16_shared_imports.ipynb   ViT training (paper fold 0)
  vit_paper_fold_validation_error_analysis.ipynb                 ViT evaluation and error analysis
splits/                    folds.csv, folds_paper.csv (committed; do not regenerate)
results.csv                runs on random / grouped splits (+ baselines)
results2.csv               runs on paper splits
results_mobilenet_paper.csv  MobileNet per-fold validation (3 LRs) and test results
ViT_validation_comparison.csv  ViT scratch vs pretrained, paper fold 0 validation
make_paper_folds.py        builds splits/folds_paper.csv from the official subset CSVs
```

## Setup

```bash
git clone https://github.com/Teamoreteateo/CS3244-DeepWeeds.git
cd CS3244-DeepWeeds
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install transformers          # only needed for the ViT notebooks
```

Data is not stored in git. Follow `data/README.md`:
download `images.zip` and `labels.csv` from https://github.com/AlexOlsen/DeepWeeds,
unzip the images to `data/images/` (17,509 files) and copy `labels.csv` to `data/labels.csv`.

A GPU is needed for fine-tuning. We trained on Google Colab (T4); the notebooks contain Colab/Google
Drive paths that must be changed to run elsewhere.

## Reproducing the results

Scripts in `src/` import each other by module name, so run them from inside `src/`:

```bash
cd src

# Baselines
python baseline.py                       # always-Negative
python features.py                       # cache hand-crafted features (once)
python smoke_test.py                     # random forest

# ResNet-50
python embed.py && python eval_embeddings.py                      # frozen + logistic regression
python resnet_finetuning/run_all_folds.py --schemes paper         # fine-tuning, 5 folds
python train_resnet.py --mode scratch --arch resnet18 --scheme paper --fold 0 --epochs 40

# ConvNeXt-Tiny (one fold per call; repeat for --fold 0..4)
python train_convnext.py --fold 0 --epochs 20

# Error analysis and Grad-CAM (ResNet checkpoint)
python error_analysis.py --ckpt <checkpoint.pt> --arch resnet50
python gradcam.py --ckpt <checkpoint.pt> --errors <misclassified_csv> --arch resnet50

# Image brightness/contrast vs errors
python image_stats.py
```

- **MobileNetV3-Large:** run `notebooks/03_mobilenet_paper.ipynb` (set `RUN_CV = True` to train the
  15 LR × fold models, then run test inference for the selected LR).
- **Vision Transformer:** run the V3 notebook to train, then
  `vit_paper_fold_validation_error_analysis.ipynb` to evaluate.

All runs use seed 42. Results are appended to `results.csv` / `results2.csv` via `metrics.log_result`.

## References

Olsen, A., Konovalov, D. A., Philippa, B., Ridd, P., Wood, J. C., Johns, J., Banks, W., Girgenti, B.,
Kenny, O., Whinney, J., Calvert, B., Rahimi Azghadi, M., & White, R. D. (2019). DeepWeeds: A multiclass
weed species image dataset for deep learning. *Scientific Reports, 9*, Article 2058.
https://doi.org/10.1038/s41598-018-38343-3

Full reference list: see the presentation slides.
