"""
convnext tiny -- training, ablation, evaluation
"""
import argparse
import json
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from data import load_folds, ROOT, IMG_DIR
from data_pipeline import DeepWeedsDataset, make_transforms
from metrics import evaluate, summarise, log_result, SPECIES


# daatset
# class DeepWeedsDataset(Dataset):
#     def __init__(self, df: pd.DataFrame, transform=None):
#         self.df = df.reset_index(drop=True)
#         self.transform = transform
#         self.filenames = self.df["Filename"].values
#         self.labels = self.df["Label"].values

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         path = IMG_DIR / self.filenames[idx]
#         image = Image.open(path).convert("RGB")
#         label = self.labels[idx]
#         if self.transform:
#             image = self.transform(image)
#         return image, label, self.filenames[idx]

# # transform
# def get_transforms():
#     train_tf = transforms.Compose([
#         transforms.Resize(256), # shorter edge 256 px
#         transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # take a random crop 224 by 224
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.RandomRotation(degrees=30),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2), # "to address severe field illumination and orientation shifts" from proposal
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225]), # for convnext
#     ])
#     val_tf = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224), # no augmentation for val data
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225]),
#     ])
#     return train_tf, val_tf


# model
def build_convnext_tiny(num_classes=9, freeze_backbone=True):
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    model = convnext_tiny(weights=weights)

    # trf learning; freeze the early blocks, fine-tune only the final stage and head
    if freeze_backbone:
        for param in model.features[:-1].parameters():
            param.requires_grad = False

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


# train, eval
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_trues = [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_trues.extend(labels.numpy())

    metrics = evaluate(all_trues, all_preds)
    return metrics, np.array(all_preds), np.array(all_trues)


# exc
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0, help="Fold index (0-4)")
    parser.add_argument("--scheme", type=str, default="grouped", choices=["random", "grouped"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weighted_loss", action="store_true", help="Use class-weighted cross-entropy")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load splits
    df = load_folds()
    pool = df[~df["is_test"]].reset_index(drop=True)
    col = f"fold_{args.scheme}"

    tr_df = pool[pool[col] != args.fold]
    va_df = pool[pool[col] == args.fold]
    print(f"Scheme: {args.scheme} | Fold: {args.fold} | Train: {len(tr_df)} | Val: {len(va_df)}")

    tr_ds = DeepWeedsDataset(tr_df, IMG_DIR, make_transforms(True), SPECIES)
    va_ds = DeepWeedsDataset(va_df, IMG_DIR, make_transforms(False), SPECIES)

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # class-imbalance ablation support
    if args.weighted_loss:
        counts = tr_df["Label"].value_counts().sort_index().values
        weights = len(tr_df) / (9.0 * counts)
        class_weights = torch.tensor(weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using Class-Weighted Cross-Entropy Loss")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using Standard Cross-Entropy Loss")

    model = build_convnext_tiny(num_classes=9).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    best_macro_f1 = 0.0
    best_results = None
    misclassifications = None
    val_files = va_df["Filename"].values             # Extract filenames in order

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, tr_loader, criterion, optimizer, device)
        val_metrics, preds, trues = evaluate_model(model, va_loader, device)

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Loss: {loss:.4f} | "
              f"Macro F1: {val_metrics['macro_f1']:.4f} | Acc: {val_metrics['accuracy']:.4f}")

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_results = val_metrics

            # extract misclassified instances
            wrong_mask = preds != trues
            misclassifications = pd.DataFrame({
                "Filename": val_files[wrong_mask],   # Pair with validation filenames
                "True_Label": trues[wrong_mask],
                "Pred_Label": preds[wrong_mask],
                "True_Species": [SPECIES[l] for l in trues[wrong_mask]],
                "Pred_Species": [SPECIES[l] for l in preds[wrong_mask]]
            })

    # save summary & misclassifications
    summary = {k: (float(best_results[k]), 0.0) for k in ["macro_f1", "weighted_f1", "accuracy", "weed_to_negative_rate"]}
    model_tag = f"convnext_tiny_{'weighted' if args.weighted_loss else 'ce'}"

    # TUNED: run 2
    base = 'weighted' if args.weighted_loss else 'ce'
    model_tag = f"convnext_tiny_{base}_tuned_2"
    params = {"fold": args.fold, "lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs, "scheduler": "cosine", "weights": "sqrt_damped" if args.weighted_loss else "none"}
    log_result(model_tag, args.scheme, params, summary)

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    misclassifications.to_csv(out_dir / f"misclassifications_{model_tag}_f{args.fold}.csv", index=False)
    print(f"\nresults appended to results.csv, misclassified samples exported to outputs/.")


if __name__ == "__main__":
    main()