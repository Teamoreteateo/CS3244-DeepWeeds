"""Train frozen-feature and end-to-end ResNet-50 DeepWeeds models.

Run from the repository root, for example:

    python src/train_resnet.py --mode frozen --scheme grouped --epochs 10
    python src/train_resnet.py --mode finetune --scheme grouped --epochs 20

The held-out test set is never used for model selection. Each cross-validation
fold reports validation metrics and evaluates the same fold model on the
held-out test set for reference.
"""
from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from torchvision import transforms

from data import IMG_DIR, ROOT, load_folds
from metrics import evaluate, log_result, summarise

SEED = 42
N_FOLDS = 5
NUM_CLASSES = 9
IMAGE_SIZE = 224


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DeepWeedsDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform: transforms.Compose):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.dataframe.iloc[index]
        with Image.open(IMG_DIR / row["Filename"]) as image:
            image = image.convert("RGB")
        return self.transform(image), int(row["Label"]), str(row["Filename"])


def make_transforms(train: bool) -> transforms.Compose:
    weights = ResNet50_Weights.DEFAULT
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(weights.transforms().mean, weights.transforms().std),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(weights.transforms().mean, weights.transforms().std),
    ])


def build_model(mode: str, device: torch.device) -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    if mode == "frozen":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    model.to(device)
    return model


def make_loaders(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int,
    workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train = DeepWeedsDataset(train_df, make_transforms(train=True))
    validation = DeepWeedsDataset(validation_df, make_transforms(train=False))
    test = DeepWeedsDataset(test_df, make_transforms(train=False))
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=workers),
        DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=workers),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=workers),
    )


def make_criterion(train_df: pd.DataFrame, weighted: bool, device: torch.device) -> nn.Module:
    if not weighted:
        return nn.CrossEntropyLoss()
    counts = train_df["Label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
    if (counts == 0).any():
        raise ValueError(f"Training fold is missing classes: {counts[counts == 0].index.tolist()}")
    weights = len(train_df) / (NUM_CLASSES * counts.to_numpy(dtype=np.float32))
    return nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    mode: str,
) -> float:
    training = optimizer is not None
    model.train(training)
    if mode == "frozen":
        # Keep pretrained BatchNorm running statistics fixed as well as weights.
        model.eval()
        model.fc.train(training)
    total_loss = 0.0
    total_items = 0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss = criterion(model(images), labels)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(labels)
        total_items += len(labels)
    return total_loss / total_items


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions, labels, filenames = [], [], []
    for images, batch_labels, batch_filenames in loader:
        predictions.extend(model(images.to(device)).argmax(dim=1).cpu().numpy())
        labels.extend(batch_labels.numpy())
        filenames.extend(batch_filenames)
    return np.asarray(labels), np.asarray(predictions), np.asarray(filenames)


def train_fold(
    mode: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    fold: int,
) -> tuple[dict, dict]:
    train_loader, validation_loader, test_loader = make_loaders(
        train_df, validation_df, test_df, args.batch_size, args.workers
    )
    model = build_model(mode, device)
    criterion = make_criterion(train_df, args.weighted_loss, device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)

    best_score = -1.0
    best_state = None
    best_validation = None
    for epoch in range(1, args.epochs + 1):
        loss = run_epoch(model, train_loader, criterion, optimizer, device, mode)
        true, predicted, _ = predict(model, validation_loader, device)
        validation = evaluate(true, predicted)
        print(
            f"fold={fold} epoch={epoch}/{args.epochs} "
            f"loss={loss:.4f} macro_f1={validation['macro_f1']:.4f} "
            f"accuracy={validation['accuracy']:.4f}"
        )
        if validation["macro_f1"] > best_score:
            best_score = validation["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_validation = validation

    if best_state is None or best_validation is None:
        raise RuntimeError("No model checkpoint was produced.")
    model.load_state_dict(best_state)
    test_true, test_predicted, _ = predict(model, test_loader, device)
    test_metrics = evaluate(test_true, test_predicted)

    checkpoint_dir = ROOT / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"resnet50_{mode}_{args.scheme}_fold{fold}.pt"
    torch.save(
        {
            "mode": mode,
            "scheme": args.scheme,
            "fold": fold,
            "seed": SEED,
            "state_dict": best_state,
            "validation_metrics": best_validation,
        },
        checkpoint_path,
    )
    print(
        f"fold={fold} best_val_macro_f1={best_validation['macro_f1']:.4f} "
        f"test_accuracy={test_metrics['accuracy']:.4f} "
        f"checkpoint={checkpoint_path}"
    )
    return best_validation, test_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("frozen", "finetune"), required=True)
    parser.add_argument("--scheme", choices=("random", "grouped"), default="grouped")
    parser.add_argument("--fold", type=int, choices=range(N_FOLDS), default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--weighted-loss", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    args.lr = args.lr or (1e-3 if args.mode == "frozen" else 1e-4)
    if args.lr <= 0:
        raise ValueError("lr must be positive")
    seed_everything(SEED)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (
            args.device == "auto" and torch.cuda.is_available()
        ) else "cpu"
    )
    print(f"mode={args.mode} scheme={args.scheme} device={device}")

    folds = load_folds()
    test_df = folds.loc[folds["is_test"]].copy()
    pool = folds.loc[~folds["is_test"]].copy()
    selected_folds = [args.fold] if args.fold is not None else list(range(N_FOLDS))
    validation_results, test_results = [], []
    fold_column = f"fold_{args.scheme}"

    for fold in selected_folds:
        train_df = pool.loc[pool[fold_column] != fold].copy()
        validation_df = pool.loc[pool[fold_column] == fold].copy()
        validation, test = train_fold(
            args.mode, train_df, validation_df, test_df, args, device, fold
        )
        validation_results.append(validation)
        test_results.append(test)

    validation_summary = summarise(validation_results)
    test_summary = summarise(test_results)
    print("\nValidation summary (mean +/- std):")
    for metric, (mean, std) in validation_summary.items():
        print(f"  {metric}: {mean:.4f} +/- {std:.4f}")
    print("\nHeld-out test reference (mean +/- std across fold models):")
    for metric, (mean, std) in test_summary.items():
        print(f"  {metric}: {mean:.4f} +/- {std:.4f}")

    params = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "weighted_loss": args.weighted_loss,
        "folds": selected_folds,
    }
    log_result(f"resnet50_{args.mode}", args.scheme, params, validation_summary)
    print("\nValidation summary logged to results.csv.")


if __name__ == "__main__":
    main()
