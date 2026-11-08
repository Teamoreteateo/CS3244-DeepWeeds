
"""Fine-tune pretrained MobileNetV3-Large on DeepWeeds."""

import argparse
import copy
import random

import numpy as np
import torch
from torch import nn
from torchvision.models import (
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
)

from data import ROOT
from data_pipeline import get_dataloaders
from metrics import evaluate, summarise, log_result

SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        total_samples += len(labels)

    return total_loss / total_samples


@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    actual, predicted = [], []

    for images, labels, _ in loader:
        outputs = model(images.to(device))
        predictions = outputs.argmax(dim=1).cpu().numpy()

        actual.extend(labels.numpy())
        predicted.extend(predictions)

    return evaluate(np.array(actual), np.array(predicted))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--scheme", choices=["grouped", "random"], default="grouped")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one batch only; do not save or log results.",
    )
    args = parser.parse_args()

    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    train_loader, val_loader, _, classes = get_dataloaders(
        val_fold=args.fold,
        split=args.scheme,
        batch_size=args.batch_size,
        augment=not args.no_augment,
        weighted_sampling=False,
        include_test=False,
        num_workers=args.workers,
    )

    model = mobilenet_v3_large(
        weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2
    )
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(classes),
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()

    if args.smoke_test:
        model.train()
        images, labels, _ = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

        print("Smoke-test loss:", round(loss.item(), 4))
        print("Smoke test passed; no checkpoint or CSV logged.")
        return

    best_f1 = -1.0
    best_state = None
    best_metrics = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        metrics = validate(model, val_loader, device)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Macro-F1: {metrics['macro_f1']:.4f} | "
            f"Accuracy: {metrics['accuracy']:.4f}",
            flush=True,
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())

    output_dir = ROOT / "outputs" / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / (
        f"mobilenetv3_finetune_{args.scheme}_fold{args.fold}"
        f"{'_noaug' if args.no_augment else '_aug'}.pt"
    )

    torch.save(
        {
            "state_dict": best_state,
            "fold": args.fold,
            "validation_metrics": {
                "macro_f1": best_metrics["macro_f1"],
                "weighted_f1": best_metrics["weighted_f1"],
                "accuracy": best_metrics["accuracy"],
                "weed_to_negative_rate":
                    best_metrics["weed_to_negative_rate"],
            },
        },
        checkpoint_path,
    )

    summary = summarise([best_metrics])

    log_result(
        model="mobilenetv3_large_finetune",
        scheme=args.scheme,
        params={
            "arch": "mobilenet_v3_large",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": 1e-4,
            "folds": [args.fold],
            "augmentation": not args.no_augment,
            "weighted_sampling": False,
            "pretrained": "IMAGENET1K_V2",
        },
        summary=summary,
    )

    print("\nBest validation macro-F1:", round(best_f1, 4))
    print("Checkpoint saved:", checkpoint_path)
    print("Result logged to results.csv")


if __name__ == "__main__":
    main()
