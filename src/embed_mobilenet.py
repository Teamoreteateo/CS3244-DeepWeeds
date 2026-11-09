
"""Extract frozen MobileNetV3-Large embeddings for DeepWeeds."""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import (
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
)
from tqdm import tqdm

from data import load_folds, IMG_DIR, OUTPUTS
from data_pipeline import DeepWeedsDataset, make_transforms
from metrics import SPECIES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit images for a quick local CPU test.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Device:", device)

    # Load pretrained MobileNetV3-Large.
    model = mobilenet_v3_large(
        weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2
    )

    # Removing the classifier gives 960-dimensional features.
    model.classifier = torch.nn.Identity()
    model.eval().to(device)

    # Use the existing frozen manifest and preprocessing.
    folds = load_folds()

    if args.max_images is not None:
        if args.max_images < 1:
            raise ValueError("--max-images must be positive")
        folds = folds.head(args.max_images).copy()

    dataset = DeepWeedsDataset(
        folds,
        IMG_DIR,
        make_transforms(augment=False),
        SPECIES,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    features = []
    labels = []

    with torch.inference_mode():
        for images, batch_labels, filenames in tqdm(
            loader, desc="MobileNetV3 embeddings"
        ):
            embeddings = model(images.to(device))
            features.append(embeddings.cpu().numpy())
            labels.append(batch_labels.numpy())

    X = np.concatenate(features).astype(np.float32)
    y = np.concatenate(labels)

    assert X.shape == (len(folds), 960), X.shape
    assert np.array_equal(y, folds["Label"].to_numpy())

    print("Embedding shape:", X.shape)
    print("Label shape:", y.shape)

    # Only save complete embeddings for evaluation.
    if args.max_images is None:
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        np.save(OUTPUTS / "X_mobilenetv3.npy", X)
        folds["Filename"].to_csv(
            OUTPUTS / "feature_order_mobilenetv3.csv",
            index=False,
        )
        print("Full embeddings saved to outputs/")
    else:
        print("Smoke test only: no files saved.")

    print("MobileNetV3 extraction successful!")


if __name__ == "__main__":
    main()
