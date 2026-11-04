"""Error analysis for a saved checkpoint on the held-out test set.

Run from the repository root:

    python src/test_error_analysis.py \
        --ckpt outputs/resnet_finetuning/checkpoints/random_fold4.pt

The script writes test predictions, misclassified instances, a confusion
matrix, per-class metrics, and error rates by capture session to
``outputs/test_evaluation/``. The misclassified CSV is compatible with
``gradcam.py``.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet50

from data import IMG_DIR, ROOT, load_folds
from data_pipeline import DeepWeedsDataset, make_transforms
from metrics import SPECIES, evaluate


def main(args):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)

    model = resnet18() if args.arch == "resnet18" else resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, len(SPECIES))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().to(device)

    folds = load_folds()
    test_df = folds.loc[folds["is_test"]].copy()
    dataset = DeepWeedsDataset(
        test_df, IMG_DIR, make_transforms(augment=False), SPECIES
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    labels, predictions, confidences, filenames = [], [], [], []
    with torch.no_grad():
        for images, batch_labels, batch_filenames in loader:
            probabilities = torch.softmax(model(images.to(device)), dim=1)
            predictions.extend(probabilities.argmax(1).cpu().numpy())
            confidences.extend(probabilities.max(1).values.cpu().numpy())
            labels.extend(batch_labels.numpy())
            filenames.extend(batch_filenames)

    y_true = np.asarray(labels)
    y_pred = np.asarray(predictions)
    confidence = np.asarray(confidences)
    metrics = evaluate(y_true, y_pred)

    tag = f"{args.arch}_{checkpoint['mode']}_test"
    model_tag = f"{checkpoint['scheme']}_fold{checkpoint['fold']}"
    test_output_dir = ROOT / "outputs" / "test_evaluation" / model_tag
    error_output_dir = ROOT / "outputs" / "error_analysis" / model_tag
    test_output_dir.mkdir(parents=True, exist_ok=True)
    error_output_dir.mkdir(parents=True, exist_ok=True)

    lookup = test_df.set_index("Filename")
    results = pd.DataFrame(
        {
            "Filename": filenames,
            "true": [SPECIES[index] for index in y_true],
            "pred": [SPECIES[index] for index in y_pred],
            "confidence": confidence,
            "session": lookup.loc[filenames, "session"].values,
        }
    )
    errors = results[results["true"] != results["pred"]].sort_values(
        "confidence", ascending=False
    )
    results.to_csv(test_output_dir / f"predictions_{tag}.csv", index=False)
    errors.to_csv(error_output_dir / f"misclassified_{tag}.csv", index=False)

    pd.DataFrame(metrics["cm"], index=SPECIES, columns=SPECIES).to_csv(
        test_output_dir / f"cm_{tag}.csv"
    )
    pd.DataFrame(metrics["cm_norm"], index=SPECIES, columns=SPECIES).to_csv(
        test_output_dir / f"cm_norm_{tag}.csv"
    )
    per_class = pd.DataFrame(metrics["per_class"]).T
    per_class = per_class.loc[[str(index) for index in range(len(SPECIES))]]
    per_class.index = SPECIES
    per_class.to_csv(test_output_dir / f"per_class_{tag}.csv")

    by_session = results.assign(wrong=results["true"] != results["pred"])
    by_session = by_session.groupby("session")["wrong"].agg(["sum", "count"])
    by_session["rate"] = by_session["sum"] / by_session["count"]
    by_session.sort_values("rate", ascending=False).to_csv(
        error_output_dir / f"by_session_{tag}.csv"
    )

    scalar_metrics = {
        key: float(value)
        for key, value in metrics.items()
        if key not in ("cm", "cm_norm", "per_class")
    }
    pd.Series(scalar_metrics, name="value").to_csv(
        test_output_dir / f"metrics_{tag}.csv", header=True
    )

    print(
        f"{len(results)} test images | {len(errors)} misclassified | "
        f"{checkpoint['mode']} | {device}"
    )
    print(
        f"macro_F1={metrics['macro_f1']:.4f}  "
        f"acc={metrics['accuracy']:.4f}  "
        f"weed->neg={metrics['weed_to_negative_rate']:.4f}"
    )
    print(f"test results written to {test_output_dir}")
    print(f"error analysis written to {error_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--arch", default="resnet50", choices=("resnet18", "resnet50"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    main(parser.parse_args())
