"""Run five-fold ResNet-50 fine-tuning for both split schemes.

Run from the repository root:

    .venv\\Scripts\\python.exe src\\resnet_finetuning\\run_all_folds.py

The script selects the best epoch in each fold by validation macro-F1, then
selects the best fold checkpoint across both schemes for convenient inspection.
The held-out test set is intentionally not evaluated.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from data import load_folds  # noqa: E402
from train_resnet import (  # noqa: E402
    SEED,
    train_fold,
    seed_everything,
)
from metrics import log_result  # noqa: E402
import torch  # noqa: E402
from argparse import Namespace  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "resnet_finetuning"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CONFIG = {
    "mode": "finetune",
    "arch": "resnet50",
    "epochs": 20,
    "batch_size": 64,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "weighted_loss": False,
    "workers": 2,
}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this full five-fold ResNet-50 run. "
            "Run this script in the GPU/Colab environment."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    folds = load_folds()
    test_df = folds.loc[folds["is_test"]].copy()
    pool = folds.loc[~folds["is_test"]].copy()
    device = torch.device("cuda")
    args = Namespace(**CONFIG, scheme="grouped", eval_test=False)
    results: list[dict] = []

    for scheme in ("random", "grouped"):
        fold_column = f"fold_{scheme}"
        args.scheme = scheme
        for fold in range(5):
            seed_everything(SEED)
            train_df = pool.loc[pool[fold_column] != fold].copy()
            validation_df = pool.loc[pool[fold_column] == fold].copy()
            validation, _ = train_fold(
                CONFIG["mode"],
                train_df,
                validation_df,
                test_df,
                args,
                device,
                fold,
            )
            source = ROOT / "outputs" / "checkpoints" / (
                f"{CONFIG['arch']}_{CONFIG['mode']}_{scheme}_fold{fold}.pt"
            )
            destination = CHECKPOINT_DIR / f"{scheme}_fold{fold}.pt"
            shutil.copy2(source, destination)
            results.append({
                "scheme": scheme,
                "fold": fold,
                **validation,
                "checkpoint": str(destination.relative_to(ROOT)),
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "fold_results.csv", index=False)
    summaries = {}
    for scheme, group in results_df.groupby("scheme"):
        summaries[scheme] = {
            metric: (
                float(group[metric].mean()),
                float(group[metric].std(ddof=0)),
            )
            for metric in (
                "macro_f1",
                "weighted_f1",
                "accuracy",
                "weed_to_negative_rate",
            )
        }
        log_result(
            "resnet50_finetune",
            scheme,
            CONFIG | {"folds": list(range(5))},
            summaries[scheme],
        )
    best = results_df.loc[results_df["macro_f1"].idxmax()]
    best_source = ROOT / best["checkpoint"]
    best_destination = OUTPUT_DIR / "best_model.pt"
    shutil.copy2(best_source, best_destination)
    report = {
        "hyperparameters": CONFIG,
        "seed": SEED,
        "selection_metric": "validation macro_f1",
        "summaries": summaries,
        "best_fold": {
            "scheme": best["scheme"],
            "fold": int(best["fold"]),
            "macro_f1": float(best["macro_f1"]),
            "accuracy": float(best["accuracy"]),
            "checkpoint": str(best_destination.relative_to(ROOT)),
        },
        "test_evaluated": False,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
