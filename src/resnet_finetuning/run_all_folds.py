"""Run five-fold ResNet-50 fine-tuning for each split scheme.

    python src/resnet_finetuning/run_all_folds.py --schemes paper
    python src/resnet_finetuning/run_all_folds.py --schemes random grouped

Best epoch per fold is selected by validation macro-F1. The held-out test set
is not evaluated.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from data import load_folds, load_folds_paper  # noqa: E402
from train_resnet import SEED, train_fold, seed_everything  # noqa: E402
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
    p = argparse.ArgumentParser()
    p.add_argument("--schemes", nargs="+", default=["random", "grouped"],
                   choices=["random", "grouped", "paper"])
    cli = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this full five-fold ResNet-50 run. "
            "Run this script in the GPU/Colab environment."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    args = Namespace(**CONFIG, scheme=cli.schemes[0], eval_test=False)
    results: list[dict] = []

    for scheme in cli.schemes:
        args.scheme = scheme
        paper = scheme == "paper"
        if paper:
            pool = load_folds_paper()
            test_df = None
        else:
            folds = load_folds()
            test_df = folds.loc[folds["is_test"]].copy()
            pool = folds.loc[~folds["is_test"]].copy()
            fold_column = f"fold_{scheme}"

        for fold in range(5):
            seed_everything(SEED)
            if paper:
                col = f"paper_fold{fold}"
                train_df      = pool.loc[pool[col] == "train"].copy()
                validation_df = pool.loc[pool[col] == "val"].copy()
                test_df       = pool.loc[pool[col] == "test"].copy()
            else:
                train_df      = pool.loc[pool[fold_column] != fold].copy()
                validation_df = pool.loc[pool[fold_column] == fold].copy()

            print(f"[{scheme} fold {fold}] train={len(train_df)} "
                  f"val={len(validation_df)}")
            validation, _ = train_fold(
                CONFIG["mode"], train_df, validation_df, test_df,
                args, device, fold,
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
    out_csv = OUTPUT_DIR / "fold_results.csv"
    if out_csv.exists():
        results_df = pd.concat([pd.read_csv(out_csv), results_df])
    results_df.to_csv(out_csv, index=False)

    summaries = {}
    for scheme, group in results_df.groupby("scheme"):
        summaries[scheme] = {
            metric: (float(group[metric].mean()),
                     float(group[metric].std(ddof=0)))
            for metric in ("macro_f1", "weighted_f1", "accuracy",
                           "weed_to_negative_rate")
        }
        log_result("resnet50_finetune", scheme,
                   CONFIG | {"folds": list(range(5))}, summaries[scheme])

    report = {
        "hyperparameters": CONFIG,
        "seed": SEED,
        "selection_metric": "validation macro_f1",
        "schemes_run": cli.schemes,
        "summaries": summaries,
        "test_evaluated": False,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()