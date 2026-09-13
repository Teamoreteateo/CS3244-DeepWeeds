"""Shared evaluation. Every model in this project reports through evaluate()."""
import os, json, datetime
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, classification_report)

LABELS = list(range(9))
NEGATIVE = 8          # verified against labels.csv


def evaluate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    weed = [l for l in LABELS if l != NEGATIVE]
    weed_total = cm[weed, :].sum()
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro",
                             labels=LABELS, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted",
                                labels=LABELS, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "weed_to_negative_rate": (cm[weed, NEGATIVE].sum() / weed_total
                                  if weed_total else np.nan),
        "cm": cm,
        "cm_norm": confusion_matrix(y_true, y_pred, labels=LABELS, normalize="true"),
        "per_class": classification_report(y_true, y_pred, labels=LABELS,
                                           zero_division=0, output_dict=True),
    }


def summarise(fold_results):
    keys = ["macro_f1", "weighted_f1", "accuracy", "weed_to_negative_rate"]
    return {k: (float(np.mean([r[k] for r in fold_results])),
                float(np.std([r[k] for r in fold_results]))) for k in keys}


def log_result(model, scheme, params, summary, path="results.csv"):
    row = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
           "model": model, "split_scheme": scheme,
           "params": json.dumps(params), "seed": 42}
    for k, (m, s) in summary.items():
        row[f"{k}_mean"], row[f"{k}_std"] = m, s
    pd.DataFrame([row]).to_csv(path, mode="a",
                               header=not os.path.exists(path), index=False)
