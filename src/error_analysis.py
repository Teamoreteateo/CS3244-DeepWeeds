"""Instance-level error analysis on a saved checkpoint. Validation fold only.

    python src/error_analysis.py --ckpt outputs/checkpoints/resnet50_finetune_grouped_fold0.pt
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


def main(a):
    dev = torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    scheme, fold = ck["scheme"], ck["fold"]
    print(f"{ck['mode']} | {scheme} | fold {fold} | device {dev}")

    model = resnet18() if a.arch == "resnet18" else resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, 9)
    model.load_state_dict(ck["state_dict"])
    model.eval().to(dev)

    f = load_folds()
    pool = f[~f["is_test"]]
    val_df = pool[pool[f"fold_{scheme}"] == fold].copy()
    print(f"validation images: {len(val_df)}")

    ds = DeepWeedsDataset(val_df, IMG_DIR, make_transforms(augment=False), SPECIES)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=a.workers)

    preds, trues, names, confs = [], [], [], []
    with torch.no_grad():
        for xb, yb, fn in dl:
            out = torch.softmax(model(xb.to(dev)), dim=1)
            p = out.argmax(1).cpu().numpy()
            preds.append(p)
            confs.append(out.max(1).values.cpu().numpy())
            trues.append(yb.numpy())
            names.extend(fn)
    y, yhat = np.concatenate(trues), np.concatenate(preds)
    conf = np.concatenate(confs)

    r = evaluate(y, yhat)
    print(f"\nmacro_F1={r['macro_f1']:.4f}  acc={r['accuracy']:.4f} "
          f"weed->neg={r['weed_to_negative_rate']:.4f}")

    out = ROOT / "outputs" / "error_analysis"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.arch}_{ck['mode']}_{scheme}_fold{fold}"

    # 1. row-normalised confusion matrix, named axes
    cm = pd.DataFrame(r["cm_norm"], index=SPECIES, columns=SPECIES).round(3)
    cm.to_csv(out / f"cm_{tag}.csv")
    print("\nrow-normalised confusion matrix (true = rows):")
    print(cm.to_string())

    # 2. per-class precision / recall / F1
    per = pd.DataFrame(r["per_class"]).T.loc[[str(i) for i in range(9)]]
    per.index = SPECIES
    per.to_csv(out / f"per_class_{tag}.csv")
    print("\nper-class:")
    print(per[["precision", "recall", "f1-score", "support"]].round(3).to_string())

    # 3. misclassified instances, most confident errors first
    m = pd.DataFrame({
        "Filename": names,
        "true": [SPECIES[i] for i in y],
        "pred": [SPECIES[i] for i in yhat],
        "confidence": conf,
        "session": val_df.set_index("Filename").loc[names, "session"].values,
    })
    err = m[m.true != m.pred].sort_values("confidence", ascending=False)
    err.to_csv(out / f"misclassified_{tag}.csv", index=False)
    ok = m[m.true == m.pred].sort_values("confidence", ascending=False)
    ok.groupby("true").head(3).to_csv(out / f"correct_{tag}.csv", index=False)
    print(f"\n{len(err)} / {len(m)} misclassified ({len(err)/len(m):.1%})")

    # 4. most frequent confusion pairs
    pairs = (err.groupby(["true", "pred"]).size()
                .sort_values(ascending=False).head(12))
    print("\ntop confusion pairs (count):")
    print(pairs.to_string())

    # 5. the paper's reported confusions, tested
    print("\nOlsen et al. reported pairs:")
    for t, p in [("Chinee apple", "Snake weed"), ("Snake weed", "Chinee apple"),
                 ("Parkinsonia", "Prickly acacia"), ("Prickly acacia", "Parkinsonia")]:
        n_true = (m.true == t).sum()
        n_conf = ((m.true == t) & (m.pred == p)).sum()
        rate = n_conf / n_true if n_true else float("nan")
        print(f"  {t:16s} -> {p:16s}  {n_conf:4d}/{n_true:4d} = {rate:.3%}")

    # 6. errors by session - does one capture session dominate?
    by_sess = (m.assign(wrong=m.true != m.pred)
                 .groupby("session")["wrong"].agg(["sum", "count"]))
    by_sess["rate"] = by_sess["sum"] / by_sess["count"]
    by_sess.sort_values("rate", ascending=False).to_csv(out / f"by_session_{tag}.csv")
    print(f"\nworst sessions by error rate:")
    print(by_sess.sort_values("rate", ascending=False).head(8).round(3).to_string())

    print(f"\nwritten to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--arch", default="resnet50", choices=("resnet18", "resnet50"))
    p.add_argument("--workers", type=int, default=2)
    main(p.parse_args())