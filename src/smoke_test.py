"""End-to-end check: cached features -> Pipeline -> shared metrics.

Run from src/:  python smoke_test.py
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from data import load_folds, load_folds_paper, OUTPUTS
from metrics import evaluate, summarise, log_result, SPECIES

SIZE = 64
SCHEME = "grouped"      # "grouped", "random", or "paper"


def main():
    X = np.load(OUTPUTS / f"X_{SIZE}.npy")
    y = np.load(OUTPUTS / "y.npy")
    paper = SCHEME == "paper"
    f = load_folds_paper() if paper else load_folds()

    # Row alignment is load-bearing: features were extracted in folds.csv order.
    order = pd.read_csv(OUTPUTS / "feature_order.csv")["Filename"].tolist()
    assert order == f["Filename"].tolist(), "feature/manifest row misalignment"
    assert len(X) == len(f) == len(y), "length mismatch"
    assert (y == f["Label"].values).all(), "label mismatch"
    print(f"aligned: X={X.shape}  y={y.shape}  scheme={SCHEME}")

    results = []
    for k in range(5):
        if paper:
            col = f"paper_fold{k}"
            tr = (f[col] == "train").values
            va = (f[col] == "val").values
        else:
            mask = (~f["is_test"]).values
            fold = f[f"fold_{SCHEME}"].values
            tr = mask & (fold != k)
            va = mask & (fold == k)
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42,
                                          n_jobs=-1)),
        ])
        pipe.fit(X[tr], y[tr])            # scaler fitted on TRAIN ONLY
        r = evaluate(y[va], pipe.predict(X[va]))
        results.append(r)
        print(f"  fold {k}: macro_F1={r['macro_f1']:.4f}  acc={r['accuracy']:.4f} "
              f"(train={tr.sum()} val={va.sum()})")

    s = summarise(results)
    print("\n--- summary (mean +/- std over 5 folds) ---")
    for name, (m, sd) in s.items():
        print(f"  {name:24s} {m:.4f} +/- {sd:.4f}")

    print("\nrow-normalised confusion matrix (last fold):")
    cm = pd.DataFrame(results[-1]["cm_norm"], index=SPECIES, columns=SPECIES)
    print(cm.round(2).to_string())

    log_result("rf_smoke", SCHEME, {"n_estimators": 200, "size": SIZE}, s)
    print("\nlogged to results.csv")


if __name__ == "__main__":
    main()