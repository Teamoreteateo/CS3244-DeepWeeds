"""End-to-end check: cached features -> Pipeline -> shared metrics.

Confirms the W5-W6 infrastructure works before any real modelling.
Run from src/:  python smoke_test.py
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from data import load_folds, OUTPUTS
from metrics import evaluate, summarise, log_result, SPECIES

SIZE = 64
SCHEME = "grouped"      # "grouped" or "random"


def main():
    X = np.load(OUTPUTS / f"X_{SIZE}.npy")
    y = np.load(OUTPUTS / "y.npy")
    f = load_folds()

    # Row alignment is load-bearing: features were extracted in folds.csv order.
    order = pd.read_csv(OUTPUTS / "feature_order.csv")["Filename"].tolist()
    assert order == f["Filename"].tolist(), "feature/manifest row misalignment"
    assert len(X) == len(f) == len(y), "length mismatch"
    assert (y == f["Label"].values).all(), "label mismatch"
    print(f"aligned: X={X.shape}  y={y.shape}")

    mask = (~f["is_test"]).values
    Xp, yp = X[mask], y[mask]
    fold = f.loc[mask, f"fold_{SCHEME}"].values
    print(f"pool={mask.sum()}  held-out test={(~mask).sum()}  scheme={SCHEME}")

    results = []
    for k in range(5):
        tr, va = fold != k, fold == k
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42,
                                          n_jobs=-1)),
        ])
        pipe.fit(Xp[tr], yp[tr])          # scaler fitted on TRAIN ONLY
        r = evaluate(yp[va], pipe.predict(Xp[va]))
        results.append(r)
        print(f"  fold {k}: macro_F1={r['macro_f1']:.4f}  acc={r['accuracy']:.4f}")

    s = summarise(results)
    print("\n--- summary (mean +/- std over 5 folds) ---")
    for k, (m, sd) in s.items():
        print(f"  {k:24s} {m:.4f} +/- {sd:.4f}")

    print("\nrow-normalised confusion matrix (last fold):")
    cm = pd.DataFrame(results[-1]["cm_norm"], index=SPECIES, columns=SPECIES)
    print(cm.round(2).to_string())

    log_result("rf_smoke", SCHEME, {"n_estimators": 200, "size": SIZE},
               s)
    print("\nlogged to results.csv")


if __name__ == "__main__":
    main()