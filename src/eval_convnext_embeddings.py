import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from data import load_folds_paper, OUTPUTS
from metrics import evaluate, summarise, log_result, SPECIES


def run(balanced=False):
    X = np.load(OUTPUTS / "X_convnext_tiny.npy")
    f = load_folds_paper()

    order = pd.read_csv(OUTPUTS / "feature_order_convnext_tiny.csv")["Filename"].tolist()
    pos = pd.Series(range(len(order)), index=order)
    X = X[pos.loc[f["Filename"]].to_numpy()]
    y = f["Label"].to_numpy()
    # mask = (~f["is_test"]).values
    # Xp, yp = X[mask], y[mask]
    # fold = f.loc[mask, f"fold_{scheme}"].values

    res = []
    for k in range(5):
        # tr, va = fold != k, fold == k
        role = f[f"paper_fold{k}"].to_numpy()
        tr, va = role == "train", role == "val"
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, n_jobs=-1,
                                      class_weight="balanced" if balanced else None)),
        ])
        pipe.fit(X[tr], y[tr])
        r = evaluate(y[va], pipe.predict(X[va]))
        res.append(r)
        print(f"  fold {k}: macroF1={r['macro_f1']:.4f} acc={r['accuracy']:.4f}")

    s = summarise(res)
    print(f"\nbalanced={balanced}")
    for k, (m, sd) in s.items():
        print(f"  {k:24s} {m:.4f} +/- {sd:.4f}")
    print(pd.DataFrame(res[-1]["cm_norm"], index=SPECIES, columns=SPECIES).round(2).to_string())

    log_result(f"convnext_tiny_frozen_logreg{'_bal' if balanced else ''}_val",
               "paper", {"head": "logreg", "balanced": balanced}, s)


if __name__ == "__main__":
    for bal in (False, True):
      run(bal)