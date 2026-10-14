import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from data import load_folds, OUTPUTS
from metrics import evaluate, summarise, log_result, SPECIES


def run(scheme, balanced=False):
    X = np.load(OUTPUTS / "X_convnext_tiny.npy")
    y = np.load(OUTPUTS / "y.npy")
    f = load_folds()

    order = pd.read_csv(OUTPUTS / "feature_order_convnext_tiny.csv")["Filename"].tolist()
    assert order == f["Filename"].tolist(), "row misalignment"

    mask = (~f["is_test"]).values
    Xp, yp = X[mask], y[mask]
    fold = f.loc[mask, f"fold_{scheme}"].values

    res = []
    for k in range(5):
        tr, va = fold != k, fold == k
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, n_jobs=-1,
                                      class_weight="balanced" if balanced else None)),
        ])
        pipe.fit(Xp[tr], yp[tr])
        r = evaluate(yp[va], pipe.predict(Xp[va]))
        res.append(r)
        print(f"  fold {k}: macroF1={r['macro_f1']:.4f} acc={r['accuracy']:.4f}")

    s = summarise(res)
    print(f"\n{scheme} balanced={balanced}")
    for k, (m, sd) in s.items():
        print(f"  {k:24s} {m:.4f} +/- {sd:.4f}")
    print(pd.DataFrame(res[-1]["cm_norm"], index=SPECIES, columns=SPECIES).round(2).to_string())

    log_result(f"convnext_tiny_frozen_logreg{'_bal' if balanced else ''}",
               scheme, {"head": "logreg", "balanced": balanced}, s)


if __name__ == "__main__":
    for scheme in ("grouped", "random"):
        for bal in (False, True):
            run(scheme, bal)