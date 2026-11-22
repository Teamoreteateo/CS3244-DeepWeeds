"""Linear heads on frozen ResNet-50 features. 5 folds, all schemes."""
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from data import load_folds, load_folds_paper, OUTPUTS
from metrics import evaluate, summarise, log_result, SPECIES


def run(scheme, balanced=False):
    X = np.load(OUTPUTS / "X_resnet50.npy")
    y = np.load(OUTPUTS / "y.npy")
    paper = scheme == "paper"
    f = load_folds_paper() if paper else load_folds()

    order = pd.read_csv(OUTPUTS / "feature_order_resnet50.csv")["Filename"].tolist()
    assert order == f["Filename"].tolist(), "row misalignment"

    res = []
    for k in range(5):
        if paper:
            col = f"paper_fold{k}"
            tr = (f[col] == "train").values
            va = (f[col] == "val").values
        else:
            mask = (~f["is_test"]).values
            fold = f[f"fold_{scheme}"].values
            tr = mask & (fold != k)
            va = mask & (fold == k)
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
    print(f"\n{scheme} balanced={balanced}")
    for name, (m, sd) in s.items():
        print(f"  {name:24s} {m:.4f} +/- {sd:.4f}")
    print(pd.DataFrame(res[-1]["cm_norm"], index=SPECIES, columns=SPECIES).round(2).to_string())

    log_result(f"resnet50_frozen_logreg{'_bal' if balanced else ''}",
               scheme, {"head": "logreg", "balanced": balanced}, s)


if __name__ == "__main__":
    for scheme in ("grouped", "random", "paper"):
        for bal in (False, True):
            run(scheme, bal)