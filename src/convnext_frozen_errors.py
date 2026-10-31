import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from data import load_folds, OUTPUTS
from metrics import evaluate, SPECIES

TAG = "convnext_tiny_frozen_logreg"

X = np.load(OUTPUTS / "X_convnext_tiny.npy")
f = load_folds()
y = f["Label"].values
order = pd.read_csv(OUTPUTS / "feature_order_convnext_tiny.csv")["Filename"].tolist()
assert order == f["Filename"].tolist(), "row misalignment"

mask = (~f["is_test"]).values
Xp, yp = X[mask], y[mask]
files = f.loc[mask, "Filename"].values

for scheme in ("grouped", "random"):
    fold = f.loc[mask, f"fold_{scheme}"].values
    scores = []
    for k in range(5):
        tr, va = fold != k, fold == k
        pipe = Pipeline([("sc", StandardScaler()),
                         ("lr", LogisticRegression(max_iter=2000, n_jobs=-1))])
        pipe.fit(Xp[tr], yp[tr])
        pred = pipe.predict(Xp[va])
        scores.append(evaluate(yp[va], pred)["macro_f1"])
        wrong = pred != yp[va]
        pd.DataFrame({
            "Filename": files[va][wrong],
            "True_Label": yp[va][wrong],
            "Pred_Label": pred[wrong],
            "True_Species": [SPECIES[i] for i in yp[va][wrong]],
            "Pred_Species": [SPECIES[i] for i in pred[wrong]],
        }).to_csv(OUTPUTS / f"misclassifications_{TAG}_{scheme}_f{k}.csv", index=False)
    print(f"{scheme}: macro F1 per fold {np.round(scores, 3).tolist()} | mean {np.mean(scores):.3f}")