import sys
import numpy as np, pandas as pd
from data import load_folds, ROOT
from metrics import SPECIES

tag = sys.argv[1] if len(sys.argv) > 1 else "convnext_tiny_ce"
scheme = sys.argv[2] if len(sys.argv) > 2 else "grouped"
n = len(SPECIES); names = [SPECIES[i] for i in range(n)]
pool = load_folds().query("not is_test").reset_index(drop=True)

def session(fn):
    return fn.split("-")[0] + "_" + fn.split(".")[0].split("-")[-1]

conf = np.zeros((n, n), dtype=int); sess = []
for k in range(5):
    va = pool[pool[f"fold_{scheme}"] == k]
    new = ROOT / "outputs" / f"misclassifications_{tag}_{scheme}_f{k}.csv"
    old = ROOT / "outputs" / f"misclassifications_{tag}_f{k}.csv"   # old naming, grouped runs only
    wrong = pd.read_csv(new if (new.exists() or scheme != "grouped") else old)
    for t, c in va["Label"].value_counts().items():
        conf[t, t] += c
    for (t, p), c in wrong.groupby(["True_Label", "Pred_Label"]).size().items():
        conf[t, t] -= c; conf[t, p] += c
    sess.append(pd.DataFrame({"session": va["Filename"].map(session).values,
                              "err": va["Filename"].isin(wrong["Filename"]).values}))

print(pd.DataFrame({"species": names, "n": conf.sum(1),
                    "recall": (conf.diagonal() / conf.sum(1)).round(3)}))
off = sorted(((conf[i, j], names[i], names[j]) for i in range(n) for j in range(n) if i != j), reverse=True)
print("\nTop confusions (count, true, predicted):"); [print(o) for o in off[:10]]
s = pd.concat(sess).groupby("session")["err"].agg(error_rate="mean", n="size")
print("\nWorst sessions:"); print(s.sort_values("error_rate", ascending=False).head(15))
pd.DataFrame(conf, index=names, columns=names).to_csv(ROOT / "outputs" / f"confusion_{tag}_{scheme}.csv")