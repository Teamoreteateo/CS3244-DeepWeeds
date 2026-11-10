import sys
import numpy as np, pandas as pd
from data import load_folds, ROOT
from metrics import SPECIES

tag = sys.argv[1]
scheme = sys.argv[2] if len(sys.argv) > 2 else "grouped"
n = len(SPECIES); NEG = SPECIES.index("Negative")
out = ROOT / "outputs"
pool = load_folds().query("not is_test").reset_index(drop=True)

conf = np.zeros((n, n), dtype=int); sess = []
for k in range(5):
    va = pool[pool[f"fold_{scheme}"] == k]
    new = out / f"misclassifications_{tag}_{scheme}_f{k}.csv"
    old = out / f"misclassifications_{tag}_f{k}.csv"          # old naming, grouped runs only
    path = new if (new.exists() or scheme != "grouped") else old
    if not path.exists():
        sys.exit(f"missing {path.name}")
    wrong = pd.read_csv(path)
    for t, c in va["Label"].value_counts().items():
        conf[t, t] += c
    for (t, p), c in wrong.groupby(["True_Label", "Pred_Label"]).size().items():
        conf[t, t] -= c; conf[t, p] += c
    sess.append(pd.DataFrame({"session": va["session"].values,
                              "err": va["Filename"].isin(wrong["Filename"]).values}))

print(f"== {tag} | {scheme} | best-epoch predictions, 5 folds pooled ==")
print(pd.DataFrame({"species": SPECIES, "n": conf.sum(1),
                    "recall": (conf.diagonal() / conf.sum(1)).round(3)}).to_string(index=False))

w = [i for i in range(n) if i != NEG]
w2n = conf[w, NEG].sum(); n2w = conf[NEG, w].sum()
print(f"\nweed -> Negative (missed weeds): {w2n}  ({w2n / conf[w].sum():.3f} of weed images)")
print(f"Negative -> weed (false alarms): {n2w}  ({n2w / conf[NEG].sum():.3f} of Negative images)")

off = sorted(((conf[i, j], SPECIES[i], SPECIES[j]) for i in range(n) for j in range(n) if i != j), reverse=True)
print("\nTop confusions (count, true, predicted):")
for o in off[:10]: print(o)

s = pd.concat(sess).groupby("session")["err"].agg(errors="sum", n="size")
s["error_rate"] = (s["errors"] / s["n"]).round(3)
s["share_of_all_errors"] = (s["errors"] / s["errors"].sum()).round(3)
print("\nWorst sessions with at least 30 images:")
print(s[s["n"] >= 30].sort_values("error_rate", ascending=False).head(15).to_string())

pd.DataFrame(conf, index=SPECIES, columns=SPECIES).to_csv(out / f"confusion_{tag}_{scheme}.csv")