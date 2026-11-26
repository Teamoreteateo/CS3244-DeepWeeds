"""
Per-image CE vs CutMix comparison from the existing misclassification CSVs.
No GPU, no images, no checkpoints needed.

Usage (from repo root):
    python src/compare_ce_cutmix.py
    python src/compare_ce_cutmix.py convnext_tiny_ce convnext_tiny_ce_cutmix   # custom tags

Writes outputs/per_image_ce_vs_cutmix.csv and prints evidence tables.
Note: a validation image NOT listed in a misclassification CSV was predicted correctly.
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score
from data import load_folds_paper, ROOT
from metrics import SPECIES

A = sys.argv[1] if len(sys.argv) > 1 else "convnext_tiny_ce"          # baseline
B = sys.argv[2] if len(sys.argv) > 2 else "convnext_tiny_ce_cutmix"   # best model
OUT = ROOT / "outputs"
NEG = SPECIES.index("Negative")
df = load_folds_paper()

rows = []
for k in range(5):
    va = df[df[f"paper_fold{k}"] == "val"][["Filename", "Label", "Species", "session"]].copy()
    va["fold"] = k
    for tag, name in ((A, "ce"), (B, "cm")):
        w = pd.read_csv(OUT / f"misclassifications_{tag}_val_paper_f{k}.csv").set_index("Filename")
        va[f"pred_{name}"] = va["Filename"].map(w["Pred_Label"]).fillna(va["Label"]).astype(int)
    rows.append(va)
d = pd.concat(rows, ignore_index=True)
d["ok_ce"] = d.pred_ce == d.Label
d["ok_cm"] = d.pred_cm == d.Label
d["status"] = np.select(
    [d.ok_ce & d.ok_cm, ~d.ok_ce & d.ok_cm, d.ok_ce & ~d.ok_cm,
     ~d.ok_ce & ~d.ok_cm & (d.pred_ce == d.pred_cm)],
    ["both_correct", "fixed_by_cutmix", "broken_by_cutmix", "both_wrong_same"],
    default="both_wrong_diff")
d["pred_ce_sp"] = d.pred_ce.map(dict(enumerate(SPECIES)))
d["pred_cm_sp"] = d.pred_cm.map(dict(enumerate(SPECIES)))
d.to_csv(OUT / "per_image_ce_vs_cutmix.csv", index=False)

print(f"validation predictions pooled over 5 folds: {len(d)}\n")
print(d.status.value_counts().to_string(), "\n")

# 1) Is CutMix's gain real? exact McNemar test on discordant pairs
b, c = (d.status == "fixed_by_cutmix").sum(), (d.status == "broken_by_cutmix").sum()
p = binomtest(b, b + c, 0.5).pvalue
print(f"McNemar (exact): CutMix fixes {b}, breaks {c}, net {b - c:+d}, p = {p:.4f}")

# 2) Bootstrap CI on pooled macro-F1 difference (resample images)
rng = np.random.default_rng(0)
y, pa, pb = d.Label.values, d.pred_ce.values, d.pred_cm.values
diff = []
for _ in range(1000):
    i = rng.integers(0, len(d), len(d))
    diff.append(f1_score(y[i], pb[i], average="macro") - f1_score(y[i], pa[i], average="macro"))
lo, hi = np.percentile(diff, [2.5, 97.5])
print(f"Pooled macro-F1 difference (CutMix - CE): "
      f"{f1_score(y, pb, average='macro') - f1_score(y, pa, average='macro'):+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]\n")

# 3) Where does CutMix help or hurt? per true species
g = d.groupby("Species")["status"].value_counts().unstack(fill_value=0)
g["net_gain"] = g.get("fixed_by_cutmix", 0) - g.get("broken_by_cutmix", 0)
print(g.reindex(SPECIES).to_string(), "\n")

# 4) What kind of mistake is fixed / broken? (true -> CE pred -> CutMix pred)
for s in ("fixed_by_cutmix", "broken_by_cutmix"):
    t = (d[d.status == s].groupby(["Species", "pred_ce_sp", "pred_cm_sp"]).size()
         .sort_values(ascending=False).head(8))
    print(f"Top patterns: {s}\n{t.to_string()}\n")

# 5) Chinee apple <-> Snake weed, and weed <-> Negative, under both models
for t_, p_ in (("Chinee apple", "Snake weed"), ("Snake weed", "Chinee apple")):
    n = (d.Species == t_).sum()
    print(f"{t_} -> {p_}: CE {((d.Species == t_) & (d.pred_ce_sp == p_)).sum()}, "
          f"CutMix {((d.Species == t_) & (d.pred_cm_sp == p_)).sum()} (of {n})")
weed = d.Label != NEG
print(f"weed -> Negative: CE {(weed & (d.pred_ce == NEG)).sum()}, CutMix {(weed & (d.pred_cm == NEG)).sum()}")
print(f"Negative -> weed: CE {((d.Label == NEG) & (d.pred_ce != NEG)).sum()}, "
      f"CutMix {((d.Label == NEG) & (d.pred_cm != NEG)).sum()}")

# 6) Sessions: do both models fail on the same sessions? (points to data, not model)
s = d.assign(e_ce=~d.ok_ce, e_cm=~d.ok_cm).groupby("session").agg(
    n=("Filename", "size"), err_ce=("e_ce", "mean"), err_cm=("e_cm", "mean"))
s = s[s.n >= 30]
print(f"\nSession error-rate correlation CE vs CutMix (n>=30 images): {s.err_ce.corr(s.err_cm):.3f}")
print("Images wrong under BOTH models (shared 'hard' images):",
      int((~d.ok_ce & ~d.ok_cm).sum()), "of", int((~d.ok_ce).sum()), "CE errors")

# 7) Images (not fold-instances): some images are validation images in several folds.
#    "Persistent" errors = wrong under CutMix in EVERY fold where the image was validated -> genuinely hard images
im = d.groupby("Filename").agg(species=("Species", "first"), session=("session", "first"), n_seen=("fold", "size"),
                               wrong_ce=("ok_ce", lambda x: (~x).sum()), wrong_cm=("ok_cm", lambda x: (~x).sum()))
im["persistent_cm_error"] = (im.wrong_cm == im.n_seen) & (im.n_seen >= 2)
im.to_csv(OUT / "per_unique_image_summary.csv")
print(f"\n{len(im)} unique validation images ({int((im.n_seen >= 2).sum())} validated in >=2 folds)")
print(f"Persistent CutMix errors (wrong every time, seen >=2x): {int(im.persistent_cm_error.sum())}")
print(im[im.persistent_cm_error].species.value_counts().to_string())