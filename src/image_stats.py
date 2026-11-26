"""
Objective image properties (brightness, contrast, sharpness, colour) for errors vs correct images.
Turns hypotheses like "dark / blurry photos fail more" into evidence.

Usage (repo root, needs data/images and outputs/per_image_ce_vs_cutmix.csv):
    python src/image_stats.py
Writes outputs/image_stats.csv (cached) and prints comparison tables.
"""
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import laplace
from scipy.stats import mannwhitneyu
from tqdm import tqdm
from data import IMG_DIR, ROOT

OUT = ROOT / "outputs"
d = pd.read_csv(OUT / "per_image_ce_vs_cutmix.csv", dtype={"session": str})
cache = OUT / "image_stats.csv"

if cache.exists():
    st = pd.read_csv(cache)
else:
    recs = []
    for fn in tqdm(d.Filename.unique()):
        im = Image.open(IMG_DIR / fn).convert("RGB")
        a = np.asarray(im, dtype=np.float32) / 255
        g = a.mean(2)
        hsv = np.asarray(im.convert("HSV"), dtype=np.float32) / 255
        recs.append(dict(Filename=fn, brightness=g.mean(), contrast=g.std(),
                         sharpness=laplace(g).var(),          # low = blurry
                         saturation=hsv[..., 1].mean(),
                         green_ratio=a[..., 1].mean() / (a.mean() + 1e-6)))
    st = pd.DataFrame(recs); st.to_csv(cache, index=False)

# an image can be a validation image in several folds (different models), so keep one row per (image, fold)
m = d.merge(st, on="Filename")
m["err_cm"] = ~m.ok_cm
feats = ["brightness", "contrast", "sharpness", "saturation", "green_ratio"]

print("Median feature value: CutMix-correct vs CutMix-wrong (Mann-Whitney p), per true species")
rows = []
for sp, g in m.groupby("Species"):
    for f in feats:
        a, b = g.loc[~g.err_cm, f], g.loc[g.err_cm, f]
        if len(b) >= 10:
            rows.append(dict(species=sp, feature=f, median_correct=a.median(), median_wrong=b.median(),
                             n_wrong=len(b), p=mannwhitneyu(a, b).pvalue))
t = pd.DataFrame(rows)
t["sig(p<0.01)"] = t.p < 0.01
print(t.round(4).to_string(index=False))

print("\nHigh-error sessions vs rest (n>=30 images): session-level mean feature vs error rate")
s = m.groupby("session").agg(n=("Filename", "size"), err=("err_cm", "mean"),
                             **{f: (f, "mean") for f in feats})
s = s[s.n >= 30]
print(s[["err"] + feats].corr(method="spearman")["err"].drop("err").round(3).to_string())
print("\nWorst sessions:\n", s.sort_values("err", ascending=False).head(8).round(3).to_string())