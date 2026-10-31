import sys
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from data import ROOT, IMG_DIR

tag, scheme, true_sp, pred_sp = sys.argv[1:5]
n = int(sys.argv[5]) if len(sys.argv) > 5 else 12
frames = []
for k in range(5):
    new = ROOT / "outputs" / f"misclassifications_{tag}_{scheme}_f{k}.csv"
    old = ROOT / "outputs" / f"misclassifications_{tag}_f{k}.csv"
    frames.append(pd.read_csv(new if (new.exists() or scheme != "grouped") else old))
df = pd.concat(frames)
sel = df[(df.True_Species == true_sp) & (df.Pred_Species == pred_sp)]
print(len(sel), "images")
sel = sel.sample(min(n, len(sel)), random_state=0)

cols = 4; rows = (len(sel) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows))
axes = axes.ravel()
for ax in axes: ax.axis("off")
for ax, fn in zip(axes, sel.Filename):
    ax.imshow(Image.open(IMG_DIR / fn).convert("RGB"))
    ax.set_title(fn[:-4], fontsize=7)
fig.suptitle(f"True: {true_sp} -> Pred: {pred_sp} ({tag}, {scheme})")
out = ROOT / "outputs" / f"errors_{tag}_{scheme}_{true_sp.replace(' ', '')}_to_{pred_sp.replace(' ', '')}.png"
fig.tight_layout(); fig.savefig(out, dpi=120); print("saved", out)