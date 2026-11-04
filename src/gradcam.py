"""Grad-CAM heatmaps for misclassified DeepWeeds images.

    python src/gradcam.py --ckpt outputs/checkpoints/resnet50_finetune_grouped_fold0.pt \
        --errors outputs/error_analysis/misclassified_resnet50_finetune_grouped_fold0.csv --n 16
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from data import IMG_DIR, ROOT
from data_pipeline import make_transforms, MEAN, STD
from metrics import SPECIES


class GradCAM:
    """Activations and gradients from the last conv block."""

    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(self._fwd)
        layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, out):
        self.acts = out.detach()

    def _bwd(self, _m, _gi, go):
        self.grads = go[0].detach()

    def __call__(self, x, class_idx):
        self.model.zero_grad()
        logits = self.model(x)
        logits[0, class_idx].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)        # GAP over spatial dims
        cam = F.relu((w * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def denorm(t):
    img = t.cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * np.array(STD) + np.array(MEAN), 0, 1)


def main(a):
    dev = torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)

    from torchvision.models import resnet18, resnet50
    model = resnet18() if a.arch == "resnet18" else resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, 9)
    model.load_state_dict(ck["state_dict"])
    model.eval().to(dev)

    cam = GradCAM(model, model.layer4[-1])      # last conv block of ResNet
    tf = make_transforms(augment=False)
    df = pd.read_csv(a.errors).head(a.n)
    print(f"{len(df)} images | {ck['mode']} {ck['scheme']} fold {ck['fold']} | {dev}")

    cols = 4
    rows = int(np.ceil(len(df) / cols)) * 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_2d(axes)

    for i, (_, r) in enumerate(df.iterrows()):
        x = tf(Image.open(IMG_DIR / r.Filename).convert("RGB")).unsqueeze(0).to(dev)
        pred_idx = SPECIES.index(r.pred)
        heat = cam(x, pred_idx)                 # why it predicted what it did
        img = denorm(x[0])

        br, bc = (i // cols) * 2, i % cols
        axes[br, bc].imshow(img)
        axes[br, bc].set_title(f"true {r.true}", fontsize=7)
        axes[br, bc].axis("off")
        axes[br + 1, bc].imshow(img)
        axes[br + 1, bc].imshow(heat, cmap="jet", alpha=0.45)
        axes[br + 1, bc].set_title(f"pred {r.pred} ({r.confidence:.2f})", fontsize=7)
        axes[br + 1, bc].axis("off")

    for ax in axes.flat:
        if not ax.has_data():
            ax.axis("off")

    out = Path(a.output_dir) if a.output_dir else ROOT / "outputs" / "error_analysis"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"gradcam_{Path(a.errors).stem}.png"
    plt.tight_layout()
    plt.savefig(p, dpi=150)
    print("wrote", p)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--errors", required=True)
    p.add_argument("--arch", default="resnet50", choices=("resnet18", "resnet50"))
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--output-dir", default=None)
    main(p.parse_args())