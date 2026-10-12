"""Classical descriptors for the DeepWeeds rungs. Run once; cache to outputs/.

Usage:
    python features.py 500      # smoke test on 500 random images
    python features.py          # full dataset
"""
import json
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage import color, feature, filters

from data import load_folds, IMG_DIR, OUTPUTS

SIZE = 64
HSV_BINS = 32
LBP_P, LBP_R, LBP_BINS = 8, 1, 10


def extract(path, size=SIZE):
    """Return a 1-D float32 feature vector for one image.

    Blocks, in order: HSV histograms | LBP histogram | HOG | edge/intensity stats.
    """
    img_u8 = np.asarray(
        Image.open(path).convert("RGB").resize((size, size)), dtype=np.uint8)
    img = img_u8.astype(np.float32) / 255.0

    hsv = color.rgb2hsv(img)
    grey = color.rgb2gray(img)
    grey_u8 = (grey * 255).astype(np.uint8)   # LBP needs integer dtype

    # --- colour: normalised HSV histograms -------------------------------
    hist = np.concatenate([
        np.histogram(hsv[..., c], bins=HSV_BINS, range=(0, 1))[0]
        for c in range(3)
    ]).astype(np.float32)
    hist /= hist.sum() + 1e-8

    # --- texture: uniform LBP histogram ----------------------------------
    lbp = feature.local_binary_pattern(grey_u8, P=LBP_P, R=LBP_R, method="uniform")
    lbp_h = np.histogram(lbp, bins=LBP_BINS, range=(0, LBP_BINS))[0].astype(np.float32)
    lbp_h /= lbp_h.sum() + 1e-8

    # --- shape: histogram of oriented gradients --------------------------
    hog = feature.hog(grey, orientations=9, pixels_per_cell=(16, 16),
                      cells_per_block=(2, 2), feature_vector=True).astype(np.float32)

    # --- edges and global intensity --------------------------------------
    edges = feature.canny(grey, sigma=1.0)
    stats = np.array([edges.mean(), filters.sobel(grey).mean(),
                      grey.mean(), grey.std()], dtype=np.float32)

    return np.concatenate([hist, lbp_h, hog, stats])


def main(limit=None):
    f = load_folds()
    if limit:
        f = f.sample(limit, random_state=0).reset_index(drop=True)
        print(f"SMOKE TEST on {limit} images")

    probe = extract(IMG_DIR / f["Filename"].iloc[0])
    n_hsv, n_lbp, n_edge = 3 * HSV_BINS, LBP_BINS, 4
    n_hog = len(probe) - n_hsv - n_lbp - n_edge
    blocks = {
        "hsv":  [0, n_hsv],
        "lbp":  [n_hsv, n_hsv + n_lbp],
        "hog":  [n_hsv + n_lbp, n_hsv + n_lbp + n_hog],
        "edge": [n_hsv + n_lbp + n_hog, len(probe)],
    }
    print(f"feature dim = {len(probe)}   blocks = {blocks}")

    X = np.zeros((len(f), len(probe)), dtype=np.float32)
    for i, fn in enumerate(tqdm(f["Filename"], desc="extracting")):
        X[i] = extract(IMG_DIR / fn)

    # --- health checks ----------------------------------------------------
    print(f"\nNaN present:        {np.isnan(X).any()}")
    print(f"Inf present:        {np.isinf(X).any()}")
    print(f"constant columns:   {(X.std(axis=0) == 0).sum()} / {X.shape[1]}")
    print(f"value range:        [{X.min():.4f}, {X.max():.4f}]")

    OUTPUTS.mkdir(exist_ok=True)
    suffix = f"_{limit}" if limit else ""
    np.save(OUTPUTS / f"X_{SIZE}{suffix}.npy", X)
    np.save(OUTPUTS / f"y{suffix}.npy", f["Label"].values)
    f["Filename"].to_csv(OUTPUTS / f"feature_order{suffix}.csv", index=False)
    if not limit:
        json.dump(blocks, open(OUTPUTS / "feature_blocks.json", "w"), indent=2)
    print(f"\ncached {X.shape} -> outputs/X_{SIZE}{suffix}.npy")


if __name__ == "__main__":
    main(limit=int(sys.argv[1]) if len(sys.argv) > 1 else None)