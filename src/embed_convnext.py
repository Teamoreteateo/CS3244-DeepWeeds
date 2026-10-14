import numpy as np, torch
from torch.utils.data import DataLoader
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from tqdm import tqdm

from data import load_folds, IMG_DIR, OUTPUTS
from data_pipeline import DeepWeedsDataset, make_transforms
from metrics import SPECIES

BATCH, WORKERS = 64, 4


def main():
    dev = torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", dev)

    m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    m.classifier[2] = torch.nn.Identity()
    m.eval().to(dev)

    f = load_folds()
    ds = DeepWeedsDataset(f, IMG_DIR, make_transforms(False), SPECIES)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=WORKERS)

    out, labs = [], []
    with torch.no_grad():
        for xb, yb in tqdm(dl, desc="embedding"):
            out.append(m(xb.to(dev)).cpu().numpy())
            labs.append(yb.numpy())
    X = np.concatenate(out).astype(np.float32)
    y = np.concatenate(labs)

    assert (y == f["Label"].values).all(), "row order drifted"
    np.save(OUTPUTS / "X_convnext_tiny.npy", X)
    f["Filename"].to_csv(OUTPUTS / "feature_order_convnext_tiny.csv", index=False)
    print("cached", X.shape)


if __name__ == "__main__":
    main()