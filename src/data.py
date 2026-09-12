"""Manifest construction, validation, and split loading."""
import re, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "images"
LABELS_CSV = ROOT / "data" / "labels.csv"
SPLITS = ROOT / "splits"
OUTPUTS = ROOT / "outputs"

FNAME = re.compile(r"^(\d{8})-(\d{6})-(\d)\.jpg$")
SEED = 42


def build_manifest(verbose=True):
    df = pd.read_csv(LABELS_CSV)
    df.columns = [c.strip() for c in df.columns]
    if verbose:
        print("columns:", list(df.columns), "| rows:", len(df))

    parts = df["Filename"].str.extract(FNAME)
    bad = parts[0].isna()
    if verbose:
        print(f"unparsed filenames: {bad.sum()}")
        if bad.any():
            print(df.loc[bad, "Filename"].head(10).tolist())

    df["date"] = parts[0]
    df["time"] = parts[1]
    df["instrument"] = parts[2]
    df["session"] = df["date"] + "-" + df["instrument"]
    df["path"] = df["Filename"].map(lambda f: str(IMG_DIR / f))
    df["exists"] = df["path"].map(lambda p: Path(p).exists())
    if verbose:
        print(f"missing image files: {(~df['exists']).sum()}")
    return df


def validate(df):
    return {
        "n_rows": len(df),
        "label_species_1to1": bool(
            df.groupby("Label")["Species"].nunique().max() == 1
            and df.groupby("Species")["Label"].nunique().max() == 1),
        "duplicate_filenames": int(df["Filename"].duplicated().sum()),
        "missing_files": int((~df["exists"]).sum()),
        "unparsed": int(df["session"].isna().sum()),
        "n_sessions": int(df["session"].nunique()),
        "n_dates": int(df["date"].nunique()),
        "class_counts": df["Species"].value_counts().to_dict(),
    }


def content_hashes(df):
    from tqdm import tqdm
    h = [hashlib.md5(Path(p).read_bytes()).hexdigest()
         for p in tqdm(df["path"], desc="hashing")]
    df = df.assign(md5=h)
    print(f"images with a byte-identical twin: {df['md5'].duplicated(keep=False).sum()}")
    return df


def load_folds():
    return pd.read_csv(SPLITS / "folds.csv", dtype={"session": str, "date": str})