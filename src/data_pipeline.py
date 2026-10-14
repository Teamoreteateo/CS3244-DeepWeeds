"""Reusable DeepWeeds split, augmentation, and PyTorch DataLoader utilities.

Place in CS3244-DeepWeeds/src/data_pipeline.py next to data.py.
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import v2

try:
    from .data import IMG_DIR, load_folds
except ImportError:  # supports `sys.path.insert(0, '../src'); from data_pipeline import ...`
    from data import IMG_DIR, load_folds

try:
    from .metrics import SPECIES
except ImportError:
    from metrics import SPECIES

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class RandomRotate90:
    """Square-preserving rotation by 0/90/180/270 degrees."""

    def __call__(self, image):
        return TF.rotate(image, random.choice((0, 90, 180, 270)))


def make_transforms(augment: bool = False):
    """Use identical deterministic preprocessing for baseline/validation/test."""
    if augment:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            RandomRotate90(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=(0.4, 1.3), contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


class DeepWeedsDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform, classes):
        self.df = dataframe.reset_index(drop=True).copy()
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.classes = list(classes)
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        with Image.open(self.img_dir / row['Filename']) as img:
            image = img.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row['Label'])


def get_dataloaders(
    val_fold: int = 0,
    split: str = 'random',
    batch_size: int = 32,
    augment: bool = False,
    weighted_sampling: bool = False,
    include_test: bool = False,
    num_workers: int = 0,
    img_dir=None,
):
    """Create consistent DeepWeeds train/val/test datasets and loaders.

    Args:
        val_fold: Fold selected for validation among NON-TEST rows.
        split: 'random' (fold_random) or 'grouped' (fold_grouped).
        augment: Apply random transforms to training only.
        weighted_sampling: Inverse-class-frequency sampling with replacement for train.
        include_test: Return a held-out test DataLoader; use only for FINAL evaluation.

    Returns:
        train_loader, val_loader, test_loader, classes
        test_loader is None when include_test=False.
        Each loader's `.dataset` gives its underlying PyTorch Dataset.
    """
    if split not in ('random', 'grouped'):
        raise ValueError("split must be 'random' or 'grouped'")
    if batch_size < 1:
        raise ValueError('batch_size must be >= 1')

    df = load_folds()
    required = {'Filename', 'Label', 'Species', 'is_test', 'session', 'fold_random', 'fold_grouped'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing split metadata columns: {sorted(missing)}')
    classes = (df[['Label', 'Species']].drop_duplicates()
             .sort_values('Label')['Species'].tolist())
    if len(classes) != 9:
        raise ValueError(f'Expected 9 DeepWeeds classes; found {len(classes)}')
    if classes != SPECIES:
        raise ValueError(f'class order mismatch:\n{classes}\n{SPECIES}')
    non_test = df.loc[~df['is_test']].copy()
    test_df = df.loc[df['is_test']].copy()
    fold_column = f'fold_{split}'
    if df['is_test'].isna().any():
        raise ValueError('is_test contains missing values')
    if df['Filename'].duplicated().any():
        raise ValueError('Fold manifest contains duplicate filenames')
    known_folds = set(non_test[fold_column].dropna().tolist())
    if val_fold not in known_folds:
        raise ValueError(f'val_fold {val_fold} not found; available folds: {sorted(known_folds)}')
    if non_test[fold_column].isna().any() or (non_test[fold_column] < 0).any():
        raise ValueError(f'{fold_column} has missing or invalid assignments in non-test rows')

    train_df = non_test.loc[non_test[fold_column] != val_fold].copy()
    val_df = non_test.loc[non_test[fold_column] == val_fold].copy()
    if train_df.empty or val_df.empty:
        raise ValueError('Training or validation partition is empty')
    if not set(train_df['Filename']).isdisjoint(val_df['Filename']):
        raise ValueError('Training/validation filename overlap')
    if not set(train_df['Filename']).isdisjoint(test_df['Filename']) or not set(val_df['Filename']).isdisjoint(test_df['Filename']):
        raise ValueError('Held-out test filenames overlap train/validation')
    if not set(non_test['session']).isdisjoint(test_df['session']):
        raise ValueError('Held-out test sessions overlap non-test data')
    if split == 'grouped' and not set(train_df['session']).isdisjoint(val_df['session']):
        raise ValueError('Grouped train/validation partitions share sessions')

    image_root = Path(img_dir) if img_dir is not None else Path(IMG_DIR)
    if not image_root.is_dir():
        raise FileNotFoundError(f'Image directory does not exist: {image_root}')

    train_dataset = DeepWeedsDataset(train_df, image_root, make_transforms(augment), classes)
    val_dataset = DeepWeedsDataset(val_df, image_root, make_transforms(False), classes)
    test_dataset = DeepWeedsDataset(test_df, image_root, make_transforms(False), classes) if include_test else None

    sampler = None
    if weighted_sampling:
        counts = train_dataset.df['Species'].value_counts()
        sample_weights = (1.0 / train_dataset.df['Species'].map(counts)).to_numpy()
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_dataset),
            replacement=True,
        )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=(sampler is None), sampler=sampler, num_workers=num_workers,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = (
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        if include_test else None
    )
    return train_loader, val_loader, test_loader, classes


def apply_cutmix(images, labels, num_classes, probability: float = 0.5, alpha: float = 1.0):
    """Optional batch-level training transform; never apply for evaluation."""
    if not 0 <= probability <= 1:
        raise ValueError('probability must be between 0 and 1')
    if torch.rand(()).item() < probability:
        return v2.CutMix(num_classes=num_classes, alpha=alpha)(images, labels)
    return images, labels
