"""Dataset loading and splitting.

The important thing in this file is split_by_image(). Every split happens on
unique image paths before any augmentation or oversampling, so the same file
can never land in both train and test. An earlier version of this project
duplicated the path list before splitting, which put about 60% of the test set
into training and inflated the augmented models by roughly 4 points. See
tests/test_splits.py.
"""

import os
import random

import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .config import (BATCH_SIZE, IMG_SIZE, MAX_PER_CLASS, MEAN, NUM_WORKERS,
                     STD)
from .splits import split_by_image


def find_dataset():
    """Return the dataset root, preferring Kaggle's mounted copy."""
    native = "/kaggle/input/brain-tumor-mri-dataset"
    if os.path.exists(native):
        return native

    import kagglehub
    path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
    for root, dirs, _ in os.walk(path):
        if "Training" in dirs or "Testing" in dirs:
            return root
    return path


def collect_paths(root, max_per_class=MAX_PER_CLASS, seed=42):
    """Walk the dataset folders and return (paths, labels).

    The upstream dataset ships its own Training/Testing folders. I merge them
    and re-split myself so the split is controlled and reproducible here.
    """
    rng = random.Random(seed)
    paths, labels = [], []

    for split in ("Training", "Testing"):
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for cls in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            files = sorted(
                f for f in os.listdir(cls_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            if max_per_class and len(files) > max_per_class:
                rng.shuffle(files)
                files = files[:max_per_class]
            for f in files:
                paths.append(os.path.join(cls_dir, f))
                labels.append(cls.lower())

    if not paths:
        raise RuntimeError(f"no images found under {root}")

    encoder = LabelEncoder().fit(labels)
    return paths, list(encoder.transform(labels)), encoder


class MRIDataset(Dataset):
    """Reads images off disk one batch at a time.

    Loading 2400 MRIs into RAM up front works on Kaggle but not on my laptop,
    so this stays lazy.
    """

    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = np.array(Image.open(self.paths[i]).convert("RGB"))
        return self.transform(image=img)["image"], int(self.labels[i])


def train_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.RandomBrightnessContrast(0.1, 0.1, p=0.3),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def heavy_transform():
    """Wider augmentation, used for the 'augmented' arm of the comparison.

    This is stronger augmentation, not generated data. Naming it synthetic
    would overstate what it does.
    """
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=20, p=0.6),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.4),
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.0), p=0.3),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def eval_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def make_loaders(paths, labels, seed, heavy_aug=False, repeats=1):
    """Build train / val / test loaders from a list of unique image paths.

    repeats > 1 lengthens the training epoch by revisiting each training image
    more than once under fresh augmentation. It only ever touches the training
    indices, so it cannot leak into val or test.
    """
    train_idx, val_idx, test_idx = split_by_image(paths, labels, seed)

    train_paths = [paths[i] for i in train_idx] * repeats
    train_labels = [labels[i] for i in train_idx] * repeats

    def loader(p, l, tfm, shuffle):
        return DataLoader(
            MRIDataset(p, l, tfm),
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
            prefetch_factor=2 if NUM_WORKERS > 0 else None,
        )

    tfm = heavy_transform() if heavy_aug else train_transform()
    return (
        loader(train_paths, train_labels, tfm, True),
        loader([paths[i] for i in val_idx], [labels[i] for i in val_idx],
               eval_transform(), False),
        loader([paths[i] for i in test_idx], [labels[i] for i in test_idx],
               eval_transform(), False),
        [labels[i] for i in train_idx],
    )
