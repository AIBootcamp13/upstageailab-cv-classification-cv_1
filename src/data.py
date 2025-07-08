import os
from typing import Tuple, Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import torch

from augment import get_album_transform, get_augraphy_pipeline


def _should_use_pin_memory() -> bool:
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return False
        elif torch.cuda.is_available():
            return True
        else:
            return False
    except Exception:
        return False


class ImageDataset(Dataset):
    def __init__(self, csv, path: str, transform: Optional[A.Compose] = None):
        if isinstance(csv, str):
            self.df = pd.read_csv(csv).values
        else:
            self.df = csv.values
        self.path = path
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        name, target = self.df[idx]
        img = np.array(Image.open(os.path.join(self.path, name)))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, target


class IndexedImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, path: str, transform: Optional[A.Compose] = None):
        self.df = df
        self.path = path
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        name, target = self.df.iloc[idx]
        img = np.array(Image.open(os.path.join(self.path, name)))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, target


class AugmentedDataset(Dataset):
    """Dataset wrapper creating multiple augmented copies per image."""

    def __init__(self, base_dataset: Dataset, num_aug: int, aug_transform: A.Compose, base_transform: A.Compose):
        self.base_dataset = base_dataset
        self.num_aug = max(0, int(num_aug))
        self.aug_transform = aug_transform
        self.base_transform = base_transform

    def __len__(self) -> int:
        if self.num_aug == 0:
            return len(self.base_dataset)
        return len(self.base_dataset) * self.num_aug

    def __getitem__(self, idx: int):
        base_idx = idx % len(self.base_dataset)
        img, target = self.base_dataset[base_idx]
        if self.num_aug == 0:
            return self.base_transform(image=img)["image"], target
        img = self.aug_transform(image=img)["image"]
        return img, target


def _basic_transform(img_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def _aug_transform(img_size: int) -> A.Compose:
    album = get_album_transform(img_size)
    pipeline = get_augraphy_pipeline()

    def _apply(image, **_):
        try:
            image = pipeline(image)
        except Exception:
            pass
        return image

    return A.Compose([A.Lambda(image=_apply)] + album.transforms)


def get_transforms(cfg, mode: str) -> A.Compose:
    img_size = cfg.data.img_size
    if mode in {"train", "valid"}:
        return _aug_transform(img_size)
    return _basic_transform(img_size)


def _split_holdout(full_df: pd.DataFrame, ratio: float, stratify: bool, seed: int):
    if stratify:
        return train_test_split(
            full_df,
            test_size=1 - ratio,
            stratify=full_df["target"],
            random_state=seed,
        )
    return train_test_split(full_df, test_size=1 - ratio, random_state=seed)


def prepare_data_loaders(cfg, seed: int):
    train_images_path = cfg.data.train_images_path
    test_images_path = cfg.data.test_images_path
    train_csv_path = cfg.data.train_csv_path
    test_csv_path = cfg.data.test_csv_path

    batch_size = cfg.train.batch_size
    num_workers = cfg.data.num_workers

    train_t = get_transforms(cfg, "train")
    val_t = get_transforms(cfg, "valid")
    test_t = get_transforms(cfg, "test")

    full_train_df = pd.read_csv(train_csv_path)

    test_dataset = ImageDataset(test_csv_path, test_images_path, transform=test_t)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=_should_use_pin_memory(),
    )

    aug_cfg = getattr(cfg, "augment", {})
    strategy = cfg.validation.strategy

    if strategy == "holdout":
        train_df, val_df = _split_holdout(
            full_train_df,
            cfg.validation.holdout.train_ratio,
            cfg.validation.holdout.stratify,
            seed,
        )
        train_ds = IndexedImageDataset(train_df, train_images_path, transform=None)
        val_ds = IndexedImageDataset(val_df, train_images_path, transform=None)
        if aug_cfg.get("train_aug_count", 0) > 0:
            train_ds = AugmentedDataset(train_ds, aug_cfg.get("train_aug_count", 0), train_t, test_t)
        else:
            train_ds = IndexedImageDataset(train_df, train_images_path, transform=train_t)
        if aug_cfg.get("valid_aug_count", 0) > 0:
            val_ds = AugmentedDataset(val_ds, aug_cfg.get("valid_aug_count", 0), val_t, test_t)
        else:
            val_ds = IndexedImageDataset(val_df, train_images_path, transform=val_t)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=_should_use_pin_memory(),
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=_should_use_pin_memory(),
        )
        return train_loader, val_loader, test_loader, None

    if strategy == "kfold":
        folds = _prepare_kfold_splits(cfg, full_train_df, seed)
        return None, None, test_loader, (
            folds,
            full_train_df,
            train_images_path,
            train_t,
            val_t,
            test_t,
        )

    if strategy == "none":
        train_ds = IndexedImageDataset(full_train_df, train_images_path, transform=None)
        if aug_cfg.get("train_aug_count", 0) > 0:
            train_ds = AugmentedDataset(train_ds, aug_cfg.get("train_aug_count", 0), train_t, test_t)
        else:
            train_ds = IndexedImageDataset(full_train_df, train_images_path, transform=train_t)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=_should_use_pin_memory(),
            drop_last=False,
        )
        return train_loader, None, test_loader, None

    raise ValueError(f"Unknown validation strategy: {strategy}")


def _prepare_kfold_splits(cfg, full_train_df: pd.DataFrame, seed: int):
    n_splits = cfg.validation.kfold.n_splits
    stratify = cfg.validation.kfold.stratify
    if stratify:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(skf.split(full_train_df, full_train_df["target"]))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(full_train_df))


def get_kfold_loaders(
    fold_idx: int,
    folds,
    full_train_df: pd.DataFrame,
    train_images_path: str,
    train_transform: A.Compose,
    val_transform: A.Compose,
    cfg,
):
    train_idx, val_idx = folds[fold_idx]
    train_df = full_train_df.iloc[train_idx]
    val_df = full_train_df.iloc[val_idx]

    aug_cfg = getattr(cfg, "augment", {})
    base_t = _basic_transform(cfg.data.img_size)

    train_ds = IndexedImageDataset(train_df, train_images_path, transform=None)
    if aug_cfg.get("train_aug_count", 0) > 0:
        train_ds = AugmentedDataset(train_ds, aug_cfg.get("train_aug_count", 0), train_transform, base_t)
    else:
        train_ds = IndexedImageDataset(train_df, train_images_path, transform=train_transform)

    val_ds = IndexedImageDataset(val_df, train_images_path, transform=None)
    if aug_cfg.get("valid_aug_count", 0) > 0:
        val_ds = AugmentedDataset(val_ds, aug_cfg.get("valid_aug_count", 0), val_transform, base_t)
    else:
        val_ds = IndexedImageDataset(val_df, train_images_path, transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=_should_use_pin_memory(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=_should_use_pin_memory(),
    )
    return train_loader, val_loader, train_df, val_df
