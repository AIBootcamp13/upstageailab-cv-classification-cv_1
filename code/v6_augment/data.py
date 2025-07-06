# -*- coding: utf-8 -*-
"""
데이터 관련 기능들을 담은 모듈
- 데이터셋 클래스들
- 데이터 로딩 및 전처리
- Transform 정의
- 데이터 분할 로직
"""

import os
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold


class ImageDataset(Dataset):
    """기본 이미지 데이터셋 클래스"""
    def __init__(self, csv, path, transform=None):
        if isinstance(csv, str):
            self.df = pd.read_csv(csv).values
        else:
            self.df = csv.values
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name, target = self.df[idx]
        img = np.array(Image.open(os.path.join(self.path, name)))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, target


class IndexedImageDataset(Dataset):
    """인덱스 기반 이미지 데이터셋 클래스"""
    def __init__(self, df, path, transform=None):
        self.df = df
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name, target = self.df.iloc[idx]
        img = np.array(Image.open(os.path.join(self.path, name)))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, target


class AugmentedDataset(Dataset):
    """원본 데이터셋을 여러 번 반복하여 증강을 적용하는 래퍼"""

    def __init__(self, base_dataset: Dataset, num_aug: int = 1):
        self.base_dataset = base_dataset
        self.num_aug = max(0, int(num_aug))

    def __len__(self):
        return len(self.base_dataset) * (self.num_aug + 1)

    def __getitem__(self, idx):
        base_idx = idx % len(self.base_dataset)
        return self.base_dataset[base_idx]


def _create_augraphy_lambda(intensity: float):
    """A.Lambda wrapper for simple Augraphy-like effects"""
    try:
        from augraphy import AugraphyPipeline, Geometric, Brightness, Blur, Noise

        pipeline = AugraphyPipeline(
            ink_phase=[],
            paper_phase=[],
            post_phase=[
                Geometric(rotate_range=(-15 * intensity, 15 * intensity), p=0.5 * intensity),
                Brightness(brightness_range=(1 - 0.3 * intensity, 1 + 0.3 * intensity), p=0.5 * intensity),
                Blur(blur_value=(1, 3), p=0.2 * intensity),
                Noise(noise_type="gauss", noise_value=(0, 10 * intensity), p=0.3 * intensity),
            ],
        )

        def _aug(image, **_):
            try:
                return {"image": pipeline(image)}
            except Exception:
                return {"image": image}

        return A.Lambda(image=_aug)
    except Exception:
        # Augraphy 설치되어 있지 않은 경우
        return None


def get_transforms(cfg):
    """이미지 변환을 위한 transform들을 반환"""
    img_size = cfg.data.img_size
    aug_cfg = getattr(cfg, "augmentation", {})
    method = getattr(aug_cfg, "method", "none").lower()
    intensity = float(getattr(aug_cfg, "intensity", 0))

    train_ops = []
    if method in ("albumentations", "mix") and intensity > 0:
        train_ops.extend([
            A.HorizontalFlip(p=0.5 * intensity),
            A.Rotate(limit=int(15 * intensity), p=0.5 * intensity),
            A.RandomBrightnessContrast(
                brightness_limit=0.2 * intensity,
                contrast_limit=0.2 * intensity,
                p=0.5 * intensity,
            ),
            A.Blur(blur_limit=3, p=0.2 * intensity),
        ])

    if method in ("augraphy", "mix") and intensity > 0:
        augraphy_aug = _create_augraphy_lambda(intensity)
        if augraphy_aug is not None:
            train_ops.append(augraphy_aug)

    train_ops.extend([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    train_transform = A.Compose(train_ops)

    test_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    return train_transform, test_transform


def prepare_data_loaders(cfg, seed):
    """설정에 따라 데이터 로더들을 준비"""
    data_path = cfg.data.data_path
    img_size = cfg.data.img_size
    batch_size = cfg.training.batch_size
    num_workers = cfg.data.num_workers

    # Transform 준비
    train_transform, test_transform = get_transforms(cfg)
    
    # 전체 훈련 데이터 로드
    full_train_df = pd.read_csv(f"{data_path}/train.csv")
    
    # 테스트 데이터 로더 준비
    test_dataset = ImageDataset(
        f"{data_path}/sample_submission.csv",
        f"{data_path}/test/",
        transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # augmentation 설정
    aug_cfg = getattr(cfg, "augmentation", {})

    # 검증 전략에 따른 데이터 분할
    validation_strategy = cfg.validation.strategy
    
    if validation_strategy == "holdout":
        train_loader, val_loader = _prepare_holdout_loaders(
            cfg, full_train_df, data_path, train_transform, test_transform, seed
        )
        return train_loader, val_loader, test_loader, None
        
    elif validation_strategy == "kfold":
        folds = _prepare_kfold_splits(cfg, full_train_df, seed)
        return None, None, test_loader, (folds, full_train_df, data_path, train_transform, test_transform)
        
    elif validation_strategy == "none":
        train_dataset = IndexedImageDataset(
            full_train_df,
            f"{data_path}/train/",
            transform=train_transform
        )
        if getattr(aug_cfg, "train_count", 0) > 0:
            train_dataset = AugmentedDataset(train_dataset, getattr(aug_cfg, "train_count", 0))
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True, 
            drop_last=False
        )
        return train_loader, None, test_loader, None
        
    else:
        raise ValueError(f"Unknown validation strategy: {validation_strategy}")


def _prepare_holdout_loaders(cfg, full_train_df, data_path, train_transform, test_transform, seed):
    """Holdout 검증을 위한 데이터 로더 준비"""
    train_ratio = cfg.validation.holdout.train_ratio
    stratify = cfg.validation.holdout.stratify
    batch_size = cfg.training.batch_size
    num_workers = cfg.data.num_workers
    
    if stratify:
        train_df, val_df = train_test_split(
            full_train_df, 
            test_size=1-train_ratio, 
            stratify=full_train_df['target'], 
            random_state=seed
        )
    else:
        train_df, val_df = train_test_split(
            full_train_df, 
            test_size=1-train_ratio, 
            random_state=seed
        )
    
    # Dataset 정의
    train_dataset = IndexedImageDataset(
        train_df,
        f"{data_path}/train/",
        transform=train_transform
    )
    if getattr(aug_cfg, "train_count", 0) > 0:
        train_dataset = AugmentedDataset(train_dataset, getattr(aug_cfg, "train_count", 0))

    val_dataset = IndexedImageDataset(
        val_df,
        f"{data_path}/train/",
        transform=test_transform
    )
    if getattr(aug_cfg, "valid_count", 0) > 0:
        val_dataset = AugmentedDataset(val_dataset, getattr(aug_cfg, "valid_count", 0))
    
    # DataLoader 정의
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True, 
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader


def _prepare_kfold_splits(cfg, full_train_df, seed):
    """K-Fold 교차 검증을 위한 분할 준비"""
    n_splits = cfg.validation.kfold.n_splits
    stratify = cfg.validation.kfold.stratify
    
    if stratify:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(skf.split(full_train_df, full_train_df['target']))
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(kf.split(full_train_df))
    
    return folds


def get_kfold_loaders(fold_idx, folds, full_train_df, data_path, train_transform, test_transform, cfg):
    """특정 fold에 대한 데이터 로더 반환"""
    train_idx, val_idx = folds[fold_idx]
    
    # 현재 fold의 데이터 분할
    train_df = full_train_df.iloc[train_idx]
    val_df = full_train_df.iloc[val_idx]
    
    # Dataset 정의
    train_dataset = IndexedImageDataset(
        train_df,
        f"{data_path}/train/",
        transform=train_transform
    )
    if getattr(aug_cfg, "train_count", 0) > 0:
        train_dataset = AugmentedDataset(train_dataset, getattr(aug_cfg, "train_count", 0))
    val_dataset = IndexedImageDataset(
        val_df,
        f"{data_path}/train/",
        transform=test_transform
    )
    if getattr(aug_cfg, "valid_count", 0) > 0:
        val_dataset = AugmentedDataset(val_dataset, getattr(aug_cfg, "valid_count", 0))
    
    # DataLoader 정의
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=True, 
        num_workers=cfg.data.num_workers, 
        pin_memory=True, 
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=False, 
        num_workers=cfg.data.num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader, train_df, val_df 