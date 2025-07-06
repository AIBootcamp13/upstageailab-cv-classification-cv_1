import os
import sys
import tempfile
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import ImageDataset, AugmentedDataset, get_transforms
from inference import predict_single_model
import pytest
import albumentations as A


def test_augmented_dataset_len():
    tmp = tempfile.mkdtemp()
    img_dir = os.path.join(tmp, "imgs")
    os.makedirs(img_dir)
    df = pd.DataFrame({'ID': ["a.jpg", "b.jpg"], 'target': [0, 1]})
    for name in df['ID']:
        Image.new('RGB', (10, 10), color='white').save(os.path.join(img_dir, name))
    dataset = ImageDataset(df, img_dir)
    aug = AugmentedDataset(dataset, num_aug=2)
    assert len(aug) == len(dataset) * 3


def test_predict_single_model_tta():
    tmp = tempfile.mkdtemp()
    img_dir = os.path.join(tmp, "imgs")
    os.makedirs(img_dir)
    df = pd.DataFrame({'ID': [f'{i}.jpg' for i in range(4)], 'target': [0]*4})
    for name in df['ID']:
        Image.new('RGB', (32, 32), color='white').save(os.path.join(img_dir, name))
    cfg = OmegaConf.create({'data': {'img_size': 32}, 'augmentation': {'method': 'albumentations', 'intensity': 0.5}})
    train_t, test_t = get_transforms(cfg)
    dataset = ImageDataset(df, img_dir, transform=test_t)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(32*32*3, 2))
    preds = predict_single_model(model, loader, torch.device('cpu'), tta_transform=train_t, tta_count=1)
    assert len(preds) == len(dataset)


def test_get_transforms_many_ops():
    cfg = OmegaConf.create({'data': {'img_size': 32}, 'augmentation': {'method': 'albumentations', 'intensity': 1.0}})
    train_t, _ = get_transforms(cfg)
    assert len(train_t.transforms) > 10


def test_get_transforms_augraphy_lambda():
    import importlib
    if importlib.util.find_spec('augraphy') is None:
        pytest.skip('augraphy not installed')
    cfg = OmegaConf.create({'data': {'img_size': 32}, 'augmentation': {'method': 'augraphy', 'intensity': 1.0}})
    train_t, _ = get_transforms(cfg)
    assert any(isinstance(t, A.Lambda) for t in train_t.transforms)
