import os
import tempfile
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

from data import ImageDataset, AugmentedDataset, get_transforms
from inference import predict_single_model


def _create_dummy_dataset(tmpdir, num=2):
    img_dir = os.path.join(tmpdir, "imgs")
    os.makedirs(img_dir)
    df = pd.DataFrame({"ID": [f"{i}.jpg" for i in range(num)], "target": [0] * num})
    for name in df["ID"]:
        Image.new("RGB", (32, 32), color="white").save(os.path.join(img_dir, name))
    return df, img_dir


def test_augmented_dataset_len():
    tmp = tempfile.mkdtemp()
    df, img_dir = _create_dummy_dataset(tmp, 2)
    dataset = ImageDataset(df, img_dir)
    aug_t = Compose([Resize(32, 32), Normalize(), ToTensorV2()])
    base_t = Compose([Resize(32, 32), Normalize(), ToTensorV2()])
    aug = AugmentedDataset(dataset, num_aug=2, aug_transform=aug_t, base_transform=base_t)
    assert len(aug) == len(dataset) * 2


def test_predict_single_model_tta():
    tmp = tempfile.mkdtemp()
    df, img_dir = _create_dummy_dataset(tmp, 4)
    cfg = OmegaConf.create({"data": {"img_size": 32}, "augment": {"test_tta_enabled": True, "test_tta_count": 1}})
    tta_t = get_transforms(cfg, "test")
    dataset = ImageDataset(df, img_dir, transform=tta_t)
    loader = DataLoader(dataset, batch_size=2)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(32 * 32 * 3, 2))
    preds = predict_single_model(model, loader, torch.device("cpu"))
    assert len(preds) == len(dataset)
