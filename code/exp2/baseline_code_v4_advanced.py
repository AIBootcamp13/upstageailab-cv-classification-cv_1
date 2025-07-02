# -*- coding: utf-8 -*-
"""baseline_code_advanced.py

Advanced version of the document type classification baseline code with performance improvements.

## Improvements Applied:
- Better model architecture (EfficientNet)
- Larger image size (224x224)
- Enhanced data augmentation
- Learning rate scheduler
- Validation data split
- Label smoothing loss
- Mixed precision training
- Test time augmentation (TTA)
- More training epochs

## Contents
- Prepare Environments
- Import Library & Define Functions
- Hyper-parameters
- Load Data
- Train Model
- Inference & Save File
"""

import os
import time
import random

import timm
import torch
import albumentations as A
import pandas as pd
import numpy as np
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# 로그 유틸리티 import
import utils.log_util as log

# 시드를 고정합니다.
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)

# CUDA 10.2+ 환경에서 결정적 연산을 위한 환경 변수 설정
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 완전한 재현성을 위한 설정
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# CUDA 환경에서 결정적 연산 설정 (선택적)
try:
    torch.use_deterministic_algorithms(True)
    log.info("완전한 재현성 설정이 활성화되었습니다.")
except Exception as e:
    log.info(f"torch.use_deterministic_algorithms(True) 설정 실패: {e}")
    log.info("기본 재현성 설정만 사용됩니다.")

# 현재 스크립트 위치를 작업 디렉토리로 설정
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Label Smoothing Loss 클래스 정의
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.classes = classes
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

# 데이터셋 클래스를 정의합니다.
class ImageDataset(Dataset):
    def __init__(self, csv_data, path, transform=None):
        if isinstance(csv_data, str):
            self.df = pd.read_csv(csv_data).values
        else:
            self.df = csv_data.values
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

# one epoch 학습을 위한 함수입니다.
def train_one_epoch(loader, model, optimizer, loss_fn, device, scaler):
    model.train()
    train_loss = 0
    preds_list = []
    targets_list = []

    pbar = tqdm(loader, desc="Training")
    for image, targets in pbar:
        image = image.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        with autocast():
            preds = model(image)
            loss = loss_fn(preds, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
        targets_list.extend(targets.detach().cpu().numpy())

        pbar.set_description(f"Training - Loss: {loss.item():.4f}")

    train_loss /= len(loader)
    train_acc = accuracy_score(targets_list, preds_list)
    train_f1 = f1_score(targets_list, preds_list, average='macro')

    ret = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_f1": train_f1,
    }

    return ret

# 검증을 위한 함수입니다.
def validate_one_epoch(loader, model, loss_fn, device):
    model.eval()
    val_loss = 0
    preds_list = []
    targets_list = []

    pbar = tqdm(loader, desc="Validation")
    with torch.no_grad():
        for image, targets in pbar:
            image = image.to(device)
            targets = targets.to(device)

            with autocast():
                preds = model(image)
                loss = loss_fn(preds, targets)

            val_loss += loss.item()
            preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
            targets_list.extend(targets.detach().cpu().numpy())

            pbar.set_description(f"Validation - Loss: {loss.item():.4f}")

    val_loss /= len(loader)
    val_acc = accuracy_score(targets_list, preds_list)
    val_f1 = f1_score(targets_list, preds_list, average='macro')

    ret = {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f1": val_f1,
    }

    return ret

# Test Time Augmentation을 위한 예측 함수
def predict_with_tta(model, loader, device, tta_count=5):
    model.eval()
    predictions = []
    
    pbar = tqdm(loader, desc="TTA Prediction")
    for image, _ in pbar:
        batch_preds = []
        for _ in range(tta_count):
            with torch.no_grad():
                with autocast():
                    pred = model(image.to(device))
                    batch_preds.append(pred.softmax(dim=1))
        
        # 평균내기
        final_pred = torch.stack(batch_preds).mean(0)
        predictions.extend(final_pred.argmax(dim=1).cpu().numpy())
    
    return predictions

"""## Hyper-parameters
* 학습 및 추론에 필요한 하이퍼파라미터들을 정의합니다.
"""

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f"Using device: {device}")

# data config
data_path = '../input/data'

# model config
model_name = 'efficientnet_b3'  # 더 좋은 모델 사용

# training config
img_size = 224  # 이미지 크기 대폭 확대
LR = 2e-4  # 더 낮은 학습률
EPOCHS = 20  # 충분한 epoch
BATCH_SIZE = 16  # 큰 모델에 맞춰 배치 크기 조정
num_workers = 0
weight_decay = 1e-4
label_smoothing = 0.1

"""## Load Data
* 학습, 검증, 테스트 데이터셋과 로더를 정의합니다.
"""

# 강화된 augmentation을 위한 transform 코드
trn_transform = A.Compose([
    A.Resize(height=img_size, width=img_size),
    # 다양한 데이터 증강 기법들
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    A.Blur(blur_limit=3, p=0.1),
    A.CLAHE(clip_limit=2.0, p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# validation image 변환을 위한 transform 코드
val_transform = A.Compose([
    A.Resize(height=img_size, width=img_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# test image 변환을 위한 transform 코드 (TTA용으로 약간의 augmentation 포함)
tst_transform = A.Compose([
    A.Resize(height=img_size, width=img_size),
    A.HorizontalFlip(p=0.5),  # TTA를 위한 가벼운 augmentation
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# 검증 데이터 분할
train_df = pd.read_csv(f"{data_path}/train.csv")
train_data, val_data = train_test_split(
    train_df, 
    test_size=0.2, 
    stratify=train_df['target'], 
    random_state=SEED
)

log.info(f"Train samples: {len(train_data)}")
log.info(f"Validation samples: {len(val_data)}")

# Dataset 정의
trn_dataset = ImageDataset(
    train_data,
    f"{data_path}/train/",
    transform=trn_transform
)
val_dataset = ImageDataset(
    val_data,
    f"{data_path}/train/",
    transform=val_transform
)
tst_dataset = ImageDataset(
    f"{data_path}/sample_submission.csv",
    f"{data_path}/test/",
    transform=val_transform  # TTA는 별도로 처리
)

log.info(f"Train dataset: {len(trn_dataset)}")
log.info(f"Validation dataset: {len(val_dataset)}")
log.info(f"Test dataset: {len(tst_dataset)}")

# DataLoader 정의
trn_loader = DataLoader(
    trn_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)
tst_loader = DataLoader(
    tst_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

"""## Train Model
* 모델을 로드하고, 학습을 진행합니다.
"""

# load model
model = timm.create_model(
    model_name,
    pretrained=True,
    num_classes=17
).to(device)

log.info(f"Model: {model_name}")
log.info(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss function, optimizer, scheduler 정의
loss_fn = LabelSmoothingLoss(classes=17, smoothing=label_smoothing)
optimizer = Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler = GradScaler()

# 최고 성능 모델 저장을 위한 변수
best_val_f1 = 0.0
best_model_path = "best_model.pth"

log.info("Starting training...")
for epoch in range(EPOCHS):
    log.info(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
    
    # Training
    train_ret = train_one_epoch(trn_loader, model, optimizer, loss_fn, device, scaler)
    
    # Validation
    val_ret = validate_one_epoch(val_loader, model, loss_fn, device)
    
    # Learning rate scheduler step
    scheduler.step()
    
    # 결과 출력
    current_lr = optimizer.param_groups[0]['lr']
    
    train_ret['epoch'] = epoch
    val_ret['epoch'] = epoch
    val_ret['lr'] = current_lr
    
    # 최고 성능 모델 저장
    if val_ret['val_f1'] > best_val_f1:
        best_val_f1 = val_ret['val_f1']
        torch.save(model.state_dict(), best_model_path)
        log.info(f"💾 Best model saved! F1: {best_val_f1:.4f}")
    
    log_message = ""
    for k, v in train_ret.items():
        log_message += f"train_{k}: {v:.4f} | "
    for k, v in val_ret.items():
        log_message += f"{k}: {v:.4f} | "
    
    log.info(log_message.rstrip(" | "))

# 최고 성능 모델 로드
log.info(f"\n🏆 Loading best model (F1: {best_val_f1:.4f})")
model.load_state_dict(torch.load(best_model_path))

"""## Inference & Save File
* 테스트 이미지에 대한 추론을 진행하고, 결과 파일을 저장합니다.
"""

log.info("\nStarting inference with Test Time Augmentation...")

# TTA를 위한 테스트 데이터셋 재정의
tst_dataset_tta = ImageDataset(
    f"{data_path}/sample_submission.csv",
    f"{data_path}/test/",
    transform=tst_transform
)
tst_loader_tta = DataLoader(
    tst_dataset_tta,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# TTA로 예측
preds_list = predict_with_tta(model, tst_loader_tta, device, tta_count=5)

# 결과 저장
pred_df = pd.DataFrame(tst_dataset.df, columns=['ID', 'target'])
pred_df['target'] = preds_list

sample_submission_df = pd.read_csv(f"{data_path}/sample_submission.csv")
assert (sample_submission_df['ID'] == pred_df['ID']).all()

output_path = "../output"
os.makedirs(output_path, exist_ok=True)
pred_df.to_csv(f"{output_path}/pred_advanced.csv", index=False)

log.info(f"\n✅ Prediction completed and saved to {output_path}/pred_advanced.csv")
log.info(f"Best validation F1 score: {best_val_f1:.4f}")

# 메모리 정리
if os.path.exists(best_model_path):
    os.remove(best_model_path)

pred_df.head()