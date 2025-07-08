# -*- coding: utf-8 -*-
"""
모델 관련 기능들을 담은 모듈
- 모델 생성
- 모델 로딩/저장
- 모델 관련 유틸리티
"""

import os
import timm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    StepLR, 
    ReduceLROnPlateau, 
    CosineAnnealingWarmRestarts
)

import log_util as log


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Loss 구현"""
    
    def __init__(self, num_classes, smoothing=0.1):
        """
        Args:
            num_classes (int): 클래스 수
            smoothing (float): 스무딩 정도 (0.0 ~ 1.0)
        """
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): 예측값 (batch_size, num_classes)
            target (torch.Tensor): 정답 라벨 (batch_size,)
        """
        log_probs = torch.log_softmax(pred, dim=-1)
        
        # 원-핫 인코딩으로 변환
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


def create_model(model_name, pretrained=True, num_classes=17):
    """TIMM을 사용해 모델 생성"""
    try:
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )
    except Exception:
        # 프리트레인 모델 다운로드 실패 시 pretrained=False로 재시도
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
        )
    return model


def create_scheduler(optimizer, cfg):
    """설정에 따라 스케쥴러 생성"""
    sched_cfg = getattr(cfg.train, "scheduler", {"enabled": False, "name": "none"})
    if not sched_cfg.get("enabled", False):
        return None

    scheduler_name = sched_cfg.get("name", "none").lower()
    
    if scheduler_name == "cosine":
        cosine_cfg = sched_cfg.get("cosine", {})
        T_max = cosine_cfg.get("T_max", 100)
        if T_max == 100:
            T_max = cfg.train.epochs

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=cosine_cfg.get("eta_min", 1e-6),
            last_epoch=cosine_cfg.get("last_epoch", -1)
        )
        log.info(f"CosineAnnealingLR 스케쥴러 생성 - T_max: {T_max}, eta_min: {cosine_cfg.get('eta_min', 1e-6)}")
        
    elif scheduler_name == "step":
        step_cfg = sched_cfg.get("step", {})
        scheduler = StepLR(
            optimizer,
            step_size=step_cfg.get("step_size", 30),
            gamma=step_cfg.get("gamma", 0.1),
            last_epoch=step_cfg.get("last_epoch", -1)
        )
        log.info(f"StepLR 스케쥴러 생성 - step_size: {step_cfg.get('step_size',30)}, gamma: {step_cfg.get('gamma',0.1)}")
        
    elif scheduler_name == "plateau":
        p_cfg = sched_cfg.get("plateau", {})
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=p_cfg.get("mode", "min"),
            factor=p_cfg.get("factor", 0.5),
            patience=p_cfg.get("patience", 5),
            threshold=p_cfg.get("threshold", 1e-4),
            threshold_mode=p_cfg.get("threshold_mode", "rel"),
            cooldown=p_cfg.get("cooldown", 0),
            min_lr=p_cfg.get("min_lr", 1e-8),
            eps=p_cfg.get("eps", 1e-8)
        )
        log.info(f"ReduceLROnPlateau 스케쥴러 생성 - mode: {p_cfg.get('mode','min')}, patience: {p_cfg.get('patience',5)}")
        
    elif scheduler_name == "cosine_warm":
        cw_cfg = sched_cfg.get("cosine_warm", {})
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cw_cfg.get("T_0", 10),
            T_mult=cw_cfg.get("T_mult", 1),
            eta_min=cw_cfg.get("eta_min", 1e-6),
            last_epoch=cw_cfg.get("last_epoch", -1)
        )
        log.info(f"CosineAnnealingWarmRestarts 스케쥴러 생성 - T_0: {cw_cfg.get('T_0',10)}")
        
    elif scheduler_name == "none":
        scheduler = None
        log.info("스케쥴러 사용 안함")
        
    else:
        raise ValueError(f"지원하지 않는 스케쥴러: {scheduler_name}")
    
    return scheduler


def setup_model_and_optimizer(cfg, device):
    """모델, 옵티마이저, 스케쥴러를 설정"""
    model = create_model(
        model_name=cfg.model.name,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=cfg.train.lr)
    
    # Label Smoothing Loss 또는 CrossEntropyLoss 선택
    ls_cfg = getattr(cfg.train, "label_smoothing", {"enabled": False})
    if ls_cfg.get("enabled", False):
        loss_fn = LabelSmoothingLoss(
            num_classes=cfg.model.num_classes,
            smoothing=ls_cfg.get("smoothing", 0.1)
        )
        log.info(f"LabelSmoothingLoss 사용 - smoothing: {ls_cfg.get('smoothing', 0.1)}")
    else:
        loss_fn = nn.CrossEntropyLoss()
        log.info("CrossEntropyLoss 사용")
    
    # 스케쥴러 생성
    scheduler = create_scheduler(optimizer, cfg)
    
    return model, optimizer, loss_fn, scheduler


def save_model(model, path):
    """모델 저장"""
    # 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    log.info(f"모델 저장 완료: {path}")


def save_model_with_metadata(model, path, metadata=None):
    """모델을 메타데이터와 함께 저장"""
    # 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 저장할 데이터 준비
    save_data = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata or {}
    }
    
    torch.save(save_data, path)
    log.info(f"모델 메타데이터와 함께 저장 완료: {path}")


def load_model(model, path):
    """모델 로드"""
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        log.info(f"모델 로드 완료: {path}")
        return model
    else:
        log.error(f"모델 파일이 존재하지 않습니다: {path}")
        return None


def load_model_with_metadata(model, path):
    """모델을 메타데이터와 함께 로드"""
    if os.path.exists(path):
        save_data = torch.load(path)
        model.load_state_dict(save_data['model_state_dict'])
        metadata = save_data.get('metadata', {})
        log.info(f"모델 메타데이터와 함께 로드 완료: {path}")
        return model, metadata
    else:
        log.error(f"모델 파일이 존재하지 않습니다: {path}")
        return None, None


def get_model_save_path(cfg, model_type):
    """모델 저장 경로 생성"""
    if cfg.model_save.enabled:
        model_dir = cfg.model_save.dir
        os.makedirs(model_dir, exist_ok=True)
        
        # 모델 파일명 생성
        model_name = cfg.model.name
        filename = f"{model_name}_{model_type}.pth"
        return os.path.join(model_dir, filename)
    else:
        return None


def get_seed_fold_model_path(cfg, seed: int, fold: int) -> str:
    """Return a model file path based on seed and fold."""
    model_dir = getattr(cfg, "model_save", {}).get("dir", "models")
    os.makedirs(model_dir, exist_ok=True)
    model_name = cfg.model.name
    filename = f"{model_name}_seed{seed}_fold{fold}.pth"
    return os.path.join(model_dir, filename)


def load_model_for_inference(cfg, path: str, device: torch.device):
    """Load a saved model for inference."""
    model = create_model(
        cfg.model.name,
        pretrained=False,
        num_classes=cfg.model.num_classes,
    ).to(device)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    return model


def get_model_info(model):
    """모델 정보 반환"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_name": model.__class__.__name__
    } 