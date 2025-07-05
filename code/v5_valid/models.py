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


def create_model(model_name, pretrained=True, num_classes=17):
    """TIMM을 사용해 모델 생성"""
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model


def create_scheduler(optimizer, cfg):
    """설정에 따라 스케쥴러 생성"""
    if not cfg.training.scheduler.enabled:
        return None
    
    scheduler_name = cfg.training.scheduler.name.lower()
    
    if scheduler_name == "cosine":
        # T_max를 epochs로 자동 설정 (설정에서 지정하지 않은 경우)
        T_max = cfg.training.scheduler.cosine.T_max
        if T_max == 100:  # 기본값인 경우 epochs로 변경
            T_max = cfg.training.epochs
            
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=cfg.training.scheduler.cosine.eta_min,
            last_epoch=cfg.training.scheduler.cosine.last_epoch
        )
        log.info(f"CosineAnnealingLR 스케쥴러 생성 - T_max: {T_max}, eta_min: {cfg.training.scheduler.cosine.eta_min}")
        
    elif scheduler_name == "step":
        scheduler = StepLR(
            optimizer,
            step_size=cfg.training.scheduler.step.step_size,
            gamma=cfg.training.scheduler.step.gamma,
            last_epoch=cfg.training.scheduler.step.last_epoch
        )
        log.info(f"StepLR 스케쥴러 생성 - step_size: {cfg.training.scheduler.step.step_size}, gamma: {cfg.training.scheduler.step.gamma}")
        
    elif scheduler_name == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=cfg.training.scheduler.plateau.mode,
            factor=cfg.training.scheduler.plateau.factor,
            patience=cfg.training.scheduler.plateau.patience,
            threshold=cfg.training.scheduler.plateau.threshold,
            threshold_mode=cfg.training.scheduler.plateau.threshold_mode,
            cooldown=cfg.training.scheduler.plateau.cooldown,
            min_lr=cfg.training.scheduler.plateau.min_lr,
            eps=cfg.training.scheduler.plateau.eps
        )
        log.info(f"ReduceLROnPlateau 스케쥴러 생성 - mode: {cfg.training.scheduler.plateau.mode}, patience: {cfg.training.scheduler.plateau.patience}")
        
    elif scheduler_name == "cosine_warm":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.training.scheduler.cosine_warm.T_0,
            T_mult=cfg.training.scheduler.cosine_warm.T_mult,
            eta_min=cfg.training.scheduler.cosine_warm.eta_min,
            last_epoch=cfg.training.scheduler.cosine_warm.last_epoch
        )
        log.info(f"CosineAnnealingWarmRestarts 스케쥴러 생성 - T_0: {cfg.training.scheduler.cosine_warm.T_0}")
        
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
    
    optimizer = Adam(model.parameters(), lr=cfg.training.lr)
    loss_fn = nn.CrossEntropyLoss()
    
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


def get_model_info(model):
    """모델 정보 반환"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_name": model.__class__.__name__
    } 