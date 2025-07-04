# -*- coding: utf-8 -*-
"""
모델 관련 기능들을 담은 모듈
- 모델 생성
- 모델 로딩/저장
- 모델 관련 유틸리티
"""

import timm
import torch
import torch.nn as nn
from torch.optim import Adam


def create_model(model_name, pretrained=True, num_classes=17):
    """TIMM을 사용해 모델 생성"""
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model


def setup_model_and_optimizer(cfg, device):
    """모델과 옵티마이저를 설정"""
    model = create_model(
        model_name=cfg.model.name,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=cfg.training.lr)
    loss_fn = nn.CrossEntropyLoss()
    
    return model, optimizer, loss_fn


def save_model(model, path):
    """모델 저장"""
    torch.save(model.state_dict(), path)


def load_model(model, path):
    """모델 로드"""
    model.load_state_dict(torch.load(path))
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