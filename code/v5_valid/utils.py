# -*- coding: utf-8 -*-
"""
유틸리티 함수들을 담은 모듈
- EarlyStopping 클래스
- 시드 고정
- 기타 헬퍼 함수들
"""

import os
import random
import torch
import numpy as np
import wandb

import log_util as log


class EarlyStopping:
    """Early Stopping 클래스"""
    def __init__(self, patience=10, min_delta=0.001, monitor='val_loss', mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        
    def __call__(self, val_metrics):
        score = val_metrics[self.monitor]
        
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        elif self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            self.should_stop = True
            
        return self.should_stop


def set_seed(seed):
    """시드 고정"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    log.info(f"시드 고정 완료: {seed}")


def setup_wandb(cfg):
    """wandb 초기화"""
    if cfg.wandb.enabled:
        wandb_config = {
            "learning_rate": cfg.training.lr,
            "epochs": cfg.training.epochs,
            "batch_size": cfg.training.batch_size,
            "model_name": cfg.model.name,
            "img_size": cfg.data.img_size,
            "seed": cfg.training.seed,
            "num_classes": cfg.model.num_classes,
            "pretrained": cfg.model.pretrained,
        }
        
        # .env 파일에서 wandb 설정 불러오기
        wandb_entity = os.getenv("WANDB_ENTITY") or cfg.wandb.entity
        wandb_project = os.getenv("WANDB_PROJECT") or cfg.wandb.project
        
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            config=wandb_config,
            name=cfg.wandb.run_name,
            tags=cfg.wandb.tags,
            notes=cfg.wandb.notes,
        )
        log.info(f"wandb 초기화 완료 - 프로젝트: {wandb_project}")
    else:
        log.info("wandb 비활성화됨")


def get_device(cfg):
    """디바이스 설정"""
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.device == 'cuda' else 'cpu')
    log.info(f"사용 장치: {device}")
    return device


def log_hyperparameters(cfg):
    """하이퍼파라미터 로깅"""
    log.info(f"하이퍼파라미터 설정 - 모델: {cfg.model.name}, 이미지 크기: {cfg.data.img_size}, "
             f"학습률: {cfg.training.lr}, 에포크: {cfg.training.epochs}, 배치 크기: {cfg.training.batch_size}")


def finish_wandb(cfg):
    """wandb 세션 종료"""
    if cfg.wandb.enabled:
        wandb.finish()
        log.info("wandb 세션 종료") 