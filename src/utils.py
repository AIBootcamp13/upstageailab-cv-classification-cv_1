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
    
    # GPU 모드일 때만 재현성 확보를 위한 CUDA 설정
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        log.info(f"시드 고정 완료: {seed} (GPU 재현성 모드 활성화)")
    else:
        log.info(f"시드 고정 완료: {seed}")


def setup_wandb(cfg):
    """wandb 초기화"""
    if cfg.wandb.enabled:
        wandb_config = {
            # Data configuration
            "data_train_images_path": cfg.data.train_images_path,
            "data_test_images_path": cfg.data.test_images_path,
            "data_train_csv_path": cfg.data.train_csv_path,
            "data_test_csv_path": cfg.data.test_csv_path,
            "data_img_size": cfg.data.img_size,
            "data_num_workers": cfg.data.num_workers,
            
            # Model configuration
            "model_name": cfg.model.name,
            "model_num_classes": cfg.model.num_classes,
            "model_pretrained": cfg.model.pretrained,
            
            # Training configuration
            "train_lr": cfg.train.lr,
            "train_epochs": cfg.train.epochs,
            "train_batch_size": cfg.train.batch_size,
            "train_seed": cfg.train.seed,
            
            # Label Smoothing
            "label_smoothing_enabled": cfg.train.label_smoothing.enabled,
            "label_smoothing_smoothing": cfg.train.label_smoothing.smoothing,
            
            # Mixed Precision
            "mixed_precision_enabled": cfg.train.mixed_precision.enabled,
            
            # Data Augmentation
            "augment_train_aug_count": cfg.augment.train_aug_count,
            "augment_valid_aug_count": cfg.augment.valid_aug_count,
            "augment_test_tta_enabled": cfg.augment.test_tta_enabled,
            
            # Scheduler
            "scheduler_enabled": cfg.train.scheduler.enabled,
            "scheduler_name": cfg.train.scheduler.name,
            # "scheduler_cosine_T_max": cfg.training.scheduler.cosine.T_max,
            # "scheduler_cosine_eta_min": cfg.training.scheduler.cosine.eta_min,
            # "scheduler_cosine_last_epoch": cfg.training.scheduler.cosine.last_epoch,
            # "scheduler_step_step_size": cfg.training.scheduler.step.step_size,
            # "scheduler_step_gamma": cfg.training.scheduler.step.gamma,
            # "scheduler_step_last_epoch": cfg.training.scheduler.step.last_epoch,
            # "scheduler_plateau_mode": cfg.training.scheduler.plateau.mode,
            # "scheduler_plateau_factor": cfg.training.scheduler.plateau.factor,
            # "scheduler_plateau_patience": cfg.training.scheduler.plateau.patience,
            # "scheduler_plateau_threshold": cfg.training.scheduler.plateau.threshold,
            # "scheduler_plateau_threshold_mode": cfg.training.scheduler.plateau.threshold_mode,
            # "scheduler_plateau_cooldown": cfg.training.scheduler.plateau.cooldown,
            # "scheduler_plateau_min_lr": cfg.training.scheduler.plateau.min_lr,
            # "scheduler_plateau_eps": cfg.training.scheduler.plateau.eps,
            # "scheduler_cosine_warm_T_0": cfg.training.scheduler.cosine_warm.T_0,
            # "scheduler_cosine_warm_T_mult": cfg.training.scheduler.cosine_warm.T_mult,
            # "scheduler_cosine_warm_eta_min": cfg.training.scheduler.cosine_warm.eta_min,
            # "scheduler_cosine_warm_last_epoch": cfg.training.scheduler.cosine_warm.last_epoch,
            
            # Validation configuration
            "val_strategy": cfg.validation.strategy,
            "val_holdout_train_ratio": cfg.validation.holdout.train_ratio,
            "val_holdout_stratify": cfg.validation.holdout.stratify,
            "val_kfold_n_splits": cfg.validation.kfold.n_splits,
            "val_kfold_stratify": cfg.validation.kfold.stratify,
            "val_early_stopping_enabled": cfg.validation.early_stopping.enabled,
            "val_early_stopping_patience": cfg.validation.early_stopping.patience,
            "val_early_stopping_min_delta": cfg.validation.early_stopping.min_delta,
            "val_early_stopping_monitor": cfg.validation.early_stopping.monitor,
            "val_early_stopping_mode": cfg.validation.early_stopping.mode,
            
            # Device configuration
            "device": cfg.device,
            
            # Output configuration
            "output_dir": cfg.output.dir,
            "output_filename": cfg.output.filename,
            
            # Model saving configuration
            "model_save_enabled": cfg.model_save.enabled,
            # "model_save_dir": cfg.model_save.dir,
            # "model_save_save_best": cfg.model_save.save_best,
            # "model_save_wandb_artifact": cfg.model_save.wandb_artifact,
            
            # Random seed ensemble configuration
            "random_seed_ensemble_enabled": cfg.random_seed_ensemble.enabled,
            "random_seed_ensemble_count": cfg.random_seed_ensemble.count,
            
            # W&B configuration
            # "wandb_project": cfg.wandb.project,
            # "wandb_entity": cfg.wandb.entity,
            # "wandb_run_name": cfg.wandb.run_name,
            # "wandb_tags": cfg.wandb.tags,
            # "wandb_notes": cfg.wandb.notes,
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
    """디바이스 설정 (CUDA, MPS, CPU 지원)"""
    if cfg.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        log.info(f"사용 장치: {device} (Apple Silicon GPU)")
    elif cfg.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        log.info(f"사용 장치: {device}")
    elif cfg.device == 'auto':
        # 자동 선택: MPS > CUDA > CPU 순서
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            log.info(f"사용 장치: {device} (자동 선택 - Apple Silicon GPU)")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            log.info(f"사용 장치: {device} (자동 선택)")
        else:
            device = torch.device('cpu')
            log.info(f"사용 장치: {device} (자동 선택)")
    else:
        device = torch.device('cpu')
        log.info(f"사용 장치: {device}")
    
    return device


def log_hyperparameters(cfg):
    """하이퍼파라미터 로깅"""
    log.info(f"하이퍼파라미터 설정 - 모델: {cfg.model.name}, 이미지 크기: {cfg.data.img_size}, "
             f"학습률: {cfg.train.lr}, 에포크: {cfg.train.epochs}, 배치 크기: {cfg.train.batch_size}")


def save_model_as_artifact(model_path, cfg, model_type="best", metadata=None):
    """모델을 wandb 아티팩트로 저장"""
    if not cfg.wandb.enabled:
        return
    
    try:
        # 아티팩트 생성
        artifact_name = f"{cfg.model.name}_model_{model_type}"
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description=f"{cfg.model.name} {model_type} model",
            metadata=metadata
        )
        
        # 모델 파일 추가
        artifact.add_file(model_path)
        
        # 아티팩트 로깅
        wandb.log_artifact(artifact)
        log.info(f"wandb 아티팩트로 모델 저장 완료: {artifact_name}")
        
    except Exception as e:
        log.error(f"wandb 아티팩트 저장 실패: {e}")


def log_model_metrics(metrics, step=None):
    """모델 메트릭을 wandb에 로깅"""
    if wandb.run is not None:
        wandb.log(metrics, step=step)


def finish_wandb(cfg):
    """wandb 세션 종료"""
    if cfg.wandb.enabled:
        wandb.finish()
        log.info("wandb 세션 종료") 