# -*- coding: utf-8 -*-
"""
학습 및 검증 관련 기능들을 담은 모듈
- 학습 및 검증 함수들
- 메트릭 계산
- 학습 루프 관리
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import wandb

# Mixed Precision Training 지원
try:
    from torch.amp.autocast_mode import autocast
    from torch.amp.grad_scaler import GradScaler
    AMP_AVAILABLE = True
except ImportError:
    try:
        # 이전 버전 호환성
        from torch.cuda.amp import autocast, GradScaler
        AMP_AVAILABLE = True
    except ImportError:
        # PyTorch 버전이 낮은 경우
        AMP_AVAILABLE = False

import pandas as pd
from torch.utils.data import DataLoader
import log_util as log
from data import (
    get_kfold_loaders,
    ImageDataset,
    IndexedImageDataset,
    AugmentedDataset,
)
from models import (
    setup_model_and_optimizer,
    save_model_with_metadata,
    get_seed_fold_model_path,
)
from utils import EarlyStopping


def train_one_epoch(loader, model, optimizer, loss_fn, device, scaler=None):
    """한 에포크 학습"""
    model.train()
    train_loss = 0
    preds_list = []
    targets_list = []
    
    use_amp = scaler is not None

    pbar = tqdm(loader)
    for batch in pbar:
        # AugmentedDataset인 경우 (image, target, base_idx) 3개 값, 일반적인 경우 (image, target) 2개 값
        if len(batch) == 3:
            image, targets, _ = batch  # base_idx 무시
        else:
            image, targets = batch
            
        image = image.to(device)
        targets = targets.to(device)

        model.zero_grad(set_to_none=True)

        if use_amp:
            # Mixed Precision Training
            with autocast(device.type):
                preds = model(image)
                loss = loss_fn(preds, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 일반 Training
            preds = model(image)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()

        train_loss += loss.item()
        preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
        targets_list.extend(targets.detach().cpu().numpy())

        pbar.set_description(f"Loss: {loss.item():.4f}")

    train_loss /= len(loader)
    train_acc = accuracy_score(targets_list, preds_list)
    train_f1 = f1_score(targets_list, preds_list, average='macro')

    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_f1": train_f1,
    }


def _predict_probs(model, loader, device):
    """데이터 로더에 대한 소프트맥스 확률 반환"""
    probs = []
    with torch.no_grad():
        for batch in loader:
            # AugmentedDataset인 경우 (image, target, base_idx) 3개 값, 일반적인 경우 (image, target) 2개 값
            if len(batch) == 3:
                images, _, _ = batch  # target과 base_idx 무시
            else:
                images, _ = batch  # target 무시
                
            images = images.to(device)
            logits = model(images)
            probs.append(logits.softmax(dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def _clone_dataset_with_transform(dataset, transform):
    # AugmentedDataset인 경우 base_dataset을 사용
    if hasattr(dataset, "base_dataset"):
        base_dataset = dataset.base_dataset
        if hasattr(base_dataset, "df") and hasattr(base_dataset, "path"):
            return IndexedImageDataset(base_dataset.df.copy(), base_dataset.path, transform=transform)
    
    if hasattr(dataset, "df") and hasattr(dataset, "path"):
        if isinstance(dataset, ImageDataset):
            df = pd.DataFrame(dataset.df, columns=["ID", "target"])
            return ImageDataset(df, dataset.path, transform=transform)
        elif isinstance(dataset, IndexedImageDataset):
            return IndexedImageDataset(dataset.df.copy(), dataset.path, transform=transform)
    raise ValueError("Unsupported dataset type for TTA")


def validate_one_epoch(loader, model, loss_fn, device):
    """한 에포크 검증"""
    model.eval()
    val_loss = 0
    targets_list = []

    base_probs_list = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            # AugmentedDataset인 경우 (image, target, base_idx) 3개 값, 일반적인 경우 (image, target) 2개 값
            if len(batch) == 3:
                image, targets, _ = batch  # base_idx 무시
            else:
                image, targets = batch
                
            image = image.to(device)
            targets = targets.to(device)

            preds = model(image)
            loss = loss_fn(preds, targets)

            val_loss += loss.item()
            base_probs_list.append(preds.softmax(dim=1).cpu().numpy())
            targets_list.extend(targets.cpu().numpy())

    probs = np.concatenate(base_probs_list, axis=0)

    preds_list = probs.argmax(axis=1)

    val_loss /= len(loader)
    val_acc = accuracy_score(targets_list, preds_list)
    val_f1 = f1_score(targets_list, preds_list, average="macro")

    return {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f1": val_f1,
    }


def update_scheduler(scheduler, val_metrics=None, cfg=None):
    """스케쥴러 업데이트"""
    if scheduler is None:
        return None
    
    # ReduceLROnPlateau의 경우 검증 메트릭이 필요
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        if val_metrics is not None and cfg is not None:
            monitor_metric = cfg.train.scheduler.plateau.mode
            if monitor_metric == "min":
                # val_loss 기준
                scheduler.step(val_metrics.get("val_loss", 0))
            elif monitor_metric == "max":
                # val_acc 또는 val_f1 기준
                scheduler.step(val_metrics.get("val_f1", 0))
        else:
            log.warning("ReduceLROnPlateau 스케쥴러를 위한 검증 메트릭이 없습니다")
    else:
        # 다른 스케쥴러들은 단순히 step() 호출
        scheduler.step()
    
    # 현재 학습률 로깅 (ReduceLROnPlateau는 get_last_lr이 없음)
    current_lr = None
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        # ReduceLROnPlateau의 경우 optimizer에서 직접 가져오기
        current_lr = scheduler.optimizer.param_groups[0]['lr']
    else:
        # 다른 스케쥴러들은 get_last_lr() 사용
        current_lr = scheduler.get_last_lr()[0]
    
    if current_lr is not None:
        log.info(f"현재 학습률: {current_lr:.6f}")
        return current_lr
    return None


def train_single_model(cfg, train_loader, val_loader, device, save_to_disk: bool = False, seed: int | None = None, fold: int = 0):
    """단일 모델 학습 (Holdout 또는 No validation)"""
    model, optimizer, loss_fn, scheduler = setup_model_and_optimizer(cfg, device)
    
    # Mixed Precision Training 설정
    scaler = None
    mp_cfg = getattr(cfg.train, "mixed_precision", {"enabled": False})
    if mp_cfg.get("enabled", False) and AMP_AVAILABLE and device.type == "cuda":
        scaler = GradScaler()
        log.info("Mixed Precision Training 활성화")
    elif mp_cfg.get("enabled", False) and not AMP_AVAILABLE:
        log.warning("Mixed Precision Training이 요청되었지만 PyTorch AMP가 사용 불가능합니다")
    elif mp_cfg.get("enabled", False) and device.type != "cuda":
        log.warning("Mixed Precision Training은 CUDA 환경에서만 지원됩니다")
    
    # Early stopping 초기화
    early_stopping = None
    if val_loader is not None and cfg.validation.early_stopping.enabled:
        early_stopping = EarlyStopping(
            patience=cfg.validation.early_stopping.patience,
            min_delta=cfg.validation.early_stopping.min_delta,
            monitor=cfg.validation.early_stopping.monitor,
            mode=cfg.validation.early_stopping.mode
        )

    aug_cfg = getattr(cfg, "augment", {})
    
    log.info("학습 시작")
    
    # 최고 성능 추적을 위한 변수
    best_metric = None
    best_epoch = 0
    
    save_path = None
    if save_to_disk:
        save_path = get_seed_fold_model_path(cfg, seed or cfg.train.seed, fold)

    for epoch in range(cfg.train.epochs):
        # 훈련
        train_ret = train_one_epoch(train_loader, model, optimizer, loss_fn, device, scaler)
        ret = {**train_ret, 'epoch': epoch}
        
        # 검증 (holdout인 경우)
        if val_loader is not None:
            val_ret = validate_one_epoch(val_loader, model, loss_fn, device)
            ret.update(val_ret)
            
            log_message = f"Epoch {epoch+1}/{cfg.train.epochs} 완료 - "
            log_message += f"train_loss: {ret['train_loss']:.4f}, "
            log_message += f"train_acc: {ret['train_acc']:.4f}, "
            log_message += f"val_loss: {ret['val_loss']:.4f}, "
            log_message += f"val_acc: {ret['val_acc']:.4f}, "
            log_message += f"val_f1: {ret['val_f1']:.4f}"
            log.info(log_message)
            
            # 스케쥴러 업데이트 (검증 메트릭 포함)
            current_lr = update_scheduler(scheduler, ret, cfg)
            if current_lr is not None:
                ret['learning_rate'] = current_lr
            
            # wandb 로깅
            if cfg.wandb.enabled:
                wandb_log = {
                    "epoch": epoch + 1,
                    "train_loss": ret['train_loss'],
                    "train_acc": ret['train_acc'],
                    "train_f1": ret['train_f1'],
                    "val_loss": ret['val_loss'],
                    "val_acc": ret['val_acc'],
                    "val_f1": ret['val_f1'],
                }
                if current_lr is not None:
                    wandb_log["learning_rate"] = current_lr
                wandb.log(wandb_log)
            
            # 최고 성능 모델 추적 (F1 스코어 기준)
            if best_metric is None or ret['val_f1'] > best_metric:
                best_metric = ret['val_f1']
                best_epoch = epoch + 1

                # 최고 성능 모델 저장 (fold 정보 포함)
                if save_to_disk and save_path is not None:
                    metadata = {
                        "fold": fold,
                        "epoch": best_epoch,
                        "val_f1": best_metric,
                        "val_acc": ret['val_acc'],
                        "val_loss": ret['val_loss'],
                        "model_name": cfg.model.name,
                    }
                    save_model_with_metadata(model, save_path, metadata)
        else:
            # No validation
            log_message = f"Epoch {epoch+1}/{cfg.train.epochs} 완료 - "
            log_message += f"train_loss: {ret['train_loss']:.4f}, "
            log_message += f"train_acc: {ret['train_acc']:.4f}, "
            log_message += f"train_f1: {ret['train_f1']:.4f}"
            log.info(log_message)
            
            # 스케쥴러 업데이트 (검증 메트릭 없음)
            current_lr = update_scheduler(scheduler, None, cfg)
            if current_lr is not None:
                ret['learning_rate'] = current_lr
            
            # wandb 로깅
            if cfg.wandb.enabled:
                wandb_log = {
                    "epoch": epoch + 1,
                    "train_loss": ret['train_loss'],
                    "train_acc": ret['train_acc'],
                    "train_f1": ret['train_f1'],
                }
                if current_lr is not None:
                    wandb_log["learning_rate"] = current_lr
                wandb.log(wandb_log)
        
        # Early stopping 체크
        if early_stopping is not None:
            if early_stopping(ret):
                log.info(f"Early stopping at epoch {epoch + 1}")
                break
    
    if save_to_disk and save_path is not None:
        if val_loader is None and os.path.exists(save_path) is False:
            metadata = {
                "epoch": epoch + 1,
                "model_name": cfg.model.name,
                "final_train_loss": ret["train_loss"],
                "final_train_acc": ret["train_acc"],
                "final_train_f1": ret["train_f1"],
            }
            save_model_with_metadata(model, save_path, metadata)
        del model
        torch.cuda.empty_cache()
        return save_path

    return model


def train_kfold_models(cfg, kfold_data, device, save_to_disk: bool = False, seed: int | None = None):
    """K-Fold 교차 검증 학습"""
    folds, full_train_df, data_path, train_transform, val_transform, test_transform = kfold_data
    n_splits = len(folds)
    models = []
    aug_cfg = getattr(cfg, "augment", {})
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        log.info(f"========== Fold {fold_idx + 1}/{n_splits} ==========")
        disk_path = None
        if save_to_disk:
            disk_path = get_seed_fold_model_path(cfg, seed or cfg.train.seed, fold_idx + 1)
        
        # 현재 fold의 데이터 로더 준비
        train_loader, val_loader, train_df, val_df = get_kfold_loaders(
            fold_idx,
            folds,
            full_train_df,
            data_path,
            train_transform,
            val_transform,
            cfg,
        )
        
        # 모델 초기화
        model, optimizer, loss_fn, scheduler = setup_model_and_optimizer(cfg, device)
        
        # Mixed Precision Training 설정 (fold별로 설정)
        scaler = None
        mp_cfg = getattr(cfg.train, "mixed_precision", {"enabled": False})
        if mp_cfg.get("enabled", False) and AMP_AVAILABLE and device.type == "cuda":
            scaler = GradScaler()
            if fold_idx == 0:
                log.info("Mixed Precision Training 활성화")
        elif mp_cfg.get("enabled", False) and not AMP_AVAILABLE and fold_idx == 0:
            log.warning("Mixed Precision Training이 요청되었지만 PyTorch AMP가 사용 불가능합니다")
        elif mp_cfg.get("enabled", False) and device.type != "cuda" and fold_idx == 0:
            log.warning("Mixed Precision Training은 CUDA 환경에서만 지원됩니다")
        
        log.info(f"Fold {fold_idx + 1} 모델 로드 완료 - 훈련: {len(train_df)}개, 검증: {len(val_df)}개")
        
        # Early stopping 초기화
        early_stopping = None
        if cfg.validation.early_stopping.enabled:
            early_stopping = EarlyStopping(
                patience=cfg.validation.early_stopping.patience,
                min_delta=cfg.validation.early_stopping.min_delta,
                monitor=cfg.validation.early_stopping.monitor,
                mode=cfg.validation.early_stopping.mode
            )
        
        # 최고 성능 추적을 위한 변수
        best_metric = None
        best_epoch = 0
        
        # 학습 시작
        for epoch in range(cfg.train.epochs):
            # 훈련
            train_ret = train_one_epoch(train_loader, model, optimizer, loss_fn, device, scaler)
            # 검증
            val_ret = validate_one_epoch(
                val_loader,
                model,
                loss_fn,
                device,
            )
            
            # 결과 합치기
            ret = {**train_ret, **val_ret, 'epoch': epoch, 'fold': fold_idx + 1}
            
            log_message = f"Fold {fold_idx + 1} Epoch {epoch+1}/{cfg.train.epochs} 완료 - "
            log_message += f"train_loss: {ret['train_loss']:.4f}, "
            log_message += f"train_acc: {ret['train_acc']:.4f}, "
            log_message += f"val_loss: {ret['val_loss']:.4f}, "
            log_message += f"val_acc: {ret['val_acc']:.4f}, "
            log_message += f"val_f1: {ret['val_f1']:.4f}"
            log.info(log_message)
            
            # 스케쥴러 업데이트
            current_lr = update_scheduler(scheduler, ret, cfg)
            if current_lr is not None:
                ret['learning_rate'] = current_lr
            
            # wandb 로깅
            if cfg.wandb.enabled:
                wandb_log = {
                    "fold": fold_idx + 1,
                    "epoch": epoch + 1,
                    "train_loss": ret['train_loss'],
                    "train_acc": ret['train_acc'],
                    "train_f1": ret['train_f1'],
                    "val_loss": ret['val_loss'],
                    "val_acc": ret['val_acc'],
                    "val_f1": ret['val_f1'],
                }
                if current_lr is not None:
                    wandb_log["learning_rate"] = current_lr
                wandb.log(wandb_log)
            
            # 최고 성능 모델 추적 (F1 스코어 기준)
            if best_metric is None or ret['val_f1'] > best_metric:
                best_metric = ret['val_f1']
                best_epoch = epoch + 1

                # 최고 성능 모델 저장 (fold 정보 포함)
                if save_to_disk and disk_path is not None:
                    metadata = {
                        "fold": fold_idx + 1,
                        "epoch": best_epoch,
                        "val_f1": best_metric,
                        "val_acc": ret['val_acc'],
                        "val_loss": ret['val_loss'],
                        "model_name": cfg.model.name,
                    }
                    save_model_with_metadata(model, disk_path, metadata)
            
            # Early stopping 체크
            if early_stopping is not None:
                if early_stopping(ret):
                    log.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        if save_to_disk and disk_path is not None:
            models.append(disk_path)
            del model
            torch.cuda.empty_cache()
        else:
            models.append(model)
        log.info(f"Fold {fold_idx + 1} 완료")

    return models
