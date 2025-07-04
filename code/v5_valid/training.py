# -*- coding: utf-8 -*-
"""
학습 및 검증 관련 기능들을 담은 모듈
- 학습 및 검증 함수들
- 메트릭 계산
- 학습 루프 관리
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import wandb

import log_util as log
from data import get_kfold_loaders, get_transforms
from models import setup_model_and_optimizer
from utils import EarlyStopping


def train_one_epoch(loader, model, optimizer, loss_fn, device):
    """한 에포크 학습"""
    model.train()
    train_loss = 0
    preds_list = []
    targets_list = []

    pbar = tqdm(loader)
    for image, targets in pbar:
        image = image.to(device)
        targets = targets.to(device)

        model.zero_grad(set_to_none=True)

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


def validate_one_epoch(loader, model, loss_fn, device):
    """한 에포크 검증"""
    model.eval()
    val_loss = 0
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for image, targets in tqdm(loader, desc="Validating"):
            image = image.to(device)
            targets = targets.to(device)

            preds = model(image)
            loss = loss_fn(preds, targets)

            val_loss += loss.item()
            preds_list.extend(preds.argmax(dim=1).cpu().numpy())
            targets_list.extend(targets.cpu().numpy())

    val_loss /= len(loader)
    val_acc = accuracy_score(targets_list, preds_list)
    val_f1 = f1_score(targets_list, preds_list, average='macro')

    return {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f1": val_f1,
    }


def train_single_model(cfg, train_loader, val_loader, device):
    """단일 모델 학습 (Holdout 또는 No validation)"""
    model, optimizer, loss_fn = setup_model_and_optimizer(cfg, device)
    
    # Early stopping 초기화
    early_stopping = None
    if val_loader is not None and cfg.validation.early_stopping.enabled:
        early_stopping = EarlyStopping(
            patience=cfg.validation.early_stopping.patience,
            min_delta=cfg.validation.early_stopping.min_delta,
            monitor=cfg.validation.early_stopping.monitor,
            mode=cfg.validation.early_stopping.mode
        )
    
    log.info("학습 시작")
    
    for epoch in range(cfg.training.epochs):
        # 훈련
        train_ret = train_one_epoch(train_loader, model, optimizer, loss_fn, device)
        ret = {**train_ret, 'epoch': epoch}
        
        # 검증 (holdout인 경우)
        if val_loader is not None:
            val_ret = validate_one_epoch(val_loader, model, loss_fn, device)
            ret.update(val_ret)
            
            log_message = f"Epoch {epoch+1}/{cfg.training.epochs} 완료 - "
            log_message += f"train_loss: {ret['train_loss']:.4f}, "
            log_message += f"train_acc: {ret['train_acc']:.4f}, "
            log_message += f"val_loss: {ret['val_loss']:.4f}, "
            log_message += f"val_acc: {ret['val_acc']:.4f}, "
            log_message += f"val_f1: {ret['val_f1']:.4f}"
            log.info(log_message)
            
            # wandb 로깅
            if cfg.wandb.enabled:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": ret['train_loss'],
                    "train_acc": ret['train_acc'],
                    "train_f1": ret['train_f1'],
                    "val_loss": ret['val_loss'],
                    "val_acc": ret['val_acc'],
                    "val_f1": ret['val_f1'],
                })
        else:
            # No validation
            log_message = f"Epoch {epoch+1}/{cfg.training.epochs} 완료 - "
            log_message += f"train_loss: {ret['train_loss']:.4f}, "
            log_message += f"train_acc: {ret['train_acc']:.4f}, "
            log_message += f"train_f1: {ret['train_f1']:.4f}"
            log.info(log_message)
            
            # wandb 로깅
            if cfg.wandb.enabled:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": ret['train_loss'],
                    "train_acc": ret['train_acc'],
                    "train_f1": ret['train_f1'],
                })
        
        # Early stopping 체크
        if early_stopping is not None:
            if early_stopping(ret):
                log.info(f"Early stopping at epoch {epoch + 1}")
                break
    
    return model


def train_kfold_models(cfg, kfold_data, device):
    """K-Fold 교차 검증 학습"""
    folds, full_train_df, data_path, train_transform, test_transform = kfold_data
    n_splits = len(folds)
    models = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        log.info(f"========== Fold {fold_idx + 1}/{n_splits} ==========")
        
        # 현재 fold의 데이터 로더 준비
        train_loader, val_loader, train_df, val_df = get_kfold_loaders(
            fold_idx, folds, full_train_df, data_path, train_transform, test_transform, cfg
        )
        
        # 모델 초기화
        model, optimizer, loss_fn = setup_model_and_optimizer(cfg, device)
        
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
        
        # 학습 시작
        for epoch in range(cfg.training.epochs):
            # 훈련
            train_ret = train_one_epoch(train_loader, model, optimizer, loss_fn, device)
            # 검증
            val_ret = validate_one_epoch(val_loader, model, loss_fn, device)
            
            # 결과 합치기
            ret = {**train_ret, **val_ret, 'epoch': epoch, 'fold': fold_idx + 1}
            
            log_message = f"Fold {fold_idx + 1} Epoch {epoch+1}/{cfg.training.epochs} 완료 - "
            log_message += f"train_loss: {ret['train_loss']:.4f}, "
            log_message += f"train_acc: {ret['train_acc']:.4f}, "
            log_message += f"val_loss: {ret['val_loss']:.4f}, "
            log_message += f"val_acc: {ret['val_acc']:.4f}, "
            log_message += f"val_f1: {ret['val_f1']:.4f}"
            log.info(log_message)
            
            # wandb 로깅
            if cfg.wandb.enabled:
                wandb.log({
                    "fold": fold_idx + 1,
                    "epoch": epoch + 1,
                    "train_loss": ret['train_loss'],
                    "train_acc": ret['train_acc'],
                    "train_f1": ret['train_f1'],
                    "val_loss": ret['val_loss'],
                    "val_acc": ret['val_acc'],
                    "val_f1": ret['val_f1'],
                })
            
            # Early stopping 체크
            if early_stopping is not None:
                if early_stopping(ret):
                    log.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        models.append(model)
        log.info(f"Fold {fold_idx + 1} 완료")
    
    return models 