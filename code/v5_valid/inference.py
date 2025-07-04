# -*- coding: utf-8 -*-
"""
추론 관련 기능들을 담은 모듈
- 추론 실행
- 결과 후처리
- 앙상블 로직
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb

import log_util as log


def predict_single_model(model, test_loader, device):
    """단일 모델로 추론"""
    log.info("추론 시작")
    
    preds_list = []
    model.eval()
    
    with torch.no_grad():
        for image, _ in tqdm(test_loader, desc="Inference"):
            image = image.to(device)
            preds = model(image)
            preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
    
    return preds_list


def predict_kfold_ensemble(models, test_loader, device):
    """K-Fold 모델들로 앙상블 추론"""
    log.info("K-Fold 앙상블 추론 시작")
    
    all_predictions = []
    
    for fold_idx, model in enumerate(models):
        log.info(f"Fold {fold_idx + 1} 추론 시작")
        
        fold_predictions = []
        model.eval()
        
        with torch.no_grad():
            for image, _ in tqdm(test_loader, desc=f"Fold {fold_idx + 1} Inference"):
                image = image.to(device)
                preds = model(image)
                fold_predictions.extend(preds.softmax(dim=1).cpu().numpy())
        
        all_predictions.append(fold_predictions)
        log.info(f"Fold {fold_idx + 1} 추론 완료")
    
    # 앙상블 예측
    log.info("K-Fold 앙상블 예측 계산 중...")
    ensemble_predictions = np.mean(all_predictions, axis=0)
    final_preds = np.argmax(ensemble_predictions, axis=1)
    
    return final_preds


def save_predictions(predictions, test_dataset, cfg):
    """예측 결과를 CSV 파일로 저장"""
    # 예측 결과 DataFrame 생성
    pred_df = pd.DataFrame(test_dataset.df, columns=['ID', 'target'])
    pred_df['target'] = predictions
    
    # 검증: sample_submission과 ID 순서가 같은지 확인
    sample_submission_df = pd.read_csv(f"{cfg.data.data_path}/sample_submission.csv")
    assert (sample_submission_df['ID'] == pred_df['ID']).all(), "ID 순서가 sample_submission과 다릅니다"
    
    # 출력 디렉토리 생성
    output_path = cfg.output.dir
    os.makedirs(output_path, exist_ok=True)
    
    # 파일 저장
    output_file = f"{output_path}/{cfg.output.filename}"
    pred_df.to_csv(output_file, index=False)
    
    log.info(f"예측 결과 저장 완료: {output_file}")
    
    return pred_df


def upload_to_wandb(pred_df, cfg):
    """wandb에 예측 결과 업로드"""
    if cfg.wandb.enabled:
        # 아티팩트 생성 및 업로드
        artifact = wandb.Artifact(
            name="predictions",
            type="predictions",
            description=f"Model predictions using {cfg.model.name}",
            metadata={
                "model_name": cfg.model.name,
                "epochs": cfg.training.epochs,
                "batch_size": cfg.training.batch_size,
                "learning_rate": cfg.training.lr,
                "img_size": cfg.data.img_size,
                "num_predictions": len(pred_df),
                "output_filename": cfg.output.filename,
                "validation_strategy": cfg.validation.strategy,
            }
        )
        
        # 결과 파일을 아티팩트에 추가
        output_file = f"{cfg.output.dir}/{cfg.output.filename}"
        artifact.add_file(output_file)
        
        # 아티팩트 로깅
        wandb.log_artifact(artifact)
        log.info(f"wandb 아티팩트 업로드 완료 - 예측 결과 파일: {cfg.output.filename}")


def run_inference(models_or_model, test_loader, test_dataset, cfg, device, is_kfold=False):
    """추론 실행 및 결과 저장"""
    # 추론 실행
    if is_kfold:
        predictions = predict_kfold_ensemble(models_or_model, test_loader, device)
    else:
        predictions = predict_single_model(models_or_model, test_loader, device)
    
    # 결과 저장
    pred_df = save_predictions(predictions, test_dataset, cfg)
    
    # wandb 업로드
    upload_to_wandb(pred_df, cfg)
    
    log.info("추론 완료")
    
    return pred_df 