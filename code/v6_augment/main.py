# -*- coding: utf-8 -*-
"""
메인 파이프라인 - 문서 타입 분류 학습 및 추론

이 파일은 전체 워크플로우를 조율하는 역할만 담당합니다.
각 기능은 별도의 모듈로 분리되어 있습니다.
"""

import os
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

import log_util as log
from utils import set_seed, setup_wandb, get_device, log_hyperparameters, finish_wandb, save_model_as_artifact
from data import prepare_data_loaders
from training import train_single_model, train_kfold_models
from inference import run_inference
from models import get_model_save_path


# 현재 스크립트 위치를 작업 디렉토리로 설정
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# .env 파일 로드
load_dotenv()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """메인 파이프라인 함수"""
    
    # 1. 설정 및 환경 초기화
    log.info("=== 설정 및 환경 초기화 ===")
    log.info(f"설정 로드 완료:")
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")
    
    # 시드 고정
    set_seed(cfg.training.seed)
    
    # wandb 초기화
    setup_wandb(cfg)
    
    # 디바이스 설정
    device = get_device(cfg)
    
    # 하이퍼파라미터 로깅
    log_hyperparameters(cfg)
    
    log.info("환경 설정 완료")
    
    # 2. 데이터 준비
    log.info("=== 데이터 준비 ===")
    train_loader, val_loader, test_loader, kfold_data = prepare_data_loaders(cfg, cfg.training.seed)
    
    # 데이터 로딩 완료 로그
    if kfold_data is not None:
        log.info(f"K-Fold 데이터 준비 완료 - {len(kfold_data[0])}개 fold")
    elif val_loader is not None and train_loader is not None:
        log.info(f"Holdout 데이터 준비 완료 - 훈련: {len(train_loader.dataset)}개, 검증: {len(val_loader.dataset)}개")  # type: ignore
    elif train_loader is not None:
        log.info(f"No validation 데이터 준비 완료 - 훈련: {len(train_loader.dataset)}개")  # type: ignore
    
    if test_loader is not None:
        log.info(f"테스트 데이터: {len(test_loader.dataset)}개")  # type: ignore
    
    # 3. 모델 학습
    log.info("=== 모델 학습 ===")
    validation_strategy = cfg.validation.strategy
    
    if validation_strategy == "kfold":
        # K-Fold 교차 검증
        models = train_kfold_models(cfg, kfold_data, device)
        log.info("K-Fold 교차 검증 학습 완료")
        
        # wandb 아티팩트 등록 (K-Fold)
        model_save_cfg = getattr(cfg, "model_save", {})
        if model_save_cfg.get("enabled", False) and model_save_cfg.get("wandb_artifact", False):
            for fold_idx in range(len(models)):
                # 각 fold의 best 모델 등록
                if model_save_cfg.get("save_best", False):
                    best_model_path = get_model_save_path(cfg, f"best_fold{fold_idx + 1}")
                    metadata = {"fold": fold_idx + 1, "type": "best"}
                    save_model_as_artifact(best_model_path, cfg, f"best_fold{fold_idx + 1}", metadata)
                
                # 각 fold의 last 모델 등록
                if model_save_cfg.get("save_last", False):
                    last_model_path = get_model_save_path(cfg, f"last_fold{fold_idx + 1}")
                    metadata = {"fold": fold_idx + 1, "type": "last"}
                    save_model_as_artifact(last_model_path, cfg, f"last_fold{fold_idx + 1}", metadata)
    else:
        # Holdout 또는 No validation
        model = train_single_model(cfg, train_loader, val_loader, device)
        log.info("단일 모델 학습 완료")
        
        # wandb 아티팩트 등록 (단일 모델)
        model_save_cfg = getattr(cfg, "model_save", {})
        if model_save_cfg.get("enabled", False) and model_save_cfg.get("wandb_artifact", False):
            # best 모델 등록
            if model_save_cfg.get("save_best", False):
                best_model_path = get_model_save_path(cfg, "best")
                metadata = {"type": "best"}
                save_model_as_artifact(best_model_path, cfg, "best", metadata)
            
            # last 모델 등록
            if model_save_cfg.get("save_last", False):
                last_model_path = get_model_save_path(cfg, "last")
                metadata = {"type": "last"}
                save_model_as_artifact(last_model_path, cfg, "last", metadata)
    
    # 4. 추론 및 결과 저장
    log.info("=== 추론 및 결과 저장 ===")
    if validation_strategy == "kfold":
        pred_df = run_inference(
            models, test_loader, test_loader.dataset, cfg, device, is_kfold=True
        )
    else:
        pred_df = run_inference(
            model, test_loader, test_loader.dataset, cfg, device, is_kfold=False
        )
    
    # 5. 완료 및 정리
    log.info("=== 완료 및 정리 ===")
    log.info("전체 프로세스 완료")
    
    # 결과 미리보기
    print("\n=== 예측 결과 미리보기 ===")
    print(pred_df.head())
    print(f"\n총 예측 개수: {len(pred_df)}")
    
    # wandb 세션 종료
    finish_wandb(cfg)


if __name__ == "__main__":
    main()
