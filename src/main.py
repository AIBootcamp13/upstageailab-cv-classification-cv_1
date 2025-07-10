# -*- coding: utf-8 -*-
"""
메인 파이프라인 - 문서 타입 분류 학습 및 추론

이 파일은 전체 워크플로우를 조율하는 역할만 담당합니다.
각 기능은 별도의 모듈로 분리되어 있습니다.
"""

import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

import log_util as log
from utils import set_seed, setup_wandb, get_device, log_hyperparameters, finish_wandb, save_model_as_artifact
from data import prepare_data_loaders
from training import train_single_model, train_kfold_models
from inference import (
    run_inference,
    predict_single_model,
    predict_kfold_ensemble,
    save_predictions,
    upload_to_wandb,
)
from augment import get_tta_transforms
from data import get_transforms
import numpy as np
from models import (
    get_seed_fold_model_path,
    load_model_for_inference,
)


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
    set_seed(cfg.train.seed)
    
    # wandb 초기화
    setup_wandb(cfg)
    
    # 디바이스 설정
    device = get_device(cfg)
    
    # 하이퍼파라미터 로깅
    log_hyperparameters(cfg)
    
    log.info("환경 설정 완료")

    # 랜덤 시드 앙상블 기능 확인
    seed_ensemble_cfg = getattr(cfg, "seed_ensemble", {"enabled": False})
    if seed_ensemble_cfg.get("enabled", False):
        ensemble_count = int(seed_ensemble_cfg.get("count", 1))
        model_paths: list[str] = []
        for idx in range(ensemble_count):
            current_seed = cfg.train.seed + idx
            log.info(f"=== Random seed ensemble {idx + 1}/{ensemble_count} - seed {current_seed} ===")

            set_seed(current_seed)
            train_loader, val_loader, test_loader, kfold_data = prepare_data_loaders(cfg, current_seed)

            valid_strategy = cfg.valid.strategy

            if valid_strategy == "kfold":
                items = train_kfold_models(cfg, kfold_data, device, save_to_disk=True, seed=current_seed)
                model_paths.extend(items)
            else:
                item = train_single_model(cfg, train_loader, val_loader, device, save_to_disk=True, seed=current_seed)
                model_paths.append(item)

        # Load saved models and ensemble predictions
        aug_cfg = getattr(cfg, "augment", {})
        tta_transforms = None
        if aug_cfg.get("test_tta_enabled", True):
            tta_transforms = get_tta_transforms(cfg.data.img_size)

        all_probs = []
        for item in model_paths:
            if isinstance(item, str):
                model = load_model_for_inference(cfg, item, device)
            else:
                model = item
            probs = predict_single_model(
                model,
                test_loader,
                device,
                tta_transforms=tta_transforms,
                return_probs=True,
            )
            all_probs.append(probs)
            del model
            torch.cuda.empty_cache()

        ensemble_probs = np.mean(all_probs, axis=0)
        final_preds = ensemble_probs.argmax(axis=1)
        pred_df = save_predictions(final_preds, test_loader.dataset, cfg)
        upload_to_wandb(pred_df, cfg)
        log.info("랜덤 시드 앙상블 추론 완료")
    else:
        # 2. 데이터 준비
        log.info("=== 데이터 준비 ===")
        train_loader, val_loader, test_loader, kfold_data = prepare_data_loaders(cfg, cfg.train.seed)

        if kfold_data is not None:
            log.info(f"K-Fold 데이터 준비 완료 - {len(kfold_data[0])}개 fold")
        elif val_loader is not None and train_loader is not None:
            log.info(f"Holdout 데이터 준비 완료 - 훈련: {len(train_loader.dataset)}개, 검증: {len(val_loader.dataset)}개")  # type: ignore
        elif train_loader is not None:
            log.info(f"No validation 데이터 준비 완료 - 훈련: {len(train_loader.dataset)}개")  # type: ignore

        if test_loader is not None:
            log.info(f"테스트 데이터: {len(test_loader.dataset)}개")  # type: ignore

        log.info("=== 모델 학습 ===")
        valid_strategy = cfg.valid.strategy

        if valid_strategy == "kfold":
            models = train_kfold_models(cfg, kfold_data, device)
            log.info("K-Fold 교차 검증 학습 완료")
            if cfg.model_save.wandb_artifact:
                for fold_idx in range(len(models)):
                    best_model_path = get_seed_fold_model_path(cfg, cfg.train.seed, fold_idx + 1)
                    metadata = {"fold": fold_idx + 1, "type": "best"}
                    save_model_as_artifact(best_model_path, cfg, f"best_fold{fold_idx + 1}", metadata)
            pred_df = run_inference(
                models, test_loader, test_loader.dataset, cfg, device, is_kfold=True
            )
        else:
            # Best model을 디스크에 저장하고 로드해서 사용
            best_model_path = train_single_model(cfg, train_loader, val_loader, device, save_to_disk=True)
            log.info("단일 모델 학습 완료")
            
            # Best model 로드
            model = load_model_for_inference(cfg, best_model_path, device)
            log.info("Best model 로드 완료")
            
            if cfg.model_save.wandb_artifact:
                metadata = {"fold": 0, "type": "best"}
                save_model_as_artifact(best_model_path, cfg, "best", metadata)
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
