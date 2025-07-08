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
import pandas as pd
from torch.utils.data import DataLoader
from data import ImageDataset, IndexedImageDataset, get_transforms, CachedTransformDataset
from augment import get_tta_transforms


def _predict_probs(model, loader, device):
    probs = []
    with torch.no_grad():
        for batch in loader:
            # AugmentedDataset인 경우 (image, target, base_idx) 3개 값, 일반적인 경우 (image, target) 2개 값
            if len(batch) == 3:
                image, _, _ = batch  # target과 base_idx 무시
            else:
                image, _ = batch  # target 무시
                
            image = image.to(device)
            logits = model(image)
            probs.append(logits.softmax(dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def _clone_dataset_with_transform(dataset, transform, cache_info=None, tta_idx=1):
    cache_root = None
    seed = None
    img_size = None
    if cache_info is not None:
        cache_root, seed, img_size = cache_info

    if hasattr(dataset, "df") and hasattr(dataset, "path"):
        if isinstance(dataset, ImageDataset):
            df = dataset.df if isinstance(dataset.df, pd.DataFrame) else pd.DataFrame(dataset.df, columns=["ID", "target"])
            base = ImageDataset(df, dataset.path, transform=None, return_filename=True)
        else:
            base = IndexedImageDataset(dataset.df.copy(), dataset.path, transform=None, return_filename=True)

        if cache_root and seed is not None and img_size is not None:
            return CachedTransformDataset(base, transform, cache_root, seed, img_size, tta_idx)
        return type(base)(base.df, base.path, transform=transform)

    class WrappedDataset(torch.utils.data.Dataset):
        def __init__(self, base, tf):
            self.base = base
            self.tf = tf

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            img, target = self.base[idx]
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).cpu().numpy()
            else:
                img = np.array(img)
            img = self.tf(image=img)["image"]
            return img, target

    if cache_root and seed is not None:
        # Without file names, caching is not possible, fallback
        return WrappedDataset(dataset, transform)

    return WrappedDataset(dataset, transform)


def predict_single_model(model, test_loader, device, tta_transforms=None, return_probs=False, cache_info=None):
    """단일 모델로 추론

    Args:
        model: 학습된 모델
        test_loader: 테스트 데이터 로더
        device: 실행 디바이스
        tta_transforms: 리스트 형태의 TTA transforms
        return_probs: True면 소프트맥스 확률을 반환
    """
    log.info("추론 시작")

    model.eval()

    if tta_transforms:
        tta_probs = []
        for idx, t in enumerate(tta_transforms, start=1):
            t_dataset = _clone_dataset_with_transform(test_loader.dataset, t, cache_info, idx)
            t_loader = DataLoader(
                t_dataset,
                batch_size=test_loader.batch_size,
                shuffle=False,
                num_workers=test_loader.num_workers,
            )
            tta_probs.append(_predict_probs(model, t_loader, device))

        probs = np.mean(tta_probs, axis=0)
    else:
        probs = _predict_probs(model, test_loader, device)

    if return_probs:
        return probs

    preds_list = probs.argmax(axis=1).tolist()
    return preds_list


def predict_kfold_ensemble(models, test_loader, device, tta_transforms=None, return_probs=False, cache_info=None):
    """K-Fold 모델들로 앙상블 추론

    Args:
        models: 학습된 모델 리스트
        test_loader: 테스트 데이터 로더
        device: 실행 디바이스
        tta_transforms: 리스트 형태의 TTA transforms
        return_probs: True면 fold 앙상블 확률을 반환
    """
    log.info("K-Fold 앙상블 추론 시작")
    
    all_predictions = []
    
    for fold_idx, model in enumerate(models):
        log.info(f"Fold {fold_idx + 1} 추론 시작")
        
        model.eval()

        if tta_transforms:
            tta_probs = []
            for idx_t, t in enumerate(tta_transforms, start=1):
                t_dataset = _clone_dataset_with_transform(test_loader.dataset, t, cache_info, idx_t)
                t_loader = DataLoader(
                    t_dataset,
                    batch_size=test_loader.batch_size,
                    shuffle=False,
                    num_workers=test_loader.num_workers,
                )
                tta_probs.append(_predict_probs(model, t_loader, device))

            probs = np.mean(tta_probs, axis=0)
        else:
            probs = _predict_probs(model, test_loader, device)

        fold_predictions = probs
        
        all_predictions.append(fold_predictions)
        log.info(f"Fold {fold_idx + 1} 추론 완료")
    
    # 앙상블 예측
    log.info("K-Fold 앙상블 예측 계산 중...")
    ensemble_predictions = np.mean(all_predictions, axis=0)
    if return_probs:
        return ensemble_predictions

    final_preds = np.argmax(ensemble_predictions, axis=1)

    return final_preds


def save_predictions(predictions, test_dataset, cfg):
    """예측 결과를 CSV 파일로 저장"""
    # 예측 결과 DataFrame 생성
    pred_df = pd.DataFrame(test_dataset.df, columns=['ID', 'target'])
    pred_df['target'] = predictions
    
    # 검증: sample_submission과 ID 순서가 같은지 확인
    test_csv_path = cfg.data.test_csv_path
    sample_submission_df = pd.read_csv(test_csv_path)
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
                "epochs": cfg.train.epochs,
                "batch_size": cfg.train.batch_size,
                "learning_rate": cfg.train.lr,
                "seed": cfg.train.seed,
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
    aug_cfg = getattr(cfg, "augment", {})
    tta_transforms = None
    img_size = getattr(getattr(cfg, "data", {}), "img_size", None)
    if aug_cfg.get("test_tta_enabled", True) and img_size is not None:
        tta_transforms = get_tta_transforms(img_size)
    train_img_path = getattr(cfg.data, "train_images_path", os.path.dirname(cfg.data.test_csv_path))
    cache_root = os.path.join(os.path.dirname(train_img_path), "train_cache")
    seed = getattr(getattr(cfg, "train", {}), "seed", 42)
    cache_info = (cache_root, seed, img_size)
    if is_kfold:
        predictions = predict_kfold_ensemble(
            models_or_model,
            test_loader,
            device,
            tta_transforms=tta_transforms,
            cache_info=cache_info,
        )
    else:
        predictions = predict_single_model(
            models_or_model,
            test_loader,
            device,
            tta_transforms=tta_transforms,
            cache_info=cache_info,
        )
    
    # 결과 저장
    pred_df = save_predictions(predictions, test_dataset, cfg)
    
    # wandb 업로드
    upload_to_wandb(pred_df, cfg)
    
    log.info("추론 완료")
    
    return pred_df 