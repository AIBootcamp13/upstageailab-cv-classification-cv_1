import os
import numpy as np
import torch
import timm
import glob
import albumentations as A
import cv2
from tqdm import tqdm
from Load_Data import ImageDataset
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from PIL import Image


class Model_Ensemble:
    def __init__(self, model_name, fold_paths_dir, fold_weights, num_classes, drop_out, device, k_fold=True):
        self.device = device

        if k_fold:
            fold_paths = sorted(glob.glob(os.path.join(fold_paths_dir, "**/model_Fold*.pth"), recursive=True))
            if not fold_paths:
                raise ValueError(f"[Error] No model_Fold*.pth files found recursively in: {fold_paths_dir}")
        else:
            # Holdout 모델은 상위 폴더에 바로 저장됨
            fold_paths = sorted(glob.glob(os.path.join(fold_paths_dir, "model_Holdout*.pth")))
            if not fold_paths:
                raise ValueError(f"[Error] No model_Holdout.pth file found in: {fold_paths_dir}")

        self.models = []
        self.weights = np.array(fold_weights) / np.sum(fold_weights)
        self._load_models(model_name, fold_paths, num_classes, drop_out)

    def _load_models(self, model_name, fold_paths, num_classes, drop_out):
        for fold_path in fold_paths:
            model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=num_classes,
                drop_path_rate=drop_out
            )
            model.load_state_dict(torch.load(fold_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            self.models.append(model)


def tta(img_size):
    base_transform = [
        A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255)),
    ]
    post_transform = [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    tta_transforms = []
    tta_transforms.append(A.Compose(base_transform + [A.InvertImg(p=1.0)] + post_transform))
    tta_transforms.append(A.Compose(base_transform + [A.MedianBlur(blur_limit=5, p=1.0)] + post_transform))
    tta_transforms.append(A.Compose(base_transform + [A.HorizontalFlip(p=1.0)] + post_transform))
    tta_transforms.append(A.Compose(base_transform + [A.Rotate(limit=7, p=1.0, border_mode=cv2.BORDER_CONSTANT, fill=(255,255,255))] + post_transform))
    return tta_transforms

def get_img_resize(img_size):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255)),
    ])

def basic_transform():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def run_inference(ensembler, submission_df, test_path, img_size, save_path, batch_size, num_workers, use_tta=False):
    tta_transforms = tta(img_size) if use_tta else None

    initial_inference_transform = get_img_resize(img_size)
    test_dataset = ImageDataset(submission_df, path=test_path, transform=initial_inference_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # 기본 예측 및 TTA 이후에 사용할 최종 정규화/텐서 변환 함수
    final_norm_to_tensor_transform = basic_transform()
    submission_preds = []
    ensembler.models = [m.eval() for m in ensembler.models]

    if use_tta:
        base_name, ext = os.path.splitext(save_path)
        save_path = f"{base_name}-TTA{ext}"

    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Inference"):
            images_tensor_batch, _, img_ids = batch_data 
            
            # TTA 없이 예측
            all_probs_list = []
            processed_images_base_tensors = []
            for img_tensor in images_tensor_batch:
                img_np = img_tensor.cpu().numpy() 
                processed_images_base_tensors.append(final_norm_to_tensor_transform(image=img_np.astype(np.float32))['image'])
            images_base_tensor = torch.stack(processed_images_base_tensors).to(ensembler.device)
            
            weighted_probs_original = None
            for weight, model in zip(ensembler.weights, ensembler.models):
                outputs = model(images_base_tensor)
                probs = torch.softmax(outputs, dim=1)
                if weighted_probs_original is None:
                    weighted_probs_original = probs * weight
                else:
                    weighted_probs_original += probs * weight
            all_probs_list.append(weighted_probs_original)

            # TTA 예측
            if tta_transforms is not None:
                for tta_transform in tta_transforms:
                    processed_images_tta_tensors = []
                    for img_tensor in images_tensor_batch:
                        img_np = img_tensor.cpu().numpy() 
                        processed_images_tta_tensors.append(tta_transform(image=img_np.astype(np.float32))['image'])
                    imgs_tta_tensor = torch.stack(processed_images_tta_tensors).to(ensembler.device)
                    
                    weighted_probs_tta = None
                    for weight, model in zip(ensembler.weights, ensembler.models):
                        outputs = model(imgs_tta_tensor)
                        probs = torch.softmax(outputs, dim=1)
                        if weighted_probs_tta is None:
                            weighted_probs_tta = probs * weight
                        else:
                            weighted_probs_tta += probs * weight
                    all_probs_list.append(weighted_probs_tta)
            
            # 모든 예측 결과의 평균 계산
            avg_probs = torch.mean(torch.stack(all_probs_list), dim=0)
            preds = torch.argmax(avg_probs, dim=1)
            submission_preds.extend(preds.cpu().numpy())

    submission_df["target"] = submission_preds
    submission_df.to_csv(save_path, index=False)
    print(f"[✓] Saved submission to: {save_path}")













# class Model_Ensemble:
#     # [수정] 여러 모델 그룹의 설정을 리스트로 받도록 변경
#     def __init__(self, model_groups, num_classes, drop_out, device):
#         self.device = device
#         self.models_info = [] # (모델 객체, 이미지 크기, 최종 가중치)를 저장할 리스트
        
#         # 1. 전체 그룹의 가중치 합으로 정규화
#         total_group_weight = sum(group['group_weight'] for group in model_groups)
#         if total_group_weight == 0: total_group_weight = 1 # 가중치 합이 0일 경우 대비

#         # 2. 각 그룹별로 모델 로드 및 개별 가중치 계산
#         for group in model_groups:
#             group_path_dir = group['path_dir']
#             group_model_name = group['model_name']
#             group_img_size = group['img_size']
#             # 정규화된 그룹 가중치
#             normalized_group_weight = group['group_weight'] / total_group_weight

#             # 그룹 내의 모든 모델 경로 탐색 (하위 폴더까지 검색)
#             model_paths = sorted(glob.glob(os.path.join(group_path_dir, "**/model_*.pth"), recursive=True))
#             if not model_paths:
#                 print(f"Warning: No models found in {group_path_dir}. Skipping this group.")
#                 continue

#             # 그룹 내 모델 개수로 그룹 가중치를 나눠서 개별 모델의 최종 가중치 계산
#             individual_weight = normalized_group_weight / len(model_paths)
            
#             print(f"Loading {len(model_paths)} models from '{group['version']}' group with weight: {individual_weight:.4f} each...")
#             for model_path in model_paths:
#                 model = timm.create_model(
#                     group_model_name,
#                     pretrained=False,
#                     num_classes=num_classes,
#                     drop_path_rate=drop_out
#                 )
#                 model.load_state_dict(torch.load(model_path, map_location=self.device))
#                 model.to(self.device)
#                 model.eval()
                
#                 # 각 모델의 정보(모델 객체, 이미지 크기, 최종 가중치)를 저장
#                 self.models_info.append({
#                     "model": model,
#                     "img_size": group_img_size,
#                     "weight": individual_weight
#                 })




# def run_inference(ensembler, submission_df, test_path, save_path, batch_size, num_workers, use_tta=False):
#     # [수정] DataLoader는 리사이즈 없이 원본 이미지만 불러오도록 transform=None 설정
#     test_dataset = ImageDataset(submission_df, path=test_path, transform=None)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#     submission_preds = []

#     with torch.no_grad():
#         for batch in tqdm(test_loader, desc="Inference"):
#             images_np, _, _ = batch # 원본 크기의 이미지 배치 (NumPy 배열)
            
#             final_probs_for_batch = None

#             # [핵심 수정] 각 모델 정보에 따라 동적으로 변환 및 추론
#             for model_info in ensembler.models_info:
#                 model = model_info['model']
#                 img_size = model_info['img_size']
#                 weight = model_info['weight']

#                 # 이 모델에 맞는 변환기 생성
#                 base_transform = A.Compose([
#                     A.LongestMaxSize(max_size=img_size),
#                     A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255)),
#                     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#                     ToTensorV2(),
#                 ])
                
#                 tta_list = [base_transform] # 1. 원본 예측을 위해 기본 변환 추가
#                 if use_tta:
#                     tta_list.extend(tta(img_size)) # 2. TTA 변환들 추가

#                 model_all_views_probs = []
                
#                 # 원본 + TTA 예측
#                 for transform in tta_list:
#                     # 배치 내의 각 이미지를 변환 후 다시 텐서로 묶음
#                     transformed_batch = torch.stack([transform(image=img.numpy())['image'] for img in images_np]).to(ensembler.device)
#                     outputs = model(transformed_batch)
#                     probs = torch.softmax(outputs, dim=1)
#                     model_all_views_probs.append(probs)
                
#                 # TTA가 적용된 경우, 모든 view의 예측 확률을 평균
#                 model_avg_probs = torch.mean(torch.stack(model_all_views_probs), dim=0)

#                 # 이 모델의 최종 예측에 가중치를 적용
#                 weighted_model_probs = model_avg_probs * weight

#                 # 모든 모델의 가중치 적용된 예측을 합산
#                 if final_probs_for_batch is None:
#                     final_probs_for_batch = weighted_model_probs
#                 else:
#                     final_probs_for_batch += weighted_model_probs

#             preds = torch.argmax(final_probs_for_batch, dim=1)
#             submission_preds.extend(preds.cpu().numpy())

#     submission_df["target"] = submission_preds
#     submission_df.to_csv(save_path, index=False)
#     print(f"[✓] Saved submission to: {save_path}")