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
    
    # test_transform 기존 정의를 주석 처리 또는 제거
    # test_transform = A.Compose([
    #     A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
    #     A.PadIfNeeded(min_height=img_size, min_width=img_size,
    #                   border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255)),
    #     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #     ToTensorV2(),
    # ])

    # 수정: DataLoader에 전달할 transform 변경
    initial_inference_transform = get_img_resize(img_size)
    test_dataset = ImageDataset(submission_df, path=test_path, transform=initial_inference_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # 기본 예측 및 TTA 이후에 사용할 최종 정규화/텐서 변환 함수
    final_norm_to_tensor_transform = basic_transform()

    submission_preds = []
    ensembler.models = [m.eval() for m in ensembler.models]

    
    if use_tta:
        # 기존 파일명에서 .csv를 찾아서 그 앞에 -TTA를 추가
        base_name, ext = os.path.splitext(save_path)
        save_path = f"{base_name}-TTA{ext}"

    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Inference"):
            # 수정: DataLoader에서 반환되는 이미지는 Tensor이므로, 변수명을 변경
            # 그리고 Albumentations에 전달하기 위해 NumPy 배열로 변환 필요
            images_tensor_batch, _, img_ids = batch_data 
            
            # TTA 없이 예측
            all_probs_list = []
            
            # 수정: Tensor 배치를 순회하며 각 이미지를 NumPy로 변환 후, 최종 정규화/텐서 변환 적용
            processed_images_base_tensors = []
            for img_tensor in images_tensor_batch:
                # Tensor를 HWC NumPy 배열로 변환 (Albumentations는 HWC를 기대)
                img_np = img_tensor.cpu().numpy() # DataLoader가 uint8 HWC NumPy를 uint8 Tensor로 변환했을 것
                # Albumentations의 Normalize는 float32를 선호하므로 명시적 캐스팅.
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
                    # 수정: Tensor 배치를 순회하며 각 이미지를 NumPy로 변환 후, TTA 변환 적용
                    processed_images_tta_tensors = []
                    for img_tensor in images_tensor_batch:
                        # Tensor를 HWC NumPy 배열로 변환
                        img_np = img_tensor.cpu().numpy() # DataLoader가 uint8 HWC NumPy를 uint8 Tensor로 변환했을 것
                        # Albumentations의 Normalize는 float32를 선호하므로 명시적 캐스팅.
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