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


class Model_Ensemble:
    def __init__(self, model_name, fold_paths_dir, fold_weights, num_classes, drop_out, device, k_fold=True):
        self.device = device

        if k_fold:
            fold_paths = sorted(glob.glob(os.path.join(fold_paths_dir, "model_Fold*.pth")))
            if not fold_paths:
                raise ValueError(f"[Error] No model_Fold*.pth files found in: {fold_paths_dir}")
        else:
            fold_paths = sorted(glob.glob(os.path.join(fold_paths_dir, "model_Holdout.pth")))
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

    # 모든 변환에 공통적으로 적용될 전처리
    base_transform = [
        A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255)),
    ]
    # 모든 변환 마지막에 공통적으로 적용될 후처리
    post_transform = [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]

    tta_transforms = []

    # 색상 반전
    tta_transforms.append(A.Compose(base_transform + [
        A.InvertImg(p=1.0)
    ] + post_transform))
    
    # 노이즈 완화
    tta_transforms.append(A.Compose(base_transform + [
        A.MedianBlur(blur_limit=5, p=1.0)
    ] + post_transform))

    # 좌우 대칭성 
    tta_transforms.append(A.Compose(base_transform + [
        A.HorizontalFlip(p=1.0)
    ] + post_transform))
    
    # 회전 대응
    tta_transforms.append(A.Compose(base_transform + [
        A.Rotate(limit=10, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255))
    ] + post_transform))

    # 색상 변형 대응
    tta_transforms.append(A.Compose(base_transform + [
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0)
    ] + post_transform))
    
    return tta_transforms


def run_inference(ensembler, submission_df, test_path, img_size, save_path, batch_size, num_workers, use_tta=False): # [수정] 파라미터 이름 use_tta로 명확화
    tta_transforms = tta(img_size) if use_tta else None
    
    test_transform = A.Compose([
        A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255)),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    test_dataset = ImageDataset(submission_df, path=test_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    submission_preds = []
    ensembler.models = [m.eval() for m in ensembler.models]

    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Inference"):
            images = images.to(ensembler.device)

            # TTA 로직 전체 수정
            all_probs_list = []

            # 원본 이미지에 대한 예측을 먼저 수행하여 리스트에 추가
            weighted_probs_original = None
            for weight, model in zip(ensembler.weights, ensembler.models):
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                if weighted_probs_original is None:
                    weighted_probs_original = probs * weight
                else:
                    weighted_probs_original += probs * weight
            all_probs_list.append(weighted_probs_original)

            # TTA가 활성화된 경우, 추가 예측 수행
            if tta_transforms is not None:
                for tta_transform in tta_transforms:
                    imgs_np = images.cpu().permute(0, 2, 3, 1).numpy()
                    transformed_images = [tta_transform(image=img)['image'] for img in imgs_np]
                    imgs_tta = torch.stack(transformed_images).to(ensembler.device)
                    
                    weighted_probs_tta = None
                    for weight, model in zip(ensembler.weights, ensembler.models):
                        outputs = model(imgs_tta)
                        probs = torch.softmax(outputs, dim=1)
                        if weighted_probs_tta is None:
                            weighted_probs_tta = probs * weight
                        else:
                            weighted_probs_tta += probs * weight
                    all_probs_list.append(weighted_probs_tta)
            
            # 모든 예측 결과(원본 + TTA)의 평균을 계산
            avg_probs = torch.mean(torch.stack(all_probs_list), dim=0)

            preds = torch.argmax(avg_probs, dim=1)
            submission_preds.extend(preds.cpu().numpy())

    submission_df["target"] = submission_preds
    submission_df.to_csv(save_path, index=False)
    print(f"[✓] Saved submission to: {save_path}")