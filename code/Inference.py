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

    def predict(self, image, tta_transforms=None):
        tta_probs = []
        if tta_transforms:
            for tta_transform in tta_transforms:
                img_tta = tta_transform(image=image)['image'].unsqueeze(0).to(self.device)
                weighted_probs = None
                for weight, model in zip(self.weights, self.models):
                    outputs = model(img_tta)
                    probs = torch.softmax(outputs, dim=1)
                    if weighted_probs is None:
                        weighted_probs = weight * probs
                    else:
                        weighted_probs += weight * probs
                tta_probs.append(weighted_probs)
            avg_probs = torch.mean(torch.stack(tta_probs), dim=0)
        else:
            img_tensor = torch.tensor(image).permute(2,0,1).unsqueeze(0).float().to(self.device)
            weighted_probs = None
            for weight, model in zip(self.weights, self.models):
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                if weighted_probs is None:
                    weighted_probs = weight * probs
                else:
                    weighted_probs += weight * probs
            avg_probs = weighted_probs
        return avg_probs

def tta(img_size):
    tta_transforms = []

    # 공통 리사이즈+패딩 transform
    resize_and_pad = [
        A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255), p=1),
    ]

    # 소규모 변화 + 밝기/회전
    tta_transforms.append(A.Compose(resize_and_pad + [
        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.02, rotate_limit=5,
                               border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255), p=0.7),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255), p=0.7),
        ], p=0.7),
        A.RandomBrightnessContrast(limit=0.1, p=0.4),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]))

    # 90, 180, 270도 고정 회전
    for angle in [90, 180, 270]:
        tta_transforms.append(A.Compose(resize_and_pad + [
            A.Rotate(limit=[angle, angle], p=1.0, border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255)),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]))
    return tta_transforms



def run_inference(ensembler, submission_df, test_path, img_size, save_path, batch_size, num_workers, tta_transforms=False):

    tta_transforms = tta(img_size) if tta_transforms else None
    test_transform = A.Compose([
        A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255), p=1),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    test_dataset = ImageDataset(submission_df, path=test_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    submission_preds = []
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Inference"):
            images = images.cpu().permute(0, 2, 3, 1).numpy()  # N,H,W,C

            for img in images:
                if tta_transforms is None:
                    # TTA 없이 단일 예측: 함수 내부에서 만든 transform 사용
                    img_transformed = test_transform(image=img)['image'].unsqueeze(0).to(ensembler.device)
                    weighted_probs = None
                    for weight, model in zip(ensembler.weights, ensembler.models):
                        outputs = model(img_transformed)
                        probs = torch.softmax(outputs, dim=1)
                        if weighted_probs is None:
                            weighted_probs = weight * probs
                        else:
                            weighted_probs += weight * probs
                    avg_probs = weighted_probs
                else:
                    # TTA 수행 후 평균
                    tta_probs = []
                    for tta_transform in tta_transforms:
                        img_tta = tta_transform(image=img)['image'].unsqueeze(0).to(ensembler.device)
                        weighted_probs = None
                        for weight, model in zip(ensembler.weights, ensembler.models):
                            outputs = model(img_tta)
                            probs = torch.softmax(outputs, dim=1)
                            if weighted_probs is None:
                                weighted_probs = weight * probs
                            else:
                                weighted_probs += weight * probs
                        tta_probs.append(weighted_probs)
                    avg_probs = torch.mean(torch.stack(tta_probs), dim=0)

                pred = torch.argmax(avg_probs, dim=1)
                submission_preds.extend(pred.cpu().numpy())

    submission_df["target"] = submission_preds
    submission_df.to_csv(save_path, index=False)
    print(f"[✓] Saved submission to: {save_path}")