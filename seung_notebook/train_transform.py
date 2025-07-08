
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    # Resize : 서칭 후 근거 찾고 수정 !!!
    A.Resize(32, 32),  # baseline_code.py : img_size=32

    # Rotation & Skew
    A.OneOf([
        A.Rotate(limit=30, p=0.7),   # ±30° 기울기
        A.RandomRotate90(p=0.3),     # ±90° 회전
    ], p=0.5),  # 전체 batch 중 약 50% 적용

    # Blur & Motion blur
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.MotionBlur(blur_limit=7, p=0.5),
    ], p=0.5),  # 전체 batch 중 약 50% 적용

    # Noise
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.7),  # 기본 noise
        A.GaussNoise(var_limit=(50.0, 100.0), p=0.3), # 고강도 noise
    ], p=0.2),  # 전체 batch 중 약 20% 적용 (강한 noise 비율 ≈ 1.6% 반영)

    # Brightness & Contrast (조명 불균형 대응)
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),

    # Normalize & Tensor
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])