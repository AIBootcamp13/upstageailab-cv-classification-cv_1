import albumentations as A
from albumentations.pytorch import ToTensorV2

# ✅ 최종 transform (조합 기반 augmentation 적용)
train_transform = A.Compose([
    # Resize
    A.Resize(256, 256),  # ← baseline과 동일

    # === 조합 기반 augmentation 시작 ===

    # 1️⃣ Blur & Motion Blur 조합
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.MotionBlur(blur_limit=7, p=0.5),
    ], p=0.5),  # Test set blur ≈ 47.5% 근거

    # 2️⃣ Noise + BrightnessContrast 조합
    A.OneOf([
        A.Compose([  # 조합형 augmentation
            A.GaussNoise(var_limit=(10, 50), p=0.7),  # 기본 noise
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        ]),
        A.GaussNoise(var_limit=(50, 100), p=0.3),    # 고강도 noise (1.6% 근거)
    ], p=0.2), # 전체 batch 중 20% 적용

    # 3️⃣ Skew (Rotation) 조합
    A.OneOf([
        A.Rotate(limit=30, p=0.7),  # ±30° skew
        A.RandomRotate90(p=0.3),    # ±90° rotation
    ], p=0.5),  # Test set skew ≈ 50.7% 근거

    # === 조합 기반 augmentation 끝 ===

    # Normalize
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])