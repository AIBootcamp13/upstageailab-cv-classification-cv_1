"""Simple augmentation utilities."""

import random

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from augraphy import (
    AugraphyPipeline,
    InkBleed,
    BleedThrough,
    ColorPaper,
    OneOf,
    NoiseTexturize,
    SubtleNoise,
    LightingGradient,
    ShadowCast,
)

def get_album_transform(img_size: int) -> A.Compose:
    """Return simple Albumentations augmentation pipeline."""
    return A.Compose([
    # 다양한 데이터 증강 기법들
    # A.HorizontalFlip(p=0.5),
    # A.VerticalFlip(p=0.2),
    # A.RandomRotate90(p=0.5),
    # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    # A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    # A.Blur(blur_limit=3, p=0.1),
    # A.CLAHE(clip_limit=2.0, p=0.2),

    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255)),
    # A.OneOf([
    #     A.Blur(blur_limit=2),
    #     A.MotionBlur(blur_limit=5),
    #     A.Defocus(radius=(1, 3)),
    # ], p=0.5),
    A.GaussNoise(var_limit=(0.0000002, 0.000001), mean=0, p=0.3),  # 은은한 미세 노이즈
    # A.OneOf([
    #     A.GaussNoise(var_limit=(0.0000002, 0.000001), mean=0, p=1.0),  # 은은한 미세 노이즈
    #     # A.ImageCompression(quality_lower=40, quality_upper=60, p=1.0),  # 압축으로 전체적 얼룩
    #     # A.CoarseDropout(max_holes=8, max_height=img_size//10, max_width=img_size//10),
    # ], p=0.3),
    # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    # A.OneOf([
    #     A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5),
    #     A.GridDropout(ratio=0.5, p=0.5),
    # ], p=0.4),
    
    A.Resize(height=img_size, width=img_size),
    # A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
    # A.PadIfNeeded(min_height=img_size, min_width=img_size,
    #                 border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=1),    
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_augraphy_pipeline():
    """Augraphy 파이프라인을 반환합니다."""
    
    return AugraphyPipeline(
        ink_phase=[
            InkBleed(p=0.3), # 잉크 번짐
            BleedThrough(p=0.3), # 뒷면 잉크 비침
        ],
        paper_phase=[
            ColorPaper(p=0.3), # 종이 색상 변경
            OneOf([
                NoiseTexturize( # 테스트 데이터랑 비슷한 노이즈
                    sigma_range=(5, 15),
                    turbulence_range=(3, 9),
                    texture_width_range=(50, 500),
                    texture_height_range=(50, 500),
                    p=0.6
                ),
                SubtleNoise(
                    subtle_range=50,
                    p=0.4
                )
            ], p=0.3),
        ],
        post_phase=[
            LightingGradient( # 조명 그라데이션
                light_position=None,
                direction=90,
                max_brightness=255,
                min_brightness=0,
                mode="gaussian",
                transparency=0.5,
                p=0.3
            ),
            ShadowCast( # 그림자
                shadow_side=random.choice(["top", "bottom", "left", "right"]), # 그림자 위치
                shadow_vertices_range=(2, 3),
                shadow_width_range=(0.5, 0.8),
                shadow_height_range=(0.5, 0.8),
                shadow_color=(0, 0, 0),
                shadow_opacity_range=(0.5, 0.6),
                shadow_iterations_range=(1, 2),
                shadow_blur_kernel_range=(101, 301),
                p=0.3
            ),
        ],
    )

def get_tta_transforms(img_size):
    """TTA를 위한 고정된 transform들 (간단하고 안정적인 변형들)"""
    return [
        # 원본
        A.Compose([
            # A.Resize(height=img_size, width=img_size),
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=img_size, min_width=img_size,
                          border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 좌우 반전
        A.Compose([
            # A.Resize(height=img_size, width=img_size),
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=img_size, min_width=img_size,
                          border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=1),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 상하 반전 (문서 이미지에 유용할 수 있음)
        A.Compose([
            # A.Resize(height=img_size, width=img_size),
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=img_size, min_width=img_size,
                          border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=1),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 회전 (약간의 회전)
        A.Compose([
            # A.Resize(height=img_size, width=img_size),
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=img_size, min_width=img_size,
                          border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=1),
            A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=15, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 밝기 조정
        A.Compose([
            # A.Resize(height=img_size, width=img_size),
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=img_size, min_width=img_size,
                          border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=1),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    ]
