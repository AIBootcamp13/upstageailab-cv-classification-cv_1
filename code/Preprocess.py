import os
import shutil
import numpy as np
import pandas as pd
import cv2
import random
from tqdm import tqdm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from augraphy import AugraphyPipeline
from augraphy.base.oneof import OneOf
from augraphy.augmentations import (
    InkBleed, BleedThrough, ColorPaper,
    NoiseTexturize, SubtleNoise,
    LightingGradient, ShadowCast
)

def resize_and_pad(img, img_size):
    transform = A.Compose([
        A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
        # A.Resize(height=img_size, width=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255), p=1),
    ])
    return transform(image=img)["image"]


class Augmentation:
    def __init__(self, img_size):
        self.img_size = img_size
        self.augmentations = self._get_augmentation(img_size)
        self.augraphy_aug = self._get_augraphy()

    def _get_augmentation(self, img_size):
        aug_list = [
            A.OneOf([ # 흐릿함: blur
                A.Defocus(radius=(1, 3)),
                A.MotionBlur(blur_limit=3),
                A.Blur(blur_limit=3),
            ], p=0.4),

            A.OneOf([ # 이미지 잘림: Crop
                A.RandomCrop(height=int(0.5 * img_size), width=img_size, p=0.3),
                A.RandomSizedCrop(
                    min_max_height=(int(0.9 * img_size), img_size),
                    height=img_size, width=img_size,
                    size=(img_size, img_size),
                    p=0.3),
            ], p=0.4),

            A.OneOf([ # 회전
                A.Rotate(limit=180, border_mode=0, fill=(255, 255, 255), keep_size=True, p=0.6),
                A.HorizontalFlip(p=0.6),
                A.VerticalFlip(p=0.6),
                A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.02, rotate_limit=3, border_mode=0, fill=(255, 255, 255), p=0.2),
            ], p=0.4),

            A.OneOf([ # 밝기/대비
                A.RandomGamma(gamma_limit=(80, 120), p=0.2),
                A.CLAHE(clip_limit=1.5, p=0.2),
                A.RandomBrightnessContrast(limit=0.1, p=0.4),
            ], p=0.2), 
        ]
        return [A.Compose(aug_list)]

    def _get_augraphy(self):
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
                ], p=0.4),
            ],
            post_phase=[
                LightingGradient( # 조명 그라데이션
                    light_position=None,
                    direction=90,
                    max_brightness=255,
                    min_brightness=0,
                    mode="gaussian",
                    transparency=0.5,
                    p=0.4
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
                    p=0.4
                ),
            ],
        )

    def mixup(self, image1, image2, label1, label2, alpha=0.5):
        lam = np.random.beta(alpha, alpha)
        mixup_image = lam * image1.astype(np.float32) + (1 - lam) * image2.astype(np.float32)
        mixup_image = np.clip(mixup_image, 0, 255).astype(np.uint8)
        return mixup_image, label1

    def cutmix(self, image1, image2, label1, label2):
        height, width, _ = image1.shape
        center_x, center_y = width // 2, height // 2
        quarter = random.randint(0, 3)
        if quarter == 0: x1, y1, x2, y2 = 0, 0, center_x, center_y
        elif quarter == 1: x1, y1, x2, y2 = center_x, 0, width, center_y
        elif quarter == 2: x1, y1, x2, y2 = 0, center_y, center_x, height
        else: x1, y1, x2, y2 = center_x, center_y, width, height
        
        image1[y1:y2, x1:x2] = image2[y1:y2, x1:x2]
        return image1, label1


    def run(self, df_to_augment, original_data_path, save_path, target_count):
        os.makedirs(save_path, exist_ok=True)
        
        augmented_records = []
        all_original_images = []

        for _, row in df_to_augment.iterrows():
            img_path = os.path.join(original_data_path, row["ID"])
            img = Image.open(img_path).convert("RGB")
            img = np.array(img)
            resized_img = resize_and_pad(img, self.img_size)
            all_original_images.append((resized_img, row["ID"], row["target"]))
            # all_original_images.append((img, row["ID"], row["target"]))

        for target_class in sorted(df_to_augment["target"].unique()):
            class_df = df_to_augment[df_to_augment["target"] == target_class]
            current_count = len(class_df)
            
            for img, img_name, label in [item for item in all_original_images if item[2] == target_class]:
                dst = os.path.join(save_path, img_name)
                Image.fromarray(img).save(dst)

            n_to_augment = max(0, target_count - current_count)
            if n_to_augment > 0:
                class_images_for_aug = [(img, name, lbl) for img, name, lbl in all_original_images if lbl == target_class]

                for i in tqdm(range(n_to_augment), desc=f"Augmented for class {target_class}"):
                    img, img_name, label1 = random.choice(class_images_for_aug)
                    
                    aug_img = img.copy() # 원본 이미지 복사
                    rand_val = random.random()
                    if rand_val < 0.2: # Mixup: 20%
                        mix_img_data = random.choice(all_original_images)
                        aug_img, label1 = self.mixup(aug_img, mix_img_data[0], label1, mix_img_data[2], alpha=0.4)
                        aug_name = f"mixup_{target_class}_{i}_{img_name}"

                    elif rand_val < 0.3: # Cutmix: 10%
                        mix_img_data = random.choice(all_original_images)
                        aug_img, label1 = self.cutmix(aug_img, mix_img_data[0], label1, mix_img_data[2])
                        aug_name = f"cutmix_{target_class}_{i}_{img_name}"

                    else: # 일반 증강: 70%
                        if (i % 2 == 0): # Augraphy: 35%
                            aug_img = self.augraphy_aug(image=aug_img)
                            aug_name = f"augraphy_{target_class}_{i}_{img_name}"
                        else: # Albumentations: 35%
                            aug_pipeline = self.augmentations[i % len(self.augmentations)]
                            aug_img = aug_pipeline(image=aug_img)["image"]
                            aug_name = f"alb_{target_class}_{i}_{img_name}"
                    
                    aug_img = resize_and_pad(aug_img, self.img_size)
                    aug_path = os.path.join(save_path, aug_name)
                    Image.fromarray(aug_img).save(aug_path)
                    augmented_records.append({"ID": aug_name, "target": label1})

        aug_df = pd.DataFrame(augmented_records)
        return aug_df