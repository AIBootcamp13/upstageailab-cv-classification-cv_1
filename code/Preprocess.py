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
    InkBleed, BleedThrough, Markup, ColorPaper,
    NoiseTexturize, BrightnessTexturize, SubtleNoise,
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
    def __init__(self, train_df, train_path, save_path, img_size, save_csv_path=None):
        self.train_df = train_df
        self.train_path = train_path
        self.save_dir = save_path
        self.img_size = img_size
        self.augmentations = self._get_augmentation(img_size)
        self.augraphy_aug = self._get_augraphy()
        self.document_classes = set(range(0, 2)) | set(range(3, 16))  # Class 2,16 제외
        self.save_csv_path = save_csv_path
        os.makedirs(save_path, exist_ok=True)

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
            ], p=0.3),

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

    def mixup(self, image1, image2, label1, label2, alpha=0.5):
        # image1 = cv2.resize(image1, (self.img_size, self.img_size))
        # image2 = cv2.resize(image2, (self.img_size, self.img_size))
        lam = np.random.beta(alpha, alpha)
        mixup_image = lam * image1.astype(np.float32) + (1 - lam) * image2.astype(np.float32)
        mixup_image = np.clip(mixup_image, 0, 255).astype(np.uint8)
        mixup_label = lam * label1 + (1 - lam) * label2
        return mixup_image, mixup_label

    def cutmix(self, image1, image2, label1, label2):
        height, width, _ = image1.shape
        center_x, center_y = width // 2, height // 2
        quarter = random.randint(0, 3)
        if quarter == 0:
            x1, y1, x2, y2 = 0, 0, center_x, center_y
        elif quarter == 1:
            x1, y1, x2, y2 = center_x, 0, width, center_y
        elif quarter == 2:
            x1, y1, x2, y2 = 0, center_y, center_x, height
        else:
            x1, y1, x2, y2 = center_x, center_y, width, height
        area = (x2 - x1) * (y2 - y1)
        total_area = height * width
        lam = 1 - (area / total_area)
        image1[y1:y2, x1:x2] = image2[y1:y2, x1:x2]
        cutmix_label = lam * label1 + (1 - lam) * label2
        return image1, cutmix_label

    def run(self, target_count):
        augmented_records = []

        all_images = []
        for _, row in self.train_df.iterrows():
            img_path = os.path.join(self.train_path, row["ID"])
            img = Image.open(img_path).convert("RGB")
            img = np.array(img)
            resized_img = resize_and_pad(img, self.img_size)
            all_images.append((resized_img, row["ID"], row["target"]))

        for target_class in sorted(self.train_df["target"].unique()):
            class_df = self.train_df[self.train_df["target"] == target_class]
            current_count = len(class_df)
            print(f"[INFO] Class {target_class}: 현재 {current_count}개 → 목표 {target_count}개로 증강 중...")

            # 원본 이미지를 증강된 폴더로 복사(이미지 리사이징 후 복사)
            for img, img_name, label in tqdm(
                [item for item in all_images if item[2] == target_class],
                total=current_count, desc=f"[Class {target_class}] 원본 복사"):
                
                dst = os.path.join(self.save_dir, img_name)
                Image.fromarray(img).save(dst)
                augmented_records.append({"ID": img_name, "target": target_class})

            n_to_augment = max(0, target_count - current_count)
            print(f"[INFO] Class {target_class}: 추가 증강 {n_to_augment}개 생성 중...")

            class_images = [
                (img, img_name, label)
                for img, img_name, label in all_images
                if label == target_class
            ]

            for i in tqdm(range(n_to_augment), desc=f"[Class {target_class}] 증강 생성"):
                img, img_name, label1 = random.choice(class_images)
                if target_class in self.document_classes and (i % 2 == 0):
                    aug_img = self.augraphy_aug(image=img)
                    aug_name = f"augraphy_{target_class}_{i}_{img_name}"
                else:
                    aug_pipeline = self.augmentations[i % len(self.augmentations)]
                    aug_img = aug_pipeline(image=img)["image"]
                    aug_name = f"alb_{target_class}_{i}_{img_name}"
                rand_val = random.random()

                aug_img = resize_and_pad(aug_img, self.img_size)
                if rand_val < 0.1:
                    mix_img, _, label2 = random.choice(all_images)
                    aug_img, _ = self.mixup(aug_img, mix_img, label1, label2, alpha=0.4)
                    aug_name = f"mixup_{target_class}_{i}_{img_name}"
                elif rand_val < 0.2:
                    mix_img, _, label2 = random.choice(all_images)
                    aug_img, _ = self.cutmix(aug_img, mix_img, label1, label2)
                    aug_name = f"cutmix_{target_class}_{i}_{img_name}"
                
                # aug_img = resize_and_pad(aug_img, self.img_size)
                aug_path = os.path.join(self.save_dir, aug_name)
                Image.fromarray(aug_img).save(aug_path)
                augmented_records.append({"ID": aug_name, "target": target_class})

        print("[완료] 클래스별 증강 완료")
        aug_df = pd.DataFrame(augmented_records)
        
        aug_image_paths = sorted([os.path.join(self.save_dir, fname) for fname in os.listdir(self.save_dir)])
        widths, heights, ids = [], [], []

        for img_path in tqdm(aug_image_paths, desc="Generating augmentation metadata"):
            img_id = os.path.basename(img_path)
            with Image.open(img_path) as img:
                w, h = img.size
            widths.append(w)
            heights.append(h)
            ids.append(img_id)

        aug_df = pd.DataFrame({
            "ID": ids,
            "width": widths,
            "height": heights,
        })
        aug_df["aspect_ratio"] = aug_df["width"] / aug_df["height"]
        aug_df["aspect_bin"] = pd.cut(aug_df["aspect_ratio"], bins=4, labels=False)
        
        aug_df = aug_df.merge(pd.DataFrame(augmented_records), on="ID", how="left")
        aug_df["strata"] = aug_df["target"].astype(str) + "_" + aug_df["aspect_bin"].astype(str)

        aug_df.to_csv(self.save_csv_path, index=False)
        print(f"[완료] 증강 CSV 저장 완료: {self.save_csv_path}")
        return aug_df









# def get_albumentations(self, img_size):
#         return A.Compose([
#             A.OneOf([
#                 A.Defocus(radius=(1, 3), p=1.0),
#                 A.MotionBlur(blur_limit=3, p=1.0),
#                 A.Blur(blur_limit=3, p=1.0),
#             ], p=0.4),

#             A.OneOf([
#                 A.RandomCrop(height=int(0.5 * img_size), width=img_size, p=1.0),
#                 A.RandomSizedCrop(
#                     min_max_height=(int(0.9 * img_size), img_size),
#                     height=img_size, width=img_size,
#                     size=(img_size, img_size),
#                     p=1.0),
#             ], p=0.4),

#             A.OneOf([
#                 A.Rotate(limit=180, border_mode=0, fill=(255, 255, 255), keep_size=True, p=1.0),
#                 A.HorizontalFlip(p=1.0),
#                 A.VerticalFlip(p=1.0),
#                 A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.02, rotate_limit=3,
#                                 border_mode=0, fill=(255, 255, 255), p=1.0),
#             ], p=0.5),

#             A.OneOf([
#                 A.RandomGamma(gamma_limit=(80, 120), p=1.0),
#                 A.CLAHE(clip_limit=1.5, p=1.0),
#                 A.RandomBrightnessContrast(limit=0.1, p=1.0),
#             ], p=0.2),
#         ])

#     def get_augraphy_pipeline(self):
#         return AugraphyPipeline(
#             ink_phase=[
#                 InkBleed(p=0.3),
#                 BleedThrough(p=0.3),
#             ],
#             paper_phase=[
#                 ColorPaper(p=0.3),
#                 OneOf([
#                     NoiseTexturize(
#                         sigma_range=(5, 15),
#                         turbulence_range=(3, 9),
#                         texture_width_range=(50, 500),
#                         texture_height_range=(50, 500),
#                         p=0.6
#                     ),
#                     SubtleNoise(
#                         subtle_range=50,
#                         p=0.4
#                     )
#                 ], p=0.4),
#             ],
#             post_phase=[
#                 LightingGradient(
#                     light_position=None,
#                     direction=90,
#                     max_brightness=255,
#                     min_brightness=0,
#                     mode="gaussian",
#                     transparency=0.5,
#                     p=0.3
#                 ),
#                 ShadowCast(
#                     shadow_side=random.choice(["top", "bottom", "left", "right"]),
#                     shadow_vertices_range=(2, 3),
#                     shadow_width_range=(0.5, 0.8),
#                     shadow_height_range=(0.5, 0.8),
#                     shadow_color=(0, 0, 0),
#                     shadow_opacity_range=(0.5, 0.6),
#                     shadow_iterations_range=(1, 2),
#                     shadow_blur_kernel_range=(101, 301),
#                     p=0.4
#                 ),
#             ],
#         )






#         aug_df.to_csv(self.save_csv_path, index=False)
#         print(f"[완료] 증강 CSV 저장 완료: {self.save_csv_path}")
#         return aug_df



# def _get_augmentation_pipeline(self):
#     aug_list = []

#     # 기존 Albumentations 증강
#     albumentations_pipeline = A.Compose([
#         A.OneOf([  # 흐릿함: blur
#             A.Defocus(radius=(1, 3), p=1.0),
#             A.MotionBlur(blur_limit=3, p=1.0),
#             A.Blur(blur_limit=3, p=1.0),
#         ], p=0.3),

#         A.OneOf([  # 이미지 잘림: Crop
#             A.RandomCrop(height=int(0.5 * self.img_size), width=self.img_size, p=1.0),
#             A.RandomSizedCrop(
#                 min_max_height=(int(0.9 * self.img_size), self.img_size),
#                 height=self.img_size, width=self.img_size,
#                 size=(self.img_size, self.img_size),
#                 p=1.0),
#         ], p=0.3),

#         A.OneOf([  # 회전 및 기하 변형
#             A.Rotate(limit=180, border_mode=0, fill=(255, 255, 255), keep_size=True, p=1.0),
#             A.HorizontalFlip(p=1.0),
#             A.VerticalFlip(p=1.0),
#             A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.02, rotate_limit=3, border_mode=0, fill=(255, 255, 255), p=1.0),
#         ], p=0.4),

#         A.OneOf([  # 밝기/대비
#             A.RandomGamma(gamma_limit=(80, 120), p=1.0),
#             A.CLAHE(clip_limit=1.5, p=1.0),
#             A.RandomBrightnessContrast(limit=0.1, p=1.0),
#         ], p=0.2),
#     ])
#     aug_list.append(albumentations_pipeline)

#     # Augraphy 증강
#     augraphy_pipeline = AugraphyPipeline(
#         ink_phase=[
#             InkBleed(p=0.3),  # 잉크 번짐
#             BleedThrough(p=0.3),  # 뒷면 잉크 비침
#         ],
#         paper_phase=[
#             ColorPaper(p=0.3),  # 종이 색상 변경
#             AugraphyOneOf([
#                 NoiseTexturize(  # 테스트 데이터와 비슷한 노이즈
#                     sigma_range=(5, 15),
#                     turbulence_range=(3, 9),
#                     texture_width_range=(50, 500),
#                     texture_height_range=(50, 500),
#                     p=0.6
#                 ),
#                 SubtleNoise(
#                     subtle_range=50,
#                     p=0.4
#                 )
#             ], p=0.3),
#         ],
#         post_phase=[
#             LightingGradient(  # 조명 그라데이션
#                 light_position=None,
#                 direction=90,
#                 max_brightness=255,
#                 min_brightness=0,
#                 mode="gaussian",
#                 transparency=0.5,
#                 p=0.3
#             ),
#             ShadowCast(  # 그림자
#                 shadow_side=random.choice(["top", "bottom", "left", "right"]),
#                 shadow_vertices_range=(2, 3),
#                 shadow_width_range=(0.5, 0.8),
#                 shadow_height_range=(0.5, 0.8),
#                 shadow_color=(0, 0, 0),
#                 shadow_opacity_range=(0.5, 0.6),
#                 shadow_iterations_range=(1, 2),
#                 shadow_blur_kernel_range=(101, 301),
#                 p=0.3
#             ),
#         ],
#     )
#     aug_list.append(augraphy_pipeline)

#     return aug_list







# def _get_augmentation_pipeline(self):
#         aug_list = []

#         # Augraphy 증강
#         augraphy_pipeline = AugraphyPipeline(
#             ink_phase=[
#                 InkBleed(p=0.5),
#                 LowInkRandomLines(p=0.5),
#                 BleedThrough(p=0.3),
#                 AugraphyOneOf([
#                     DirtyDrum(p=0.5),
#                     BadPhotoCopy(p=0.5),
#                 ], p=0.5),
#                 AugraphyOneOf([
#                     SubtleNoise(p=0.4),
#                     LowLightNoise(p=0.4),
#                     NoiseTexturize(p=0.4),
#                 ], p=0.4),
#             ],
#             paper_phase=[
#                 LightingGradient(p=0.5),
#                 ShadowCast(p=0.5),
#                 Geometric(scale=(0.9, 1.1), p=1.0),
#             ],
#             post_phase=[]
#         )
#         aug_list.append(augraphy_pipeline)

#         # Albumentations 증강
#         albumentations_pipeline = A.Compose([
#             A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, border_mode=0, fill=(255, 255, 255), p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.HorizontalFlip(p=0.5),
#             A.RandomBrightnessContrast(p=0.3),
#             A.OneOf([
#                 A.RandomRotate90(p=0.5),
#                 A.Rotate(limit=90, border_mode=0, fill=(255, 255, 255), p=0.5),
#             ], p=0.5),
#             A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
#             A.ImageCompression(quality_lower=40, quality_upper=80, compression_type='jpeg', p=0.5)
#         ])
#         aug_list.append(albumentations_pipeline)

#         return aug_list




# def _get_augmentation_pipeline(self):
#         aug_list = []

#         # ▶ Albumentations 증강
#         albumentations_pipeline = A.Compose([
#             A.OneOf([  # 흐릿함: blur
#                 A.Defocus(radius=(1, 3), p=1.0),
#                 A.MotionBlur(blur_limit=3, p=1.0),
#                 A.Blur(blur_limit=3, p=1.0),
#             ], p=0.4),

#             A.OneOf([  # 이미지 잘림: Crop
#                 A.RandomCrop(height=int(0.5 * self.img_size), width=self.img_size, p=1.0),
#                 A.RandomSizedCrop(
#                     min_max_height=(int(0.9 * self.img_size), self.img_size),
#                     height=self.img_size, width=self.img_size,
#                     size=(self.img_size, self.img_size),
#                     p=1.0),
#             ], p=0.4),

#             A.OneOf([  # 회전 및 기하 변형
#                 A.Rotate(limit=180, border_mode=0, fill=(255, 255, 255), keep_size=True, p=1.0),
#                 A.HorizontalFlip(p=1.0),
#                 A.VerticalFlip(p=1.0),
#                 A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.02, rotate_limit=3,
#                                 border_mode=0, fill=(255, 255, 255), p=1.0),
#             ], p=0.5),

#             A.OneOf([  # 밝기/대비
#                 A.RandomGamma(gamma_limit=(80, 120), p=1.0),
#                 A.CLAHE(clip_limit=1.5, p=1.0),
#                 A.RandomBrightnessContrast(limit=0.1, p=1.0),
#             ], p=0.2),
#         ])
#         aug_list.append(albumentations_pipeline)

#         # ▶ Augraphy 증강
#         augraphy_pipeline = AugraphyPipeline(
#             ink_phase=[
#                 InkBleed(p=0.3),
#                 BleedThrough(p=0.3),
#             ],
#             paper_phase=[
#                 ColorPaper(p=0.3),
#                 OneOf([
#                     NoiseTexturize(
#                         sigma_range=(5, 15),
#                         turbulence_range=(3, 9),
#                         texture_width_range=(50, 500),
#                         texture_height_range=(50, 500),
#                         p=0.6
#                     ),
#                     SubtleNoise(
#                         subtle_range=50,
#                         p=0.4
#                     )
#                 ], p=0.4),
#             ],
#             post_phase=[
#                 LightingGradient(
#                     light_position=None,
#                     direction=90,
#                     max_brightness=255,
#                     min_brightness=0,
#                     mode="gaussian",
#                     transparency=0.5,
#                     p=0.3
#                 ),
#                 ShadowCast(
#                     shadow_side=random.choice(["top", "bottom", "left", "right"]),
#                     shadow_vertices_range=(2, 3),
#                     shadow_width_range=(0.5, 0.8),
#                     shadow_height_range=(0.5, 0.8),
#                     shadow_color=(0, 0, 0),
#                     shadow_opacity_range=(0.5, 0.6),
#                     shadow_iterations_range=(1, 2),
#                     shadow_blur_kernel_range=(101, 301),
#                     p=0.4
#                 ),
#             ],
#         )
#         aug_list.append(augraphy_pipeline)

#         return aug_list