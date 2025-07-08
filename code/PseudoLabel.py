# PseudoLabel.py

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image
import os
import shutil

from Load_Data import ImageDataset
from Inference import tta
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PseudoLabeler:
    def __init__(self, ensembler, device, img_size, batch_size, num_workers):
        self.ensembler = ensembler
        self.device = device
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tta_transforms = tta(img_size)

    def run(self, original_aug_df, original_aug_path, test_df, test_path, 
            confidence_threshold, save_base_dir, run_name):
        
        # 경로 생성
        new_data_dir = os.path.join(save_base_dir, run_name)
        new_image_dir = os.path.join(new_data_dir, "images")
        new_csv_path = os.path.join(new_data_dir, f"{run_name}-combined.csv")
        os.makedirs(new_image_dir, exist_ok=True)

        # 의사 레이블 생성
        confident_pseudo_df = self._create_pseudo_labels(test_df, test_path, confidence_threshold)

        # 이미지 파일 복사
        for img_id in tqdm(original_aug_df['ID'], desc="Copying original images"):
            shutil.copy(os.path.join(original_aug_path, img_id), os.path.join(new_image_dir, img_id))
        for img_id in tqdm(confident_pseudo_df['ID'], desc="Copying pseudo-labeled images"):
            shutil.copy(os.path.join(test_path, img_id), os.path.join(new_image_dir, img_id))

        # 데이터프레임 결합
        cols_to_keep = ['ID', 'target', 'strata']
        combined_df = pd.concat([
            original_aug_df[cols_to_keep],
            confident_pseudo_df[cols_to_keep]
        ], ignore_index=True)

        # 최종 CSV 저장
        combined_df.to_csv(new_csv_path, index=False)        
        return combined_df, new_image_dir

    def _create_pseudo_labels(self, test_df, test_path, confidence_threshold):
        test_transform = A.Compose([
            A.LongestMaxSize(max_size=self.img_size),
            A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, border_mode=0, value=(255, 255, 255)),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        
        test_dataset = ImageDataset(test_df.copy(), path=test_path, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        self.ensembler.models = [m.eval() for m in self.ensembler.models]
        
        all_image_ids, all_final_probs, all_final_preds = [], [], []

        with torch.no_grad():
            for images, _, image_ids in tqdm(test_loader, desc="Pseudo-Labeling Inference"):
                images = images.to(self.device)
                all_image_ids.extend(image_ids)
                
                all_probs_list = [self._get_ensemble_prediction(images)]
                for tta_transform in self.tta_transforms:
                    imgs_np = images.cpu().permute(0, 2, 3, 1).numpy()
                    transformed_images = [tta_transform(image=img)['image'] for img in imgs_np]
                    imgs_tta = torch.stack(transformed_images).to(self.device)
                    all_probs_list.append(self._get_ensemble_prediction(imgs_tta))

                avg_probs = torch.mean(torch.stack(all_probs_list), dim=0)
                final_probs, final_preds = torch.max(avg_probs, dim=1)

                all_final_probs.extend(final_probs.cpu().numpy())
                all_final_preds.extend(final_preds.cpu().numpy())

        pseudo_df = pd.DataFrame({'ID': all_image_ids, 'probability': all_final_probs, 'target': all_final_preds})
        confident_pseudo_df = pseudo_df[pseudo_df['probability'] >= confidence_threshold].copy()
        
        # 종횡비 기반 strata 생성
        widths, heights = [], []
        for img_id in tqdm(confident_pseudo_df['ID'], desc="Generating strata"):
            with Image.open(os.path.join(test_path, img_id)) as img:
                w, h = img.size
            widths.append(w); heights.append(h)

        confident_pseudo_df['width'] = widths
        confident_pseudo_df['height'] = heights
        confident_pseudo_df['aspect_ratio'] = confident_pseudo_df['width'] / confident_pseudo_df['height']
        confident_pseudo_df['aspect_bin'] = pd.cut(confident_pseudo_df['aspect_ratio'], bins=4, labels=False)
        confident_pseudo_df['strata'] = confident_pseudo_df['target'].astype(str) + "_" + confident_pseudo_df['aspect_bin'].astype(str)
        
        return confident_pseudo_df

    def _get_ensemble_prediction(self, images_batch):
        weighted_probs = None
        for weight, model in zip(self.ensembler.weights, self.ensembler.models):
            outputs = model(images_batch); probs = torch.softmax(outputs, dim=1)
            if weighted_probs is None: weighted_probs = probs * weight
            else: weighted_probs += probs * weight
        return weighted_probs