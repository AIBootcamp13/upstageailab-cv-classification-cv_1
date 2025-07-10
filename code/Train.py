import os
import torch
import timm
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from Load_Data import ImageDataset, get_transforms
from Preprocess import Augmentation

class Trainer:
    def __init__(self, df, original_data_path, augmented_save_path, augmented_csv_save_path, model_name, epochs, batch_size, lr, drop_out,
                 img_size, num_workers, device, save_dir, run_name_prefix, num_classes,
                 n_splits, patience, weight_decay, k_fold=True, augmentation_target_count=300):
        
        self.df = df
        self.original_data_path = original_data_path
        self.augmented_save_path = augmented_save_path
        self.augmented_csv_save_path = augmented_csv_save_path
        self.augmentation_target_count = augmentation_target_count
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.drop_out = drop_out
        self.img_size = img_size
        self.num_workers = num_workers
        self.device = device
        self.save_dir = save_dir
        self.run_name_prefix = run_name_prefix
        self.num_classes = num_classes
        self.n_splits = n_splits
        self.patience = patience
        self.weight_decay = weight_decay
        self.k_fold = k_fold
        self.augmenter = Augmentation(img_size)

    def _add_strata_to_df(self):
        widths, heights = [], []
        for img_id in tqdm(self.df['ID'], desc="Calculating aspect ratios"):
            img_path = os.path.join(self.original_data_path, img_id)
            with Image.open(img_path) as img:
                w, h = img.size
            widths.append(w)
            heights.append(h)
        
        df_copy = self.df.copy()
        df_copy['width'] = widths
        df_copy['height'] = heights
        df_copy['aspect_ratio'] = df_copy['width'] / df_copy['height']
        df_copy['aspect_bin'] = pd.cut(df_copy['aspect_ratio'], bins=4, labels=False)
        df_copy['strata'] = df_copy['target'].astype(str) + "_" + df_copy['aspect_bin'].astype(str)
        self.df = df_copy

    def run(self):
        self._add_strata_to_df()
        fold_f1s = []
        all_folds_train_dfs = [] 

        if self.k_fold:
            strata_counts = self.df["strata"].value_counts()
            safe_strata = strata_counts[strata_counts >= self.n_splits].index
            problem_strata = strata_counts[strata_counts < self.n_splits].index

            df_safe = self.df[self.df["strata"].isin(safe_strata)].reset_index(drop=True)
            df_problem = self.df[self.df["strata"].isin(problem_strata)].reset_index(drop=True)
            
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

            for fold, (train_idx_safe, val_idx_safe) in enumerate(skf.split(df_safe, df_safe["strata"])):
                
                fold_train_df_orig = pd.concat([df_safe.iloc[train_idx_safe], df_problem]).reset_index(drop=True)
                fold_val_df = df_safe.iloc[val_idx_safe].reset_index(drop=True)

                print(f"\n{'='*20} Processing Fold {fold+1} {'='*20}\n")
                
                fold_aug_image_path = os.path.join(self.augmented_save_path, f"fold_{fold+1}")
                augmented_info_df = self.augmenter.run(
                    df_to_augment=fold_train_df_orig,
                    original_data_path=self.original_data_path,
                    save_path=fold_aug_image_path,
                    target_count=self.augmentation_target_count
                )
                
                final_train_df = pd.concat([fold_train_df_orig, augmented_info_df[~augmented_info_df['ID'].isin(fold_train_df_orig['ID'])]]).reset_index(drop=True)
                
                # 이 Fold의 최종 학습 데이터 정보를 리스트에 추가
                all_folds_train_dfs.append(final_train_df)
                print(f"\n=== Fold {fold+1}: Train={len(final_train_df)}, Val={len(fold_val_df)} ===")
                best_f1 = self.train_fold(fold, final_train_df, fold_val_df, fold_aug_image_path)
                fold_f1s.append(best_f1)
        
        else: # Holdout
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, val_idx = next(sss.split(self.df, self.df["strata"]))

            holdout_train_df_orig = self.df.iloc[train_idx].reset_index(drop=True)
            holdout_val_df = self.df.iloc[val_idx].reset_index(drop=True)
            
            holdout_aug_image_path = os.path.join(self.augmented_save_path, "holdout")
            augmented_info_df = self.augmenter.run(
                df_to_augment=holdout_train_df_orig,
                original_data_path=self.original_data_path,
                save_path=holdout_aug_image_path,
                target_count=self.augmentation_target_count
            )
            final_train_df = pd.concat([holdout_train_df_orig, augmented_info_df[~augmented_info_df['ID'].isin(holdout_train_df_orig['ID'])]]).reset_index(drop=True)
            
            # Holdout의 최종 학습 데이터 정보를 리스트에 추가
            all_folds_train_dfs.append(final_train_df)
            best_f1 = self.train_fold(0, final_train_df, holdout_val_df, holdout_aug_image_path)
            fold_f1s.append(best_f1)

        # 모든 루프가 끝난 후, 종합 CSV 파일 저장
        master_train_df = pd.concat(all_folds_train_dfs, ignore_index=True)
        master_train_df.drop_duplicates(subset=['ID'], keep='first', inplace=True)
        master_train_df.to_csv(self.augmented_csv_save_path, index=False)
        print(f"Augmented CSV file with {len(master_train_df)} records saved to: {self.augmented_csv_save_path}")

        f1_df = pd.DataFrame({'fold': list(range(1, len(fold_f1s)+1)), 'f1': fold_f1s})
        return f1_df

    def train_fold(self, fold, train_df, val_df, train_data_path):
        train_transform, val_transform = get_transforms(self.img_size)
        
        train_dataset = ImageDataset(train_df, path=train_data_path, transform=train_transform)
        val_dataset = ImageDataset(val_df, path=self.original_data_path, transform=val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        model = timm.create_model(self.model_name, pretrained=True, num_classes=self.num_classes, drop_path_rate=self.drop_out).to(self.device)
        loss_fn = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.0000005) # 5e-7
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-7)
        
        best_f1, trigger = -1.0, 0
        fold_label = f"Fold {fold+1}" if self.k_fold else "Holdout"
        os.makedirs(self.save_dir, exist_ok=True) # 모델 저장
        os.environ["WANDB_DIR"] = "../"           

        wandb.init(
            project="Document Classification",
            name=f"{self.run_name_prefix}-{fold_label}",
            config={
                "epochs": self.epochs, "batch_size": self.batch_size, "learning_rate": self.lr,
                "Drop_out": self.drop_out, "weight_decay": self.weight_decay,
                "model_name": self.model_name, "img_size": self.img_size,
                "Aug_cnt": self.augmentation_target_count
            }
        )

        for epoch in range(1, self.epochs + 1):
            model.train()
            train_loss = 0.0
            train_bar = tqdm(train_loader, desc=f"[{fold_label}][Epoch {epoch}/{self.epochs}] Training")
            for batch in train_bar:
                images, targets, _ = batch # ImageDataset이 ID도 반환
                images, targets = images.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_bar.set_postfix(loss=loss.item())

            avg_train_loss = train_loss / len(train_loader)

            model.eval()
            val_loss, correct, total = 0.0, 0, 0
            all_preds, all_targets = [], []
            current_lr = optimizer.param_groups[0]['lr']

            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f"[{fold_label}][Epoch {epoch}/{self.epochs}] Validation")
                for batch in val_bar:
                    images, targets, _ = batch
                    images, targets = images.to(self.device), targets.to(self.device)
                    outputs = model(images)
                    loss = loss_fn(outputs, targets)
                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)

            avg_val_loss = val_loss / len(val_loader)
            val_acc = correct / total
            val_f1 = f1_score(all_targets, all_preds, average='macro')

            print(f"[{fold_label}] Ep{epoch} - Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

            wandb.log({
                "fold": fold_label, "train_loss": avg_train_loss, "val_loss": avg_val_loss,
                "val_acc": val_acc, "val_f1": val_f1, "Train_Data": len(train_df),
                "Val_Data": len(val_df), "scheduler_lr": current_lr
            })

            if val_f1 > best_f1:
                best_f1, trigger = val_f1, 0
                model_save_name = f"model_{fold_label.replace(' ', '_')}.pth"
                torch.save(model.state_dict(), os.path.join(self.save_dir, model_save_name))
            else:
                trigger += 1
                if trigger >= self.patience:
                    print(f"[{fold_label}] Early stopping. Best F1: {best_f1:.4f}")
                    break
            
            scheduler.step()
            wandb.log({"best_val_f1": best_f1})

        wandb.finish()
        return best_f1