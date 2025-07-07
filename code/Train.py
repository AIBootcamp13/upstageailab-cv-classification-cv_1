import os
import torch
import timm
import wandb
import numpy as np
import pandas as pd
import albumentations as A
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from Load_Data import ImageDataset, get_transforms

class Trainer:
    def __init__(self, df, data_path, model_name, epochs, batch_size, lr, drop_out,
                 img_size, num_workers, device, save_dir, run_name_prefix, num_classes,
                 n_splits, patience, weight_decay, k_fold=False):  
        self.df = df
        self.data_path = data_path
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
        self.weight_decay = weight_decay  # 추가
        self.k_fold = k_fold # 추가

        self.base_train_transform, self.val_transform = get_transforms()
        self.train_transform = self.base_train_transform  # aug_transform 제거, base transform만 사용

    def run(self):
        fold_f1s = []
        
        if self.k_fold:  # 수정 부분: k_fold가 True면 기존 K-Fold 수행
            kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            for fold, (train_idx, val_idx) in enumerate(kf.split(self.df, self.df['target'])):
                train_df = self.df.iloc[train_idx].reset_index(drop=True)
                val_df = self.df.iloc[val_idx].reset_index(drop=True)
                print(f"\n=== Fold {fold+1}: Train={len(train_df)}, Val={len(val_df)} ===")
                best_f1 = self.train_fold(fold, train_df, val_df)
                fold_f1s.append(best_f1)
                
        else:  # 수정 부분: k_fold=False면 Holdout
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, val_idx = next(sss.split(self.df, self.df['target']))
            train_df = self.df.iloc[train_idx].reset_index(drop=True)
            val_df = self.df.iloc[val_idx].reset_index(drop=True)
            print(f"\n=== Holdout: Train={len(train_df)}, Val={len(val_df)} ===")
            best_f1 = self.train_fold(0, train_df, val_df)
            fold_f1s.append(best_f1)

        f1_df = pd.DataFrame({'fold': list(range(1, len(fold_f1s)+1)), 'f1': fold_f1s})
        return f1_df

    def train_fold(self, fold, train_df, val_df):
        train_dataset = ImageDataset(train_df, path=self.data_path, transform=self.train_transform)
        val_dataset = ImageDataset(val_df, path=self.data_path, transform=self.val_transform)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        model = timm.create_model(self.model_name, pretrained=True, num_classes=self.num_classes, drop_path_rate=self.drop_out).to(self.device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=7, T_mult=2, eta_min=1e-6)
        best_f1, trigger = -1.0, 0

        # fold_label 생성: k-fold와 홀드아웃 구분
        fold_label = f"Fold {fold+1}" if self.k_fold else "Holdout"

        os.environ["WANDB_DIR"] = "../../"
        wandb.init(
            project="Document Classification",
            entity="moonstalker9010-none",
            name=f"{self.run_name_prefix}-{fold_label}",
            config={
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.lr,
                "Drop_out": self.drop_out,
                "weight_decay": self.weight_decay,
                "model_name": self.model_name,
                "img_size": self.img_size,
            }
        )

        for epoch in range(1, self.epochs + 1):
            # 학습
            model.train()
            train_loss = 0.0
            train_bar = tqdm(train_loader, desc=f"[{fold_label}][Epoch {epoch}/{self.epochs}] Training")
            for images, targets in train_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_bar.set_postfix(loss=loss.item())

            avg_train_loss = train_loss / len(train_loader)

            # 검증
            model.eval()
            val_loss, correct, total = 0.0, 0, 0
            all_preds, all_targets = [], []
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f"[{fold_label}][Epoch {epoch}/{self.epochs}] Validation")
                for images, targets in val_bar:
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
                "fold": fold_label,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "Train_Data": len(train_df),
                "Val_Data": len(val_df),
                "scheduler_lr": current_lr     
            })

            # EarlyStopping
            if val_f1 > best_f1:
                best_f1, trigger = val_f1, 0
                os.makedirs(self.save_dir, exist_ok=True)
                model_save_name = f"model_{fold_label.replace(' ', '_')}.pth"
                torch.save(model.state_dict(), os.path.join(self.save_dir, model_save_name))
            else:
                trigger += 1
                if trigger >= self.patience:
                    print(f"[{fold_label}] Early stopping. Best F1: {best_f1:.4f}")
                    break

            wandb.log({"best_val_f1": best_f1})

        wandb.finish()
        return best_f1