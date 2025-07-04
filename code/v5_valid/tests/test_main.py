"""
Test cases for main.py functionality
"""
import pytest
import torch
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import patch, MagicMock
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold

# Import functions and classes from main.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    ImageDataset, 
    IndexedImageDataset, 
    EarlyStopping, 
    train_one_epoch, 
    validate_one_epoch
)


class TestImageDataset:
    """Test ImageDataset class"""
    
    def setup_method(self):
        # Create temporary test data
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, "test.csv")
        self.img_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(self.img_dir)
        
        # Create test CSV
        test_data = {
            'ID': ['img1.jpg', 'img2.jpg', 'img3.jpg'],
            'target': [0, 1, 2]
        }
        df = pd.DataFrame(test_data)
        df.to_csv(self.csv_path, index=False)
        
        # Create dummy images
        for img_name in test_data['ID']:
            img = Image.new('RGB', (32, 32), color='red')
            img.save(os.path.join(self.img_dir, img_name))
    
    def test_image_dataset_csv_path(self):
        """Test ImageDataset with CSV path"""
        dataset = ImageDataset(self.csv_path, self.img_dir)
        assert len(dataset) == 3
        
        # Test __getitem__
        img, target = dataset[0]
        assert isinstance(img, np.ndarray)
        assert target == 0
    
    def test_image_dataset_dataframe(self):
        """Test ImageDataset with DataFrame"""
        df = pd.read_csv(self.csv_path)
        dataset = ImageDataset(df, self.img_dir)
        assert len(dataset) == 3
        
        img, target = dataset[0]
        assert isinstance(img, np.ndarray)
        assert target == 0


class TestIndexedImageDataset:
    """Test IndexedImageDataset class"""
    
    def setup_method(self):
        # Create test DataFrame
        self.df = pd.DataFrame({
            'ID': ['img1.jpg', 'img2.jpg', 'img3.jpg'],
            'target': [0, 1, 2]
        })
        
        # Create temporary image directory
        self.temp_dir = tempfile.mkdtemp()
        self.img_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(self.img_dir)
        
        # Create dummy images
        for img_name in self.df['ID']:
            img = Image.new('RGB', (32, 32), color='red')
            img.save(os.path.join(self.img_dir, img_name))
    
    def test_indexed_image_dataset(self):
        """Test IndexedImageDataset"""
        dataset = IndexedImageDataset(self.df, self.img_dir)
        assert len(dataset) == 3
        
        img, target = dataset[0]
        assert isinstance(img, np.ndarray)
        assert target == 0


class TestEarlyStopping:
    """Test EarlyStopping class"""
    
    def test_early_stopping_min_mode(self):
        """Test early stopping with min mode (for loss)"""
        early_stopping = EarlyStopping(patience=2, min_delta=0.01, monitor='val_loss', mode='min')
        
        # Test improving loss
        assert not early_stopping({'val_loss': 1.0})
        assert not early_stopping({'val_loss': 0.9})
        assert not early_stopping({'val_loss': 0.8})
        
        # Test non-improving loss
        assert not early_stopping({'val_loss': 0.82})  # within min_delta
        assert early_stopping({'val_loss': 0.83})  # should stop after patience
    
    def test_early_stopping_max_mode(self):
        """Test early stopping with max mode (for accuracy)"""
        early_stopping = EarlyStopping(patience=2, min_delta=0.01, monitor='val_acc', mode='max')
        
        # Test improving accuracy
        assert not early_stopping({'val_acc': 0.8})
        assert not early_stopping({'val_acc': 0.9})
        assert not early_stopping({'val_acc': 0.95})
        
        # Test non-improving accuracy
        assert not early_stopping({'val_acc': 0.94})  # within min_delta
        assert early_stopping({'val_acc': 0.93})  # should stop after patience


class TestTrainingFunctions:
    """Test training and validation functions"""
    
    def setup_method(self):
        # Create dummy model and data
        self.device = torch.device('cpu')
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32*32*3, 10),
            torch.nn.LogSoftmax(dim=1)
        )
        
        # Create dummy data
        self.images = torch.randn(10, 3, 32, 32)
        self.targets = torch.randint(0, 10, (10,))
        self.dataset = torch.utils.data.TensorDataset(self.images, self.targets)
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=2)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = torch.nn.NLLLoss()
    
    def test_train_one_epoch(self):
        """Test train_one_epoch function"""
        result = train_one_epoch(self.loader, self.model, self.optimizer, self.loss_fn, self.device)
        
        # Check return structure
        assert 'train_loss' in result
        assert 'train_acc' in result
        assert 'train_f1' in result
        
        # Check value types
        assert isinstance(result['train_loss'], float)
        assert isinstance(result['train_acc'], float)
        assert isinstance(result['train_f1'], float)
        
        # Check reasonable ranges
        assert result['train_loss'] >= 0
        assert 0 <= result['train_acc'] <= 1
        assert 0 <= result['train_f1'] <= 1
    
    def test_validate_one_epoch(self):
        """Test validate_one_epoch function"""
        result = validate_one_epoch(self.loader, self.model, self.loss_fn, self.device)
        
        # Check return structure
        assert 'val_loss' in result
        assert 'val_acc' in result
        assert 'val_f1' in result
        
        # Check value types
        assert isinstance(result['val_loss'], float)
        assert isinstance(result['val_acc'], float)
        assert isinstance(result['val_f1'], float)
        
        # Check reasonable ranges
        assert result['val_loss'] >= 0
        assert 0 <= result['val_acc'] <= 1
        assert 0 <= result['val_f1'] <= 1


class TestValidationStrategies:
    """Test validation strategies"""
    
    def setup_method(self):
        # Create sample data
        np.random.seed(42)
        self.n_samples = 100
        self.n_classes = 5
        
        # Create balanced dataset
        self.df = pd.DataFrame({
            'ID': [f'img_{i}.jpg' for i in range(self.n_samples)],
            'target': np.random.randint(0, self.n_classes, self.n_samples)
        })
    
    def test_holdout_split(self):
        """Test holdout data split"""
        train_ratio = 0.8
        train_df, val_df = train_test_split(
            self.df, 
            test_size=1-train_ratio, 
            stratify=self.df['target'], 
            random_state=42
        )
        
        # Check split ratios
        assert len(train_df) == int(self.n_samples * train_ratio)
        assert len(val_df) == self.n_samples - int(self.n_samples * train_ratio)
        
        # Check no overlap
        train_ids = set(train_df['ID'].tolist())
        val_ids = set(val_df['ID'].tolist())
        assert len(train_ids.intersection(val_ids)) == 0
    
    def test_kfold_split(self):
        """Test K-Fold data split"""
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        folds = list(skf.split(self.df, self.df['target']))
        
        # Check number of folds
        assert len(folds) == n_splits
        
        # Check fold sizes
        for train_idx, val_idx in folds:
            # Check no overlap
            assert len(set(train_idx).intersection(set(val_idx))) == 0
            
            # Check sizes
            assert len(train_idx) + len(val_idx) == self.n_samples
            assert len(val_idx) == self.n_samples // n_splits or len(val_idx) == self.n_samples // n_splits + 1
        
        # Check all samples are used
        all_val_indices = set()
        for train_idx, val_idx in folds:
            all_val_indices.update(val_idx)
        
        assert len(all_val_indices) == self.n_samples


class TestPipelineComponents:
    """Test pipeline components"""
    
    def setup_method(self):
        # Create minimal test data
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "data")
        os.makedirs(os.path.join(self.data_dir, "train"))
        os.makedirs(os.path.join(self.data_dir, "test"))
        
        # Create train data
        n_train = 20
        train_data = {
            'ID': [f'train_{i}.jpg' for i in range(n_train)],
            'target': np.random.randint(0, 3, n_train)
        }
        train_df = pd.DataFrame(train_data)
        train_df.to_csv(os.path.join(self.data_dir, "train.csv"), index=False)
        
        # Create test data
        n_test = 10
        test_data = {
            'ID': [f'test_{i}.jpg' for i in range(n_test)],
            'target': [0] * n_test  # dummy targets
        }
        test_df = pd.DataFrame(test_data)
        test_df.to_csv(os.path.join(self.data_dir, "sample_submission.csv"), index=False)
        
        # Create dummy images
        for img_name in train_data['ID']:
            img = Image.new('RGB', (32, 32), color='red')
            img.save(os.path.join(self.data_dir, "train", img_name))
        
        for img_name in test_data['ID']:
            img = Image.new('RGB', (32, 32), color='blue')
            img.save(os.path.join(self.data_dir, "test", img_name))
    
    def test_data_loading(self):
        """Test data loading"""
        # Test train data loading
        train_df = pd.read_csv(f"{self.data_dir}/train.csv")
        assert len(train_df) == 20
        assert 'ID' in train_df.columns
        assert 'target' in train_df.columns
        
        # Test test data loading
        test_df = pd.read_csv(f"{self.data_dir}/sample_submission.csv")
        assert len(test_df) == 10
        assert 'ID' in test_df.columns
        assert 'target' in test_df.columns
    
    def test_holdout_validation_setup(self):
        """Test holdout validation setup"""
        from omegaconf import OmegaConf
        
        # Create test config
        cfg = OmegaConf.create({
            'validation': {
                'strategy': 'holdout',
                'holdout': {
                    'train_ratio': 0.8,
                    'stratify': True
                }
            },
            'training': {
                'seed': 42
            }
        })
        
        # Load and split data
        full_train_df = pd.read_csv(f"{self.data_dir}/train.csv")
        
        if cfg.validation.holdout.stratify:
            train_df, val_df = train_test_split(
                full_train_df, 
                test_size=1-cfg.validation.holdout.train_ratio, 
                stratify=full_train_df['target'], 
                random_state=cfg.training.seed
            )
        else:
            train_df, val_df = train_test_split(
                full_train_df, 
                test_size=1-cfg.validation.holdout.train_ratio, 
                random_state=cfg.training.seed
            )
        
        # Test split results
        assert len(train_df) + len(val_df) == len(full_train_df)
        assert len(train_df) == int(len(full_train_df) * cfg.validation.holdout.train_ratio)
        
        # Test no overlap
        train_ids = set(train_df['ID'].tolist())
        val_ids = set(val_df['ID'].tolist())
        assert len(train_ids.intersection(val_ids)) == 0
    
    def test_kfold_validation_setup(self):
        """Test K-Fold validation setup"""
        from omegaconf import OmegaConf
        
        # Create test config
        cfg = OmegaConf.create({
            'validation': {
                'strategy': 'kfold',
                'kfold': {
                    'n_splits': 3,
                    'stratify': True
                }
            },
            'training': {
                'seed': 42
            }
        })
        
        # Load data
        full_train_df = pd.read_csv(f"{self.data_dir}/train.csv")
        
        # Setup K-Fold
        if cfg.validation.kfold.stratify:
            skf = StratifiedKFold(n_splits=cfg.validation.kfold.n_splits, shuffle=True, random_state=cfg.training.seed)
            folds = list(skf.split(full_train_df, full_train_df['target']))
        else:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=cfg.validation.kfold.n_splits, shuffle=True, random_state=cfg.training.seed)
            folds = list(kf.split(full_train_df))
        
        # Test fold results
        assert len(folds) == cfg.validation.kfold.n_splits
        
        # Test each fold
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            train_df = full_train_df.iloc[train_idx]
            val_df = full_train_df.iloc[val_idx]
            
            # Check split integrity
            assert len(train_df) + len(val_df) == len(full_train_df)
            assert len(set(train_df.index).intersection(set(val_df.index))) == 0
    
    def test_predictions_format(self):
        """Test predictions format"""
        # Create dummy predictions
        test_df = pd.read_csv(f"{self.data_dir}/sample_submission.csv")
        
        # Simulate predictions
        pred_df = test_df.copy()
        pred_df['target'] = np.random.randint(0, 3, len(pred_df))
        
        # Test format
        assert 'ID' in pred_df.columns
        assert 'target' in pred_df.columns
        assert len(pred_df) == 10
        
        # Test value ranges
        assert all(0 <= target <= 2 for target in pred_df['target'])
        
        # Test saving and loading
        pred_path = f"{self.temp_dir}/test_predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        
        # Verify file exists and format
        assert os.path.exists(pred_path)
        loaded_pred = pd.read_csv(pred_path)
        assert list(loaded_pred.columns) == ['ID', 'target']
        assert len(loaded_pred) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 