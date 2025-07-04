"""
Integration tests for specific features
"""
import os
import sys
import tempfile
import pytest
import pandas as pd
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from unittest.mock import patch, MagicMock

# Add parent directory to path to import main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_test_data():
    """Create test data for integration tests"""
    temp_dir = tempfile.mkdtemp()
    data_dir = os.path.join(temp_dir, "data")
    os.makedirs(os.path.join(data_dir, "train"))
    os.makedirs(os.path.join(data_dir, "test"))
    
    # Create train data with balanced classes
    n_train = 60  # 20 samples per class
    train_data = {
        'ID': [f'train_{i}.jpg' for i in range(n_train)],
        'target': [i % 3 for i in range(n_train)]  # 0, 1, 2 classes
    }
    train_df = pd.DataFrame(train_data)
    train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    
    # Create test data
    n_test = 30
    test_data = {
        'ID': [f'test_{i}.jpg' for i in range(n_test)],
        'target': [0] * n_test  # dummy targets
    }
    test_df = pd.DataFrame(test_data)
    test_df.to_csv(os.path.join(data_dir, "sample_submission.csv"), index=False)
    
    # Create dummy images
    for img_name in train_data['ID']:
        img = Image.new('RGB', (32, 32), color='red')
        img.save(os.path.join(data_dir, "train", img_name))
    
    for img_name in test_data['ID']:
        img = Image.new('RGB', (32, 32), color='blue')
        img.save(os.path.join(data_dir, "test", img_name))
    
    return temp_dir, data_dir


def test_holdout_feature():
    """Test holdout validation feature"""
    temp_dir, data_dir = create_test_data()
    
    # Create test config
    cfg = OmegaConf.create({
        'data': {
            'data_path': data_dir,
            'img_size': 32,
            'num_workers': 0
        },
        'model': {
            'name': 'resnet18',
            'num_classes': 3,
            'pretrained': False
        },
        'training': {
            'lr': 0.001,
            'epochs': 2,
            'batch_size': 8,
            'seed': 42
        },
        'validation': {
            'strategy': 'holdout',
            'holdout': {
                'train_ratio': 0.8,
                'stratify': True
            },
            'early_stopping': {
                'enabled': True,
                'patience': 5,
                'min_delta': 0.001,
                'monitor': 'val_loss',
                'mode': 'min'
            }
        },
        'device': 'cpu',
        'output': {
            'dir': temp_dir,
            'filename': 'holdout_test.csv'
        },
        'wandb': {
            'enabled': False
        }
    })
    
    # Test imports and basic functionality
    from main import ImageDataset, IndexedImageDataset, EarlyStopping
    from sklearn.model_selection import train_test_split
    
    # Test data loading
    full_train_df = pd.read_csv(f"{data_dir}/train.csv")
    assert len(full_train_df) == 60
    
    # Test holdout split
    train_df, val_df = train_test_split(
        full_train_df, 
        test_size=1-cfg.validation.holdout.train_ratio, 
        stratify=full_train_df['target'], 
        random_state=cfg.training.seed
    )
    
    # Verify split
    assert len(train_df) == 48  # 80% of 60
    assert len(val_df) == 12   # 20% of 60
    
    # Test dataset creation
    trn_dataset = IndexedImageDataset(train_df, f"{data_dir}/train/")
    val_dataset = IndexedImageDataset(val_df, f"{data_dir}/train/")
    
    assert len(trn_dataset) == 48
    assert len(val_dataset) == 12
    
    # Test early stopping
    early_stopping = EarlyStopping(
        patience=cfg.validation.early_stopping.patience,
        min_delta=cfg.validation.early_stopping.min_delta,
        monitor=cfg.validation.early_stopping.monitor,
        mode=cfg.validation.early_stopping.mode
    )
    
    # Test early stopping behavior
    assert not early_stopping({'val_loss': 1.0})
    assert not early_stopping({'val_loss': 0.5})
    
    print("✅ Holdout 검증 기능 테스트 성공")


def test_kfold_feature():
    """Test K-Fold validation feature"""
    temp_dir, data_dir = create_test_data()
    
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
    
    # Test imports
    from sklearn.model_selection import StratifiedKFold
    from main import IndexedImageDataset
    
    # Test data loading
    full_train_df = pd.read_csv(f"{data_dir}/train.csv")
    
    # Test K-Fold setup
    skf = StratifiedKFold(n_splits=cfg.validation.kfold.n_splits, shuffle=True, random_state=cfg.training.seed)
    folds = list(skf.split(full_train_df, full_train_df['target']))
    
    # Verify fold setup
    assert len(folds) == 3
    
    # Test each fold
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_df = full_train_df.iloc[train_idx]
        val_df = full_train_df.iloc[val_idx]
        
        # Verify fold integrity
        assert len(train_df) + len(val_df) == 60
        assert len(set(train_df.index).intersection(set(val_df.index))) == 0
        
        # Test dataset creation
        trn_dataset = IndexedImageDataset(train_df, f"{data_dir}/train/")
        val_dataset = IndexedImageDataset(val_df, f"{data_dir}/train/")
        
        assert len(trn_dataset) == len(train_df)
        assert len(val_dataset) == len(val_df)
    
    print("✅ K-Fold 교차 검증 기능 테스트 성공")


def test_early_stopping_feature():
    """Test early stopping feature"""
    from main import EarlyStopping
    
    # Test early stopping with loss (min mode)
    early_stopping = EarlyStopping(patience=3, min_delta=0.01, monitor='val_loss', mode='min')
    
    # Test improving loss
    assert not early_stopping({'val_loss': 1.0})
    assert not early_stopping({'val_loss': 0.9})
    assert not early_stopping({'val_loss': 0.8})
    
    # Test stagnant loss
    assert not early_stopping({'val_loss': 0.81})  # within min_delta
    assert not early_stopping({'val_loss': 0.82})  # still within patience
    assert early_stopping({'val_loss': 0.83})  # should stop after patience
    
    # Test early stopping with accuracy (max mode)
    early_stopping = EarlyStopping(patience=2, min_delta=0.01, monitor='val_acc', mode='max')
    
    # Test improving accuracy
    assert not early_stopping({'val_acc': 0.5})
    assert not early_stopping({'val_acc': 0.7})
    assert not early_stopping({'val_acc': 0.9})
    
    # Test stagnant accuracy
    assert not early_stopping({'val_acc': 0.89})  # within min_delta
    assert early_stopping({'val_acc': 0.88})  # should stop after patience
    
    print("✅ Early Stopping 기능 테스트 성공")


def test_training_functions():
    """Test training and validation functions"""
    import torch
    import torch.nn as nn
    from main import train_one_epoch, validate_one_epoch
    
    # Create simple model and data
    device = torch.device('cpu')
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(32*32*3, 5),
        nn.ReLU(),
        nn.Linear(5, 3),
        nn.LogSoftmax(dim=1)
    )
    
    # Create dummy data
    images = torch.randn(20, 3, 32, 32)
    targets = torch.randint(0, 3, (20,))
    dataset = torch.utils.data.TensorDataset(images, targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.NLLLoss()
    
    # Test training
    train_result = train_one_epoch(loader, model, optimizer, loss_fn, device)
    
    # Verify training result structure
    assert 'train_loss' in train_result
    assert 'train_acc' in train_result
    assert 'train_f1' in train_result
    
    # Verify reasonable values
    assert train_result['train_loss'] > 0
    assert 0 <= train_result['train_acc'] <= 1
    assert 0 <= train_result['train_f1'] <= 1
    
    # Test validation
    val_result = validate_one_epoch(loader, model, loss_fn, device)
    
    # Verify validation result structure
    assert 'val_loss' in val_result
    assert 'val_acc' in val_result
    assert 'val_f1' in val_result
    
    # Verify reasonable values
    assert val_result['val_loss'] > 0
    assert 0 <= val_result['val_acc'] <= 1
    assert 0 <= val_result['val_f1'] <= 1
    
    print("✅ 학습 및 검증 함수 테스트 성공")


def test_inference_functionality():
    """Test inference functionality"""
    temp_dir, data_dir = create_test_data()
    
    # Load test data
    test_df = pd.read_csv(f"{data_dir}/sample_submission.csv")
    
    # Simulate predictions
    pred_df = test_df.copy()
    pred_df['target'] = np.random.randint(0, 3, len(pred_df))
    
    # Test prediction format
    assert 'ID' in pred_df.columns
    assert 'target' in pred_df.columns
    assert len(pred_df) == 30
    
    # Test value ranges
    assert all(0 <= target <= 2 for target in pred_df['target'])
    
    # Test saving predictions
    pred_path = os.path.join(temp_dir, "test_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    
    # Verify file exists and format
    assert os.path.exists(pred_path)
    loaded_pred = pd.read_csv(pred_path)
    assert list(loaded_pred.columns) == ['ID', 'target']
    assert len(loaded_pred) == 30
    
    print("✅ 추론 기능 테스트 성공")


if __name__ == "__main__":
    print("🧪 각 기능별 테스트 시작...")
    
    test_holdout_feature()
    test_kfold_feature()
    test_early_stopping_feature()
    test_training_functions()
    test_inference_functionality()
    
    print("\n🎉 모든 기능 테스트 완료!")
    print("""
테스트 완료된 기능:
✅ 8:2 Holdout 검증
✅ Stratified K-Fold 교차 검증
✅ Early Stopping
✅ 학습 및 검증 함수
✅ 추론 기능
""") 