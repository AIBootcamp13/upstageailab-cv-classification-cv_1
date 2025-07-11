# -*- coding: utf-8 -*-
"""
data.py 모듈 테스트
사용자 정의 데이터셋 클래스와 데이터 로더 준비 함수들을 테스트
"""
import os
import sys
import pytest
import tempfile
import pandas as pd
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import torch

# 상위 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import (
    ImageDataset,
    IndexedImageDataset,
    get_transforms,
    prepare_data_loaders,
    get_kfold_loaders
)


class TestImageDataset:
    """ImageDataset 클래스 테스트"""
    
    def setup_method(self):
        """테스트 데이터 준비"""
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, "test.csv")
        self.img_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(self.img_dir)
        
        # 테스트용 CSV 생성
        test_data = {
            'ID': ['img1.jpg', 'img2.jpg', 'img3.jpg'],
            'target': [0, 1, 2]
        }
        df = pd.DataFrame(test_data)
        df.to_csv(self.csv_path, index=False)
        
        # 더미 이미지 생성
        for img_name in test_data['ID']:
            img = Image.new('RGB', (32, 32), color='red')
            img.save(os.path.join(self.img_dir, img_name))
    
    def test_dataset_with_csv_path(self):
        """CSV 경로로 데이터셋 생성 테스트"""
        dataset = ImageDataset(self.csv_path, self.img_dir)
        assert len(dataset) == 3
        
        # 첫 번째 아이템 확인
        img, target = dataset[0]
        assert isinstance(img, np.ndarray)
        assert target == 0
        assert img.shape == (32, 32, 3)
    
    def test_dataset_with_dataframe(self):
        """DataFrame으로 데이터셋 생성 테스트"""
        df = pd.read_csv(self.csv_path)
        dataset = ImageDataset(df, self.img_dir)
        assert len(dataset) == 3
        
        # 모든 아이템 확인
        for i in range(len(dataset)):
            img, target = dataset[i]
            assert isinstance(img, np.ndarray)
            assert target == i
    
    def test_dataset_with_transform(self):
        """Transform 적용 테스트"""
        from albumentations import Compose, Resize, Normalize
        from albumentations.pytorch import ToTensorV2
        
        transform = Compose([
            Resize(16, 16),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        dataset = ImageDataset(self.csv_path, self.img_dir, transform=transform)
        img, target = dataset[0]
        
        # Transform 후 텐서 확인
        assert hasattr(img, 'shape')
        assert img.shape == (3, 16, 16)  # C, H, W 형태


class TestIndexedImageDataset:
    """IndexedImageDataset 클래스 테스트"""
    
    def setup_method(self):
        """테스트 데이터 준비"""
        self.temp_dir = tempfile.mkdtemp()
        self.img_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(self.img_dir)
        
        # 테스트용 DataFrame
        self.df = pd.DataFrame({
            'ID': ['img1.jpg', 'img2.jpg', 'img3.jpg'],
            'target': [0, 1, 2]
        })
        
        # 더미 이미지 생성
        for img_name in self.df['ID']:
            img = Image.new('RGB', (32, 32), color='blue')
            img.save(os.path.join(self.img_dir, img_name))
    
    def test_indexed_dataset(self):
        """IndexedImageDataset 기본 동작 테스트"""
        dataset = IndexedImageDataset(self.df, self.img_dir)
        assert len(dataset) == 3
        
        # 각 아이템 확인
        for i in range(len(dataset)):
            img, target = dataset[i]
            assert isinstance(img, np.ndarray)
            assert target == i
            assert img.shape == (32, 32, 3)
    
    def test_indexed_dataset_with_transform(self):
        """Transform 적용 테스트"""
        from albumentations import Compose, Resize
        from albumentations.pytorch import ToTensorV2
        
        transform = Compose([
            Resize(64, 64),
            ToTensorV2()
        ])
        
        dataset = IndexedImageDataset(self.df, self.img_dir, transform=transform)
        img, target = dataset[0]
        
        # Transform 후 텐서 확인
        assert hasattr(img, 'shape')
        assert img.shape == (3, 64, 64)


class TestTransforms:
    """Transform 함수 테스트"""
    
    def test_get_transforms(self):
        """get_transforms 함수 테스트"""
        img_size = 224
        cfg = OmegaConf.create({'data': {'img_size': img_size}})
        train_transform = get_transforms(cfg, 'train')
        test_transform = get_transforms(cfg, 'test')
        
        # Transform 객체 확인
        assert train_transform is not None
        assert test_transform is not None
        
        # 더미 이미지로 테스트
        dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Train transform 적용
        train_result = train_transform(image=dummy_img)
        train_img = train_result['image']
        assert train_img.shape == (3, img_size, img_size)
        
        # Test transform 적용
        test_result = test_transform(image=dummy_img)
        test_img = test_result['image']
        assert test_img.shape == (3, img_size, img_size)


class TestDataLoaderPreparation:
    """데이터 로더 준비 함수 테스트"""
    
    def setup_method(self):
        """테스트 데이터 준비"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "data")
        os.makedirs(os.path.join(self.data_dir, "train"))
        os.makedirs(os.path.join(self.data_dir, "test"))
        
        # 훈련 데이터 생성 (60개 샘플, 3개 클래스)
        n_train = 60
        train_data = {
            'ID': [f'train_{i}.jpg' for i in range(n_train)],
            'target': [i % 3 for i in range(n_train)]
        }
        train_df = pd.DataFrame(train_data)
        train_df.to_csv(os.path.join(self.data_dir, "train.csv"), index=False)
        
        # 테스트 데이터 생성
        n_test = 30
        test_data = {
            'ID': [f'test_{i}.jpg' for i in range(n_test)],
            'target': [0] * n_test
        }
        test_df = pd.DataFrame(test_data)
        test_df.to_csv(os.path.join(self.data_dir, "sample_submission.csv"), index=False)
        
        # 더미 이미지 생성
        for img_name in train_data['ID']:
            img = Image.new('RGB', (32, 32), color='red')
            img.save(os.path.join(self.data_dir, "train", img_name))
        
        for img_name in test_data['ID']:
            img = Image.new('RGB', (32, 32), color='blue')
            img.save(os.path.join(self.data_dir, "test", img_name))
    
    def test_holdout_data_loaders(self):
        """Holdout 검증 데이터 로더 테스트"""
        cfg = OmegaConf.create({
            'data': {
                'train_images_path': os.path.join(self.data_dir, "train"),
                'test_images_path': os.path.join(self.data_dir, "test"),
                'train_csv_path': os.path.join(self.data_dir, "train.csv"),
                'test_csv_path': os.path.join(self.data_dir, "sample_submission.csv"),
                'img_size': 32,
                'num_workers': 0
            },
            'train': {
                'batch_size': 8,
                'seed': 42
            },
            'valid': {
                'strategy': 'holdout',
                'holdout': {
                    'train_ratio': 0.8,
                    'stratify': True
                }
            }
        })
        
        train_loader, val_loader, test_loader, kfold_data = prepare_data_loaders(cfg, 42)
        
        # 반환값 확인
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        assert kfold_data is None
        
        # 데이터 로더 크기 확인
        assert len(train_loader.dataset) == 48  # 80% of 60  # type: ignore
        assert len(val_loader.dataset) == 12   # 20% of 60  # type: ignore
        assert len(test_loader.dataset) == 30  # type: ignore
    
    def test_kfold_data_loaders(self):
        """K-Fold 검증 데이터 로더 테스트"""
        cfg = OmegaConf.create({
            'data': {
                'train_images_path': os.path.join(self.data_dir, "train"),
                'test_images_path': os.path.join(self.data_dir, "test"),
                'train_csv_path': os.path.join(self.data_dir, "train.csv"),
                'test_csv_path': os.path.join(self.data_dir, "sample_submission.csv"),
                'img_size': 32,
                'num_workers': 0
            },
            'train': {
                'batch_size': 8,
                'seed': 42
            },
            'valid': {
                'strategy': 'kfold',
                'kfold': {
                    'n_splits': 3,
                    'stratify': True
                }
            }
        })
        
        train_loader, val_loader, test_loader, kfold_data = prepare_data_loaders(cfg, 42)
        
        # 반환값 확인
        assert train_loader is None
        assert val_loader is None
        assert test_loader is not None
        assert kfold_data is not None
        
        # K-Fold 데이터 확인
        folds, full_train_df, train_images_path, train_transform, val_transform, test_transform, cache_root = kfold_data
        assert len(folds) == 3
        assert len(full_train_df) == 60
        assert train_images_path == os.path.join(self.data_dir, "train")
        assert train_transform is not None
        assert val_transform is not None
        assert test_transform is not None
    
    def test_no_validation_data_loaders(self):
        """No validation 데이터 로더 테스트"""
        cfg = OmegaConf.create({
            'data': {
                'train_images_path': os.path.join(self.data_dir, "train"),
                'test_images_path': os.path.join(self.data_dir, "test"),
                'train_csv_path': os.path.join(self.data_dir, "train.csv"),
                'test_csv_path': os.path.join(self.data_dir, "sample_submission.csv"),
                'img_size': 32,
                'num_workers': 0
            },
            'train': {
                'batch_size': 8,
                'seed': 42
            },
            'valid': {
                'strategy': 'none'
            }
        })
        
        train_loader, val_loader, test_loader, kfold_data = prepare_data_loaders(cfg, 42)
        
        # 반환값 확인
        assert train_loader is not None
        assert val_loader is None
        assert test_loader is not None
        assert kfold_data is None
        
        # 전체 데이터 사용 확인
        assert len(train_loader.dataset) == 60  # 전체 데이터  # type: ignore
        assert len(test_loader.dataset) == 30  # type: ignore


class TestKFoldLoaders:
    """K-Fold 로더 함수 테스트"""
    
    def setup_method(self):
        """테스트 데이터 준비"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "data")
        os.makedirs(os.path.join(self.data_dir, "train"))
        
        # 훈련 데이터 생성
        n_train = 60
        train_data = {
            'ID': [f'train_{i}.jpg' for i in range(n_train)],
            'target': [i % 3 for i in range(n_train)]
        }
        self.full_train_df = pd.DataFrame(train_data)
        
        # 더미 이미지 생성
        for img_name in train_data['ID']:
            img = Image.new('RGB', (32, 32), color='red')
            img.save(os.path.join(self.data_dir, "train", img_name))
        
        # K-Fold 설정
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        self.folds = list(skf.split(self.full_train_df, self.full_train_df['target']))
        
        cfg = OmegaConf.create({'data': {'img_size': 32}})
        self.train_transform = get_transforms(cfg, 'train')
        self.val_transform = get_transforms(cfg, 'valid')
        
        # 설정 객체
        self.cfg = OmegaConf.create({
            'train': {'batch_size': 8, 'seed': 42},
            'data': {'num_workers': 0, 'img_size': 32}
        })
    
    def test_get_kfold_loaders(self):
        """get_kfold_loaders 함수 테스트"""
        fold_idx = 0
        
        train_loader, val_loader, train_df, val_df = get_kfold_loaders(
            fold_idx,
            self.folds,
            self.full_train_df,
            os.path.join(self.data_dir, "train"),
            self.train_transform,
            self.val_transform,
            self.cfg,
            os.path.join(self.data_dir, "cache"),
        )
        
        # 반환값 확인
        assert train_loader is not None
        assert val_loader is not None
        assert train_df is not None
        assert val_df is not None
        
        # 데이터 분할 확인
        assert len(train_df) + len(val_df) == 60
        assert len(set(train_df.index).intersection(set(val_df.index))) == 0
        
        # 데이터 로더 확인
        assert len(train_loader.dataset) == len(train_df)  # type: ignore
        assert len(val_loader.dataset) == len(val_df)  # type: ignore


class TestCachedBasicTransformDataset:
    """CachedBasicTransformDataset 클래스 테스트"""
    
    def setup_method(self):
        """테스트 데이터 준비"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "data")
        os.makedirs(self.data_dir)
        
        # 더미 이미지 생성
        self.img_name = "test.jpg"
        img = Image.new('RGB', (32, 32), color='red')
        img.save(os.path.join(self.data_dir, self.img_name))
        
        # 데이터프레임 생성
        self.df = pd.DataFrame({
            'ID': [self.img_name],
            'target': [0]
        })
        
        # Transform 설정
        from albumentations import Compose, Resize, Normalize
        from albumentations.pytorch import ToTensorV2
        
        self.transform = Compose([
            Resize(16, 16),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # 캐시 디렉토리
        self.cache_root = os.path.join(self.temp_dir, "cache")
        self.seed = 42
        self.img_size = 16
        
        # 기본 데이터셋
        from data import IndexedImageDataset
        self.base_dataset = IndexedImageDataset(self.df, self.data_dir, transform=None, return_filename=True)
    
    def teardown_method(self):
        """테스트 후 정리"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_cached_basic_transform_dataset(self):
        """CachedBasicTransformDataset 기본 기능 테스트"""
        from data import CachedBasicTransformDataset
        
        # 캐시 데이터셋 생성
        cached_dataset = CachedBasicTransformDataset(
            self.base_dataset,
            self.transform,
            self.cache_root,
            self.seed,
            self.img_size
        )
        
        # 길이 확인
        assert len(cached_dataset) == 1
        
        # 첫 번째 호출 - 캐시 파일 생성
        img_tensor, target = cached_dataset[0]
        assert isinstance(img_tensor, torch.Tensor)
        assert img_tensor.shape == (3, 16, 16)  # C, H, W
        assert target == 0
        
        # 캐시 파일 존재 확인
        cache_dir = os.path.join(self.cache_root, f"img{self.img_size}_seed{self.seed}")
        cache_path = os.path.join(cache_dir, "test.pt")
        assert os.path.exists(cache_path)
        
        # 두 번째 호출 - 캐시에서 로드
        img_tensor2, target2 = cached_dataset[0]
        assert torch.equal(img_tensor, img_tensor2)
        assert target2 == 0
    
    def test_cached_basic_transform_dataset_with_existing_cache(self):
        """기존 캐시가 있는 경우 테스트"""
        from data import CachedBasicTransformDataset
        
        # 캐시 디렉토리 미리 생성
        cache_dir = os.path.join(self.cache_root, f"img{self.img_size}_seed{self.seed}")
        os.makedirs(cache_dir, exist_ok=True)
        
        # 더미 캐시 파일 생성
        cache_path = os.path.join(cache_dir, "test.pt")
        dummy_tensor = torch.randn(3, 16, 16)
        torch.save(dummy_tensor, cache_path)
        
        # 캐시 데이터셋 생성
        cached_dataset = CachedBasicTransformDataset(
            self.base_dataset,
            self.transform,
            self.cache_root,
            self.seed,
            self.img_size
        )
        
        # 캐시에서 로드되는지 확인
        img_tensor, target = cached_dataset[0]
        assert torch.equal(img_tensor, dummy_tensor)
        assert target == 0

    def test_cached_basic_transform_dataset_with_memory_cache(self):
        """메모리 캐싱 기능 테스트"""
        from data import CachedBasicTransformDataset
        
        # 메모리 캐싱 활성화된 캐시 데이터셋 생성
        cached_dataset = CachedBasicTransformDataset(
            self.base_dataset,
            self.transform,
            self.cache_root,
            self.seed,
            self.img_size,
            memory_cache=True
        )
        
        # 길이 확인
        assert len(cached_dataset) == 1
        
        # 첫 번째 호출 - 캐시 파일 생성 및 메모리에 저장
        img_tensor, target = cached_dataset[0]
        assert isinstance(img_tensor, torch.Tensor)
        assert img_tensor.shape == (3, 16, 16)  # C, H, W
        assert target == 0
        
        # 캐시 파일 존재 확인
        cache_dir = os.path.join(self.cache_root, f"img{self.img_size}_seed{self.seed}")
        cache_path = os.path.join(cache_dir, "test.pt")
        assert os.path.exists(cache_path)
        
        # 두 번째 호출 - 메모리에서 로드 (파일 I/O 없음)
        img_tensor2, target2 = cached_dataset[0]
        assert torch.equal(img_tensor, img_tensor2)
        assert target2 == 0
        
        # 메모리 캐시에 저장되었는지 확인
        assert 0 in cached_dataset._memory_cache
        assert cached_dataset._memory_cache[0] == (img_tensor, target)


if __name__ == "__main__":
    pytest.main([__file__]) 