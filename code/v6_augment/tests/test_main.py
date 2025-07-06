# -*- coding: utf-8 -*-
"""
main.py 모듈 통합 테스트
전체 파이프라인과 모듈 간 통합을 테스트
"""
import os
import sys
import pytest
import tempfile
import pandas as pd
from PIL import Image
from omegaconf import OmegaConf
from unittest.mock import patch, MagicMock

# 상위 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 분리된 모듈들 import
from main import main
from data import prepare_data_loaders
from training import train_single_model, train_kfold_models
from inference import run_inference
from utils import set_seed, get_device
from models import setup_model_and_optimizer


class TestMainPipeline:
    """메인 파이프라인 통합 테스트"""
    
    def setup_method(self):
        """테스트 데이터 준비"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "data")
        os.makedirs(os.path.join(self.data_dir, "train"))
        os.makedirs(os.path.join(self.data_dir, "test"))
        
        # 훈련 데이터 생성 (30개 샘플, 3개 클래스)
        n_train = 30
        train_data = {
            'ID': [f'train_{i}.jpg' for i in range(n_train)],
            'target': [i % 3 for i in range(n_train)]
        }
        train_df = pd.DataFrame(train_data)
        train_df.to_csv(os.path.join(self.data_dir, "train.csv"), index=False)
        
        # 테스트 데이터 생성
        n_test = 12
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
        
        # 출력 디렉토리 생성
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.output_dir)
    
    def create_test_config(self, validation_strategy='holdout'):
        """테스트용 설정 생성"""
        cfg = OmegaConf.create({
            'data': {
                'data_path': self.data_dir,
                'img_size': 32,
                'num_workers': 0
            },
            'model': {
                'name': 'resnet18',
                'pretrained': False,
                'num_classes': 3
            },
            'training': {
                'lr': 0.01,
                'epochs': 1,
                'batch_size': 4,
                'seed': 42
            },
            'validation': {
                'strategy': validation_strategy,
                'holdout': {
                    'train_ratio': 0.8,
                    'stratify': True
                },
                'kfold': {
                    'n_splits': 2,
                    'stratify': True
                },
                'early_stopping': {
                    'enabled': False
                }
            },
            'device': 'cpu',
            'output': {
                'dir': self.output_dir,
                'filename': 'test_predictions.csv'
            },
            'wandb': {
                'enabled': False
            }
        })
        return cfg
    
    @patch('main.setup_wandb')
    @patch('main.finish_wandb')
    @patch('main.log')
    def test_main_pipeline_holdout(self, mock_log, mock_finish_wandb, mock_setup_wandb):
        """Holdout 검증 메인 파이프라인 테스트"""
        # 설정 생성
        cfg = self.create_test_config('holdout')
        
        # 설정 파일 생성
        config_dir = os.path.join(self.temp_dir, "config")
        os.makedirs(config_dir)
        config_path = os.path.join(config_dir, "test_config.yaml")
        
        with open(config_path, 'w') as f:
            f.write(f"""
data:
  data_path: {self.data_dir}
  img_size: 32
  num_workers: 0
model:
  name: resnet18
  pretrained: false
  num_classes: 3
training:
  lr: 0.01
  epochs: 1
  batch_size: 4
  seed: 42
validation:
  strategy: holdout
  holdout:
    train_ratio: 0.8
    stratify: true
  early_stopping:
    enabled: false
device: cpu
output:
  dir: {self.output_dir}
  filename: test_predictions.csv
wandb:
  enabled: false
""")
        
        # 현재 디렉토리 변경
        original_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            
            # Hydra를 사용한 메인 함수 실행
            with patch('hydra.main') as mock_hydra:
                # Mock 설정
                mock_hydra.return_value = lambda func: func
                
                # 직접 함수 호출로 테스트
                with patch('main.prepare_data_loaders') as mock_prepare_data:
                    with patch('main.train_single_model') as mock_train:
                        with patch('main.run_inference') as mock_inference:
                            # Mock 반환값 설정
                            mock_prepare_data.return_value = (
                                MagicMock(), MagicMock(), MagicMock(), None
                            )
                            mock_train.return_value = MagicMock()
                            mock_inference.return_value = pd.DataFrame({
                                'ID': ['test_0.jpg', 'test_1.jpg'],
                                'target': [0, 1]
                            })
                            
                            # 메인 함수 실행
                            main(cfg)
                            
                            # 각 단계가 호출되었는지 확인
                            mock_prepare_data.assert_called_once()
                            mock_train.assert_called_once()
                            mock_inference.assert_called_once()
                            
                            # wandb 함수들이 호출되었는지 확인
                            mock_setup_wandb.assert_called_once()
                            mock_finish_wandb.assert_called_once()
        finally:
            os.chdir(original_cwd)
    
    @patch('main.setup_wandb')
    @patch('main.finish_wandb')
    @patch('main.log')
    def test_main_pipeline_kfold(self, mock_log, mock_finish_wandb, mock_setup_wandb):
        """K-Fold 검증 메인 파이프라인 테스트"""
        cfg = self.create_test_config('kfold')
        
        with patch('main.prepare_data_loaders') as mock_prepare_data:
            with patch('main.train_kfold_models') as mock_train:
                with patch('main.run_inference') as mock_inference:
                    # Mock 반환값 설정
                    mock_prepare_data.return_value = (
                        None, None, MagicMock(), MagicMock()  # kfold_data 있음
                    )
                    mock_train.return_value = [MagicMock(), MagicMock()]
                    mock_inference.return_value = pd.DataFrame({
                        'ID': ['test_0.jpg', 'test_1.jpg'],
                        'target': [0, 1]
                    })
                    
                    # 메인 함수 실행
                    main(cfg)
                    
                    # K-Fold 관련 함수들이 호출되었는지 확인
                    mock_prepare_data.assert_called_once()
                    mock_train.assert_called_once()
                    mock_inference.assert_called_once()
                    
                    # K-Fold 추론인지 확인
                    call_args = mock_inference.call_args
                    assert call_args[1]['is_kfold'] is True
    
    @patch('main.setup_wandb')
    @patch('main.finish_wandb')
    @patch('main.log')
    def test_main_pipeline_no_validation(self, mock_log, mock_finish_wandb, mock_setup_wandb):
        """검증 없는 메인 파이프라인 테스트"""
        cfg = self.create_test_config('none')
        
        with patch('main.prepare_data_loaders') as mock_prepare_data:
            with patch('main.train_single_model') as mock_train:
                with patch('main.run_inference') as mock_inference:
                    # Mock 반환값 설정
                    mock_prepare_data.return_value = (
                        MagicMock(), None, MagicMock(), None  # val_loader가 None
                    )
                    mock_train.return_value = MagicMock()
                    mock_inference.return_value = pd.DataFrame({
                        'ID': ['test_0.jpg'],
                        'target': [0]
                    })
                    
                    # 메인 함수 실행
                    main(cfg)
                    
                    # 검증 없이 학습되었는지 확인
                    train_call_args = mock_train.call_args
                    assert train_call_args[0][2] is None  # val_loader가 None


class TestModuleIntegration:
    """모듈 간 통합 테스트"""
    
    def setup_method(self):
        """테스트 데이터 준비"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "data")
        os.makedirs(os.path.join(self.data_dir, "train"))
        os.makedirs(os.path.join(self.data_dir, "test"))
        
        # 작은 데이터셋 생성
        n_train = 20
        train_data = {
            'ID': [f'train_{i}.jpg' for i in range(n_train)],
            'target': [i % 3 for i in range(n_train)]
        }
        train_df = pd.DataFrame(train_data)
        train_df.to_csv(os.path.join(self.data_dir, "train.csv"), index=False)
        
        n_test = 8
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
    
        # 설정
        self.cfg = OmegaConf.create({
            'data': {
                'data_path': self.data_dir,
                'img_size': 32,
                'num_workers': 0
            },
            'model': {
                'name': 'resnet18',
                'pretrained': False,
                'num_classes': 3
            },
            'training': {
                'lr': 0.01,
                'epochs': 1,
                'batch_size': 4,
                'seed': 42
            },
            'validation': {
                'strategy': 'holdout',
                'holdout': {
                    'train_ratio': 0.8,
                    'stratify': True
                },
                'kfold': {
                    'n_splits': 3,
                    'shuffle': True,
                    'stratify': True
                },
                'early_stopping': {
                    'enabled': False
                }
            },
            'device': 'cpu',
            'output': {
                'dir': self.temp_dir,
                'filename': 'integration_test.csv'
            },
            'wandb': {
                'enabled': False
            }
        })
    
    def test_data_to_training_integration(self):
        """데이터 준비 → 학습 모듈 통합 테스트"""
        # 시드 설정
        set_seed(self.cfg.training.seed)
        
        # 데이터 준비
        train_loader, val_loader, test_loader, kfold_data = prepare_data_loaders(
            self.cfg, self.cfg.training.seed
        )
        
        # 데이터 로더 검증
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        assert kfold_data is None  # holdout 모드
        
        # 디바이스 설정
        device = get_device(self.cfg)
        
        # 모델 및 옵티마이저 설정
        model, optimizer, loss_fn, _ = setup_model_and_optimizer(self.cfg, device)
        
        # 학습 실행
        trained_model = train_single_model(self.cfg, train_loader, val_loader, device)
        
        # 학습된 모델 검증
        assert trained_model is not None
        assert hasattr(trained_model, 'forward')
    
    def test_training_to_inference_integration(self):
        """학습 → 추론 모듈 통합 테스트"""
        # 시드 설정
        set_seed(self.cfg.training.seed)
        
        # 데이터 준비
        train_loader, val_loader, test_loader, _ = prepare_data_loaders(
            self.cfg, self.cfg.training.seed
        )
        
        # 디바이스 설정
        device = get_device(self.cfg)
        
        # 학습
        trained_model = train_single_model(self.cfg, train_loader, val_loader, device)
        
        # 추론 실행
        result_df = run_inference(
            trained_model, test_loader, test_loader.dataset, 
            self.cfg, device, is_kfold=False
        )
        
        # 추론 결과 검증
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 8  # 테스트 데이터 수
        assert 'ID' in result_df.columns
        assert 'target' in result_df.columns
        
        # 예측값이 유효한 클래스인지 확인
        for pred in result_df['target']:
            assert 0 <= pred < 3
    
    def test_full_pipeline_integration(self):
        """전체 파이프라인 통합 테스트"""
        # 시드 설정
        set_seed(self.cfg.training.seed)
        
        # 1. 데이터 준비
        train_loader, val_loader, test_loader, kfold_data = prepare_data_loaders(
            self.cfg, self.cfg.training.seed
        )
        
        # 2. 디바이스 설정
        device = get_device(self.cfg)
        
        # 3. 학습
        trained_model = train_single_model(self.cfg, train_loader, val_loader, device)
        
        # 4. 추론
        result_df = run_inference(
            trained_model, test_loader, test_loader.dataset, 
            self.cfg, device, is_kfold=False
        )
        
        # 5. 결과 파일 확인
        output_path = os.path.join(self.temp_dir, 'integration_test.csv')
        assert os.path.exists(output_path)
        
        # 6. 파일 내용 검증
        saved_df = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(result_df, saved_df)
    
    def test_kfold_pipeline_integration(self):
        """K-Fold 파이프라인 통합 테스트"""
        # K-Fold 설정으로 변경
        self.cfg.validation.strategy = 'kfold'
        self.cfg.validation.kfold.n_splits = 2
        
        # 시드 설정
        set_seed(self.cfg.training.seed)
        
        # 1. 데이터 준비
        train_loader, val_loader, test_loader, kfold_data = prepare_data_loaders(
            self.cfg, self.cfg.training.seed
        )
        
        # K-Fold 데이터 검증
        assert train_loader is None
        assert val_loader is None
        assert test_loader is not None
        assert kfold_data is not None
        
        # 2. 디바이스 설정
        device = get_device(self.cfg)
        
        # 3. K-Fold 학습
        trained_models = train_kfold_models(self.cfg, kfold_data, device)
        
        # 4. 앙상블 추론
        result_df = run_inference(
            trained_models, test_loader, test_loader.dataset, 
            self.cfg, device, is_kfold=True
        )
        
        # 5. 결과 검증
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 8
        
        # 6. 결과 파일 확인
        output_path = os.path.join(self.temp_dir, 'integration_test.csv')
        assert os.path.exists(output_path)


class TestErrorHandling:
    """오류 처리 테스트"""
    
    def test_invalid_config_handling(self):
        """잘못된 설정 처리 테스트"""
        # 잘못된 모델 이름
        with pytest.raises(Exception):
            cfg = OmegaConf.create({
                'model': {'name': 'invalid_model'},
                'training': {'lr': 0.01},
                'device': 'cpu'
            })
            device = get_device(cfg)
            setup_model_and_optimizer(cfg, device)
    
    def test_missing_data_handling(self):
        """데이터 누락 처리 테스트"""
        temp_dir = tempfile.mkdtemp()
        
        # 데이터 파일이 없는 경우
        cfg = OmegaConf.create({
            'data': {
                'data_path': temp_dir,
                'img_size': 32,
                'num_workers': 0
            },
            'training': {'batch_size': 4, 'seed': 42},
            'validation': {'strategy': 'holdout'}
        })
        
        with pytest.raises(FileNotFoundError):
            prepare_data_loaders(cfg, 42)


if __name__ == "__main__":
    pytest.main([__file__]) 