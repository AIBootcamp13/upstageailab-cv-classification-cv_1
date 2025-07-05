# -*- coding: utf-8 -*-
"""
training.py 모듈 테스트
사용자 정의 학습 및 검증 함수들을 테스트
"""
import os
import sys
import pytest
import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch

# 상위 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training import (
    train_one_epoch,
    validate_one_epoch,
    train_single_model,
    train_kfold_models
)
from models import setup_model_and_optimizer
from utils import EarlyStopping


class TestTrainOneEpoch:
    """train_one_epoch 함수 테스트"""
    
    def setup_method(self):
        """테스트 준비"""
        self.device = torch.device('cpu')
        
        # 간단한 모델 생성
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32*32*3, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10)
        )
        
        # 더미 데이터 생성
        self.images = torch.randn(20, 3, 32, 32)
        self.targets = torch.randint(0, 10, (20,))
        self.dataset = torch.utils.data.TensorDataset(self.images, self.targets)
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=4, shuffle=True)
        
        # 옵티마이저와 손실 함수
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def test_train_one_epoch_basic(self):
        """기본 학습 함수 테스트"""
        result = train_one_epoch(self.loader, self.model, self.optimizer, self.loss_fn, self.device)
        
        # 반환값 구조 확인
        assert isinstance(result, dict)
        assert 'train_loss' in result
        assert 'train_acc' in result
        assert 'train_f1' in result
        
        # 값 타입 확인
        assert isinstance(result['train_loss'], float)
        assert isinstance(result['train_acc'], float)
        assert isinstance(result['train_f1'], float)
        
        # 값 범위 확인
        assert result['train_loss'] >= 0
        assert 0 <= result['train_acc'] <= 1
        assert 0 <= result['train_f1'] <= 1
    
    def test_train_one_epoch_gradient_update(self):
        """그래디언트 업데이트 확인"""
        # 학습 전 파라미터 저장
        initial_params = [param.clone() for param in self.model.parameters()]
        
        # 학습 실행
        train_one_epoch(self.loader, self.model, self.optimizer, self.loss_fn, self.device)
        
        # 파라미터가 업데이트되었는지 확인
        updated_params = list(self.model.parameters())
        
        params_updated = False
        for initial, updated in zip(initial_params, updated_params):
            if not torch.equal(initial, updated):
                params_updated = True
                break
        
        assert params_updated, "모델 파라미터가 업데이트되지 않았습니다"
    
    def test_train_one_epoch_model_mode(self):
        """모델이 훈련 모드로 설정되는지 확인"""
        self.model.eval()  # 먼저 평가 모드로 설정
        assert not self.model.training
        
        # 학습 실행
        train_one_epoch(self.loader, self.model, self.optimizer, self.loss_fn, self.device)
        
        # 훈련 모드로 변경되었는지 확인
        assert self.model.training


class TestValidateOneEpoch:
    """validate_one_epoch 함수 테스트"""
    
    def setup_method(self):
        """테스트 준비"""
        self.device = torch.device('cpu')
        
        # 간단한 모델 생성
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32*32*3, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 5)
        )
        
        # 더미 데이터 생성
        self.images = torch.randn(15, 3, 32, 32)
        self.targets = torch.randint(0, 5, (15,))
        self.dataset = torch.utils.data.TensorDataset(self.images, self.targets)
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=3, shuffle=False)
        
        # 손실 함수
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def test_validate_one_epoch_basic(self):
        """기본 검증 함수 테스트"""
        result = validate_one_epoch(self.loader, self.model, self.loss_fn, self.device)
        
        # 반환값 구조 확인
        assert isinstance(result, dict)
        assert 'val_loss' in result
        assert 'val_acc' in result
        assert 'val_f1' in result
        
        # 값 타입 확인
        assert isinstance(result['val_loss'], float)
        assert isinstance(result['val_acc'], float)
        assert isinstance(result['val_f1'], float)
        
        # 값 범위 확인
        assert result['val_loss'] >= 0
        assert 0 <= result['val_acc'] <= 1
        assert 0 <= result['val_f1'] <= 1
    
    def test_validate_one_epoch_no_gradient(self):
        """검증 시 그래디언트가 계산되지 않는지 확인"""
        # 그래디언트 활성화
        for param in self.model.parameters():
            param.requires_grad = True
        
        # 검증 실행
        validate_one_epoch(self.loader, self.model, self.loss_fn, self.device)
        
        # 그래디언트가 None인지 확인 (계산되지 않았음을 의미)
        for param in self.model.parameters():
            assert param.grad is None, "검증 중에 그래디언트가 계산되었습니다"
    
    def test_validate_one_epoch_model_mode(self):
        """모델이 평가 모드로 설정되는지 확인"""
        self.model.train()  # 먼저 훈련 모드로 설정
        assert self.model.training
        
        # 검증 실행
        validate_one_epoch(self.loader, self.model, self.loss_fn, self.device)
        
        # 평가 모드로 변경되었는지 확인
        assert not self.model.training


class TestTrainSingleModel:
    """train_single_model 함수 테스트"""
    
    def setup_method(self):
        """테스트 준비"""
        self.device = torch.device('cpu')
        
        # 더미 데이터 생성
        self.images = torch.randn(20, 3, 32, 32)
        self.targets = torch.randint(0, 3, (20,))
        self.dataset = torch.utils.data.TensorDataset(self.images, self.targets)
        
        # 훈련/검증 데이터 분할
        train_size = 16
        val_size = 4
        train_dataset = torch.utils.data.Subset(self.dataset, range(train_size))
        val_dataset = torch.utils.data.Subset(self.dataset, range(train_size, train_size + val_size))
        
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4)
        
        # 설정
        self.cfg = OmegaConf.create({
            'model': {
                'name': 'resnet18',
                'pretrained': False,
                'num_classes': 3
            },
            'training': {
                'lr': 0.01,
                'epochs': 2
            },
            'validation': {
                'early_stopping': {
                    'enabled': False
                }
            },
            'wandb': {
                'enabled': False
            }
        })
    
    @patch('training.log')
    def test_train_single_model_holdout(self, mock_log):
        """Holdout 검증으로 단일 모델 학습 테스트"""
        model = train_single_model(self.cfg, self.train_loader, self.val_loader, self.device)
        
        # 모델이 반환되는지 확인
        assert model is not None
        assert hasattr(model, 'forward')
        
        # 로그가 호출되었는지 확인
        mock_log.info.assert_called()
    
    @patch('training.log')
    def test_train_single_model_no_validation(self, mock_log):
        """검증 없이 단일 모델 학습 테스트"""
        model = train_single_model(self.cfg, self.train_loader, None, self.device)
        
        # 모델이 반환되는지 확인
        assert model is not None
        
        # 로그가 호출되었는지 확인
        mock_log.info.assert_called()
    
    def test_train_single_model_with_early_stopping(self):
        """Early stopping 포함 학습 테스트"""
        # Early stopping 활성화
        self.cfg.validation.early_stopping.enabled = True
        self.cfg.validation.early_stopping.patience = 1
        self.cfg.validation.early_stopping.monitor = 'val_loss'
        self.cfg.validation.early_stopping.mode = 'min'
        self.cfg.validation.early_stopping.min_delta = 0.001
        
        with patch('training.log'):
            model = train_single_model(self.cfg, self.train_loader, self.val_loader, self.device)
        
        assert model is not None


class TestTrainKFoldModels:
    """train_kfold_models 함수 테스트"""
    
    def setup_method(self):
        """테스트 준비"""
        self.device = torch.device('cpu')
        
        # 설정
        self.cfg = OmegaConf.create({
            'model': {
                'name': 'resnet18',
                'pretrained': False,
                'num_classes': 3
            },
            'training': {
                'lr': 0.01,
                'epochs': 1,
                'batch_size': 4
            },
            'data': {
                'num_workers': 0
            },
            'validation': {
                'early_stopping': {
                    'enabled': False
                }
            },
            'wandb': {
                'enabled': False
            }
        })
        
        # 더미 K-Fold 데이터 준비
        import pandas as pd
        full_train_df = pd.DataFrame({
            'ID': [f'img_{i}.jpg' for i in range(30)],
            'target': [i % 3 for i in range(30)]
        })
        
        # 더미 fold 생성 (2-fold)
        folds = [
            (np.arange(15), np.arange(15, 30)),
            (np.arange(15, 30), np.arange(15))
        ]
        
        data_path = "/tmp/dummy"
        train_transform = None
        test_transform = None
        
        self.kfold_data = (folds, full_train_df, data_path, train_transform, test_transform)
    
    @patch('training.get_kfold_loaders')
    @patch('training.log')
    def test_train_kfold_models_basic(self, mock_log, mock_get_loaders):
        """기본 K-Fold 학습 테스트"""
        # 더미 데이터 로더 생성
        dummy_images = torch.randn(12, 3, 32, 32)
        dummy_targets = torch.randint(0, 3, (12,))
        dummy_dataset = torch.utils.data.TensorDataset(dummy_images, dummy_targets)
        dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=4)
        
        # Mock 설정
        mock_get_loaders.return_value = (
            dummy_loader, dummy_loader, 
            pd.DataFrame({'ID': ['img1.jpg'], 'target': [0]}),
            pd.DataFrame({'ID': ['img2.jpg'], 'target': [1]})
        )
        
        models = train_kfold_models(self.cfg, self.kfold_data, self.device)
        
        # 반환된 모델들 확인
        assert isinstance(models, list)
        assert len(models) == 2  # 2-fold
        
        for model in models:
            assert model is not None
            assert hasattr(model, 'forward')
        
        # 로그가 호출되었는지 확인
        mock_log.info.assert_called()


class TestTrainingIntegration:
    """학습 관련 통합 테스트"""
    
    def setup_method(self):
        """테스트 준비"""
        self.device = torch.device('cpu')
        
        # 설정
        self.cfg = OmegaConf.create({
            'model': {
                'name': 'resnet18',
                'pretrained': False,
                'num_classes': 5
            },
            'training': {
                'lr': 0.01,
                'epochs': 1
            }
        })
    
    def test_training_consistency(self):
        """학습 결과의 일관성 테스트"""
        # 동일한 시드로 두 번 학습
        torch.manual_seed(42)
        model1, optimizer1, loss_fn1 = setup_model_and_optimizer(self.cfg, self.device)
        
        torch.manual_seed(42)
        model2, optimizer2, loss_fn2 = setup_model_and_optimizer(self.cfg, self.device)
        
        # 더미 데이터
        torch.manual_seed(42)
        images = torch.randn(8, 3, 224, 224)
        targets = torch.randint(0, 5, (8,))
        dataset = torch.utils.data.TensorDataset(images, targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        # 두 모델을 동일하게 학습
        torch.manual_seed(42)
        result1 = train_one_epoch(loader, model1, optimizer1, loss_fn1, self.device)
        
        torch.manual_seed(42)
        result2 = train_one_epoch(loader, model2, optimizer2, loss_fn2, self.device)
        
        # 결과가 동일한지 확인
        assert abs(result1['train_loss'] - result2['train_loss']) < 1e-6
        assert abs(result1['train_acc'] - result2['train_acc']) < 1e-6
    
    def test_loss_decreases_over_epochs(self):
        """에포크에 따른 손실 감소 테스트"""
        model, optimizer, loss_fn = setup_model_and_optimizer(self.cfg, self.device)
        
        # 더미 데이터 (쉬운 패턴)
        images = torch.randn(16, 3, 224, 224)
        targets = torch.zeros(16, dtype=torch.long)  # 모두 클래스 0
        dataset = torch.utils.data.TensorDataset(images, targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        losses = []
        for epoch in range(3):
            result = train_one_epoch(loader, model, optimizer, loss_fn, self.device)
            losses.append(result['train_loss'])
        
        # 손실이 감소하는 경향이 있는지 확인 (최소한 마지막이 첫 번째보다 작아야 함)
        assert losses[-1] < losses[0], f"손실이 감소하지 않았습니다: {losses}"


class TestMixedPrecisionTraining:
    """Mixed Precision Training 테스트"""
    
    def setup_method(self):
        """테스트 준비"""
        self.device = torch.device('cpu')
        
        # 간단한 모델 생성
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32*32*3, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10)
        )
        
        # 더미 데이터 생성
        self.images = torch.randn(20, 3, 32, 32)
        self.targets = torch.randint(0, 10, (20,))
        self.dataset = torch.utils.data.TensorDataset(self.images, self.targets)
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=4, shuffle=True)
        
        # 옵티마이저와 손실 함수
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def test_train_one_epoch_without_scaler(self):
        """scaler 없이 학습 테스트 (일반 학습)"""
        result = train_one_epoch(self.loader, self.model, self.optimizer, self.loss_fn, self.device, scaler=None)
        
        # 반환값 확인
        assert isinstance(result, dict)
        assert 'train_loss' in result
        assert 'train_acc' in result
        assert 'train_f1' in result
        
        # 값 검증
        assert result['train_loss'] >= 0
        assert 0 <= result['train_acc'] <= 1
        assert 0 <= result['train_f1'] <= 1
    
    def test_train_one_epoch_with_mock_scaler(self):
        """Mock scaler로 Mixed Precision Training 테스트"""
        # Mock scaler 생성
        mock_scaler = MagicMock()
        mock_scaler.scale.return_value = MagicMock()
        mock_scaler.scale.return_value.backward = MagicMock()
        
        # 학습 실행
        result = train_one_epoch(self.loader, self.model, self.optimizer, self.loss_fn, self.device, scaler=mock_scaler)
        
        # 반환값 확인
        assert isinstance(result, dict)
        assert 'train_loss' in result
        assert 'train_acc' in result
        assert 'train_f1' in result
        
        # Mock scaler 메서드가 호출되었는지 확인
        assert mock_scaler.scale.called
        assert mock_scaler.step.called
        assert mock_scaler.update.called
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision_with_cuda(self):
        """CUDA에서 Mixed Precision Training 테스트"""
        # CUDA 설정
        device = torch.device('cuda')
        model = self.model.to(device)
        
        # 데이터를 CUDA로 이동
        images = self.images.to(device)
        targets = self.targets.to(device)
        dataset = torch.utils.data.TensorDataset(images, targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Mixed Precision 지원 확인
        try:
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
            
            # 학습 실행
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            result = train_one_epoch(loader, model, optimizer, self.loss_fn, device, scaler=scaler)
            
            # 결과 확인
            assert isinstance(result, dict)
            assert 'train_loss' in result
            assert 'train_acc' in result
            assert 'train_f1' in result
            
        except ImportError:
            pytest.skip("Mixed Precision not available")
    
    def test_mixed_precision_setting_with_cpu(self):
        """CPU에서 Mixed Precision 설정 시 경고 테스트"""
        cfg = OmegaConf.create({
            'model': {
                'name': 'resnet18',
                'pretrained': False,
                'num_classes': 10
            },
            'training': {
                'lr': 0.001,
                'epochs': 1,
                'label_smoothing': {
                    'enabled': False,
                    'smoothing': 0.1
                },
                'mixed_precision': {
                    'enabled': True  # CPU에서 활성화 요청
                },
                'scheduler': {
                    'enabled': False
                }
            },
            'validation': {
                'strategy': 'none',
                'early_stopping': {
                    'enabled': False
                }
            },
            'model_save': {
                'enabled': False
            },
            'wandb': {
                'enabled': False
            }
        })
        
        device = torch.device('cpu')
        
        # 더미 데이터 생성
        images = torch.randn(16, 3, 32, 32)
        targets = torch.randint(0, 10, (16,))
        dataset = torch.utils.data.TensorDataset(images, targets)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        # 로그 모킹
        with patch('training.log') as mock_log:
            # 학습 실행 (CPU에서 Mixed Precision 요청)
            model = train_single_model(cfg, train_loader, None, device)
            
            # 경고 로그가 출력되었는지 확인
            warning_calls = [call for call in mock_log.warning.call_args_list if 'CUDA' in str(call)]
            assert len(warning_calls) > 0, "CPU에서 Mixed Precision 요청 시 경고가 출력되지 않았습니다"
        
        # 모델이 정상적으로 학습되었는지 확인
        assert model is not None
    
    def test_mixed_precision_unavailable_warning(self):
        """Mixed Precision 미지원 시 경고 테스트"""
        cfg = OmegaConf.create({
            'model': {
                'name': 'resnet18',
                'pretrained': False,
                'num_classes': 10
            },
            'training': {
                'lr': 0.001,
                'epochs': 1,
                'label_smoothing': {
                    'enabled': False,
                    'smoothing': 0.1
                },
                'mixed_precision': {
                    'enabled': True
                },
                'scheduler': {
                    'enabled': False
                }
            },
            'validation': {
                'strategy': 'none',
                'early_stopping': {
                    'enabled': False
                }
            },
            'model_save': {
                'enabled': False
            },
            'wandb': {
                'enabled': False
            }
        })
        
        device = torch.device('cpu')
        
        # 더미 데이터 생성
        images = torch.randn(16, 3, 32, 32)
        targets = torch.randint(0, 10, (16,))
        dataset = torch.utils.data.TensorDataset(images, targets)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        # AMP_AVAILABLE을 False로 패치
        with patch('training.AMP_AVAILABLE', False):
            with patch('training.log') as mock_log:
                # 학습 실행
                model = train_single_model(cfg, train_loader, None, device)
                
                # 경고 로그가 출력되었는지 확인
                warning_calls = [call for call in mock_log.warning.call_args_list if 'PyTorch AMP' in str(call)]
                assert len(warning_calls) > 0, "AMP 미지원 시 경고가 출력되지 않았습니다"
        
        # 모델이 정상적으로 학습되었는지 확인
        assert model is not None
    
    def test_scaler_parameter_in_train_functions(self):
        """train_one_epoch 함수의 scaler 파라미터 테스트"""
        # scaler=None으로 호출 (기본값)
        result1 = train_one_epoch(self.loader, self.model, self.optimizer, self.loss_fn, self.device, scaler=None)
        
        # scaler를 명시적으로 None으로 전달
        result2 = train_one_epoch(self.loader, self.model, self.optimizer, self.loss_fn, self.device, None)
        
        # 둘 다 정상적으로 작동해야 함
        assert isinstance(result1, dict)
        assert isinstance(result2, dict)
        
        # 각 결과가 필요한 키를 가지고 있는지 확인
        for result in [result1, result2]:
            assert 'train_loss' in result
            assert 'train_acc' in result
            assert 'train_f1' in result


if __name__ == "__main__":
    pytest.main([__file__]) 