# -*- coding: utf-8 -*-
"""
utils.py 모듈 테스트
사용자 정의 유틸리티 함수들을 테스트
"""
import os
import sys
import pytest
import torch
import random
import numpy as np
from omegaconf import OmegaConf
from unittest.mock import patch, MagicMock

# 상위 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    EarlyStopping,
    set_seed,
    setup_wandb,
    get_device,
    log_hyperparameters,
    finish_wandb
)


class TestEarlyStopping:
    """EarlyStopping 클래스 테스트"""
    
    def test_early_stopping_min_mode_basic(self):
        """최소화 모드 기본 동작 테스트"""
        early_stopping = EarlyStopping(patience=3, min_delta=0.01, monitor='val_loss', mode='min')
        
        # 개선되는 경우
        assert not early_stopping({'val_loss': 1.0})
        assert not early_stopping({'val_loss': 0.9})
        assert not early_stopping({'val_loss': 0.8})
        
        # 개선되지 않는 경우
        assert not early_stopping({'val_loss': 0.81})  # min_delta 내에서 변화, counter=1
        assert not early_stopping({'val_loss': 0.82})  # counter=2
        assert early_stopping({'val_loss': 0.83})      # counter=3, should stop
    
    def test_early_stopping_max_mode_basic(self):
        """최대화 모드 기본 동작 테스트"""
        early_stopping = EarlyStopping(patience=2, min_delta=0.01, monitor='val_acc', mode='max')
        
        # 개선되는 경우
        assert not early_stopping({'val_acc': 0.8})
        assert not early_stopping({'val_acc': 0.9})
        assert not early_stopping({'val_acc': 0.95})
        
        # 개선되지 않는 경우
        assert not early_stopping({'val_acc': 0.94})  # min_delta 내에서 변화, counter=1
        assert early_stopping({'val_acc': 0.93})      # counter=2, should stop
    
    def test_early_stopping_patience(self):
        """patience 파라미터 테스트"""
        early_stopping = EarlyStopping(patience=5, min_delta=0.001, monitor='val_loss', mode='min')
        
        # 5번까지는 멈추지 않아야 함
        for i in range(5):
            assert not early_stopping({'val_loss': 1.0 + i * 0.01})
        
        # 6번째에 멈춰야 함
        assert early_stopping({'val_loss': 1.05})
    
    def test_early_stopping_min_delta(self):
        """min_delta 파라미터 테스트"""
        early_stopping = EarlyStopping(patience=2, min_delta=0.1, monitor='val_loss', mode='min')
        
        # min_delta보다 작은 개선은 개선으로 인정하지 않음
        assert not early_stopping({'val_loss': 1.0})
        assert not early_stopping({'val_loss': 0.95})  # 0.05 개선 (< 0.1), counter=1
        assert early_stopping({'val_loss': 0.94})      # counter=2, should stop
    
    def test_early_stopping_monitor_key(self):
        """monitor 키 변경 테스트"""
        early_stopping = EarlyStopping(patience=2, min_delta=0.01, monitor='custom_metric', mode='max')
        
        assert not early_stopping({'custom_metric': 0.8})
        assert not early_stopping({'custom_metric': 0.75})  # 감소, counter=1
        assert early_stopping({'custom_metric': 0.74})     # counter=2, should stop
    
    def test_early_stopping_reset_on_improvement(self):
        """개선 시 counter 리셋 테스트"""
        early_stopping = EarlyStopping(patience=2, min_delta=0.01, monitor='val_loss', mode='min')
        
        assert not early_stopping({'val_loss': 1.0})
        assert not early_stopping({'val_loss': 1.01})  # counter=1
        assert not early_stopping({'val_loss': 0.8})   # 개선 - counter 리셋
        assert not early_stopping({'val_loss': 0.81})  # counter=1 (새로 시작)
        assert early_stopping({'val_loss': 0.82})      # counter=2, should stop


class TestSetSeed:
    """set_seed 함수 테스트"""
    
    @patch('utils.log')
    def test_set_seed_basic(self, mock_log):
        """기본 시드 설정 테스트"""
        seed = 42
        set_seed(seed)
        
        # 로그가 호출되었는지 확인 (GPU 환경에서는 추가 메시지 포함)
        call_args = mock_log.info.call_args[0][0]
        assert f"시드 고정 완료: {seed}" in call_args
    
    def test_set_seed_reproducibility(self):
        """시드 설정 후 재현성 테스트"""
        seed = 123
        
        # 첫 번째 실행
        set_seed(seed)
        torch_rand1 = torch.rand(5)
        np_rand1 = np.random.rand(5)
        py_rand1 = [random.random() for _ in range(5)]
        
        # 두 번째 실행
        set_seed(seed)
        torch_rand2 = torch.rand(5)
        np_rand2 = np.random.rand(5)
        py_rand2 = [random.random() for _ in range(5)]
        
        # 결과가 동일한지 확인
        assert torch.equal(torch_rand1, torch_rand2)
        assert np.array_equal(np_rand1, np_rand2)
        assert py_rand1 == py_rand2
    
    def test_set_seed_environment_variable(self):
        """환경 변수 설정 테스트"""
        seed = 456
        set_seed(seed)
        
        # PYTHONHASHSEED 환경 변수가 설정되었는지 확인
        assert os.environ.get('PYTHONHASHSEED') == str(seed)


class TestSetupWandb:
    """setup_wandb 함수 테스트"""
    
    @patch('utils.wandb')
    @patch('utils.log')
    @patch('utils.os.getenv')
    def test_setup_wandb_enabled(self, mock_getenv, mock_log, mock_wandb):
        """wandb 활성화 테스트"""
        # Mock 설정
        mock_getenv.side_effect = lambda key, default=None: {
            'WANDB_ENTITY': 'test_entity',
            'WANDB_PROJECT': 'test_project'
        }.get(key, default)
        
        cfg = OmegaConf.create({
            'wandb': {
                'enabled': True,
                'entity': 'config_entity',
                'project': 'config_project',
                'run_name': 'test_run',
                'tags': ['test'],
                'notes': 'test notes'
            },
            'train': {
                'lr': 0.001,
                'epochs': 10,
                'batch_size': 32,
                'seed': 42,
                'label_smoothing': {'enabled': False, 'smoothing': 0.1},
                'mixed_precision': {'enabled': False},
                'scheduler': {'enabled': False, 'name': 'none'}
            },
            'model': {
                'name': 'resnet18',
                'num_classes': 17,
                'pretrained': True
            },
            'data': {
                'img_size': 224,
                'train_images_path': 'train_path',
                'test_images_path': 'test_path',
                'train_csv_path': 'train.csv',
                'test_csv_path': 'test.csv',
                'num_workers': 0
            },
            'augment': {
                'train_aug_count': 0,
                'valid_aug_count': 0,
                'test_tta_enabled': False,
                'test_tta_count': 0,
                'method': 'none',
                'intensity': 0.0,
                'valid_tta_count': 0,
                'train_aug_ops': [],
                'valid_aug_ops': [],
                'valid_tta_ops': [],
                'test_tta_ops': [],
                'train_aug_add_org': False,
                'valid_aug_add_org': False,
                'valid_tta_add_org': False,
                'test_tta_add_org': False
            },
            'validation': {
                'strategy': 'none',
                'holdout': {'train_ratio': 0.8, 'stratify': True},
                'kfold': {'n_splits': 2, 'stratify': True},
                'early_stopping': {'enabled': False, 'patience': 1, 'min_delta': 0.0, 'monitor': 'val_loss', 'mode': 'min'}
            },
            'device': 'cpu',
            'output': {'dir': 'out', 'filename': 'pred.csv'},
            'model_save': {'enabled': False},
            'random_seed_ensemble': {'enabled': False, 'count': 1}
        })
        
        setup_wandb(cfg)
        
        # wandb.init이 호출되었는지 확인
        mock_wandb.init.assert_called_once()
        
        # 호출 인자 확인
        call_args = mock_wandb.init.call_args
        assert call_args[1]['project'] == 'test_project'  # 환경변수 우선
        assert call_args[1]['entity'] == 'test_entity'    # 환경변수 우선
        assert call_args[1]['name'] == 'test_run'
        assert call_args[1]['tags'] == ['test']
        assert call_args[1]['notes'] == 'test notes'
        
        # 로그 확인
        mock_log.info.assert_called_with("wandb 초기화 완료 - 프로젝트: test_project")
    
    @patch('utils.log')
    def test_setup_wandb_disabled(self, mock_log):
        """wandb 비활성화 테스트"""
        cfg = OmegaConf.create({
            'wandb': {
                'enabled': False
            }
        })
        
        setup_wandb(cfg)
        
        # 비활성화 로그 확인
        mock_log.info.assert_called_with("wandb 비활성화됨")
    
    @patch('utils.wandb')
    @patch('utils.log')
    @patch('utils.os.getenv')
    def test_setup_wandb_config_fallback(self, mock_getenv, mock_log, mock_wandb):
        """환경변수 없을 때 config 사용 테스트"""
        # Mock 설정 - 환경변수 없음
        mock_getenv.return_value = None
        
        cfg = OmegaConf.create({
            'wandb': {
                'enabled': True,
                'entity': 'config_entity',
                'project': 'config_project',
                'run_name': 'test_run',
                'tags': [],
                'notes': ''
            },
            'train': {
                'lr': 0.001,
                'epochs': 1,
                'batch_size': 8,
                'seed': 42,
                'label_smoothing': {'enabled': False, 'smoothing': 0.1},
                'mixed_precision': {'enabled': False},
                'scheduler': {'enabled': False, 'name': 'none'}
            },
            'model': {'name': 'resnet18', 'num_classes': 10, 'pretrained': False},
            'data': {
                'img_size': 32,
                'train_images_path': 'train',
                'test_images_path': 'test',
                'train_csv_path': 'train.csv',
                'test_csv_path': 'test.csv',
                'num_workers': 0
            },
            'augment': {
                'train_aug_count': 0,
                'valid_aug_count': 0,
                'test_tta_enabled': False,
                'test_tta_count': 0,
                'method': 'none',
                'intensity': 0.0,
                'valid_tta_count': 0,
                'train_aug_ops': [],
                'valid_aug_ops': [],
                'valid_tta_ops': [],
                'test_tta_ops': [],
                'train_aug_add_org': False,
                'valid_aug_add_org': False,
                'valid_tta_add_org': False,
                'test_tta_add_org': False
            },
            'validation': {
                'strategy': 'none',
                'holdout': {'train_ratio': 0.8, 'stratify': True},
                'kfold': {'n_splits': 2, 'stratify': True},
                'early_stopping': {'enabled': False, 'patience': 1, 'min_delta': 0.0, 'monitor': 'val_loss', 'mode': 'min'}
            },
            'device': 'cpu',
            'output': {'dir': 'out', 'filename': 'pred.csv'},
            'model_save': {'enabled': False},
            'random_seed_ensemble': {'enabled': False, 'count': 1}
        })
        
        setup_wandb(cfg)
        
        # config 값이 사용되었는지 확인
        call_args = mock_wandb.init.call_args
        assert call_args[1]['project'] == 'config_project'
        assert call_args[1]['entity'] == 'config_entity'


class TestGetDevice:
    """get_device 함수 테스트"""
    
    @patch('utils.torch.cuda.is_available')
    @patch('utils.log')
    def test_get_device_cuda_available(self, mock_log, mock_cuda_available):
        """CUDA 사용 가능한 경우 테스트"""
        mock_cuda_available.return_value = True
        
        cfg = OmegaConf.create({'device': 'cuda'})
        device = get_device(cfg)
        
        assert device.type == 'cuda'
        mock_log.info.assert_called_with("사용 장치: cuda")
    
    @patch('utils.torch.cuda.is_available')
    @patch('utils.log')
    def test_get_device_cuda_not_available(self, mock_log, mock_cuda_available):
        """CUDA 사용 불가능한 경우 테스트"""
        mock_cuda_available.return_value = False
        
        cfg = OmegaConf.create({'device': 'cuda'})
        device = get_device(cfg)
        
        assert device.type == 'cpu'
        mock_log.info.assert_called_with("사용 장치: cpu")
    
    @patch('utils.log')
    def test_get_device_cpu_explicitly(self, mock_log):
        """명시적으로 CPU 사용 테스트"""
        cfg = OmegaConf.create({'device': 'cpu'})
        device = get_device(cfg)
        
        assert device.type == 'cpu'
        mock_log.info.assert_called_with("사용 장치: cpu")


class TestLogHyperparameters:
    """log_hyperparameters 함수 테스트"""
    
    @patch('utils.log')
    def test_log_hyperparameters(self, mock_log):
        """하이퍼파라미터 로깅 테스트"""
        cfg = OmegaConf.create({
            'model': {'name': 'resnet34'},
            'data': {'img_size': 224},
            'train': {
                'lr': 0.001,
                'epochs': 50,
                'batch_size': 64
            }
        })
        
        log_hyperparameters(cfg)
        
        # 로그 메시지 확인
        expected_msg = ("하이퍼파라미터 설정 - 모델: resnet34, 이미지 크기: 224, "
                       "학습률: 0.001, 에포크: 50, 배치 크기: 64")
        mock_log.info.assert_called_with(expected_msg)


class TestFinishWandb:
    """finish_wandb 함수 테스트"""
    
    @patch('utils.wandb')
    @patch('utils.log')
    def test_finish_wandb_enabled(self, mock_log, mock_wandb):
        """wandb 활성화 상태에서 종료 테스트"""
        cfg = OmegaConf.create({
            'wandb': {'enabled': True}
        })
        
        finish_wandb(cfg)
        
        # wandb.finish 호출 확인
        mock_wandb.finish.assert_called_once()
        mock_log.info.assert_called_with("wandb 세션 종료")
    
    @patch('utils.wandb')
    @patch('utils.log')
    def test_finish_wandb_disabled(self, mock_log, mock_wandb):
        """wandb 비활성화 상태에서 종료 테스트"""
        cfg = OmegaConf.create({
            'wandb': {'enabled': False}
        })
        
        finish_wandb(cfg)
        
        # wandb.finish 호출되지 않음
        mock_wandb.finish.assert_not_called()
        # 로그도 호출되지 않음
        mock_log.info.assert_not_called()


class TestUtilsIntegration:
    """유틸리티 함수 통합 테스트"""
    
    def test_seed_consistency_across_utils(self):
        """다양한 함수에서 시드 일관성 테스트"""
        seed = 789
        
        # 시드 설정 후 여러 연산
        set_seed(seed)
        initial_torch = torch.rand(3)
        initial_np = np.random.rand(3)
        
        # 시드 재설정 후 같은 연산
        set_seed(seed)
        repeated_torch = torch.rand(3)
        repeated_np = np.random.rand(3)
        
        # 결과 일치 확인
        assert torch.equal(initial_torch, repeated_torch)
        assert np.array_equal(initial_np, repeated_np)
    
    @patch('utils.torch.cuda.is_available')
    def test_device_with_different_configs(self, mock_cuda_available):
        """다양한 설정에서 디바이스 선택 테스트"""
        mock_cuda_available.return_value = True
        
        # CUDA 설정
        with patch('utils.log'):
            device_cuda = get_device(OmegaConf.create({'device': 'cuda'}))
            assert device_cuda.type == 'cuda'
        
        # CPU 설정
        with patch('utils.log'):
            device_cpu = get_device(OmegaConf.create({'device': 'cpu'}))
            assert device_cpu.type == 'cpu'
        
        # CUDA 불가능한 상황
        mock_cuda_available.return_value = False
        with patch('utils.log'):
            device_fallback = get_device(OmegaConf.create({'device': 'cuda'}))
            assert device_fallback.type == 'cpu'


if __name__ == "__main__":
    pytest.main([__file__]) 