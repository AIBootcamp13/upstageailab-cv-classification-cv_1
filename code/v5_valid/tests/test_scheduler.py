# -*- coding: utf-8 -*-
"""
스케쥴러 관련 테스트
"""

import pytest
import torch
import tempfile
import shutil
import os
from omegaconf import OmegaConf
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    StepLR, 
    ReduceLROnPlateau, 
    CosineAnnealingWarmRestarts
)

# 상위 디렉토리를 path에 추가
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_model, setup_model_and_optimizer, create_scheduler
from training import update_scheduler


class TestCreateScheduler:
    """스케쥴러 생성 테스트"""
    
    def setup_method(self):
        """테스트 준비"""
        self.device = torch.device('cpu')
        self.model = create_model("resnet18", pretrained=False, num_classes=5)
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
    
    def test_create_scheduler_disabled(self):
        """스케쥴러 비활성화 테스트"""
        cfg = OmegaConf.create({
            'training': {
                'scheduler': {
                    'enabled': False
                }
            }
        })
        
        scheduler = create_scheduler(self.optimizer, cfg)
        assert scheduler is None
    
    def test_create_scheduler_cosine(self):
        """CosineAnnealingLR 스케쥴러 생성 테스트"""
        cfg = OmegaConf.create({
            'training': {
                'epochs': 50,
                'scheduler': {
                    'enabled': True,
                    'name': 'cosine',
                    'cosine': {
                        'T_max': 100,
                        'eta_min': 1e-6,
                        'last_epoch': -1
                    }
                }
            }
        })
        
        scheduler = create_scheduler(self.optimizer, cfg)
        assert scheduler is not None
        assert isinstance(scheduler, CosineAnnealingLR)
        assert scheduler.T_max == 50  # epochs로 자동 설정됨
        assert scheduler.eta_min == 1e-6
    
    def test_create_scheduler_step(self):
        """StepLR 스케쥴러 생성 테스트"""
        cfg = OmegaConf.create({
            'training': {
                'scheduler': {
                    'enabled': True,
                    'name': 'step',
                    'step': {
                        'step_size': 30,
                        'gamma': 0.1,
                        'last_epoch': -1
                    }
                }
            }
        })
        
        scheduler = create_scheduler(self.optimizer, cfg)
        assert scheduler is not None
        assert isinstance(scheduler, StepLR)
        assert scheduler.step_size == 30
        assert scheduler.gamma == 0.1
    
    def test_create_scheduler_plateau(self):
        """ReduceLROnPlateau 스케쥴러 생성 테스트"""
        cfg = OmegaConf.create({
            'training': {
                'scheduler': {
                    'enabled': True,
                    'name': 'plateau',
                    'plateau': {
                        'mode': 'min',
                        'factor': 0.5,
                        'patience': 5,
                        'threshold': 1e-4,
                        'threshold_mode': 'rel',
                        'cooldown': 0,
                        'min_lr': 1e-8,
                        'eps': 1e-8
                    }
                }
            }
        })
        
        scheduler = create_scheduler(self.optimizer, cfg)
        assert scheduler is not None
        assert isinstance(scheduler, ReduceLROnPlateau)
        assert scheduler.mode == 'min'
        assert scheduler.factor == 0.5
        assert scheduler.patience == 5
    
    def test_create_scheduler_cosine_warm(self):
        """CosineAnnealingWarmRestarts 스케쥴러 생성 테스트"""
        cfg = OmegaConf.create({
            'training': {
                'scheduler': {
                    'enabled': True,
                    'name': 'cosine_warm',
                    'cosine_warm': {
                        'T_0': 10,
                        'T_mult': 1,
                        'eta_min': 1e-6,
                        'last_epoch': -1
                    }
                }
            }
        })
        
        scheduler = create_scheduler(self.optimizer, cfg)
        assert scheduler is not None
        assert isinstance(scheduler, CosineAnnealingWarmRestarts)
        assert scheduler.T_0 == 10
        assert scheduler.T_mult == 1
        assert scheduler.eta_min == 1e-6
    
    def test_create_scheduler_none(self):
        """스케쥴러 'none' 설정 테스트"""
        cfg = OmegaConf.create({
            'training': {
                'scheduler': {
                    'enabled': True,
                    'name': 'none'
                }
            }
        })
        
        scheduler = create_scheduler(self.optimizer, cfg)
        assert scheduler is None
    
    def test_create_scheduler_invalid_name(self):
        """잘못된 스케쥴러 이름 테스트"""
        cfg = OmegaConf.create({
            'training': {
                'scheduler': {
                    'enabled': True,
                    'name': 'invalid_scheduler'
                }
            }
        })
        
        with pytest.raises(ValueError, match="지원하지 않는 스케쥴러"):
            create_scheduler(self.optimizer, cfg)


class TestSetupModelAndOptimizer:
    """모델과 옵티마이저 설정 테스트"""
    
    def setup_method(self):
        """테스트 준비"""
        self.device = torch.device('cpu')
    
    def test_setup_with_scheduler(self):
        """스케쥴러 포함 설정 테스트"""
        cfg = OmegaConf.create({
            'model': {
                'name': 'resnet18',
                'pretrained': False,
                'num_classes': 5
            },
            'training': {
                'lr': 0.001,
                'epochs': 10,
                'scheduler': {
                    'enabled': True,
                    'name': 'cosine',
                    'cosine': {
                        'T_max': 100,
                        'eta_min': 1e-6,
                        'last_epoch': -1
                    }
                }
            }
        })
        
        model, optimizer, loss_fn, scheduler = setup_model_and_optimizer(cfg, self.device)
        
        assert model is not None
        assert optimizer is not None
        assert loss_fn is not None
        assert scheduler is not None
        assert isinstance(scheduler, CosineAnnealingLR)
        assert scheduler.T_max == 10  # epochs로 자동 설정
    
    def test_setup_without_scheduler(self):
        """스케쥴러 없이 설정 테스트"""
        cfg = OmegaConf.create({
            'model': {
                'name': 'resnet18',
                'pretrained': False,
                'num_classes': 5
            },
            'training': {
                'lr': 0.001,
                'scheduler': {
                    'enabled': False
                }
            }
        })
        
        model, optimizer, loss_fn, scheduler = setup_model_and_optimizer(cfg, self.device)
        
        assert model is not None
        assert optimizer is not None
        assert loss_fn is not None
        assert scheduler is None
    
    def test_setup_learning_rate(self):
        """학습률 설정 테스트"""
        cfg = OmegaConf.create({
            'model': {
                'name': 'resnet18',
                'pretrained': False,
                'num_classes': 5
            },
            'training': {
                'lr': 0.01,
                'scheduler': {
                    'enabled': False
                }
            }
        })
        
        model, optimizer, loss_fn, scheduler = setup_model_and_optimizer(cfg, self.device)
        
        assert optimizer.param_groups[0]['lr'] == 0.01


class TestSchedulerFunctionality:
    """스케쥴러 기능 테스트"""
    
    def setup_method(self):
        """테스트 준비"""
        self.device = torch.device('cpu')
        self.model = create_model("resnet18", pretrained=False, num_classes=5)
        self.optimizer = Adam(self.model.parameters(), lr=0.1)
    
    def test_cosine_scheduler_step(self):
        """CosineAnnealingLR 스케쥴러 스텝 테스트"""
        scheduler = CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0.001)
        
        initial_lr = self.optimizer.param_groups[0]['lr']
        
        # 여러 스텝 실행
        for _ in range(5):
            scheduler.step()
        
        new_lr = self.optimizer.param_groups[0]['lr']
        assert new_lr != initial_lr
        assert new_lr < initial_lr  # 학습률이 감소해야 함
    
    def test_step_scheduler_step(self):
        """StepLR 스케쥴러 스텝 테스트"""
        scheduler = StepLR(self.optimizer, step_size=3, gamma=0.1)
        
        initial_lr = self.optimizer.param_groups[0]['lr']
        
        # step_size 전까지는 학습률 유지
        for _ in range(2):
            scheduler.step()
        
        lr_before_step = self.optimizer.param_groups[0]['lr']
        assert lr_before_step == initial_lr
        
        # step_size 도달 시 학습률 감소
        scheduler.step()
        lr_after_step = self.optimizer.param_groups[0]['lr']
        assert lr_after_step == initial_lr * 0.1
    
    def test_plateau_scheduler_step(self):
        """ReduceLROnPlateau 스케쥴러 스텝 테스트"""
        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=0)
        
        initial_lr = self.optimizer.param_groups[0]['lr']
        
        # 손실이 개선되지 않는 경우 시뮬레이션 (patience=0으로 즉시 감소)
        scheduler.step(1.0)  # 첫 번째 값 설정
        scheduler.step(1.0)  # 동일한 손실 값으로 즉시 감소
        
        new_lr = self.optimizer.param_groups[0]['lr']
        assert new_lr == initial_lr * 0.5  # patience=0이므로 즉시 학습률 감소
    
    def test_cosine_warm_scheduler_step(self):
        """CosineAnnealingWarmRestarts 스케쥴러 스텝 테스트"""
        scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=1, eta_min=0.001)
        
        initial_lr = self.optimizer.param_groups[0]['lr']
        
        # 여러 스텝 실행
        for _ in range(3):
            scheduler.step()
        
        new_lr = self.optimizer.param_groups[0]['lr']
        assert new_lr != initial_lr
        assert new_lr < initial_lr  # 학습률이 감소해야 함


class TestUpdateScheduler:
    """스케쥴러 업데이트 테스트"""
    
    def setup_method(self):
        """테스트 준비"""
        self.device = torch.device('cpu')
        self.model = create_model("resnet18", pretrained=False, num_classes=5)
        self.optimizer = Adam(self.model.parameters(), lr=0.1)
    
    def test_update_scheduler_none(self):
        """스케쥴러가 None인 경우 테스트"""
        result = update_scheduler(None, None, None)
        assert result is None
    
    def test_update_scheduler_cosine(self):
        """CosineAnnealingLR 스케쥴러 업데이트 테스트"""
        scheduler = CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0.001)
        
        initial_lr = self.optimizer.param_groups[0]['lr']
        
        # 스케쥴러 업데이트
        result_lr = update_scheduler(scheduler, None, None)
        
        new_lr = self.optimizer.param_groups[0]['lr']
        assert new_lr != initial_lr
        assert result_lr == new_lr
    
    def test_update_scheduler_step(self):
        """StepLR 스케쥴러 업데이트 테스트"""
        scheduler = StepLR(self.optimizer, step_size=1, gamma=0.1)
        
        initial_lr = self.optimizer.param_groups[0]['lr']
        
        # 스케쥴러 업데이트
        result_lr = update_scheduler(scheduler, None, None)
        
        new_lr = self.optimizer.param_groups[0]['lr']
        assert new_lr == initial_lr * 0.1  # step_size=1이므로 바로 감소
        assert result_lr == new_lr
    
    def test_update_scheduler_plateau(self):
        """ReduceLROnPlateau 스케쥴러 업데이트 테스트"""
        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=0)
        
        cfg = OmegaConf.create({
            'training': {
                'scheduler': {
                    'plateau': {
                        'mode': 'min'
                    }
                }
            }
        })
        
        val_metrics = {"val_loss": 1.0}
        
        initial_lr = self.optimizer.param_groups[0]['lr']
        
        # 스케쥴러 업데이트
        result_lr = update_scheduler(scheduler, val_metrics, cfg)
        
        new_lr = self.optimizer.param_groups[0]['lr']
        assert result_lr == new_lr
    
    def test_update_scheduler_plateau_without_cfg(self):
        """ReduceLROnPlateau 스케쥴러 업데이트 - cfg 없는 경우 테스트"""
        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=0)
        
        val_metrics = {"val_loss": 1.0}
        
        # cfg 없이 스케쥴러 업데이트 (경고 발생)
        result_lr = update_scheduler(scheduler, val_metrics, None)
        
        # 스케쥴러 업데이트는 되지 않지만 현재 학습률은 반환
        assert result_lr is not None
    
    def test_update_scheduler_cosine_warm(self):
        """CosineAnnealingWarmRestarts 스케쥴러 업데이트 테스트"""
        scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=1, eta_min=0.001)
        
        initial_lr = self.optimizer.param_groups[0]['lr']
        
        # 스케쥴러 업데이트
        result_lr = update_scheduler(scheduler, None, None)
        
        new_lr = self.optimizer.param_groups[0]['lr']
        assert new_lr != initial_lr
        assert result_lr == new_lr


class TestSchedulerIntegration:
    """스케쥴러 통합 테스트"""
    
    def setup_method(self):
        """테스트 준비"""
        self.device = torch.device('cpu')
    
    def test_full_scheduler_workflow(self):
        """전체 스케쥴러 워크플로우 테스트"""
        cfg = OmegaConf.create({
            'model': {
                'name': 'resnet18',
                'pretrained': False,
                'num_classes': 5
            },
            'training': {
                'lr': 0.01,
                'epochs': 10,
                'scheduler': {
                    'enabled': True,
                    'name': 'cosine',
                    'cosine': {
                        'T_max': 100,
                        'eta_min': 1e-6,
                        'last_epoch': -1
                    }
                }
            }
        })
        
        # 모델 및 옵티마이저 설정
        model, optimizer, loss_fn, scheduler = setup_model_and_optimizer(cfg, self.device)
        
        # 초기 학습률 저장
        initial_lr = optimizer.param_groups[0]['lr']
        
        # 여러 에포크 시뮬레이션
        lrs = []
        for epoch in range(5):
            # 스케쥴러 업데이트
            current_lr = update_scheduler(scheduler, None, cfg)
            lrs.append(current_lr)
        
        # 검증
        assert model is not None
        assert optimizer is not None
        assert loss_fn is not None
        assert scheduler is not None
        assert isinstance(scheduler, CosineAnnealingLR)
        assert len(lrs) == 5
        assert all(lr is not None for lr in lrs)
        assert lrs[-1] != initial_lr  # 학습률이 변경되었는지 확인
    
    def test_scheduler_with_different_configs(self):
        """다양한 설정으로 스케쥴러 테스트"""
        schedulers_configs = [
            {
                'name': 'cosine',
                'cosine': {
                    'T_max': 20,
                    'eta_min': 1e-6,
                    'last_epoch': -1
                }
            },
            {
                'name': 'step',
                'step': {
                    'step_size': 5,
                    'gamma': 0.5,
                    'last_epoch': -1
                }
            },
            {
                'name': 'plateau',
                'plateau': {
                    'mode': 'min',
                    'factor': 0.5,
                    'patience': 3,
                    'threshold': 1e-4,
                    'threshold_mode': 'rel',
                    'cooldown': 0,
                    'min_lr': 1e-8,
                    'eps': 1e-8
                }
            },
            {
                'name': 'cosine_warm',
                'cosine_warm': {
                    'T_0': 10,
                    'T_mult': 1,
                    'eta_min': 1e-6,
                    'last_epoch': -1
                }
            }
        ]
        
        for scheduler_config in schedulers_configs:
            cfg = OmegaConf.create({
                'model': {
                    'name': 'resnet18',
                    'pretrained': False,
                    'num_classes': 5
                },
                'training': {
                    'lr': 0.001,
                    'epochs': 10,
                    'scheduler': {
                        'enabled': True,
                        **scheduler_config
                    }
                }
            })
            
            # 모델 및 옵티마이저 설정
            model, optimizer, loss_fn, scheduler = setup_model_and_optimizer(cfg, self.device)
            
            # 검증
            assert model is not None
            assert optimizer is not None
            assert loss_fn is not None
            assert scheduler is not None
            
            # 스케쥴러 업데이트 테스트
            if scheduler_config['name'] == 'plateau':
                val_metrics = {"val_loss": 1.0}
                result_lr = update_scheduler(scheduler, val_metrics, cfg)
            else:
                result_lr = update_scheduler(scheduler, None, cfg)
            
            assert result_lr is not None


if __name__ == "__main__":
    pytest.main([__file__]) 