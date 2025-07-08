# -*- coding: utf-8 -*-
"""
models.py 모듈 테스트
사용자 정의 모델 생성 및 관련 함수들을 테스트
"""
import os
import sys
import pytest
import torch
import tempfile
import shutil
from omegaconf import OmegaConf

# 상위 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    create_model,
    setup_model_and_optimizer,
    save_model,
    load_model,
    save_model_with_metadata,
    load_model_with_metadata,
    get_model_save_path,
    get_seed_fold_model_path,
    load_model_for_inference,
    get_model_info,
)


class TestModelCreation:
    """모델 생성 함수 테스트"""
    
    def test_create_model_pretrained(self):
        """사전 훈련된 모델 생성 테스트"""
        model_name = 'resnet18'
        num_classes = 10
        
        model = create_model(model_name, pretrained=True, num_classes=num_classes)
        
        # 모델 객체 확인
        assert model is not None
        assert hasattr(model, 'forward')
        
        # 입력/출력 차원 확인
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (1, num_classes)
    
    def test_create_model_no_pretrained(self):
        """사전 훈련 없는 모델 생성 테스트"""
        model_name = 'resnet18'
        num_classes = 5
        
        model = create_model(model_name, pretrained=False, num_classes=num_classes)
        
        # 모델 객체 확인
        assert model is not None
        
        # 출력 차원 확인
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (1, num_classes)
    
    def test_create_model_different_architectures(self):
        """다양한 모델 아키텍처 테스트"""
        models_to_test = [
            ('resnet18', 17),
            ('resnet34', 10),
            ('efficientnet_b0', 5)
        ]
        
        for model_name, num_classes in models_to_test:
            model = create_model(model_name, pretrained=False, num_classes=num_classes)
            
            # 기본 검증
            assert model is not None
            
            # 출력 차원 확인
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = model(dummy_input)
            assert output.shape == (1, num_classes)


class TestModelAndOptimizerSetup:
    """모델과 옵티마이저 설정 함수 테스트"""
    
    def test_setup_model_and_optimizer_cpu(self):
        """CPU에서 모델과 옵티마이저 설정 테스트 (스케쥴러 비활성화)"""
        cfg = OmegaConf.create({
            'model': {
                'name': 'resnet18',
                'pretrained': False,
                'num_classes': 10
            },
            'train': {
                'lr': 0.001,
                'scheduler': {
                    'enabled': False
                }
            }
        })
        device = torch.device('cpu')
        
        model, optimizer, loss_fn, scheduler = setup_model_and_optimizer(cfg, device)
        
        # 모델 확인
        assert model is not None
        assert next(model.parameters()).device == device
        
        # 옵티마이저 확인
        assert optimizer is not None
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]['lr'] == 0.001
        
        # 손실 함수 확인
        assert loss_fn is not None
        assert isinstance(loss_fn, torch.nn.CrossEntropyLoss)
        
        # 스케쥴러 확인 (비활성화됨)
        assert scheduler is None
    
    def test_setup_model_and_optimizer_with_scheduler(self):
        """스케쥴러 포함 설정 테스트"""
        cfg = OmegaConf.create({
            'model': {
                'name': 'resnet18',
                'pretrained': False,
                'num_classes': 5
            },
            'train': {
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
        device = torch.device('cpu')
        
        model, optimizer, loss_fn, scheduler = setup_model_and_optimizer(cfg, device)
        
        assert model is not None
        assert optimizer is not None
        assert loss_fn is not None
        assert scheduler is not None  # 스케쥴러가 생성되어야 함
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_setup_model_and_optimizer_cuda(self):
        """CUDA에서 모델과 옵티마이저 설정 테스트"""
        cfg = OmegaConf.create({
            'model': {
                'name': 'resnet18',
                'pretrained': False,
                'num_classes': 5
            },
            'train': {
                'lr': 0.01,
                'scheduler': {
                    'enabled': False
                }
            }
        })
        device = torch.device('cuda')
        
        model, optimizer, loss_fn, scheduler = setup_model_and_optimizer(cfg, device)
        
        # 모델이 GPU에 있는지 확인
        assert next(model.parameters()).device.type == 'cuda'
        
        # 옵티마이저 학습률 확인
        assert optimizer.param_groups[0]['lr'] == 0.01
        
        # 스케쥴러 확인 (비활성화됨)
        assert scheduler is None


class TestModelSaveLoad:
    """모델 저장/로딩 함수 테스트"""
    
    def setup_method(self):
        """테스트 준비"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.pth")
    
    def teardown_method(self):
        """테스트 정리"""
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_model(self):
        """모델 저장 및 로딩 테스트"""
        # 원본 모델 생성
        original_model = create_model('resnet18', pretrained=False, num_classes=10)
        
        # 모델 저장
        save_model(original_model, self.model_path)
        
        # 파일이 생성되었는지 확인
        assert os.path.exists(self.model_path)
        
        # 새로운 모델 생성 후 로딩
        new_model = create_model('resnet18', pretrained=False, num_classes=10)
        loaded_model = load_model(new_model, self.model_path)
        
        # 로딩된 모델 확인
        assert loaded_model is not None
        
        # 파라미터가 동일한지 확인
        original_params = list(original_model.parameters())
        loaded_params = list(loaded_model.parameters())
        
        assert len(original_params) == len(loaded_params)
        
        for orig_param, loaded_param in zip(original_params, loaded_params):
            assert torch.equal(orig_param, loaded_param)
    
    def test_save_model_creates_file(self):
        """모델 저장 시 파일 생성 확인"""
        model = create_model('resnet18', pretrained=False, num_classes=5)
        
        # 파일이 존재하지 않는지 확인
        assert not os.path.exists(self.model_path)
        
        # 모델 저장
        save_model(model, self.model_path)
        
        # 파일이 생성되었는지 확인
        assert os.path.exists(self.model_path)
        assert os.path.getsize(self.model_path) > 0
    
    def test_save_load_model_with_metadata(self):
        """메타데이터 포함 모델 저장/로드 테스트"""
        model = create_model("resnet18", pretrained=False, num_classes=5)
        save_path = os.path.join(self.temp_dir, "test_model_meta.pth")
        metadata = {
            "epoch": 10,
            "val_acc": 0.95,
            "model_name": "resnet18"
        }
        
        save_model_with_metadata(model, save_path, metadata)
        
        assert os.path.exists(save_path)
        
        # 메타데이터 포함 로드 테스트
        loaded_model = create_model("resnet18", pretrained=False, num_classes=5)
        loaded_model, loaded_metadata = load_model_with_metadata(loaded_model, save_path)
        
        assert loaded_model is not None
        assert loaded_metadata is not None
        assert loaded_metadata["epoch"] == 10
        assert loaded_metadata["val_acc"] == 0.95
        assert loaded_metadata["model_name"] == "resnet18"
    
    def test_get_model_save_path(self):
        """모델 저장 경로 생성 테스트"""
        cfg = OmegaConf.create({
            'model': {
                'name': 'resnet18'
            },
            'train': {
                'seed': 123
            },
            'model_save': {
                'dir': self.temp_dir
            }
        })
        
        best_path = get_model_save_path(cfg, "best")
        last_path = get_model_save_path(cfg, "last")
        
        assert best_path is not None
        assert last_path is not None
        assert "resnet18_seed123_best.pth" in best_path
        assert "resnet18_seed123_last.pth" in last_path
    

class TestSeedFoldPath:
    """시드 및 폴드 기반 경로 생성 및 로드 테스트"""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def test_get_seed_fold_model_path(self):
        cfg = OmegaConf.create({
            'model': {'name': 'resnet18'},
            'model_save': {'dir': self.temp_dir}
        })

        path = get_seed_fold_model_path(cfg, seed=1, fold=2)
        assert os.path.basename(path) == 'resnet18_seed1_fold2.pth'
        assert os.path.dirname(path) == self.temp_dir

    def test_load_model_for_inference(self):
        cfg = OmegaConf.create({
            'model': {'name': 'resnet18', 'num_classes': 3},
            'model_save': {'dir': self.temp_dir}
        })

        model = create_model('resnet18', pretrained=False, num_classes=3)
        save_path = get_seed_fold_model_path(cfg, seed=0, fold=1)
        save_model_with_metadata(model, save_path, {'epoch': 1})

        loaded = load_model_for_inference(cfg, save_path, torch.device('cpu'))
        assert loaded is not None
        assert isinstance(loaded, torch.nn.Module)


class TestModelInfo:
    """모델 정보 함수 테스트"""
    
    def test_get_model_info(self):
        """모델 정보 조회 테스트"""
        model = create_model('resnet18', pretrained=False, num_classes=17)
        
        info = get_model_info(model)
        
        # 반환값 구조 확인
        assert isinstance(info, dict)
        assert 'total_params' in info
        assert 'trainable_params' in info
        assert 'model_name' in info
        
        # 파라미터 수 확인
        assert isinstance(info['total_params'], int)
        assert isinstance(info['trainable_params'], int)
        assert info['total_params'] > 0
        assert info['trainable_params'] > 0
        
        # 기본적으로 모든 파라미터가 훈련 가능해야 함
        assert info['total_params'] == info['trainable_params']
        
        # 모델 이름 확인
        assert isinstance(info['model_name'], str)
    
    def test_get_model_info_frozen_params(self):
        """일부 파라미터가 동결된 모델 정보 테스트"""
        model = create_model('resnet18', pretrained=False, num_classes=10)
        
        # 첫 번째 레이어의 파라미터 동결
        first_param = next(model.parameters())
        first_param.requires_grad = False
        
        info = get_model_info(model)
        
        # 전체 파라미터 수와 훈련 가능한 파라미터 수가 달라야 함
        assert info['total_params'] > info['trainable_params']
    
    def test_get_model_info_different_models(self):
        """다양한 모델의 정보 비교 테스트"""
        small_model = create_model('resnet18', pretrained=False, num_classes=10)
        large_model = create_model('resnet34', pretrained=False, num_classes=10)
        
        small_info = get_model_info(small_model)
        large_info = get_model_info(large_model)
        
        # ResNet34가 ResNet18보다 파라미터가 많아야 함
        assert large_info['total_params'] > small_info['total_params']
        assert large_info['trainable_params'] > small_info['trainable_params']


class TestModelIntegration:
    """모델 관련 통합 테스트"""
    
    def test_model_forward_pass(self):
        """모델 순전파 테스트"""
        model = create_model('resnet18', pretrained=False, num_classes=17)
        model.eval()
        
        # 배치 크기 다양하게 테스트
        batch_sizes = [1, 4, 8]
        
        for batch_size in batch_sizes:
            dummy_input = torch.randn(batch_size, 3, 224, 224)
            
            with torch.no_grad():
                output = model(dummy_input)
            
            # 출력 크기 확인
            assert output.shape == (batch_size, 17)
            
            # 출력이 유한한 값인지 확인
            assert torch.isfinite(output).all()
    
    def test_model_training_mode(self):
        """모델 훈련/평가 모드 전환 테스트"""
        model = create_model('resnet18', pretrained=False, num_classes=10)
        
        # 기본적으로 훈련 모드
        assert model.training is True
        
        # 평가 모드로 전환
        model.eval()
        assert model.training is False
        
        # 다시 훈련 모드로 전환
        model.train()
        assert model.training is True
    
    def test_model_gradient_computation(self):
        """모델 그래디언트 계산 테스트"""
        cfg = OmegaConf.create({
            'model': {
                'name': 'resnet18',
                'pretrained': False,
                'num_classes': 5
            },
            'train': {
                'lr': 0.001,
                'scheduler': {
                    'enabled': False
                }
            }
        })
        device = torch.device('cpu')
        
        model, optimizer, loss_fn, scheduler = setup_model_and_optimizer(cfg, device)
        
        # 더미 데이터
        dummy_input = torch.randn(2, 3, 224, 224)
        dummy_target = torch.randint(0, 5, (2,))
        
        # 순전파
        output = model(dummy_input)
        loss = loss_fn(output, dummy_target)
        
        # 역전파
        loss.backward()
        
        # 그래디언트가 계산되었는지 확인
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        
        assert has_gradients, "모델 파라미터에 그래디언트가 계산되지 않았습니다"


class TestLabelSmoothing:
    """LabelSmoothingLoss 클래스 테스트"""
    
    def test_label_smoothing_loss_creation(self):
        """LabelSmoothingLoss 생성 테스트"""
        from models import LabelSmoothingLoss
        
        num_classes = 10
        smoothing = 0.1
        
        loss_fn = LabelSmoothingLoss(num_classes=num_classes, smoothing=smoothing)
        
        assert loss_fn is not None
        assert loss_fn.num_classes == num_classes
        assert loss_fn.smoothing == smoothing
        assert loss_fn.confidence == 1.0 - smoothing
    
    def test_label_smoothing_loss_forward(self):
        """LabelSmoothingLoss forward 테스트"""
        from models import LabelSmoothingLoss
        
        num_classes = 5
        smoothing = 0.1
        batch_size = 4
        
        loss_fn = LabelSmoothingLoss(num_classes=num_classes, smoothing=smoothing)
        
        # 더미 데이터 생성
        pred = torch.randn(batch_size, num_classes)
        target = torch.randint(0, num_classes, (batch_size,))
        
        # Forward pass
        loss = loss_fn(pred, target)
        
        # 결과 확인
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # 스칼라 값
        assert loss.item() >= 0  # 손실은 음수가 될 수 없음
    
    def test_label_smoothing_vs_cross_entropy(self):
        """LabelSmoothing과 CrossEntropy 비교 테스트"""
        from models import LabelSmoothingLoss
        
        num_classes = 5
        smoothing = 0.1
        batch_size = 4
        
        # 두 loss 함수 생성
        label_smoothing_loss = LabelSmoothingLoss(num_classes=num_classes, smoothing=smoothing)
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        
        # 더미 데이터 생성
        pred = torch.randn(batch_size, num_classes, requires_grad=True)
        target = torch.randint(0, num_classes, (batch_size,))
        
        # 손실 계산
        ls_loss = label_smoothing_loss(pred, target)
        ce_loss = cross_entropy_loss(pred, target)
        
        # 모두 유효한 손실 값이어야 함
        assert torch.isfinite(ls_loss)
        assert torch.isfinite(ce_loss)
        assert ls_loss.item() >= 0
        assert ce_loss.item() >= 0
    
    def test_label_smoothing_with_config(self):
        """설정을 통한 LabelSmoothing 테스트"""
        # LabelSmoothing 활성화 설정
        cfg_enabled = OmegaConf.create({
            'model': {
                'name': 'resnet18',
                'pretrained': False,
                'num_classes': 10
            },
            'train': {
                'lr': 0.001,
                'label_smoothing': {
                    'enabled': True,
                    'smoothing': 0.15
                },
                'mixed_precision': {
                    'enabled': False
                },
                'scheduler': {
                    'enabled': False
                }
            }
        })
        
        # LabelSmoothing 비활성화 설정
        cfg_disabled = OmegaConf.create({
            'model': {
                'name': 'resnet18',
                'pretrained': False,
                'num_classes': 10
            },
            'train': {
                'lr': 0.001,
                'label_smoothing': {
                    'enabled': False,
                    'smoothing': 0.1
                },
                'mixed_precision': {
                    'enabled': False
                },
                'scheduler': {
                    'enabled': False
                }
            }
        })
        
        device = torch.device('cpu')
        
        # LabelSmoothing 활성화 테스트
        model1, optimizer1, loss_fn1, scheduler1 = setup_model_and_optimizer(cfg_enabled, device)
        from models import LabelSmoothingLoss
        assert isinstance(loss_fn1, LabelSmoothingLoss)
        assert loss_fn1.smoothing == 0.15
        
        # LabelSmoothing 비활성화 테스트
        model2, optimizer2, loss_fn2, scheduler2 = setup_model_and_optimizer(cfg_disabled, device)
        assert isinstance(loss_fn2, torch.nn.CrossEntropyLoss)
    
    def test_label_smoothing_different_smoothing_values(self):
        """다양한 smoothing 값 테스트"""
        from models import LabelSmoothingLoss
        
        num_classes = 5
        batch_size = 4
        smoothing_values = [0.0, 0.1, 0.2, 0.3, 0.5]
        
        # 더미 데이터 생성
        pred = torch.randn(batch_size, num_classes)
        target = torch.randint(0, num_classes, (batch_size,))
        
        losses = []
        for smoothing in smoothing_values:
            loss_fn = LabelSmoothingLoss(num_classes=num_classes, smoothing=smoothing)
            loss = loss_fn(pred, target)
            losses.append(loss.item())
            
            # 유효한 손실 값인지 확인
            assert torch.isfinite(loss)
            assert loss.item() >= 0
        
        # smoothing=0.0일 때와 다른 값들 비교
        assert len(set(losses)) > 1  # 다른 smoothing 값들이 다른 손실을 만드는지 확인


class TestMixedPrecisionConfig:
    """Mixed Precision Training 설정 테스트"""
    
    def test_mixed_precision_config_enabled(self):
        """Mixed Precision 활성화 설정 테스트"""
        cfg = OmegaConf.create({
            'model': {
                'name': 'resnet18',
                'pretrained': False,
                'num_classes': 10
            },
            'train': {
                'lr': 0.001,
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
            }
        })
        
        device = torch.device('cpu')
        
        # 함수 호출 테스트 (실제 Mixed Precision은 CUDA에서만 동작)
        model, optimizer, loss_fn, scheduler = setup_model_and_optimizer(cfg, device)
        
        # 기본 검증
        assert model is not None
        assert optimizer is not None
        assert loss_fn is not None
        assert scheduler is None
    
    def test_mixed_precision_config_disabled(self):
        """Mixed Precision 비활성화 설정 테스트"""
        cfg = OmegaConf.create({
            'model': {
                'name': 'resnet18',
                'pretrained': False,
                'num_classes': 10
            },
            'train': {
                'lr': 0.001,
                'label_smoothing': {
                    'enabled': False,
                    'smoothing': 0.1
                },
                'mixed_precision': {
                    'enabled': False
                },
                'scheduler': {
                    'enabled': False
                }
            }
        })
        
        device = torch.device('cpu')
        
        # 함수 호출 테스트
        model, optimizer, loss_fn, scheduler = setup_model_and_optimizer(cfg, device)
        
        # 기본 검증
        assert model is not None
        assert optimizer is not None
        assert loss_fn is not None
        assert scheduler is None
    
    def test_combined_label_smoothing_and_mixed_precision(self):
        """LabelSmoothing과 Mixed Precision 동시 활성화 테스트"""
        cfg = OmegaConf.create({
            'model': {
                'name': 'resnet18',
                'pretrained': False,
                'num_classes': 10
            },
            'train': {
                'lr': 0.001,
                'label_smoothing': {
                    'enabled': True,
                    'smoothing': 0.1
                },
                'mixed_precision': {
                    'enabled': True
                },
                'scheduler': {
                    'enabled': False
                }
            }
        })
        
        device = torch.device('cpu')
        
        # 함수 호출 테스트
        model, optimizer, loss_fn, scheduler = setup_model_and_optimizer(cfg, device)
        
        # LabelSmoothing 확인
        from models import LabelSmoothingLoss
        assert isinstance(loss_fn, LabelSmoothingLoss)
        assert loss_fn.smoothing == 0.1
        
        # 기본 검증
        assert model is not None
        assert optimizer is not None
        assert scheduler is None


if __name__ == "__main__":
    pytest.main([__file__]) 