# -*- coding: utf-8 -*-
"""
inference.py 모듈 테스트
사용자 정의 추론 함수들을 테스트
"""
import os
import sys
import pytest
import torch
import numpy as np
import pandas as pd
import tempfile
from omegaconf import OmegaConf
from unittest.mock import patch, MagicMock
from torch.utils.data import DataLoader, TensorDataset

# 상위 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import (
    predict_single_model,
    predict_kfold_ensemble,
    save_predictions,
    run_inference
)
from data import ImageDataset


class TestPredictSingleModel:
    """predict_single_model 함수 테스트"""
    
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
        
        # 더미 테스트 데이터
        self.images = torch.randn(20, 3, 32, 32)
        self.targets = torch.zeros(20)  # 더미 타겟
        self.dataset = TensorDataset(self.images, self.targets)
        self.test_loader = DataLoader(self.dataset, batch_size=4, shuffle=False)
    
    @patch('inference.log')
    def test_predict_single_model_basic(self, mock_log):
        """기본 단일 모델 추론 테스트"""
        predictions = predict_single_model(self.model, self.test_loader, self.device)
        
        # 반환값 확인
        assert isinstance(predictions, list)
        assert len(predictions) == 20  # 테스트 데이터 개수
        
        # 예측값이 클래스 인덱스인지 확인
        for pred in predictions:
            assert isinstance(pred, (int, np.integer))
            assert 0 <= pred < 10  # 클래스 수
        
        # 로그 호출 확인
        mock_log.info.assert_called_with("추론 시작")
    
    @patch('inference.log')
    def test_predict_single_model_evaluation_mode(self, mock_log):
        """모델이 평가 모드로 설정되는지 확인"""
        self.model.train()  # 훈련 모드로 설정
        assert self.model.training
        
        predict_single_model(self.model, self.test_loader, self.device)
        
        # 평가 모드로 변경되었는지 확인
        assert not self.model.training
    
    @patch('inference.log')
    def test_predict_single_model_different_batch_sizes(self, mock_log):
        """다양한 배치 크기에서 추론 테스트"""
        batch_sizes = [1, 2, 5, 10]
        
        for batch_size in batch_sizes:
            loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
            predictions = predict_single_model(self.model, loader, self.device)
            
            # 항상 전체 데이터 개수와 일치해야 함
            assert len(predictions) == 20


class TestPredictKFoldEnsemble:
    """predict_kfold_ensemble 함수 테스트"""
    
    def setup_method(self):
        """테스트 준비"""
        self.device = torch.device('cpu')
        
        # 3개의 모델 생성 (K-Fold 시뮬레이션)
        self.models = []
        for i in range(3):
            model = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(32*32*3, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 5)
            )
            self.models.append(model)
        
        # 더미 테스트 데이터
        self.images = torch.randn(15, 3, 32, 32)
        self.targets = torch.zeros(15)
        self.dataset = TensorDataset(self.images, self.targets)
        self.test_loader = DataLoader(self.dataset, batch_size=3, shuffle=False)
    
    @patch('inference.log')
    def test_predict_kfold_ensemble_basic(self, mock_log):
        """기본 K-Fold 앙상블 추론 테스트"""
        predictions = predict_kfold_ensemble(self.models, self.test_loader, self.device)
        
        # 반환값 확인
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 15  # 테스트 데이터 개수
        
        # 예측값이 클래스 인덱스인지 확인
        for pred in predictions:
            assert isinstance(pred, (int, np.integer))
            assert 0 <= pred < 5  # 클래스 수
        
        # 로그 호출 확인 (마지막 호출)
        mock_log.info.assert_called_with("K-Fold 앙상블 예측 계산 중...")
    
    @patch('inference.log')
    def test_predict_kfold_ensemble_single_model(self, mock_log):
        """단일 모델로 K-Fold 앙상블 테스트"""
        single_model_list = [self.models[0]]
        predictions = predict_kfold_ensemble(single_model_list, self.test_loader, self.device)
        
        # 여전히 정상 동작해야 함
        assert len(predictions) == 15
        
        # 단일 모델 추론과 비교
        direct_predictions = predict_single_model(self.models[0], self.test_loader, self.device)
        np.testing.assert_array_equal(predictions, direct_predictions)
    
    @patch('inference.log')
    def test_predict_kfold_ensemble_all_models_eval(self, mock_log):
        """모든 모델이 평가 모드로 설정되는지 확인"""
        # 모든 모델을 훈련 모드로 설정
        for model in self.models:
            model.train()
            assert model.training
        
        predict_kfold_ensemble(self.models, self.test_loader, self.device)
        
        # 모든 모델이 평가 모드로 변경되었는지 확인
        for model in self.models:
            assert not model.training


class TestSavePredictions:
    """save_predictions 함수 테스트"""
    
    def setup_method(self):
        """테스트 준비"""
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.temp_dir, "test_predictions.csv")
        
        # 더미 테스트 데이터셋 (ID 정보 포함)
        self.test_dataset = MagicMock()
        self.test_dataset.df = pd.DataFrame({
            'ID': [f'test_{i}.jpg' for i in range(10)],
            'target': [0] * 10  # 더미 타겟
        })
        
        self.predictions = [i % 3 for i in range(10)]  # 더미 예측값
        
        # 더미 sample_submission.csv 생성
        sample_submission_path = os.path.join(self.temp_dir, "sample_submission.csv")
        sample_submission_df = pd.DataFrame({
            'ID': [f'test_{i}.jpg' for i in range(10)],
            'target': [0] * 10
        })
        sample_submission_df.to_csv(sample_submission_path, index=False)
        
        # 설정
        self.cfg = OmegaConf.create({
            'data': {
                'data_path': self.temp_dir
            },
            'output': {
                'dir': self.temp_dir,
                'filename': 'test_predictions.csv'
            }
        })
    
    @patch('inference.log')
    def test_save_predictions_basic(self, mock_log):
        """기본 예측 결과 저장 테스트"""
        pred_df = save_predictions(self.predictions, self.test_dataset, self.cfg)
        
        # 파일이 생성되었는지 확인
        output_file = f"{self.cfg.output.dir}/{self.cfg.output.filename}"
        assert os.path.exists(output_file)
        
        # 파일 내용 확인
        saved_df = pd.read_csv(output_file)
        assert len(saved_df) == 10
        assert 'ID' in saved_df.columns
        assert 'target' in saved_df.columns
        
        # 예측값 확인
        assert saved_df['target'].tolist() == self.predictions
        
        # 로그 호출 확인
        mock_log.info.assert_called_with(f"예측 결과 저장 완료: {output_file}")
    
    @patch('inference.log')
    def test_save_predictions_correct_format(self, mock_log):
        """저장된 파일의 형식 확인"""
        pred_df = save_predictions(self.predictions, self.test_dataset, self.cfg)
        
        # 파일 읽기
        output_file = f"{self.cfg.output.dir}/{self.cfg.output.filename}"
        saved_df = pd.read_csv(output_file)
        
        # 예상 형식과 일치하는지 확인
        expected_df = pd.DataFrame({
            'ID': [f'test_{i}.jpg' for i in range(10)],
            'target': [i % 3 for i in range(10)]
        })
        
        pd.testing.assert_frame_equal(saved_df, expected_df)
    
    @patch('inference.log')
    def test_save_predictions_different_lengths(self, mock_log):
        """예측값과 데이터셋 길이 불일치 테스트"""
        # 길이가 다른 예측값
        wrong_predictions = [0, 1, 2]  # 3개 (데이터셋은 10개)
        
        # 예외가 발생해야 함
        with pytest.raises(ValueError):
            save_predictions(wrong_predictions, self.test_dataset, self.cfg)


class TestRunInference:
    """run_inference 함수 테스트"""
    
    def setup_method(self):
        """테스트 준비"""
        self.device = torch.device('cpu')
        self.temp_dir = tempfile.mkdtemp()
        
        # 더미 모델
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32*32*3, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 5)
        )
        
        # 더미 테스트 데이터
        self.images = torch.randn(8, 3, 32, 32)
        self.targets = torch.zeros(8)
        self.dataset = TensorDataset(self.images, self.targets)
        self.test_loader = DataLoader(self.dataset, batch_size=2, shuffle=False)
        
        # 더미 테스트 데이터셋 (ID 정보 포함)
        self.test_dataset = MagicMock()
        self.test_dataset.df = pd.DataFrame({
            'ID': [f'test_{i}.jpg' for i in range(8)],
            'target': [0] * 8
        })
        
        # 더미 sample_submission.csv 생성
        sample_submission_path = os.path.join(self.temp_dir, "sample_submission.csv")
        sample_submission_df = pd.DataFrame({
            'ID': [f'test_{i}.jpg' for i in range(8)],
            'target': [0] * 8
        })
        sample_submission_df.to_csv(sample_submission_path, index=False)
        
        # 설정
        self.cfg = OmegaConf.create({
            'data': {
                'data_path': self.temp_dir
            },
            'output': {
                'dir': self.temp_dir,
                'filename': 'test_inference.csv'
            },
            'wandb': {
                'enabled': False
            }
        })
    
    @patch('inference.predict_single_model')
    @patch('inference.save_predictions')
    @patch('inference.upload_to_wandb')
    @patch('inference.log')
    def test_run_inference_single_model(self, mock_log, mock_upload, mock_save, mock_predict):
        """단일 모델 추론 실행 테스트"""
        # Mock 설정
        mock_predict.return_value = [i % 5 for i in range(8)]
        mock_save.return_value = pd.DataFrame({'ID': [f'test_{i}' for i in range(8)], 'target': [i % 5 for i in range(8)]})
        mock_upload.return_value = None
        
        result_df = run_inference(
            self.model, self.test_loader, self.test_dataset, 
            self.cfg, self.device, is_kfold=False
        )
        
        # 함수 호출 확인
        mock_predict.assert_called_once_with(self.model, self.test_loader, self.device)
        mock_save.assert_called_once()
        
        # 반환값 확인
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 8
    
    @patch('inference.predict_kfold_ensemble')
    @patch('inference.save_predictions')
    @patch('inference.upload_to_wandb')
    @patch('inference.log')
    def test_run_inference_kfold_ensemble(self, mock_log, mock_upload, mock_save, mock_predict):
        """K-Fold 앙상블 추론 실행 테스트"""
        # Mock 설정
        mock_predict.return_value = [i % 5 for i in range(8)]
        mock_save.return_value = pd.DataFrame({'ID': [f'test_{i}' for i in range(8)], 'target': [i % 5 for i in range(8)]})
        mock_upload.return_value = None
        
        models = [self.model, self.model]  # 2개 모델
        
        result_df = run_inference(
            models, self.test_loader, self.test_dataset, 
            self.cfg, self.device, is_kfold=True
        )
        
        # 함수 호출 확인
        mock_predict.assert_called_once_with(models, self.test_loader, self.device)
        mock_save.assert_called_once()
        
        # 반환값 확인
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 8
    
    @patch('inference.predict_single_model')
    @patch('inference.log')
    def test_run_inference_file_output(self, mock_log, mock_predict):
        """파일 출력 경로 테스트"""
        # Mock 설정
        mock_predict.return_value = [i % 5 for i in range(8)]
        
        run_inference(
            self.model, self.test_loader, self.test_dataset, 
            self.cfg, self.device, is_kfold=False
        )
        
        # 출력 파일이 생성되었는지 확인
        output_path = os.path.join(self.temp_dir, 'test_inference.csv')
        assert os.path.exists(output_path)
        
        # 파일 내용 확인
        result_df = pd.read_csv(output_path)
        assert len(result_df) == 8
        assert 'ID' in result_df.columns
        assert 'target' in result_df.columns


class TestInferenceIntegration:
    """추론 관련 통합 테스트"""
    
    def setup_method(self):
        """테스트 준비"""
        self.device = torch.device('cpu')
        self.temp_dir = tempfile.mkdtemp()
        
        # 실제 크기의 모델 생성
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32*32*3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 17)  # 실제 클래스 수
        )
        
        # 더미 테스트 데이터
        self.images = torch.randn(12, 3, 32, 32)
        self.targets = torch.zeros(12)
        self.dataset = TensorDataset(self.images, self.targets)
        self.test_loader = DataLoader(self.dataset, batch_size=4, shuffle=False)
        
        # 더미 테스트 데이터셋
        self.test_dataset = MagicMock()
        self.test_dataset.df = pd.DataFrame({
            'ID': [f'test_{i}.jpg' for i in range(12)],
            'target': [0] * 12
        })
        
        # 더미 sample_submission.csv 생성
        sample_submission_path = os.path.join(self.temp_dir, "sample_submission.csv")
        sample_submission_df = pd.DataFrame({
            'ID': [f'test_{i}.jpg' for i in range(12)],
            'target': [0] * 12
        })
        sample_submission_df.to_csv(sample_submission_path, index=False)
        
        # 설정
        self.cfg = OmegaConf.create({
            'data': {
                'data_path': self.temp_dir
            },
            'output': {
                'dir': self.temp_dir,
                'filename': 'integration_test.csv'
            },
            'wandb': {
                'enabled': False
            }
        })
    
    @patch('inference.log')
    def test_end_to_end_inference(self, mock_log):
        """전체 추론 파이프라인 테스트"""
        # 단일 모델 추론
        result_df = run_inference(
            self.model, self.test_loader, self.test_dataset, 
            self.cfg, self.device, is_kfold=False
        )
        
        # 결과 검증
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 12
        assert 'ID' in result_df.columns
        assert 'target' in result_df.columns
        
        # 예측값이 유효한 클래스 인덱스인지 확인
        for pred in result_df['target']:
            assert isinstance(pred, (int, np.integer))
            assert 0 <= pred < 17
        
        # 파일 생성 확인
        output_path = os.path.join(self.temp_dir, 'integration_test.csv')
        assert os.path.exists(output_path)
    
    @patch('inference.log')
    def test_inference_consistency(self, mock_log):
        """추론 일관성 테스트"""
        # 동일한 입력으로 두 번 추론
        result1 = run_inference(
            self.model, self.test_loader, self.test_dataset, 
            self.cfg, self.device, is_kfold=False
        )
        
        result2 = run_inference(
            self.model, self.test_loader, self.test_dataset, 
            self.cfg, self.device, is_kfold=False
        )
        
        # 결과가 동일한지 확인
        pd.testing.assert_frame_equal(result1, result2)
    
    @patch('inference.log')
    def test_kfold_vs_single_model(self, mock_log):
        """K-Fold 앙상블과 단일 모델 비교 테스트"""
        # 단일 모델 추론
        single_result = run_inference(
            self.model, self.test_loader, self.test_dataset, 
            self.cfg, self.device, is_kfold=False
        )
        
        # K-Fold 앙상블 (단일 모델로 구성)
        kfold_result = run_inference(
            [self.model], self.test_loader, self.test_dataset, 
            self.cfg, self.device, is_kfold=True
        )
        
        # 둘 다 같은 구조여야 함
        assert single_result.shape == kfold_result.shape
        assert list(single_result.columns) == list(kfold_result.columns)
        
        # 단일 모델 앙상블은 원본과 같아야 함
        pd.testing.assert_frame_equal(single_result, kfold_result)


if __name__ == "__main__":
    pytest.main([__file__]) 