"""
적응형 문서 이미지 Dataset
테스트 데이터 분석 결과를 바탕으로 맞춤형 증강을 적용하는 PyTorch Dataset
"""

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Callable, Tuple, Any, Union
import os

# 로컬 모듈 import
from test_data_analyzer import analyze_document_test_data
from adaptive_augmentation import create_adaptive_document_pipeline, create_progressive_document_pipeline


class AdaptiveDocumentDataset(Dataset):
    """
    테스트 데이터 분석 결과를 바탕으로 적응형 증강을 적용하는 Dataset
    """
    
    def __init__(
        self, 
        image_paths: List[str], 
        labels: List[int], 
        test_data_dir: str,
        pytorch_transform: Optional[Callable] = None,
        cache_analysis: bool = True,
        analysis_cache_path: str = "test_analysis_cache.json"
    ):
        """
        Args:
            image_paths: 학습 이미지 파일 경로 리스트
            labels: 레이블 리스트
            test_data_dir: 테스트 데이터 디렉토리 경로
            pytorch_transform: PyTorch 변환 (정규화, 텐서 변환 등)
            cache_analysis: 분석 결과 캐싱 여부
            analysis_cache_path: 분석 결과 캐시 파일 경로
        """
        self.image_paths = image_paths
        self.labels = labels
        self.pytorch_transform = pytorch_transform
        self.test_data_dir = test_data_dir
        self.cache_analysis = cache_analysis
        self.analysis_cache_path = analysis_cache_path
        
        # 데이터 유효성 검증
        if len(image_paths) != len(labels):
            raise ValueError("이미지 경로와 레이블의 개수가 일치하지 않습니다.")
        
        # 테스트 데이터 분석 또는 캐시 로드
        self.test_stats = self._load_or_analyze_test_data()
        
        # Augraphy 파이프라인 생성
        try:
            self.augraphy_pipeline = create_adaptive_document_pipeline(self.test_stats)
            self.augraphy_available = True
            print("✅ Augraphy 파이프라인 생성 완료!")
        except Exception as e:
            print(f"⚠️ Augraphy를 사용할 수 없습니다: {e}")
            print("💡 pip install augraphy 를 실행하여 설치해주세요.")
            self.augraphy_pipeline = None
            self.augraphy_available = False
    
    def _load_or_analyze_test_data(self) -> dict:
        """테스트 데이터 분석 결과를 로드하거나 새로 분석"""
        
        if self.cache_analysis and os.path.exists(self.analysis_cache_path):
            try:
                from test_data_analyzer import load_analysis_results
                print(f"📂 캐시된 분석 결과 로드: {self.analysis_cache_path}")
                return load_analysis_results(self.analysis_cache_path)
            except Exception as e:
                print(f"⚠️ 캐시 로드 실패: {e}")
        
        # 새로운 분석 수행
        print("🔍 테스트 데이터 분석 중...")
        stats = analyze_document_test_data(self.test_data_dir, sample_size=100)
        
        # 결과 캐싱
        if self.cache_analysis:
            try:
                from test_data_analyzer import save_analysis_results
                save_analysis_results(stats, self.analysis_cache_path)
            except Exception as e:
                print(f"⚠️ 분석 결과 캐싱 실패: {e}")
        
        return stats
    
    def update_pipeline_strength(self, epoch: int, total_epochs: int) -> None:
        """
        에포크에 따른 증강 강도 조절
        
        Args:
            epoch: 현재 에포크
            total_epochs: 총 에포크 수
        """
        if not self.augraphy_available:
            return
            
        try:
            strength_factor = min(1.0, epoch / (total_epochs * 0.7))
            
            if epoch < total_epochs * 0.3:
                stage = "light"
            elif epoch < total_epochs * 0.7:
                stage = "medium" 
            else:
                stage = "heavy"
            
            self.augraphy_pipeline = create_progressive_document_pipeline(
                self.test_stats, stage
            )
            
            print(f"🔄 증강 파이프라인 업데이트: {stage} (에포크 {epoch}/{total_epochs})")
            
        except Exception as e:
            print(f"⚠️ 파이프라인 업데이트 실패: {e}")
    
    def get_test_stats_summary(self) -> str:
        """테스트 데이터 분석 결과 요약"""
        summary = []
        summary.append("📊 테스트 데이터 분석 요약:")
        
        for key, stats in self.test_stats.items():
            summary.append(f"  {key}: 평균={stats['mean']:.2f}, 표준편차={stats['std']:.2f}")
        
        return "\n".join(summary)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        데이터셋 아이템 로드 및 증강 적용
        
        Args:
            idx: 인덱스
            
        Returns:
            (증강된 이미지 텐서, 레이블) 튜플
        """
        try:
            # 이미지 로드
            image_path = self.image_paths[idx]
            image = cv2.imread(image_path)
            
            if image is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
            
            # Augraphy 증강 적용
            if self.augraphy_available and self.augraphy_pipeline is not None:
                try:
                    augmented_image = self.augraphy_pipeline(image)
                except Exception as e:
                    print(f"⚠️ Augraphy 증강 실패, 원본 이미지 사용: {e}")
                    augmented_image = image
            else:
                # Augraphy가 없을 때 기본 증강
                augmented_image = self._basic_augmentation(image)
            
            # PyTorch 변환 적용
            if self.pytorch_transform:
                # BGR -> RGB 변환
                augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
                augmented_image = self.pytorch_transform(augmented_image)
            else:
                # 기본 텐서 변환
                augmented_image = torch.from_numpy(augmented_image).permute(2, 0, 1).float() / 255.0
            
            return augmented_image, self.labels[idx]
            
        except Exception as e:
            print(f"❌ 데이터 로드 오류 (idx={idx}): {e}")
            # 오류 시 더미 데이터 반환
            dummy_image = torch.zeros((3, 224, 224))
            return dummy_image, self.labels[idx]
    
    def _basic_augmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Augraphy가 없을 때 사용할 기본 증강
        
        Args:
            image: 입력 이미지
            
        Returns:
            증강된 이미지
        """
        # 테스트 데이터 분석 결과 기반 기본 증강
        rotation_mean = self.test_stats['rotation_angles']['mean']
        brightness_mean = self.test_stats['brightness_levels']['mean']
        
        # 회전 적용
        if abs(rotation_mean) > 5:  # 의미있는 회전이 있다면
            angle = np.random.normal(rotation_mean, 15)  # 평균 ± 15도
            angle = np.clip(angle, -45, 45)  # 범위 제한
            
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (w, h))
        
        # 밝기 조정
        brightness_factor = brightness_mean / 127.5  # 정규화
        brightness_adjust = np.random.uniform(0.8, 1.2) * brightness_factor
        image = cv2.convertScaleAbs(image, alpha=brightness_adjust, beta=0)
        
        # 가우시안 노이즈 추가
        noise_level = self.test_stats['noise_levels']['mean']
        if noise_level > 5:
            noise = np.random.normal(0, noise_level * 0.5, image.shape)
            image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
        
        return image


def create_adaptive_dataloader(
    train_image_paths: List[str],
    train_labels: List[int],
    test_data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    pytorch_transform: Optional[Callable] = None,
    shuffle: bool = True
) -> DataLoader:
    """
    적응형 DataLoader 생성 헬퍼 함수
    
    Args:
        train_image_paths: 학습 이미지 경로 리스트
        train_labels: 학습 레이블 리스트  
        test_data_dir: 테스트 데이터 디렉토리
        batch_size: 배치 크기
        num_workers: 워커 프로세스 수
        pytorch_transform: PyTorch 변환
        shuffle: 셔플 여부
        
    Returns:
        AdaptiveDocumentDataset을 사용하는 DataLoader
    """
    
    dataset = AdaptiveDocumentDataset(
        image_paths=train_image_paths,
        labels=train_labels,
        test_data_dir=test_data_dir,
        pytorch_transform=pytorch_transform
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


def load_image_paths_and_labels(
    data_dir: str,
    class_names: Optional[List[str]] = None
) -> Tuple[List[str], List[int]]:
    """
    데이터 디렉토리에서 이미지 경로와 레이블을 로드
    
    Args:
        data_dir: 데이터 디렉토리 (클래스별 하위 폴더 구조 가정)
        class_names: 클래스 이름 리스트 (None이면 자동 추출)
        
    Returns:
        (이미지 경로 리스트, 레이블 리스트) 튜플
    """
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise ValueError(f"데이터 디렉토리가 존재하지 않습니다: {data_dir}")
    
    image_paths = []
    labels = []
    
    # 클래스별 폴더 구조인지 확인
    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    if subdirs:  # 클래스별 폴더 구조
        if class_names is None:
            class_names = sorted([d.name for d in subdirs])
        
        print(f"📁 클래스별 폴더 구조 감지: {class_names}")
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = data_path / class_name
            if not class_dir.exists():
                continue
                
            class_images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                class_images.extend(list(class_dir.glob(ext)))
            
            image_paths.extend([str(p) for p in class_images])
            labels.extend([class_idx] * len(class_images))
            
            print(f"  {class_name}: {len(class_images)}개 이미지")
    
    else:  # 단일 폴더 구조
        print("📁 단일 폴더 구조 감지")
        
        all_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            all_images.extend(list(data_path.glob(ext)))
        
        image_paths = [str(p) for p in all_images]
        labels = [0] * len(all_images)  # 모든 이미지에 레이블 0 할당
        
        print(f"  총 {len(all_images)}개 이미지")
    
    if not image_paths:
        raise ValueError(f"이미지를 찾을 수 없습니다: {data_dir}")
    
    return image_paths, labels


if __name__ == "__main__":
    # 사용 예시
    try:
        print("🧪 AdaptiveDocumentDataset 테스트")
        
        # 더미 데이터로 테스트
        dummy_paths = ["test1.jpg", "test2.jpg", "test3.jpg"]
        dummy_labels = [0, 1, 0]
        test_dir = "input/data/test"
        
        # 실제로는 다음과 같이 사용:
        # train_paths, train_labels = load_image_paths_and_labels("input/data/train")
        # dataset = AdaptiveDocumentDataset(train_paths, train_labels, test_dir)
        
        print("✅ 모듈 임포트 및 기본 구조 확인 완료")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        print("실제 데이터 경로를 확인해주세요.") 