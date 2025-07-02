"""
적응형 Augraphy 증강 모듈
테스트 데이터 분석 결과를 바탕으로 맞춤형 Augraphy 파이프라인을 생성합니다.
"""

from augraphy import *
import numpy as np
from typing import Dict, Tuple, Any


def create_adaptive_document_pipeline(test_data_stats: Dict[str, Dict[str, float]]) -> AugraphyPipeline:
    """
    테스트 데이터 분석 결과를 바탕으로 맞춤형 파이프라인 생성
    
    Args:
        test_data_stats: 테스트 데이터 분석 결과
        
    Returns:
        맞춤형 AugraphyPipeline 객체
    """
    
    # 회전 파라미터 설정
    rotation_mean = test_data_stats['rotation_angles']['mean']
    rotation_std = test_data_stats['rotation_angles']['std']
    rotation_range = (
        max(-180, int(rotation_mean - 2*rotation_std)),
        min(180, int(rotation_mean + 2*rotation_std))
    )
    
    # 밝기 파라미터 설정
    brightness_mean = test_data_stats['brightness_levels']['mean']
    brightness_factor = brightness_mean / 127.5  # 정규화 (0-255 -> 0-2)
    brightness_range = (
        max(0.5, brightness_factor - 0.3),
        min(2.0, brightness_factor + 0.3)
    )
    
    # 블러 파라미터 설정
    blur_mean = test_data_stats['blur_levels']['mean']
    if blur_mean < 100:  # 낮은 선명도 = 높은 블러
        blur_intensity = "high"
        blur_range = (2, 5)
        blur_probability = 0.7
    elif blur_mean < 500:
        blur_intensity = "medium"  
        blur_range = (1, 3)
        blur_probability = 0.5
    else:
        blur_intensity = "low"
        blur_range = (0, 2)
        blur_probability = 0.3
    
    # 노이즈 파라미터 설정
    noise_mean = test_data_stats['noise_levels']['mean']
    noise_range = (0, min(50, int(noise_mean * 2)))
    noise_probability = 0.6 if noise_mean > 5 else 0.3
    
    # 대비 파라미터 설정
    contrast_mean = test_data_stats['contrast_levels']['mean']
    contrast_factor = max(0.8, min(1.2, contrast_mean / 50))  # 대략적인 정규화
    
    print(f"📊 테스트 데이터 분석 결과 기반 파라미터:")
    print(f"  - 회전 범위: {rotation_range}°")
    print(f"  - 밝기 범위: {brightness_range}")
    print(f"  - 블러 강도: {blur_intensity} (범위: {blur_range})")
    print(f"  - 노이즈 범위: {noise_range} (확률: {noise_probability})")
    print(f"  - 대비 인수: {contrast_factor}")
    
    # Paper Phase 구성
    paper_phase = []
    if noise_mean > 10:  # 노이즈가 많다면 종이 질감 추가
        paper_phase.extend([
            PaperFactory(p=0.4),
            SubtleNoise(p=0.5),
        ])
    
    if contrast_mean < 30:  # 대비가 낮다면 색상 종이 효과 추가
        paper_phase.append(ColorPaper(p=0.3))
    
    # Post Phase 구성
    post_phase = [
        # 테스트 데이터의 회전 패턴 반영
        Geometric(
            rotate_range=rotation_range,
            scale_range=(0.95, 1.05),
            translation_range=(-10, 10),
            p=0.9  # 높은 확률로 적용
        ),
        
        # 테스트 데이터의 밝기 패턴 반영
        Brightness(
            brightness_range=brightness_range,
            p=0.8
        ),
        
        # 대비 조정
        Gamma(
            gamma_range=(contrast_factor - 0.2, contrast_factor + 0.2),
            p=0.6
        ),
        
        # 테스트 데이터의 노이즈 패턴 반영
        Noise(
            noise_type="gauss",
            noise_value=noise_range,
            p=noise_probability
        ),
        
        # 추가 현실적인 문서 효과들
        LightingGradient(p=0.3),  # 조명 그라데이션
        ShadowCast(p=0.2),        # 그림자
    ]
    
    # 블러 조건부 추가
    if blur_intensity != "low":
        post_phase.append(
            Blur(blur_value=blur_range, p=blur_probability)
        )
    
    # 스캔 품질 관련 효과 추가
    if blur_mean < 300:  # 블러가 심하면 스캔 품질 저하 효과 추가
        post_phase.extend([
            BadPhotoCopy(p=0.3),
            DirtyDrum(p=0.2),
            DirtyRollers(p=0.2),
        ])
    
    return AugraphyPipeline(
        ink_phase=[],  # 필요시 잉크 효과 추가 가능
        paper_phase=paper_phase,
        post_phase=post_phase
    )


def create_progressive_document_pipeline(
    test_data_stats: Dict[str, Dict[str, float]], 
    stage: str = "light"
) -> AugraphyPipeline:
    """
    단계별 문서 증강 강도 조절
    
    Args:
        test_data_stats: 테스트 데이터 분석 결과
        stage: 증강 강도 ("light", "medium", "heavy")
        
    Returns:
        단계별 AugraphyPipeline 객체
    """
    
    # 기본 파라미터 추출
    rotation_mean = test_data_stats['rotation_angles']['mean']
    rotation_std = test_data_stats['rotation_angles']['std']
    
    if stage == "light":
        # 가벼운 증강 (초기 학습)
        rotation_range = (
            max(-30, int(rotation_mean - 0.5*rotation_std)),
            min(30, int(rotation_mean + 0.5*rotation_std))
        )
        
        return AugraphyPipeline(
            post_phase=[
                Geometric(rotate_range=rotation_range, p=0.5),
                Brightness(brightness_range=(0.9, 1.1), p=0.3),
                Noise(noise_type="gauss", noise_value=(0, 10), p=0.2)
            ]
        )
    
    elif stage == "medium":
        # 중간 증강 (중기 학습)  
        rotation_range = (
            max(-90, int(rotation_mean - rotation_std)),
            min(90, int(rotation_mean + rotation_std))
        )
        
        return AugraphyPipeline(
            paper_phase=[PaperFactory(p=0.3)],
            post_phase=[
                Geometric(rotate_range=rotation_range, p=0.7),
                Brightness(brightness_range=(0.8, 1.2), p=0.5),
                Blur(blur_value=(1, 3), p=0.3),
                ShadowCast(p=0.3),
                Noise(noise_type="gauss", noise_value=(0, 20), p=0.4)
            ]
        )
    
    else:  # "heavy"
        # 강한 증강 (후기 학습) - 전체 분석 결과 적용
        return create_adaptive_document_pipeline(test_data_stats)


def adjust_pipeline_strength(
    pipeline: AugraphyPipeline, 
    strength_factor: float
) -> AugraphyPipeline:
    """
    기존 파이프라인의 강도 조절
    
    Args:
        pipeline: 원본 파이프라인
        strength_factor: 강도 인수 (0.0-1.0)
        
    Returns:
        강도가 조절된 파이프라인
    """
    # 여기서는 간단한 예시로 확률값들을 조절
    # 실제로는 각 증강의 파라미터들을 세밀하게 조절할 수 있음
    
    for phase in [pipeline.ink_phase, pipeline.paper_phase, pipeline.post_phase]:
        if phase:
            for augmentation in phase:
                if hasattr(augmentation, 'p'):
                    augmentation.p = min(1.0, augmentation.p * strength_factor)
    
    return pipeline


def create_rotation_focused_pipeline(dominant_angles: list) -> AugraphyPipeline:
    """
    특정 회전 각도에 집중된 파이프라인 생성
    
    Args:
        dominant_angles: 주요 회전 각도들의 리스트
        
    Returns:
        회전 특화 AugraphyPipeline 객체
    """
    
    rotation_augmentations = []
    
    for angle in dominant_angles:
        rotation_augmentations.append(
            Geometric(
                rotate_range=(angle - 15, angle + 15),
                p=0.8 / len(dominant_angles)  # 각도별 확률 분산
            )
        )
    
    post_phase = rotation_augmentations + [
        # 회전으로 인한 품질 저하 시뮬레이션
        Brightness(brightness_range=(0.85, 1.15), p=0.4),
        Blur(blur_value=(0, 2), p=0.3),
        Noise(noise_type="gauss", noise_value=(0, 15), p=0.3),
    ]
    
    return AugraphyPipeline(
        ink_phase=[],
        paper_phase=[],  
        post_phase=post_phase
    )


def validate_pipeline_parameters(test_data_stats: Dict[str, Dict[str, float]]) -> bool:
    """
    파이프라인 파라미터 유효성 검증
    
    Args:
        test_data_stats: 테스트 데이터 분석 결과
        
    Returns:
        유효성 검증 결과
    """
    required_keys = ['rotation_angles', 'brightness_levels', 'blur_levels', 'noise_levels', 'contrast_levels']
    
    for key in required_keys:
        if key not in test_data_stats:
            print(f"❌ 필수 분석 결과 누락: {key}")
            return False
        
        stats = test_data_stats[key]
        if not all(stat_key in stats for stat_key in ['mean', 'std', 'min', 'max']):
            print(f"❌ {key}의 통계 정보 불완전")
            return False
    
    print("✅ 파이프라인 파라미터 유효성 검증 완료")
    return True


if __name__ == "__main__":
    # 테스트용 더미 데이터
    dummy_stats = {
        'rotation_angles': {'mean': 15.0, 'std': 30.0, 'min': -45.0, 'max': 90.0},
        'brightness_levels': {'mean': 120.0, 'std': 20.0, 'min': 80.0, 'max': 200.0},
        'blur_levels': {'mean': 250.0, 'std': 100.0, 'min': 50.0, 'max': 800.0},
        'noise_levels': {'mean': 8.0, 'std': 3.0, 'min': 2.0, 'max': 20.0},
        'contrast_levels': {'mean': 45.0, 'std': 15.0, 'min': 20.0, 'max': 80.0}
    }
    
    print("🧪 적응형 파이프라인 테스트:")
    if validate_pipeline_parameters(dummy_stats):
        pipeline = create_adaptive_document_pipeline(dummy_stats)
        print("✅ 파이프라인 생성 성공!") 