"""
테스트 데이터 분석 모듈
문서 이미지의 회전, 밝기, 블러, 노이즈 등을 분석하여 증강 파라미터를 도출합니다.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


def analyze_document_test_data(test_dir: str, sample_size: int = 100) -> Dict[str, Dict[str, float]]:
    """
    문서 이미지 테스트 데이터의 특성 분석
    
    Args:
        test_dir: 테스트 데이터 디렉토리 경로
        sample_size: 분석할 샘플 수
        
    Returns:
        분석 결과 통계 딕셔너리
    """
    test_images = list(Path(test_dir).glob('*.jpg')) + list(Path(test_dir).glob('*.png'))
    
    if len(test_images) == 0:
        raise ValueError(f"테스트 디렉토리 {test_dir}에서 이미지를 찾을 수 없습니다.")
    
    test_images = test_images[:sample_size]
    print(f"📁 {len(test_images)}개의 테스트 이미지를 분석합니다...")
    
    analysis_results = {
        'rotation_angles': [],
        'brightness_levels': [],
        'blur_levels': [],
        'noise_levels': [],
        'contrast_levels': []
    }
    
    for i, img_path in enumerate(test_images):
        if i % 20 == 0:
            print(f"  진행률: {i}/{len(test_images)} ({i/len(test_images)*100:.1f}%)")
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. 회전 각도 분석
        rotation = estimate_document_rotation(gray)
        analysis_results['rotation_angles'].append(rotation)
        
        # 2. 밝기 분석
        brightness = np.mean(gray)
        analysis_results['brightness_levels'].append(brightness)
        
        # 3. 블러 정도 분석
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        analysis_results['blur_levels'].append(blur_score)
        
        # 4. 노이즈 레벨 분석
        noise_level = estimate_noise_level(gray)
        analysis_results['noise_levels'].append(noise_level)
        
        # 5. 대비 분석
        contrast = gray.std()
        analysis_results['contrast_levels'].append(contrast)
    
    print("✅ 테스트 데이터 분석 완료!")
    
    # 통계 정보 계산
    stats = {}
    for key, values in analysis_results.items():
        if values:  # 빈 리스트가 아닌 경우만
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'percentiles': np.percentile(values, [25, 50, 75]).tolist()
            }
        else:
            stats[key] = {
                'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'percentiles': [0, 0, 0]
            }
    
    # 분석 결과 출력
    print_analysis_summary(stats)
    
    return stats


def estimate_document_rotation(gray_image: np.ndarray) -> float:
    """
    문서 이미지의 회전 각도 추정
    
    Args:
        gray_image: 그레이스케일 이미지
        
    Returns:
        추정된 회전 각도 (도)
    """
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    if lines is not None:
        angles = []
        for rho, theta in lines[:20]:
            angle = (theta * 180 / np.pi) - 90
            if abs(angle) <= 45:  # 주요 각도만 고려
                angles.append(angle)
        
        if angles:
            # 가장 빈번한 각도 반환
            hist, bins = np.histogram(angles, bins=18, range=(-45, 45))
            dominant_angle_idx = np.argmax(hist)
            return (bins[dominant_angle_idx] + bins[dominant_angle_idx + 1]) / 2
    
    return 0


def estimate_noise_level(gray_image: np.ndarray) -> float:
    """
    이미지의 노이즈 레벨 추정
    
    Args:
        gray_image: 그레이스케일 이미지
        
    Returns:
        노이즈 레벨 값
    """
    # 가우시안 블러와 원본의 차이로 노이즈 추정
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    noise = gray_image.astype(float) - blurred.astype(float)
    return float(np.std(noise))


def print_analysis_summary(stats: Dict[str, Dict[str, float]]) -> None:
    """
    분석 결과 요약 출력
    
    Args:
        stats: 분석 결과 통계
    """
    print("\n📊 테스트 데이터 분석 결과:")
    print("=" * 50)
    
    for key, values in stats.items():
        print(f"\n🔹 {key}:")
        print(f"  평균: {values['mean']:.2f}")
        print(f"  표준편차: {values['std']:.2f}")
        print(f"  범위: {values['min']:.2f} ~ {values['max']:.2f}")
        print(f"  분위수: {values['percentiles']}")


def save_analysis_results(stats: Dict[str, Dict[str, float]], output_path: str = "test_analysis_results.json") -> None:
    """
    분석 결과를 JSON 파일로 저장
    
    Args:
        stats: 분석 결과 통계
        output_path: 저장할 파일 경로
    """
    import json
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"💾 분석 결과가 {output_path}에 저장되었습니다.")


def load_analysis_results(input_path: str = "test_analysis_results.json") -> Dict[str, Dict[str, float]]:
    """
    저장된 분석 결과를 로드
    
    Args:
        input_path: 로드할 파일 경로
        
    Returns:
        분석 결과 통계
    """
    import json
    
    with open(input_path, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    
    print(f"📂 분석 결과를 {input_path}에서 로드했습니다.")
    return stats


if __name__ == "__main__":
    # 테스트 실행
    test_dir = "input/data/test"
    
    try:
        stats = analyze_document_test_data(test_dir, sample_size=50)
        save_analysis_results(stats)
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("테스트 데이터 경로를 확인해주세요.") 