#!/usr/bin/env python3
"""
빠른 시작 예시 스크립트
테스트 데이터 분석과 기본 사용법을 보여줍니다.
"""

import os
import sys
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_data_directories():
    """데이터 디렉토리 존재 여부 확인"""
    train_dir = Path("../input/data/train")
    test_dir = Path("../input/data/test")
    
    print("📂 데이터 디렉토리 확인:")
    print(f"  학습 데이터: {train_dir} {'✅' if train_dir.exists() else '❌'}")
    print(f"  테스트 데이터: {test_dir} {'✅' if test_dir.exists() else '❌'}")
    
    if not train_dir.exists():
        print(f"\n⚠️ 학습 데이터 디렉토리가 없습니다: {train_dir}")
        print("  예상 구조:")
        print("  input/data/train/")
        print("  ├── class1/")
        print("  │   ├── image1.jpg")
        print("  │   └── ...")
        print("  └── class2/")
    
    if not test_dir.exists():
        print(f"\n⚠️ 테스트 데이터 디렉토리가 없습니다: {test_dir}")
        print("  예상 구조:")
        print("  input/data/test/")
        print("  ├── test1.jpg")
        print("  ├── test2.jpg")
        print("  └── ...")
    
    return train_dir.exists() and test_dir.exists()

def demo_test_analysis():
    """테스트 데이터 분석 데모"""
    print("\n🔍 테스트 데이터 분석 데모")
    print("=" * 40)
    
    try:
        from test_data_analyzer import analyze_document_test_data, save_analysis_results
        
        test_dir = "../input/data/test"
        if not os.path.exists(test_dir):
            print(f"❌ 테스트 디렉토리가 없습니다: {test_dir}")
            return False
        
        # 분석 실행 (샘플 수를 줄여서 빠르게)
        print("분석 중... (샘플 수: 20)")
        stats = analyze_document_test_data(test_dir, sample_size=20)
        
        # 결과 저장
        save_analysis_results(stats, "demo_analysis.json")
        print("✅ 분석 완료! demo_analysis.json에 결과 저장")
        
        return True
        
    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        return False

def demo_augmentation_pipeline():
    """증강 파이프라인 데모"""
    print("\n⚙️ 증강 파이프라인 데모")
    print("=" * 40)
    
    try:
        from adaptive_augmentation import create_adaptive_document_pipeline, validate_pipeline_parameters
        import json
        
        # 분석 결과 로드
        if not os.path.exists("demo_analysis.json"):
            print("❌ 분석 결과가 없습니다. 먼저 분석을 실행하세요.")
            return False
        
        with open("demo_analysis.json", 'r') as f:
            stats = json.load(f)
        
        # 파라미터 유효성 검증
        if not validate_pipeline_parameters(stats):
            print("❌ 파라미터 유효성 검증 실패")
            return False
        
        # 파이프라인 생성
        print("파이프라인 생성 중...")
        try:
            pipeline = create_adaptive_document_pipeline(stats)
            print("✅ Augraphy 파이프라인 생성 성공!")
            return True
        except Exception as e:
            print(f"⚠️ Augraphy 파이프라인 생성 실패: {e}")
            print("💡 pip install augraphy 를 실행하여 설치해보세요.")
            return False
        
    except Exception as e:
        print(f"❌ 파이프라인 생성 실패: {e}")
        return False

def demo_dataset():
    """데이터셋 데모"""
    print("\n📊 적응형 데이터셋 데모")
    print("=" * 40)
    
    try:
        from adaptive_dataset import load_image_paths_and_labels, AdaptiveDocumentDataset
        from torchvision import transforms
        
        train_dir = "../input/data/train"
        test_dir = "../input/data/test"
        
        if not os.path.exists(train_dir):
            print(f"❌ 학습 디렉토리가 없습니다: {train_dir}")
            return False
        
        # 데이터 경로 로드
        print("데이터 경로 로드 중...")
        train_paths, train_labels = load_image_paths_and_labels(train_dir)
        print(f"✅ 학습 데이터: {len(train_paths)}개 이미지, {len(set(train_labels))}개 클래스")
        
        # 변환 정의
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 데이터셋 생성 (첫 10개만 사용)
        print("적응형 데이터셋 생성 중...")
        dataset = AdaptiveDocumentDataset(
            image_paths=train_paths[:10],  # 빠른 테스트를 위해 10개만
            labels=train_labels[:10],
            test_data_dir=test_dir,
            pytorch_transform=transform,
            cache_analysis=True,
            analysis_cache_path="demo_analysis.json"  # 이미 생성된 분석 결과 사용
        )
        
        print(f"✅ 데이터셋 생성 성공! 크기: {len(dataset)}")
        print(f"테스트 데이터 분석 요약:")
        print(dataset.get_test_stats_summary())
        
        return True
        
    except Exception as e:
        print(f"❌ 데이터셋 생성 실패: {e}")
        return False

def main():
    """메인 실행 함수"""
    print("🚀 테스트 데이터 기반 적응형 증강 시스템 - 빠른 시작")
    print("=" * 60)
    
    # 1. 데이터 디렉토리 확인
    if not check_data_directories():
        print("\n❌ 데이터 디렉토리 설정을 완료한 후 다시 실행해주세요.")
        print("\n💡 전체 시스템 실행:")
        print("   python main.py --train_dir ../input/data/train --test_dir ../input/data/test --analysis_only")
        return
    
    # 2. 테스트 데이터 분석
    if not demo_test_analysis():
        print("\n❌ 테스트 데이터 분석에 실패했습니다.")
        return
    
    # 3. 증강 파이프라인 테스트
    if not demo_augmentation_pipeline():
        print("\n⚠️ Augraphy 파이프라인을 사용할 수 없지만 기본 증강으로 동작합니다.")
    
    # 4. 데이터셋 테스트
    if not demo_dataset():
        print("\n❌ 데이터셋 생성에 실패했습니다.")
        return
    
    print("\n🎉 모든 데모 완료!")
    print("\n📚 다음 단계:")
    print("1. 전체 분석: python main.py --analysis_only")
    print("2. 학습 실행: python main.py --epochs 10")
    print("3. 자세한 사용법: cat README.md")
    
    # 생성된 파일들 정리
    cleanup_files = ["demo_analysis.json"]
    print(f"\n🧹 임시 파일 정리: {cleanup_files}")
    for file in cleanup_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"  삭제: {file}")

if __name__ == "__main__":
    main() 