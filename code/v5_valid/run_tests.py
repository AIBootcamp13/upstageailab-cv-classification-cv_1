#!/usr/bin/env python3
"""
Test runner for validation features
"""
import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and print results"""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {description} 성공")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"❌ {description} 실패")
            if result.stderr:
                print(f"에러: {result.stderr}")
            if result.stdout:
                print(f"출력: {result.stdout}")
            return False
            
    except Exception as e:
        print(f"❌ {description} 실행 중 오류 발생: {str(e)}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 검증 기능 테스트 시작")
    print("=" * 60)
    
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    tests = [
        ("uv run pytest tests/test_main.py -v", "단위 테스트 실행"),
        ("uv run pytest tests/test_features.py -v", "기능별 통합 테스트 실행"),
        ("uv run pytest tests/test_main.py tests/test_features.py -v", "전체 테스트 스위트 실행"),
        ("uv run python tests/test_features.py", "기능별 개별 테스트 실행"),
    ]
    
    success_count = 0
    total_count = len(tests)
    
    for cmd, description in tests:
        if run_command(cmd, description):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"📊 테스트 결과 요약")
    print(f"{'='*60}")
    print(f"✅ 성공: {success_count}/{total_count}")
    print(f"❌ 실패: {total_count - success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
        print("""
테스트 완료된 기능:
✅ 8:2 Holdout 검증
✅ Stratified K-Fold 교차 검증  
✅ Early Stopping
✅ 학습 및 검증 함수
✅ 추론 기능
✅ 데이터 로딩 및 전처리
✅ 예측 결과 포맷
""")
    else:
        print("\n⚠️  일부 테스트에서 오류가 발생했습니다.")
        sys.exit(1)

if __name__ == "__main__":
    main() 