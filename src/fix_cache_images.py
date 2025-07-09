#!/usr/bin/env python3
"""
캐시된 이미지의 색상 문제를 해결하는 스크립트
기존 캐시를 삭제하고 새로운 캐시를 생성합니다.
"""

import os
import shutil
import argparse
from pathlib import Path


def find_cache_directories(base_path):
    """캐시 디렉토리를 찾습니다."""
    cache_dirs = []
    base_path = Path(base_path)
    
    # train_cache 디렉토리 찾기
    train_cache_path = base_path / "train_cache"
    if train_cache_path.exists():
        cache_dirs.append(train_cache_path)
    
    # 하위 디렉토리에서 train_cache 찾기
    for subdir in base_path.rglob("train_cache"):
        if subdir.is_dir():
            cache_dirs.append(subdir)
    
    return cache_dirs


def clean_cache_directories(cache_dirs, dry_run=False):
    """캐시 디렉토리를 정리합니다."""
    cleaned_dirs = []
    
    for cache_dir in cache_dirs:
        print(f"캐시 디렉토리 발견: {cache_dir}")
        
        if dry_run:
            print(f"  [DRY RUN] 삭제 예정: {cache_dir}")
            cleaned_dirs.append(cache_dir)
        else:
            try:
                shutil.rmtree(cache_dir)
                print(f"  삭제 완료: {cache_dir}")
                cleaned_dirs.append(cache_dir)
            except Exception as e:
                print(f"  삭제 실패: {cache_dir} - {e}")
    
    return cleaned_dirs


def main():
    parser = argparse.ArgumentParser(description="캐시된 이미지 색상 문제 해결")
    parser.add_argument("--base-path", default=".", help="검색할 기본 경로 (기본값: 현재 디렉토리)")
    parser.add_argument("--dry-run", action="store_true", help="실제 삭제하지 않고 미리보기만 실행")
    parser.add_argument("--force", action="store_true", help="확인 없이 바로 삭제")
    
    args = parser.parse_args()
    
    print("캐시 디렉토리 검색 중...")
    cache_dirs = find_cache_directories(args.base_path)
    
    if not cache_dirs:
        print("캐시 디렉토리를 찾을 수 없습니다.")
        return
    
    print(f"\n발견된 캐시 디렉토리: {len(cache_dirs)}개")
    for cache_dir in cache_dirs:
        print(f"  - {cache_dir}")
    
    if args.dry_run:
        print("\n=== DRY RUN 모드 ===")
        clean_cache_directories(cache_dirs, dry_run=True)
        print("\n실제로 삭제하려면 --dry-run 옵션을 제거하고 실행하세요.")
        return
    
    if not args.force:
        response = input(f"\n{len(cache_dirs)}개의 캐시 디렉토리를 삭제하시겠습니까? (y/N): ")
        if response.lower() != 'y':
            print("취소되었습니다.")
            return
    
    print("\n캐시 디렉토리 삭제 중...")
    cleaned_dirs = clean_cache_directories(cache_dirs, dry_run=False)
    
    print(f"\n완료! {len(cleaned_dirs)}개의 캐시 디렉토리가 삭제되었습니다.")
    print("이제 모델을 다시 실행하면 올바른 색상으로 캐시된 이미지가 생성됩니다.")


if __name__ == "__main__":
    main() 