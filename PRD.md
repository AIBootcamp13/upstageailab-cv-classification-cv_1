프로그램을 아래와 같이 개선해줘.
프로그램 실행시 마다 매번 이미지 증강을 처리하다보니, 시간이 너무 오래걸리고, 증강된 이미지가 메모리 상에만 있다보니, 실제 증강된 이미지를 눈으로 확인하기가 어렵다 따라서 아래와 같은 개선이 필요하다.
- 어차피 랜덤시드와 옵션이 같으면 같은 이미지를 증강할텐데, 프로그램 실행시마다 매번 동일한 증강을 하는것은 비효율적이다. 현재 옵션들을 합친 해시값을 만들어 img_cache폴더 아래에 해시값 폴더를 만든후에 저장하고, 동일 랜덤시드, 옵션인 프로그램이 기동되면 이 값을 읽어서 사용하도록 해라.
- 이미지들을 저장할때 train_aug, valid_aug, valid_tta, test_tta 폴더에 적절히 분류해 담아라.

어떤식으로 저장할지 폴더 구조는 다음과 같다.

## 폴더 구조

```
src/
├── img_cache/                           # 메인 캐시 디렉토리
│   ├── {base_hash}/                     # 기본 설정 해시값 (kfold/seed 제외)
│   │   ├── seed_{seed_value}/           # 랜덤시드별 폴더 (상위)
│   │   │   ├── fold_{fold_idx}/         # kfold별 폴더 (하위)
│   │   │   │   ├── train_aug/           # 해당 fold의 훈련 데이터 증강 이미지
│   │   │   │   │   ├── {original_filename}_aug_{aug_idx}.jpg
│   │   │   │   │   └── ...
│   │   │   │   ├── valid_aug/           # 해당 fold의 검증 데이터 증강 이미지
│   │   │   │   │   ├── {original_filename}_aug_{aug_idx}.jpg
│   │   │   │   │   └── ...
│   │   │   │   ├── valid_tta/           # 해당 fold의 검증 TTA 이미지
│   │   │   │   │   ├── {original_filename}_tta_{tta_idx}.jpg
│   │   │   │   │   └── ...
│   │   │   │   ├── test_tta/            # 테스트 TTA 이미지 (fold와 무관)
│   │   │   │   │   ├── {original_filename}_tta_{tta_idx}.jpg
│   │   │   │   │   └── ...
│   │   │   │   ├── fold_info.json       # fold별 데이터 분할 정보
│   │   │   │   ├── cache_info.json      # 캐시 메타데이터
│   │   │   │   └── cache_stats.json     # 캐시 통계
│   │   │   └── ...
│   │   ├── cache_index.json             # seed/fold별 캐시 인덱스
│   │   └── fold_splits.json             # 전체 fold 분할 정보 (시드별)
│   └── cache_index.json                 # 전체 캐시 인덱스
```

## 구체적인 폴더 구조 예시

```
img_cache/
├── a1b2c3d4e5f6/                        # 기본 설정 해시
│   ├── seed_42/                         # 시드 42 (상위)
│   │   ├── fold_0/                      # 0번 fold
│   │   │   ├── train_aug/
│   │   │   │   ├── image_001_aug_0.jpg
│   │   │   │   ├── image_001_aug_1.jpg
│   │   │   │   ├── image_002_aug_0.jpg
│   │   │   │   └── ...
│   │   │   ├── valid_aug/
│   │   │   │   ├── image_101_aug_0.jpg
│   │   │   │   └── ...
│   │   │   ├── valid_tta/
│   │   │   │   ├── image_101_tta_0.jpg
│   │   │   │   └── ...
│   │   │   ├── test_tta/
│   │   │   │   ├── test_001_tta_0.jpg
│   │   │   │   └── ...
│   │   │   ├── fold_info.json
│   │   │   ├── cache_info.json
│   │   │   └── cache_stats.json
│   │   ├── fold_1/                      # 1번 fold
│   │   │   ├── train_aug/
│   │   │   └── ...
│   │   ├── fold_2/                      # 2번 fold
│   │   ├── fold_3/                      # 3번 fold
│   │   ├── fold_4/                      # 4번 fold
│   │   ├── cache_index.json
│   │   └── fold_splits.json
│   ├── seed_43/                         # 시드 43 (상위)
│   │   ├── fold_0/
│   │   ├── fold_1/
│   │   ├── fold_2/
│   │   ├── fold_3/
│   │   ├── fold_4/
│   │   ├── cache_index.json
│   │   └── fold_splits.json
│   ├── seed_44/                         # 시드 44 (상위)
│   │   ├── fold_0/
│   │   ├── fold_1/
│   │   ├── fold_2/
│   │   ├── fold_3/
│   │   ├── fold_4/
│   │   ├── cache_index.json
│   │   └── fold_splits.json
│   ├── cache_index.json
│   └── fold_splits.json
└── cache_index.json
```

## 메타데이터 파일 구조 (수정)

### 1. cache_index.json (전체)
```json
{
  "a1b2c3d4e5f6": {
    "base_config": {
      "augment": {...},
      "data": {...},
      "validation": {...}
    },
    "created_at": "2024-01-01T00:00:00Z",
    "seeds": {
      "42": {
        "fold_count": 5,
        "created_at": "2024-01-01T00:00:00Z",
        "total_size_mb": 512
      },
      "43": {
        "fold_count": 5,
        "created_at": "2024-01-01T00:00:00Z",
        "total_size_mb": 512
      },
      "44": {
        "fold_count": 5,
        "created_at": "2024-01-01T00:00:00Z",
        "total_size_mb": 512
      }
    },
    "total_size_mb": 1536
  }
}
```

### 2. fold_splits.json (시드별)
```json
{
  "seed": 42,
  "fold_splits": {
    "fold_0": {
      "train_indices": [0, 1, 2, ...],
      "valid_indices": [100, 101, 102, ...],
      "train_files": ["image_001.jpg", "image_002.jpg", ...],
      "valid_files": ["image_101.jpg", "image_102.jpg", ...]
    },
    "fold_1": {...},
    "fold_2": {...},
    "fold_3": {...},
    "fold_4": {...}
  },
  "created_at": "2024-01-01T00:00:00Z"
}
```

### 3. fold_info.json (fold별)
```json
{
  "seed_value": 42,
  "fold_idx": 0,
  "train_count": 800,
  "valid_count": 200,
  "aug_count": 1,
  "tta_count": 1,
  "created_at": "2024-01-01T00:00:00Z"
}
```

## 장점

1. **시드별 관리**: 시드가 변경되면 해당 시드 폴더만 삭제/관리 가능
2. **캐시 효율성**: 동일 시드 내에서 fold 간 공유 가능한 메타데이터 관리
3. **확장성**: 새로운 시드 추가 시 기존 구조 유지
4. **정리 용이성**: 특정 시드의 모든 fold 캐시를 한 번에 정리 가능

## 캐시 관리 전략

### 1. 시드별 부분 캐시
- 특정 시드의 일부 fold만 캐시가 있어도 해당 부분은 재사용
- 없는 fold만 새로 생성

### 2. 시드 간 공유
- `test_tta`는 시드와 무관하므로 첫 번째 시드에서 생성한 것을 공유
- fold 분할 정보는 시드별로 다르므로 개별 관리

### 3. 캐시 정리
- 오래된 시드 전체 삭제
- 특정 시드의 특정 fold만 삭제
- 시드별로 디스크 사용량 모니터링
