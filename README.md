# Upstage AI Lab CV Classification Project

문서 타입 분류를 위한 이미지 분류 대회 참여 프로젝트입니다. 주어진 문서 이미지를 17개 클래스 중 하나로 분류하는 과제로, 금융, 의료, 보험, 물류 등 다양한 산업 분야에서 실제 활용되는 문서 분류 자동화 기술을 개발합니다.

## 🚀 주요 기능

### 1. 다양한 검증 전략
- **8:2 Holdout 검증**: 훈련 데이터를 8:2로 분할하여 검증
- **Stratified K-Fold 교차 검증**: K개의 fold로 교차 검증 및 앙상블
- **No validation**: 전체 데이터 훈련
- **Early Stopping**: 과적합 방지를 위한 조기 종료

### 2. 고급 이미지 증강
- **Albumentations**: 일반적인 컴퓨터 비전 증강
- **Augraphy**: 문서 특화 증강 (잉크 번짐, 노이즈, 스테인 등)
- **Mix 증강**: 두 방법의 조합
- **TTA (Test Time Augmentation)**: 추론 시 증강 앙상블

### 3. 다양한 모델 지원
- **ResNet**: ResNet18, ResNet34
- **EfficientNet**: EfficientNetV2-S, EfficientNetV2-M, EfficientNetV2-L, EfficientNetV2-XL
- **TF EfficientNet**: TensorFlow 버전 EfficientNet
- **EfficientNetV2-RW**: RegNet 기반 EfficientNet

### 4. 학습 최적화
- **Learning Rate Scheduler**: Cosine, Step, Plateau, Cosine Warm Restarts
- **Mixed Precision Training**: 메모리 효율성 향상
- **Label Smoothing**: 일반화 성능 향상
- **Random Seed Ensemble**: 여러 시드 앙상블

### 5. 실험 관리
- **Hydra**: 설정 관리 및 하이퍼파라미터 실험
- **WandB**: 실험 추적 및 시각화
- **Comprehensive Testing**: pytest 기반 테스트 커버리지

## 📁 프로젝트 구조

```
upstageailab-cv-classification-cv_1_fork/
├── src/                           # 메인 소스 코드
│   ├── config/                    # Hydra 설정 파일들
│   │   ├── config.yaml           # 기본 설정
│   │   ├── holdout.yaml          # Holdout 검증 설정
│   │   ├── kfold.yaml            # K-Fold 검증 설정
│   │   ├── fast_test.yaml        # 빠른 테스트 설정
│   │   ├── high_performance.yaml # 고성능 실험 설정
│   │   ├── test_*.yaml           # 다양한 테스트 설정
│   │   ├── efnv2*.yaml           # EfficientNet 모델 설정
│   │   └── wandb_aug.yaml        # WandB 증강 실험 설정
│   ├── models/                    # 훈련된 모델 저장소
│   ├── results/                   # 예측 결과 파일들
│   ├── tests/                     # 테스트 코드
│   ├── main.py                   # 메인 실행 파일
│   ├── data.py                   # 데이터 로딩 및 증강
│   ├── models.py                 # 모델 정의
│   ├── training.py               # 훈련 로직
│   ├── inference.py              # 추론 로직
│   ├── utils.py                  # 유틸리티 함수들
│   └── log_util.py               # 로깅 유틸리티
├── input/                        # 데이터 디렉토리
│   └── data/                     # 실제 데이터
├── .venv/                        # uv 가상환경
├── pyproject.toml               # 프로젝트 설정
└── uv.lock                      # 의존성 잠금 파일
```

## 🛠️ 설치 및 설정

### 1. 환경 설정
```bash
# 프로젝트 클론
git clone <repository-url>
cd upstageailab-cv-classification-cv_1_fork

# uv 가상환경 활성화 (자동 생성됨)
# 패키지는 이미 설치되어 있음
```

### 2. 데이터 준비
```bash
# 데이터 다운로드 (필요시)
cd input
bash get_data.sh
```

## 🚀 사용법

### 기본 실행
```bash
# src 폴더로 이동
cd src

# 기본 설정으로 실행
uv run main.py
```

### 검증 전략별 실행

#### 1. Holdout 검증
```bash
# Holdout 검증 설정 사용
uv run main.py --config-name=holdout

# 또는 직접 설정
uv run main.py validation.strategy=holdout validation.holdout.train_ratio=0.8
```

#### 2. K-Fold 교차 검증
```bash
# 5-fold 교차 검증
uv run main.py --config-name=kfold

# 또는 직접 설정
uv run main.py validation.strategy=kfold validation.kfold.n_splits=5
```

#### 3. 빠른 테스트
```bash
# Holdout 빠른 테스트
uv run main.py --config-name=test_holdout

# K-Fold 빠른 테스트
uv run main.py --config-name=test_kfold
```

#### 4. 검증 없이 전체 데이터 사용
```bash
uv run main.py validation.strategy=none
```

### 커맨드라인 파라미터 오버라이드

#### 학습 파라미터 변경
```bash
# 에포크 수 변경
uv run main.py train.epochs=10

# 배치 크기와 학습률 변경
uv run main.py train.batch_size=16 train.lr=1e-4

# 모델 변경
uv run main.py model.name=efficientnetv2_s
```

#### 증강 설정 변경
```bash
# 증강 방법 변경
uv run main.py augment.method=augraphy

# 증강 강도 조절
uv run main.py augment.intensity=0.8

# TTA 활성화
uv run main.py augment.test_tta_count=5
```

#### 스케줄러 설정
```bash
# Cosine 스케줄러 사용
uv run main.py train.scheduler.name=cosine

# Step 스케줄러 사용
uv run main.py train.scheduler.name=step train.scheduler.step.step_size=30
```

### WandB 연동

#### 1. 환경 설정
```bash
# .env 파일 생성
cp src/env_template.txt src/.env
```

#### 2. .env 파일 편집
```env
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=document-classification
WANDB_ENTITY=your_wandb_entity_here
```

#### 3. WandB 활성화 실행
```bash
# WandB 활성화
uv run main.py wandb.enabled=true

# 또는 설정 파일 사용
uv run main.py --config-name=wandb_aug
```

## 🧪 테스트 실행

### 전체 테스트
```bash
# 모든 테스트 실행
uv run pytest

# 상세 출력으로 실행
uv run pytest -v
```

### 개별 테스트
```bash
# 특정 테스트 파일만 실행
uv run pytest tests/test_main.py -v
uv run pytest tests/test_data.py -v
uv run pytest tests/test_models.py -v
uv run pytest tests/test_training.py -v
uv run pytest tests/test_inference.py -v
uv run pytest tests/test_augmentation.py -v
uv run pytest tests/test_scheduler.py -v
uv run pytest tests/test_utils.py -v
```

### 테스트 커버리지
- ✅ **데이터 로딩**: ImageDataset, IndexedImageDataset 클래스
- ✅ **8:2 Holdout**: 데이터 분할 및 stratified 검증
- ✅ **K-Fold**: 교차 검증 및 앙상블 기능
- ✅ **Early Stopping**: 조기 종료 로직 및 다양한 모니터링 지표
- ✅ **학습/검증**: train_one_epoch, validate_one_epoch 함수
- ✅ **추론**: 예측 결과 생성 및 포맷 검증
- ✅ **증강**: Albumentations, Augraphy 증강 기능
- ✅ **스케줄러**: 다양한 학습률 스케줄러
- ✅ **모델**: 다양한 백본 모델 지원

## 📊 설정 옵션

### Validation 전략
```yaml
validation:
  strategy: "holdout"  # "holdout", "kfold", "none"
```

### Holdout 설정
```yaml
validation:
  holdout:
    train_ratio: 0.8    # 훈련 데이터 비율
    stratify: true      # 층화 분할 여부
```

### K-Fold 설정
```yaml
validation:
  kfold:
    n_splits: 5         # fold 개수
    stratify: true      # 층화 분할 여부
```

### Early Stopping 설정
```yaml
validation:
  early_stopping:
    enabled: true       # 활성화 여부
    patience: 10        # 개선되지 않는 epoch 수
    min_delta: 0.001    # 개선으로 인정할 최소 변화량
    monitor: "val_loss" # 모니터링 지표 ("val_loss", "val_acc", "val_f1")
    mode: "min"         # "min" (loss용) 또는 "max" (accuracy/f1용)
```

### 증강 설정
```yaml
augment:
  method: "mix"         # "none", "albumentations", "augraphy", "mix"
  intensity: 1.0        # 증강 강도 (0.0 ~ 1.0)
  train_aug_count: 1    # 훈련 데이터 증강 복사본 수
  test_tta_count: 1     # 테스트 TTA 복사본 수
  train_aug_ops: [all]  # 훈련 증강 연산자
  test_tta_ops: [rotate] # 테스트 TTA 연산자
```

### 모델 설정
```yaml
model:
  name: "resnet34"      # 모델 이름
  num_classes: 17       # 클래스 수
  pretrained: true      # 사전훈련 모델 사용 여부
```

### 훈련 설정
```yaml
train:
  lr: 1e-3              # 학습률
  epochs: 100           # 에포크 수
  batch_size: 32        # 배치 크기
  seed: 42              # 랜덤 시드
  
  # Label Smoothing
  label_smoothing:
    enabled: false
    smoothing: 0.1
  
  # Mixed Precision Training
  mixed_precision:
    enabled: false
  
  # Learning Rate Scheduler
  scheduler:
    enabled: true
    name: "cosine"       # "cosine", "step", "plateau", "cosine_warm", "none"
```

## 📈 주요 특징

1. **Stratified 분할**: 클래스 불균형을 고려한 분할
2. **Early Stopping**: 과적합 방지를 위한 조기 종료
3. **K-Fold 앙상블**: 여러 모델의 평균으로 더 안정적인 예측
4. **WandB 지원**: 모든 검증 지표를 WandB에 자동 로깅
5. **유연한 설정**: Hydra를 통한 다양한 설정 조합 가능
6. **문서 특화 증강**: Augraphy를 통한 문서 특화 증강
7. **TTA 지원**: 추론 시 증강 앙상블로 성능 향상
8. **다양한 모델**: ResNet, EfficientNet 등 다양한 백본 모델
9. **학습 최적화**: 스케줄러, Mixed Precision, Label Smoothing
10. **랜덤 시드 앙상블**: 여러 시드의 앙상블로 안정성 향상

## 📤 출력 결과

- **Holdout/No validation**: 단일 모델의 예측 결과
- **K-Fold**: 모든 fold의 앙상블 예측 결과 (softmax 평균 후 argmax)
- **Random Seed Ensemble**: 여러 시드의 앙상블 예측 결과

각 fold별 성능과 최종 앙상블 성능이 로그와 WandB에 기록됩니다.

## 🔧 환경 관리

### 패키지 설치
```bash
# 추가 패키지 설치 시 uv add 사용
uv add package_name

# 개발 의존성 설치
uv add --group dev package_name
```

### Python 스크립트 실행
```bash
# 모든 Python 스크립트는 uv run으로 실행
uv run main.py
uv run python tests/test_main.py
```

## ⚠️ 주의사항

- 메모리 사용량 고려 (GPU 메모리 모니터링)
- 클래스 불균형 문제 해결 필요
- 테스트 데이터의 다양한 변형에 대응
- 재현 가능성을 위한 시드 고정
- uv 환경 관리 주의 (패키지 설치 시 uv add 사용)

## 🎯 성능 목표

- 현재 최고 성능: 약 96% 정확도
- 클래스별 균형잡힌 성능 향상
- 테스트 데이터 변형에 대한 강건성 확보
- Leaderboard 점수 향상

## 📚 참고 문서

- [AGENTS.md](AGENTS.md): 프로젝트 상세 가이드
- [PRD.md](PRD.md): 제품 요구사항 문서
- [Hydra 공식 문서](https://hydra.cc/)
- [WandB 공식 문서](https://wandb.ai/)
- [PyTorch 공식 문서](https://pytorch.org/)
- [timm 라이브러리](https://github.com/huggingface/pytorch-image-models)
- [Albumentations](https://albumentations.ai/)
- [Augraphy](https://github.com/mindee/augraphy) 