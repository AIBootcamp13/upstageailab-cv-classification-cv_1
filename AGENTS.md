# Upstage AI Lab CV Classification Project - Cursor Rules

## 프로젝트 개요
이 프로젝트는 문서 타입 분류를 위한 이미지 분류 대회 참여를 위한 팀 프로젝트입니다. 주어진 문서 이미지를 17개 클래스 중 하나로 분류하는 과제로, 금융, 의료, 보험, 물류 등 다양한 산업 분야에서 실제 활용되는 문서 분류 자동화 기술 개발을 목표로 합니다.

## 기술 스택
- **Python 3.10.13+**
- **PyTorch 2.7.1+** - 딥러닝 프레임워크
- **timm 0.9.12** - 사전 훈련된 모델 라이브러리
- **albumentations 1.3.1** - 이미지 증강
- **augraphy 8.2.6+** - 문서 특화 이미지 증강
- **pandas 2.1.4, numpy 1.26.0** - 데이터 처리
- **scikit-learn** - 평가 메트릭
- **PIL, OpenCV** - 이미지 처리
- **hydra-core 1.3.2+** - 설정 관리 및 하이퍼파라미터 관리
- **wandb 0.21.0+** - 실험 추적 및 시각화
- **pytest 8.4.1+** - 테스트 프레임워크

## 프로젝트 구조

### 주요 디렉토리
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
│   │   ├── resnet18_*.pth        # ResNet18 모델들
│   │   ├── resnet34_*.pth        # ResNet34 모델들
│   │   ├── efficientnetv2_s_*.pth # EfficientNetV2-S 모델들
│   │   └── tf_efficientnetv2_s_*.pth # TF EfficientNetV2-S 모델들
│   ├── results/                   # 예측 결과 파일들
│   │   ├── pred.csv              # 기본 예측 결과
│   │   ├── pred_efnv2*.csv       # EfficientNet 모델 결과들
│   │   ├── pred_*_test.csv       # 테스트 결과들
│   │   └── pred_*_pure.csv       # 순수 모델 결과들
│   ├── tests/                     # 테스트 코드
│   │   ├── test_main.py          # 메인 파이프라인 테스트
│   │   ├── test_data.py          # 데이터 로딩 테스트
│   │   ├── test_models.py        # 모델 테스트
│   │   ├── test_training.py      # 훈련 로직 테스트
│   │   ├── test_inference.py     # 추론 테스트
│   │   ├── test_augmentation.py  # 증강 테스트
│   │   ├── test_scheduler.py     # 스케줄러 테스트
│   │   └── test_utils.py         # 유틸리티 테스트
│   ├── main.py                   # 메인 실행 파일
│   ├── data.py                   # 데이터 로딩 및 증강
│   ├── models.py                 # 모델 정의
│   ├── training.py               # 훈련 로직
│   ├── inference.py              # 추론 로직
│   ├── utils.py                  # 유틸리티 함수들
│   ├── log_util.py               # 로깅 유틸리티
│   ├── pytest.ini               # pytest 설정
│   └── env_template.txt          # 환경 변수 템플릿
├── input/                        # 데이터 디렉토리
│   ├── data/                     # 실제 데이터
│   │   ├── train/               # 훈련 이미지들
│   │   ├── test/                # 테스트 이미지들
│   │   ├── train.csv            # 훈련 라벨
│   │   ├── meta.csv             # 메타데이터
│   │   └── sample_submission.csv # 제출 샘플
│   ├── data.tar.gz              # 압축된 데이터
│   └── get_data.sh              # 데이터 다운로드 스크립트
├── .venv/                        # uv 가상환경
├── pyproject.toml               # 프로젝트 설정
├── uv.lock                      # 의존성 잠금 파일
├── run.sh                       # 복수 커맨드 실행 스크립트
├── AGENTS.md                    # 이 파일
├── README.md                    # 프로젝트 README
├── PRD.md                       # 제품 요구사항 문서
└── CLAUDE.md                    # Claude 설정
```

### 데이터 구조
- **Train**: 1,570장 이미지, 17개 클래스
- **Test**: 3,140장 이미지 (다양한 변형 포함)
- 클래스 불균형 존재 (일부 클래스는 샘플 수 적음)
- 테스트 데이터는 회전, 반전, 밝기 변화 등 포함

## 코딩 컨벤션

### Python 코드
- UTF-8 인코딩 사용
- 함수와 클래스는 명확한 docstring 작성
- 시드 고정: `SEED = 42`
- GPU 사용 가능 시 CUDA 우선 사용 (MPS 지원)

### 모델 관련
- `timm` 라이브러리 사용하여 사전 훈련된 모델 로드
- ResNet, EfficientNet 등 다양한 백본 모델 실험
- TTA (Test Time Augmentation) 적용
- K-fold 교차 검증 사용

### 데이터 처리
- `albumentations`와 `augraphy` 사용한 이미지 증강
- 클래스 불균형 대응 (가중치 조정, 증강)
- 테스트 데이터 특성 분석 기반 적응형 증강
- 밝기, 대비, 회전 보정 적용

## 주요 기능

### 1. 검증 전략
- **Holdout 검증**: 8:2 분할로 훈련/검증
- **K-Fold 교차 검증**: 5-fold 앙상블
- **No validation**: 전체 데이터 훈련

### 2. 이미지 증강
- **Albumentations**: 일반적인 컴퓨터 비전 증강
- **Augraphy**: 문서 특화 증강 (잉크 번짐, 노이즈 등)
- **Mix 증강**: 두 방법 조합
- **TTA**: 추론 시 증강 앙상블

### 3. 모델 지원
- **ResNet**: ResNet18, ResNet34
- **EfficientNet**: EfficientNetV2-S, EfficientNetV2-M, EfficientNetV2-L, EfficientNetV2-XL
- **TF EfficientNet**: TensorFlow 버전 EfficientNet
- **EfficientNetV2-RW**: RegNet 기반 EfficientNet

### 4. 학습 최적화
- **Learning Rate Scheduler**: Cosine, Step, Plateau, Cosine Warm Restarts
- **Mixed Precision Training**: 메모리 효율성 향상
- **Label Smoothing**: 일반화 성능 향상
- **Early Stopping**: 과적합 방지

### 5. 실험 관리
- **Hydra**: 설정 관리 및 하이퍼파라미터 실험
- **WandB**: 실험 추적 및 시각화
- **Random Seed Ensemble**: 여러 시드 앙상블

## 파일 명명 규칙
- 설정 파일: `{feature}.yaml` (holdout, kfold, fast_test 등)
- 모델 파일: `{model_name}_{type}.pth` (best, last)
- 결과 파일: `pred_{experiment_name}.csv`
- 테스트 파일: `test_{module}.py`

## 환경 설정
- uv 가상환경: `.venv/` (자동 생성)
- 환경 파일: `pyproject.toml`, `uv.lock`
- Python 버전: 3.10.13
- 실행 명령어: `uv run <script_name>`
- 가상환경은 uv로 관리되며, 패키지는 이미 설치되어 있고, 코드를 실행했을때 만약 패키지 설치가 안되서 오류가 발생하는 경우에는 해당 패키지만 추가로 설치하되 uv add 명령을 사용해서 패키지를 설치해야 한다. 그리고 .py 실행은 uv run 명령으로 실행해야한다. 이점을 주의해라.

### Hydra 사용법
Hydra를 사용하여 설정을 관리하는 실행 방법:

#### 기본 실행
```bash
# src 폴더에서 기본 설정(config.yaml) 사용
cd src
uv run main.py
```

#### 설정 파일 변경
```bash
# holdout 검증 설정 사용
uv run main.py --config-name=holdout

# kfold 검증 설정 사용
uv run main.py --config-name=kfold

# 빠른 테스트 설정 사용
uv run main.py --config-name=fast_test

# 고성능 실험 설정 사용
uv run main.py --config-name=high_performance
```

#### 커맨드라인 파라미터 오버라이드
```bash
# 에포크 수 변경
uv run main.py train.epochs=5

# 여러 파라미터 동시 변경
uv run main.py train.epochs=5 train.batch_size=16 model.name=resnet50

# 이미지 크기와 학습률 변경
uv run main.py data.img_size=224 train.lr=1e-4

# 검증 전략 변경
uv run main.py validation.strategy=kfold validation.kfold.n_splits=5
```

#### 하이퍼파라미터 설정 구조
- **data**: 데이터 관련 설정 (이미지 경로, 크기, 워커 수)
- **model**: 모델 관련 설정 (이름, 클래스 수, 사전훈련 여부)
- **train**: 훈련 관련 설정 (학습률, 에포크, 배치 크기, 시드)
- **augment**: 증강 관련 설정 (방법, 강도, TTA 등)
- **validation**: 검증 관련 설정 (전략, early stopping 등)
- **device**: 디바이스 설정 (cuda/cpu/mps)
- **output**: 출력 관련 설정 (디렉토리, 파일명)
- **wandb**: WandB 관련 설정 (프로젝트, 엔티티 등)

### W&B(Weights & Biases) 사용법
W&B를 사용하여 실험을 추적하고 시각화하는 설정 및 실행 방법:

#### 환경 설정
1. 환경 변수 설정 파일 생성:
```bash
# .env 파일 생성 (env_template.txt 참고)
cp src/env_template.txt src/.env
```

2. .env 파일에 wandb 설정 추가:
```env
# Weights & Biases API 키
WANDB_API_KEY=your_wandb_api_key_here

# 프로젝트 설정
WANDB_PROJECT=document-classification
WANDB_ENTITY=your_wandb_entity_here
```

#### 실행 방법
```bash
# WandB 활성화하여 실행
uv run main.py wandb.enabled=true

# 또는 설정 파일에서 활성화
uv run main.py --config-name=wandb_aug
```

#### wandb 설정 관리
config.yaml에서 wandb 관련 설정을 관리할 수 있습니다:
- **wandb.enabled**: wandb 사용 여부 (true/false)
- **wandb.project**: 프로젝트 이름
- **wandb.entity**: 팀/개인 계정 이름
- **wandb.run_name**: 실행 이름 (null이면 자동 생성)
- **wandb.tags**: 태그 목록
- **wandb.notes**: 실험 노트

#### 로깅 정보
wandb에서 다음 정보들을 자동으로 추적합니다:
- 에포크별 training/validation loss, accuracy, f1-score
- 하이퍼파라미터 (학습률, 배치 크기, 모델 이름 등)
- 시스템 메트릭 (GPU 사용률, 메모리 사용량 등)
- 추론 결과 파일 (Artifacts로 버전 관리)

## 테스트 실행

구현된 기능들이 올바르게 동작하는지 pytest로 확인할 수 있습니다.

### 간단한 테스트 실행
```bash
# 모든 테스트 자동 실행 (권장)
uv run pytest

# 상세한 출력으로 실행
uv run pytest -v

# 특정 테스트 파일만 실행
uv run pytest tests/test_main.py -v
uv run pytest tests/test_data.py -v
```

### 기능별 개별 테스트
```bash
# 기능별 개별 테스트 (터미널 출력 포함)
uv run python tests/test_main.py
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

모든 테스트가 통과하면 구현된 기능들이 올바르게 동작함을 확인할 수 있습니다.

## 주의사항
- 메모리 사용량 고려 (GPU 메모리 모니터링)
- 클래스 불균형 문제 해결 필요
- 테스트 데이터의 다양한 변형에 대응
- 재현 가능성을 위한 시드 고정
- uv 환경 관리 주의 (패키지 설치 시 uv add 사용)

## 성능 목표
- 현재 최고 성능: 약 96% 정확도
- 클래스별 균형잡힌 성능 향상
- 테스트 데이터 변형에 대한 강건성 확보
- Leaderboard 점수 향상 