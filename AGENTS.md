# Upstage AI Lab CV Classification Project - Cursor Rules

## 프로젝트 개요
이 프로젝트는 문서 타입 분류를 위한 이미지 분류 대회 참여를 위한 팀 프로젝트입니다. 주어진 문서 이미지를 17개 클래스 중 하나로 분류하는 과제로, 금융, 의료, 보험, 물류 등 다양한 산업 분야에서 실제 활용되는 문서 분류 자동화 기술 개발을 목표로 합니다.

## 기술 스택
- **Python 3.10.13+**
- **PyTorch 2.1.0** - 딥러닝 프레임워크
- **timm 0.9.12** - 사전 훈련된 모델 라이브러리
- **albumentations 1.3.1** - 이미지 증강
- **pandas 2.1.4, numpy 1.26.0** - 데이터 처리
- **scikit-learn** - 평가 메트릭
- **PIL, OpenCV** - 이미지 처리

## 프로젝트 구조

### 주요 디렉토리
- `code/` - 메인 코드 디렉토리
  - `baseline/` - 기본 베이스라인 모델
  - `exp1/` - k-fold 교차 검증 실험 (holdout, kfold)
  - `exp2/` - 고급 모델 실험 (TTA, EfficientNet, 앙상블)
  - `exp3/` - 최종 실험 (메인 실험)
  - `augmentation/` - 적응형 데이터 증강 시스템
  - `jupyter_notebooks/` - EDA 및 분석 노트북
    - `seung_notebook/` - 추가 분석 노트북
- `input/data/` - 데이터셋 (train, test)
- `docs/` - 문서 및 가이드 (wandb_guide.md)
- `.venv/` - uv 가상환경

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
- GPU 사용 가능 시 CUDA 우선 사용

### 모델 관련
- `timm` 라이브러리 사용하여 사전 훈련된 모델 로드
- ResNet, EfficientNet 등 다양한 백본 모델 실험
- TTA (Test Time Augmentation) 적용
- K-fold 교차 검증 사용

### 데이터 처리
- `albumentations` 사용한 이미지 증강
- 클래스 불균형 대응 (가중치 조정, 증강)
- 테스트 데이터 특성 분석 기반 적응형 증강
- 밝기, 대비, 회전 보정 적용

## 주요 기능

### 1. 베이스라인 모델
- ResNet 기반 기본 분류 모델
- 단순한 데이터 증강
- 기본 학습 파이프라인

### 2. 실험별 특징
- **exp1**: k-fold 교차 검증 실험 (holdout, kfold)
- **exp2**: 고급 모델 실험 (TTA, EfficientNet-v2, 앙상블)
- **exp3**: 최종 실험 (메인 실험)

### 3. 적응형 증강 시스템
- 테스트 데이터 분석 기반 증강
- Augraphy 기반 문서 특화 증강
- 점진적 증강 강도 조절

## 파일 명명 규칙
- 실험 파일: `baseline_code_v{version}_{feature}.py`
- 로그 파일: `{experiment_name}.log`
- 출력 파일: `pred_{experiment_name}_{score}.csv`
- 실행 스크립트: `run.sh` (exp2 실험들 자동 실행)

## 환경 설정
- uv 가상환경: `.venv/` (자동 생성)
- 환경 파일: `pyproject.toml`, `uv.lock`
- Python 버전: 3.10.13
- 실행 명령어: `uv run <script_name>`

## 주의사항
- 메모리 사용량 고려 (GPU 메모리 모니터링)
- 클래스 불균형 문제 해결 필요
- 테스트 데이터의 다양한 변형에 대응
- 재현 가능성을 위한 시드 고정

## 성능 목표
- 현재 최고 성능: 약 96% 정확도 (exp2 결과 기준)
- 클래스별 균형잡힌 성능 향상
- 테스트 데이터 변형에 대한 강건성 확보
- Leaderboard 점수 향상 (현재 최고: 90.13%) 