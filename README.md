# Document Classification Competition
## Team AigoRhythm

| ![문국현](https://avatars.githubusercontent.com/u/167870439?v=4) | ![류지헌](https://avatars.githubusercontent.com/u/10584296?v=4) | ![이승현](https://avatars.githubusercontent.com/u/126837633?v=4) | ![정재훈](https://avatars.githubusercontent.com/u/127591967?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
| [문국현](https://github.com/GH-Door) | [류지헌](https://github.com/mahomi) | [이승현](https://github.com/shyio06) | [정재훈](https://github.com/coevol) |
| 팀장, 모델링 총괄 | 모델 실험 및 최적화 | EDA 및 데이터 전처리 | 모델 검증 및 성능 분석 |


## 0. Overview
이 레포지토리는 문서 타입 분류를 위한 이미지 분류 대회 참여를 위한 팀 프로젝트 공간입니다.
주어진 문서 이미지를 17개 클래스 중 하나로 분류하는 과제로, 금융, 의료, 보험, 물류 등 다양한 산업 분야에서 실제 활용되는 문서 분류 자동화 기술 개발을 목표로 합니다.

### Environment

본 프로젝트는 **문서 타입 분류(Document Type Classification)** 태스크를 위한 Python 및 PyTorch 기반 딥러닝 환경에서 개발되었습니다.

팀 내부 환경 통일과 협업을 위해 **conda 가상환경**을 사용하며, 아래와 같은 방식으로 환경을 관리합니다.

- 가상환경 설정 파일: `environment.yml`
- 동일한 가상환경 설치 명령어:
  ```bash
  conda env create -f environment.yml
- 설치된 라이브러리 확인:
    ```bash
    conda list
- 가상환경 이름: CV_Project

- 가상환경 활성화:
    ```bash
    conda activate CV_Project
기존에 같은 이름의 가상환경이 있다면, 충돌 방지를 위해 삭제 후 설치 진행을 권장합니다.


### Requirements
본 프로젝트는 아래와 같은 주요 라이브러리를 사용하며, environment.yml에 명시되어 있습니다.

- torch
- torchvision
- timm
- albumentations
- pandas
- numpy
- scikit-learn
- tqdm
- pillow

## 1. Competiton Info

### Overview

- 대회 주제: 17개 클래스의 문서 타입 이미지 분류
- 도메인: Computer Vision - Document Classification
- 데이터: 현업 실 데이터 기반으로 제작된 문서 이미지 데이터셋
- 목표: 주어진 문서 이미지를 17개 클래스 중 하나로 정확하게 분류하는 모델 개발

### Timeline

- 대회 기간: 2025년 6월 30일 (월) 10:00 ~ 7월 10일 (목) 19:00

## 2. Components

### Directory

```
├── code/
│   ├── baseline_code_with_log.py
│   ├── baseline_code.py
│   ├── jhryu/
│   │   ├── v1_simple/                # 단순 모델 실험 디렉토리
│   │   │   ├── baseline_code_v4_*.py  # 다양한 모델 실험 파일들
│   │   │   ├── baseline_code_v5_*.py  # K-fold 교차검증 실험 파일들
│   │   │   └── ensemble_*.py         # 앙상블 모델 파일들
│   │   └── v2_hydra_wandb/           # Hydra와 WandB를 활용한 실험 디렉토리
│   │       ├── main.py               # 메인 실행 파일
│   │       ├── config/               # 설정 파일 디렉토리
│   │       ├── data.py               # 데이터 처리 모듈
│   │       ├── models.py             # 모델 정의 모듈
│   │       ├── training.py           # 훈련 로직 모듈
│   │       ├── inference.py          # 추론 로직 모듈
│   │       ├── utils.py              # 유틸리티 함수들
│   │       └── tests/                # 테스트 파일들
│   ├── jupyter_notebooks/
│   │   ├── baseline_code.ipynb       # 베이스라인 코드
│   │   ├── jhryu_eda_img_size.ipynb  # 이미지 크기 EDA
│   │   ├── Moon.ipynb                # Moon 분석 노트북
│   │   └── requirements.txt          # 노트북 의존성
│   ├── seung_notebook/
│   │   ├── baseline_code.py          # 베이스라인 코드
│   │   ├── *_eda*.ipynb             # EDA 분석 노트북들
│   │   ├── baseline_valid_transform_code.ipynb
│   │   ├── train_transform.py        # 훈련 데이터 변환 코드
│   │   └── *.md                     # 분석 문서들
│   └── utils/
│       └── log_util.py               # 로깅 유틸리티
├── docs/
│   └── wandb_guide.md               # WandB 사용 가이드
├── input/
│   ├── get_data.sh                  # 데이터 다운로드 스크립트
│   └── data/                        # 데이터 디렉토리 (다운로드 후 생성)
│       ├── train/                   # 훈련 데이터
│       └── test/                    # 테스트 데이터
├── code.tar.gz                      # 코드 아카이브
└── README.md                        # 프로젝트 설명 문서
```

## 3. Data description

### Dataset overview

- **Train 데이터**
  - 이미지: 총 1,570장
  - 클래스: 총 17개
  - `train.csv` 파일에 ID와 클래스 라벨(`target`)이 포함되어 있습니다.
  - `meta.csv` 파일에는 클래스 번호(`target`)와 클래스 이름(`class_name`) 정보가 담겨 있습니다.

- **Test 데이터**
  - 이미지: 총 3,140장
  - `sample_submission.csv` 파일에 ID가 포함되어 있으며, 예측 결과를 제출할 때 사용됩니다.
  - Test 데이터는 회전, 반전 등 다양한 변형과 훼손이 포함되어 있어, 실제 환경과 유사한 조건을 반영합니다.

### EDA

- **Train 데이터 EDA**
  - **파일 일치 확인**: CSV와 이미지 디렉토리 간 누락된 파일 없음.
  - **클래스 분포 분석**: 상위 14개 클래스는 각 100장으로 균등하지만, 일부 클래스(`resume`, `statement_of_opinion`, `application_for_payment_of_pregnancy_medical_expenses`)는 샘플 수가 적어 불균형 존재.
  - **해상도 및 비율 분석**: 클래스별로 명확한 종횡비 분포가 나타나며, 일부 클래스는 회전된 이미지가 혼재. 비율 기반으로 회전/왜곡 여부와 패턴을 파악.
  - **밝기 및 대비 분석**: 클래스별 평균 밝기와 분산을 확인하여, 저강도(어두운), 중간강도, 고강도 그룹으로 나누어 분석.
  - **마스킹 분석**: 클래스별 밝은 영역과 어두운 영역의 비율을 확인, 보안 문서류는 어두운 영역이 높음.
  - **전반 결론**: 클래스 불균형, 회전/왜곡, 밝기 차이가 존재 → 이를 고려한 데이터 증강, 클래스 가중치 조정, 밝기 보정 전략 필요.

- **Test 데이터 EDA**
  - **파일 일치 확인**: CSV와 이미지 디렉토리 간 누락된 파일 없음.
  - **해상도 및 비율 분석**: 0.75 (세로형), 1.25 (가로형) 비율 이미지가 대부분을 차지.
  - **밝기 분석**: 대부분 밝은 배경(평균 픽셀 값 180–220), train 대비 훨씬 밝고 균일함.
  - **마스킹 분석**: 어두운 영역 비율이 매우 낮아 대부분 밝은 문서. (dark ratio 거의 0)
  - **컬러/흑백 비율**: 100% 컬러 이미지.

### Data Processing

- **데이터 라벨링**
  - Train 데이터는 `train.csv`의 `target` 컬럼을 기준으로 클래스 레이블을 부여.

- **데이터 클리닝 및 전처리**
  - Train 이미지의 밝기 및 대비 보정: Test 데이터 분포(밝고 균일)에 맞도록 조정.
  - 회전 및 왜곡 보정: 클래스별 비율 패턴과 회전 상태를 분석해 자동 회전 보정 적용.
  - 배경 정규화: 배경 영역을 완전한 흰색으로 정리.
  - 노이즈 제거 및 텍스트 강화: 보안 문서류는 강한 노이즈 제거 및 에지 강화, 의료/금융 문서는 부드러운 노이즈 제거 및 선명도 향상.
  - 데이터 증강: 소수 클래스에 대한 증강, aspect ratio 기반 TTA (Test Time Augmentation) 전략 포함.
  - 마스킹 영역 고려: 클래스별 밝기/어두움 비율을 기반으로 증강 및 보정 전략 설계.

- **클래스 불균형 대응**
  - 클래스 가중치 조정 및 소수 클래스 중심 데이터 증강 전략 적용.


## 4. Modeling

### Model Description

본 프로젝트에서는 **EfficientNet 계열 모델**을 주력으로 사용하여 문서 분류 성능을 극대화했습니다.

#### 사용된 모델 아키텍처

- **EfficientNet-B3**: 초기 베이스라인 모델로 사용
- **EfficientNetV2-L**: 대용량 모델로 성능 개선
- **EfficientNetV2-XL**: 최고 성능을 위한 초대형 모델
- **EfficientNetV2-RW-M**: 효율성과 성능의 균형을 위한 모델

#### 모델 선택 이유

1. **높은 성능**: ImageNet에서 검증된 SOTA 성능
2. **효율성**: 파라미터 대비 높은 성능 효율
3. **전이학습 적합성**: 사전 훈련된 가중치를 활용한 빠른 학습
4. **문서 이미지 특성**: 세밀한 텍스트와 구조 인식에 우수한 성능

### Modeling Process

#### 1. 베이스라인 모델 (ResNet34)
- **목적**: 기본 성능 확인 및 파이프라인 검증
- **이미지 크기**: 32×32 (초기 테스트)
- **성능**: 기본적인 분류 성능 확인

#### 2. EfficientNet 시리즈 실험
- **EfficientNet-B3**: 첫 번째 주요 모델
- **EfficientNetV2-L/XL**: 대용량 모델로 성능 향상
- **이미지 크기**: 320×320 → 480×480 (점진적 증가)
- **배치 크기**: 32 → 16 (메모리 효율성 고려)

#### 3. K-Fold 교차 검증
- **Fold 수**: 5-fold 교차 검증
- **다중 시드**: 42, 123 등 다양한 시드로 안정성 확보
- **앙상블**: 10개 모델 (2개 시드 × 5-fold) 앙상블

#### 4. 데이터 증강 및 TTA
- **훈련 시 증강**: 
  - 회전 (±10도)
  - 밝기/대비 조정
  - 가우시안 노이즈 추가
  - 크기 조정 및 크롭
- **TTA (Test Time Augmentation)**:
  - 원본 이미지
  - 수평 뒤집기
  - 경미한 회전
  - 밝기 조정

#### 5. 최적화 전략
- **손실 함수**: CrossEntropyLoss
- **옵티마이저**: Adam (학습률 0.001)
- **스케줄러**: CosineAnnealingLR
- **조기 종료**: 검증 성능 기준 조기 종료
- **모델 저장**: 최고 F1 점수 기준 모델 저장

#### 6. 성능 향상 기법
- **클래스 불균형 대응**: 가중치 조정 및 소수 클래스 증강
- **캐싱 시스템**: 증강된 이미지 캐싱으로 학습 속도 향상
- **배치 정규화**: 안정적인 학습을 위한 정규화
- **드롭아웃**: 과적합 방지

## 5. Result

### Leader Board

#### 최종 성능 결과

- **최고 성능 모델**: EfficientNetV2-XL + K-Fold 앙상블 + TTA
- **검증 성능**: 
  - F1 Score: 0.93+ (검증 데이터 기준)
  - Accuracy: 0.93+ (검증 데이터 기준)
- **모델 구성**: 
  - 10개 모델 앙상블 (2개 시드 × 5-fold)
  - 이미지 크기: 320×320, 384x384
  - TTA 적용

#### 실험 결과 요약

| 모델 | 이미지 크기 | K-Fold | TTA | 검증 F1 | 검증 ACC |
|------|-------------|---------|-----|---------|----------|
| ResNet34 | 32×32 | X | X | ~0.40 | ~0.45 |
| EfficientNet-B3 | 224×224 | X | X | ~0.65 | ~0.70 |
| EfficientNetV2-L | 320×320 | 5-fold | X | ~0.80 | ~0.82 |
| EfficientNetV2-XL | 480×480 | 5-fold | O | **0.86+** | **0.87+** |

### Presentation

- 발표 자료는 프로젝트 완료 후 업데이트 예정

## etc

### Meeting Log

- 팀 회의록은 내부 협업 도구를 통해 관리
- 주요 결정사항과 실험 결과는 코드 내 주석 및 로그 파일로 관리

### Reference

#### 주요 참고 문헌

- **EfficientNet 논문**: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- **EfficientNetV2 논문**: [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
- **timm 라이브러리**: [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models)
- **Albumentations**: [Fast Image Augmentation Library](https://github.com/albumentations-team/albumentations)

#### 기술 참고 자료

- **K-Fold Cross Validation**: 모델 성능 검증을 위한 교차 검증 기법
- **Test Time Augmentation**: 추론 시 증강 기법으로 성능 향상
- **Class Imbalance Handling**: 불균형 데이터셋 대응 전략
- **Document Classification**: 문서 분류를 위한 컴퓨터 비전 기법
