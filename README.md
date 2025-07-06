# Document Classification Competition
## Team

| ![문국현](https://avatars.githubusercontent.com/u/167870439?v=4) | ![류지헌](https://avatars.githubusercontent.com/u/10584296?v=4) | ![이승현](https://avatars.githubusercontent.com/u/126837633?v=4) | ![정재훈](https://avatars.githubusercontent.com/u/127591967?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
| [문국현](https://github.com/GH-Door) | [류지헌](https://github.com/mahomi) | [이승현](https://github.com/shyio06) | [정재훈](https://github.com/coevol) |
| 팀장, 담당 역할 | 담당 역할 | 담당 역할 | 담당 역할 |


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

- _Insert your directory structure_

e.g.
```
├── code
│   ├── jupyter_notebooks
│   │   └── baseline_code.ipynb    # 베이스 라인 코드
│   └── train.py
├── docs
│   ├── pdf
│   │   └── (Template) [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디 .pptx
│   └── paper
└── input
│   ├── get_data.sh  # ./get_data.sh 실행 시, 데이터 다운로드하여 data 폴더를 생성하여 압축을 푼다
    └── data
        ├── train
        └── test
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

### Model descrition

- _Write model information and why your select this model_

### Modeling Process

- _Write model train and test process with capture_

### Random Seed Ensemble

`code/v6_augment`에서 랜덤 시드 앙상블 기능을 제공한다. 설정 파일의
`random_seed_ensemble.enabled` 값을 `true`로 두면 학습 시드를 변경해
여러 번 학습하고, 모든 모델의 예측을 평균하여 최종 결과를 생성한다.
`random_seed_ensemble.count` 로 시드 변경 횟수를 지정할 수 있다.

## 5. Result

### Leader Board

- _Insert Leader Board Capture_
- _Write rank and score_

### Presentation

- _Insert your presentaion file(pdf) link_

## etc

### Meeting Log

- _Insert your meeting log link like Notion or Google Docs_

### Reference

- _Insert related reference_
