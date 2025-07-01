# Document Classification Competition
## Team

| ![문국현](https://avatars.githubusercontent.com/u/167870439?v=4) | ![류지헌](https://avatars.githubusercontent.com/u/10584296?v=4) | ![이승현](https://avatars.githubusercontent.com/u/126837633?v=4) | ![정재훈](https://avatars.githubusercontent.com/u/127591967?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
| [문국현](https://github.com/GH-Door) | [류지헌](https://github.com/UpstageAILab) | [이승현](https://github.com/UpstageAILab) | [정재훈](https://github.com/UpstageAILab) |
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

## 3. Data descrption

### Dataset overview

- _Explain using data_

### EDA

- _Describe your EDA process and step-by-step conclusion_

### Data Processing

- _Describe data processing process (e.g. Data Labeling, Data Cleaning..)_

## 4. Modeling

### Model descrition

- _Write model information and why your select this model_

### Modeling Process

- _Write model train and test process with capture_

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
