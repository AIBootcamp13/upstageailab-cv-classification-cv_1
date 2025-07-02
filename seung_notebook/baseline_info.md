# Baseline Code 
---

## 1. 문제 정의 & 목표

- **문제 유형**: 문서 이미지 타입 분류 (document type classification)
- **목표**: 문서 이미지를 주어진 클래스(총 17개) 중 하나로 정확하게 분류
- **평가지표**: Accuracy, macro F1-score (코드에서 `f1_score (average='macro')` 사용)
- **출력**: 클래스 레이블(`target`) 예측 후 제출 파일 생성

---

## 2. 데이터 구조 & 전처리 방식

- **데이터 파일**: `train.csv`, `sample_submission.csv`
- **이미지 경로**: `train/`, `test/`
- **전처리**
  - Resize (32x32)
  - Normalize (mean: `[0.485, 0.456, 0.406]`, std: `[0.229, 0.224, 0.225]`)
  - ToTensorV2 (Albumentations 사용한 데이터 증강)
- **데이터셋 클래스**: `ImageDataset` 정의하여 CSV 기반 로딩

---

## 3. 모델 아키텍처 & 설정

- **모델**: `ResNet34` (timm 라이브러리 사용, pretrained=True)
- **출력 클래스 수**: 17
- **모델 헤드**: 마지막 linear layer가 `num_classes=17`로 자동 설정
- **활성화 함수**: 최종적으로 softmax 후 argmax (코드 상에서 명시적으로 softmax는 안 쓰고, `argmax(dim=1)` 사용)

---

## 4. Loss function & Optimizer

- **손실 함수**: CrossEntropyLoss
- **Optimizer**: Adam (learning rate: 1e-3)
- **Scheduler**: 없음 (baseline 코드에 포함되지 않음)

---

## 5. 학습 루프 및 평가 로직

- **학습 epoch 수**: 1 (baseline 기준)
- **batch size**: 32
- **metric 계산**: accuracy, macro F1-score
- **val set 분리**: 없음 (baseline은 validation 없이 전체 train dataset 사용)

---

## 6. Inference 및 제출 방식

- **test set 추론**: `tst_loader` 사용, argmax로 클래스 예측
- **결과 저장**: `pred.csv` 파일로 저장 (폴더: `output/`)
- **제출 파일 포맷**: sample_submission과 동일하게 `ID`, `target` 열 포함

---

## 7. 하이퍼파라미터 및 고정값

- **random seed**: 42 (numpy, torch, random 등 모두 고정)
- **이미지 크기**: 32x32
- **augmentation 강도**: 최소화 (resize, normalize만 사용)

---

## 8. 성능(benchmark) 결과 & 로그

- **baseline 점수**: 코드 상에 log 출력 형태로 `train_acc`, `train_f1`, `train_loss` 제공
- **로그 구조**
  ```text
  epoch: 0
  train_loss: ...
  train_acc: ...
  train_f1: ...