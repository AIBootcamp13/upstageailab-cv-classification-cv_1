# Validation 기능 사용법

## 구현된 기능

1. **8:2 Holdout 검증**
   - 훈련 데이터를 8:2로 분할하여 검증
   - Stratified 분할 지원
   - Early stopping 적용 가능

2. **Stratified K-Fold 교차 검증**
   - K개의 fold로 교차 검증
   - 각 fold별로 개별 모델 훈련
   - 앙상블 예측으로 최종 결과 생성
   - Early stopping 적용 가능

3. **Early Stopping**
   - 검증 loss, accuracy, f1-score 기준으로 조기 종료
   - patience, min_delta 설정 가능
   - validation이 있는 경우에만 동작

## 설정 옵션

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

## 사용 예시

### 1. Holdout 검증
```bash
# 기본 holdout 검증
uv run main.py --config-name=holdout

# 또는 직접 설정
uv run main.py validation.strategy=holdout validation.holdout.train_ratio=0.8
```

### 2. K-Fold 교차 검증
```bash
# 5-fold 교차 검증
uv run main.py --config-name=kfold

# 또는 직접 설정
uv run main.py validation.strategy=kfold validation.kfold.n_splits=5
```

### 3. 빠른 테스트
```bash
# Holdout 빠른 테스트
uv run main.py --config-name=test_holdout

# K-Fold 빠른 테스트
uv run main.py --config-name=test_kfold
```

### 4. 검증 없이 전체 데이터 사용
```bash
uv run main.py validation.strategy=none
```

## 주요 특징

1. **Stratified 분할**: 클래스 불균형을 고려한 분할
2. **Early Stopping**: 과적합 방지를 위한 조기 종료
3. **K-Fold 앙상블**: 여러 모델의 평균으로 더 안정적인 예측
4. **WandB 지원**: 모든 검증 지표를 WandB에 자동 로깅
5. **유연한 설정**: Hydra를 통한 다양한 설정 조합 가능

## 출력 결과

- **Holdout/No validation**: 단일 모델의 예측 결과
- **K-Fold**: 모든 fold의 앙상블 예측 결과 (softmax 평균 후 argmax)

각 fold별 성능과 최종 앙상블 성능이 로그와 WandB에 기록됩니다. 