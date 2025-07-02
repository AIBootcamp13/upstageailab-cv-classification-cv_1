# 테스트 데이터 기반 적응형 증강 시스템

문서 이미지 분류를 위한 테스트 데이터 기반 적응형 증강 시스템입니다. 테스트 데이터의 특성(회전, 밝기, 노이즈, 블러 등)을 분석하여 학습 데이터에 유사한 증강을 적용합니다.

## 🎯 주요 기능

- **테스트 데이터 자동 분석**: 회전 각도, 밝기, 노이즈, 블러 등의 특성 분석
- **Augraphy 기반 문서 증강**: 문서 이미지에 특화된 현실적인 증강
- **적응형 파이프라인**: 테스트 데이터 분석 결과를 바탕으로 맞춤형 증강 파이프라인 생성
- **점진적 증강**: 학습 과정에서 증강 강도를 점진적으로 증가
- **fallback 지원**: Augraphy가 없어도 기본 증강으로 동작

## 📁 파일 구조

```
code/
├── test_data_analyzer.py      # 테스트 데이터 분석 모듈
├── adaptive_augmentation.py   # Augraphy 기반 적응형 증강 파이프라인
├── adaptive_dataset.py        # PyTorch Dataset 클래스
├── main.py                   # 메인 실행 파일
├── requirements.txt          # 필요한 라이브러리
└── README.md                 # 사용법 설명
```

## 🚀 설치 및 설정

### 1. 필요한 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 2. Augraphy 설치 (선택사항, 권장)

```bash
pip install augraphy
```

> **참고**: Augraphy가 설치되지 않아도 기본 증강으로 동작하지만, 문서 이미지에 특화된 증강을 위해서는 Augraphy 설치를 권장합니다.

## 📊 사용법

### 1. 기본 사용법

```bash
python main.py --train_dir input/data/train --test_dir input/data/test --epochs 50
```

### 2. 상세 옵션

```bash
python main.py \
    --train_dir input/data/train \
    --test_dir input/data/test \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --image_size 224 \
    --device cuda
```

### 3. 분석만 수행 (학습 없이)

```bash
python main.py --test_dir input/data/test --analysis_only
```

### 4. 사용 가능한 옵션들

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--train_dir` | `input/data/train` | 학습 데이터 디렉토리 |
| `--test_dir` | `input/data/test` | 테스트 데이터 디렉토리 |
| `--epochs` | `50` | 학습 에포크 수 |
| `--batch_size` | `32` | 배치 크기 |
| `--learning_rate` | `0.001` | 학습률 |
| `--image_size` | `224` | 이미지 크기 |
| `--num_classes` | `auto` | 클래스 수 (자동 감지) |
| `--device` | `auto` | 디바이스 (auto, cuda, cpu) |
| `--analysis_only` | `False` | 분석만 수행 |

## 🗂️ 데이터 구조

### 학습 데이터 (클래스별 폴더 구조)
```
input/data/train/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

### 테스트 데이터 (단일 폴더)
```
input/data/test/
├── test1.jpg
├── test2.jpg
├── test3.jpg
└── ...
```

## 🔍 작동 원리

1. **테스트 데이터 분석**: 테스트 이미지들의 회전, 밝기, 노이즈, 블러 등을 분석
2. **적응형 파이프라인 생성**: 분석 결과를 바탕으로 Augraphy 파이프라인 생성
3. **점진적 증강**: 학습 과정에서 증강 강도를 점진적으로 증가
4. **모델 학습**: ResNet-50 기반 분류 모델 학습

## 📈 출력 파일

- `test_analysis_results.json`: 테스트 데이터 분석 결과
- `analysis_report.txt`: 상세 분석 보고서
- `models/best_model.pth`: 최고 성능 모델
- `test_analysis_cache.json`: 분석 결과 캐시

## 🔧 모듈별 사용법

### 1. 테스트 데이터 분석만 실행

```python
from test_data_analyzer import analyze_document_test_data

# 테스트 데이터 분석
stats = analyze_document_test_data("input/data/test", sample_size=100)
print(stats)
```

### 2. 적응형 데이터셋 직접 사용

```python
from adaptive_dataset import AdaptiveDocumentDataset, load_image_paths_and_labels
from torchvision import transforms

# 데이터 로드
train_paths, train_labels = load_image_paths_and_labels("input/data/train")

# 변환 정의
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 적응형 데이터셋 생성
dataset = AdaptiveDocumentDataset(
    image_paths=train_paths,
    labels=train_labels,
    test_data_dir="input/data/test",
    pytorch_transform=transform
)
```

### 3. 커스텀 파이프라인 생성

```python
from adaptive_augmentation import create_adaptive_document_pipeline

# 분석 결과 기반 파이프라인 생성
pipeline = create_adaptive_document_pipeline(stats)

# 이미지에 증강 적용
import cv2
image = cv2.imread("sample.jpg")
augmented = pipeline(image)
```

## ⚠️ 주의사항

1. **Augraphy 설치**: 최적의 성능을 위해 Augraphy 설치 권장
2. **메모리 사용량**: 큰 이미지나 많은 데이터의 경우 메모리 사용량 확인
3. **GPU 메모리**: 배치 크기 조절로 GPU 메모리 사용량 조절
4. **데이터 품질**: 테스트 데이터의 품질이 증강 품질에 직접적 영향

## 🛠️ 문제 해결

### Augraphy 설치 오류
```bash
# 시스템 의존성 설치 (Ubuntu/Debian)
sudo apt-get install python3-opencv

# Augraphy 설치
pip install augraphy
```

### CUDA 관련 오류
```bash
# CPU 모드로 실행
python main.py --device cpu
```

### 메모리 부족
```bash
# 배치 크기 줄이기
python main.py --batch_size 16

# 워커 프로세스 수 줄이기 (adaptive_dataset.py에서 num_workers 조정)
```

## 📊 성능 향상 팁

1. **분석 샘플 수 증가**: 더 많은 테스트 이미지로 분석하여 정확도 향상
2. **점진적 학습**: 작은 학습률로 오래 학습
3. **앙상블**: 여러 모델의 결과를 앙상블
4. **Test Time Augmentation**: 추론 시에도 증강 적용

## 📝 라이센스

이 코드는 문서 이미지 분류를 위한 연구 및 교육 목적으로 제공됩니다. 