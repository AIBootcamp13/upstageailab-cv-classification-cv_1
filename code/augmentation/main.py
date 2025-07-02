"""
테스트 데이터 기반 적응형 증강 시스템 메인 실행 파일
사용법: python main.py --train_dir input/data/train --test_dir input/data/test --epochs 50
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import os
import time
from pathlib import Path
from typing import Tuple, Dict, Any

# 로컬 모듈 import
from test_data_analyzer import analyze_document_test_data, save_analysis_results
from adaptive_augmentation import create_adaptive_document_pipeline, validate_pipeline_parameters
from adaptive_dataset import AdaptiveDocumentDataset, load_image_paths_and_labels, create_adaptive_dataloader


class AdaptiveTrainingPipeline:
    """적응형 증강 학습 파이프라인"""
    
    def __init__(
        self,
        train_dir: str,
        test_dir: str,
        num_classes: int,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        image_size: int = 224,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.num_classes = num_classes
        self.device = device
        self.image_size = image_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        print(f"🚀 적응형 학습 파이프라인 초기화")
        print(f"  디바이스: {self.device}")
        print(f"  이미지 크기: {self.image_size}")
        print(f"  배치 크기: {self.batch_size}")
        print(f"  학습률: {self.learning_rate}")
        
        # PyTorch 변환 정의
        self.pytorch_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 데이터 로드
        self.setup_data()
        
        # 모델 초기화
        self.setup_model()
    
    def setup_data(self):
        """데이터셋 설정"""
        print("\n📁 데이터 로드 중...")
        
        # 학습 데이터 로드
        self.train_paths, self.train_labels = load_image_paths_and_labels(self.train_dir)
        print(f"  학습 데이터: {len(self.train_paths)}개 이미지")
        
        # 적응형 데이터셋 생성
        self.train_dataset = AdaptiveDocumentDataset(
            image_paths=self.train_paths,
            labels=self.train_labels,
            test_data_dir=self.test_dir,
            pytorch_transform=self.pytorch_transform,
            cache_analysis=True,
            analysis_cache_path="test_analysis_results.json"
        )
        
        # 데이터로더 생성
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        print(f"  배치 수: {len(self.train_loader)}")
        print(f"  테스트 데이터 분석 결과:")
        print(self.train_dataset.get_test_stats_summary())
    
    def setup_model(self):
        """모델 및 최적화 설정"""
        print(f"\n🧠 모델 설정 (클래스 수: {self.num_classes})")
        
        # ResNet 모델 사용
        self.model = models.resnet50(pretrained=True)
        
        # 분류 헤드 교체
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)
        
        # 디바이스로 이동
        self.model = self.model.to(self.device)
        
        # 손실 함수 및 최적화기
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        
        print("✅ 모델 설정 완료")
    
    def train_epoch(self, epoch: int, total_epochs: int) -> Dict[str, float]:
        """한 에포크 학습"""
        
        # 적응형 증강 강도 업데이트
        self.train_dataset.update_pipeline_strength(epoch, total_epochs)
        
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 진행률 출력
            if batch_idx % 50 == 0:
                print(f'  배치 [{batch_idx}/{len(self.train_loader)}] '
                      f'손실: {loss.item():.4f} | 정확도: {100.*correct/total:.2f}%')
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        self.scheduler.step()
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'time': epoch_time
        }
    
    def train(self, epochs: int, save_dir: str = "models"):
        """전체 학습 실행"""
        print(f"\n🏋️ 학습 시작 ({epochs} 에포크)")
        
        # 모델 저장 디렉토리 생성
        Path(save_dir).mkdir(exist_ok=True)
        
        best_accuracy = 0
        training_history = []
        
        for epoch in range(epochs):
            print(f"\n📈 에포크 {epoch+1}/{epochs}")
            
            # 학습 실행
            epoch_results = self.train_epoch(epoch, epochs)
            training_history.append(epoch_results)
            
            # 결과 출력
            print(f"  평균 손실: {epoch_results['loss']:.4f}")
            print(f"  정확도: {epoch_results['accuracy']:.2f}%")
            print(f"  소요 시간: {epoch_results['time']:.2f}초")
            print(f"  학습률: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # 최고 모델 저장
            if epoch_results['accuracy'] > best_accuracy:
                best_accuracy = epoch_results['accuracy']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'accuracy': best_accuracy,
                    'test_stats': self.train_dataset.test_stats
                }, f"{save_dir}/best_model.pth")
                print(f"  ⭐ 최고 모델 저장 (정확도: {best_accuracy:.2f}%)")
        
        print(f"\n🎉 학습 완료! 최고 정확도: {best_accuracy:.2f}%")
        return training_history
    
    def save_analysis_report(self, output_path: str = "analysis_report.txt"):
        """분석 보고서 저장"""
        report = []
        report.append("=" * 60)
        report.append("테스트 데이터 기반 적응형 증강 분석 보고서")
        report.append("=" * 60)
        report.append("")
        
        # 테스트 데이터 분석 결과
        report.append("📊 테스트 데이터 분석 결과:")
        report.append("-" * 30)
        for key, stats in self.train_dataset.test_stats.items():
            report.append(f"{key}:")
            report.append(f"  평균: {stats['mean']:.2f}")
            report.append(f"  표준편차: {stats['std']:.2f}")
            report.append(f"  범위: {stats['min']:.2f} ~ {stats['max']:.2f}")
            report.append("")
        
        # 데이터셋 정보
        report.append("📁 데이터셋 정보:")
        report.append("-" * 30)
        report.append(f"학습 이미지 수: {len(self.train_paths)}")
        report.append(f"클래스 수: {self.num_classes}")
        report.append(f"배치 크기: {self.batch_size}")
        report.append("")
        
        # 모델 정보
        report.append("🧠 모델 정보:")
        report.append("-" * 30)
        report.append(f"모델: ResNet-50")
        report.append(f"디바이스: {self.device}")
        report.append(f"학습률: {self.learning_rate}")
        report.append("")
        
        # 증강 정보
        report.append("⚙️ 적응형 증강 정보:")
        report.append("-" * 30)
        if self.train_dataset.augraphy_available:
            report.append("Augraphy 파이프라인 사용")
        else:
            report.append("기본 증강 사용 (Augraphy 미설치)")
        report.append("")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"📄 분석 보고서 저장: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='테스트 데이터 기반 적응형 증강 학습')
    parser.add_argument('--train_dir', type=str, default='input/data/train',
                        help='학습 데이터 디렉토리')
    parser.add_argument('--test_dir', type=str, default='input/data/test',
                        help='테스트 데이터 디렉토리')
    parser.add_argument('--epochs', type=int, default=50,
                        help='학습 에포크 수')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='학습률')
    parser.add_argument('--image_size', type=int, default=224,
                        help='이미지 크기')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='클래스 수 (자동 감지하려면 None)')
    parser.add_argument('--device', type=str, default='auto',
                        help='디바이스 (auto, cuda, cpu)')
    parser.add_argument('--analysis_only', action='store_true',
                        help='분석만 수행하고 학습하지 않음')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("🔍 테스트 데이터 기반 적응형 증강 시스템")
    print("=" * 50)
    
    # 데이터 디렉토리 확인
    if not os.path.exists(args.train_dir):
        print(f"❌ 학습 데이터 디렉토리를 찾을 수 없습니다: {args.train_dir}")
        return
    
    if not os.path.exists(args.test_dir):
        print(f"❌ 테스트 데이터 디렉토리를 찾을 수 없습니다: {args.test_dir}")
        return
    
    # 클래스 수 자동 감지
    if args.num_classes is None:
        train_paths, train_labels = load_image_paths_and_labels(args.train_dir)
        args.num_classes = len(set(train_labels))
        print(f"📊 자동 감지된 클래스 수: {args.num_classes}")
    
    # 분석만 수행하는 경우
    if args.analysis_only:
        print("\n🔍 테스트 데이터 분석만 수행합니다...")
        try:
            stats = analyze_document_test_data(args.test_dir, sample_size=100)
            save_analysis_results(stats, "test_analysis_results.json")
            
            # 파이프라인 생성 테스트
            if validate_pipeline_parameters(stats):
                pipeline = create_adaptive_document_pipeline(stats)
                print("✅ 적응형 파이프라인 생성 테스트 성공!")
            
        except Exception as e:
            print(f"❌ 분석 실패: {e}")
        return
    
    # 전체 학습 파이프라인 실행
    try:
        pipeline = AdaptiveTrainingPipeline(
            train_dir=args.train_dir,
            test_dir=args.test_dir,
            num_classes=args.num_classes,
            device=device,
            image_size=args.image_size,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        # 학습 실행
        training_history = pipeline.train(args.epochs)
        
        # 분석 보고서 저장
        pipeline.save_analysis_report()
        
        print("\n🎯 학습 완료 요약:")
        final_stats = training_history[-1]
        print(f"  최종 정확도: {final_stats['accuracy']:.2f}%")
        print(f"  최종 손실: {final_stats['loss']:.4f}")
        
    except Exception as e:
        print(f"❌ 학습 파이프라인 실행 실패: {e}")
        print("디버깅을 위해 --analysis_only 옵션을 사용해보세요.")


if __name__ == "__main__":
    main() 