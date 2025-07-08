#!/bin/bash

cd src

# 이미지 사이즈 비교
# 실험결과 : 이미지 사이즈가 클수록 성능이 좋아진다. 배치 사이즈가 작을 수록 성능이 좋아진다.
uv run main.py --config-name=wandb_aug wandb.run_name=rs64_b32 data.img_size=64 train.batch_size=32
uv run main.py --config-name=wandb_aug wandb.run_name=rs128_b32 data.img_size=128 train.batch_size=32
uv run main.py --config-name=wandb_aug wandb.run_name=rs224_b32 data.img_size=224 train.batch_size=32
uv run main.py --config-name=wandb_aug wandb.run_name=rs343_b32 data.img_size=343 train.batch_size=32
uv run main.py --config-name=wandb_aug wandb.run_name=rs384_b32 data.img_size=384 train.batch_size=32
uv run main.py --config-name=wandb_aug wandb.run_name=rs480_b32 data.img_size=480 train.batch_size=32

uv run main.py --config-name=wandb_aug wandb.run_name=rs64_b16 data.img_size=64 train.batch_size=16
uv run main.py --config-name=wandb_aug wandb.run_name=rs128_b16 data.img_size=128 train.batch_size=16
uv run main.py --config-name=wandb_aug wandb.run_name=rs224_b16 data.img_size=224 train.batch_size=16
uv run main.py --config-name=wandb_aug wandb.run_name=rs343_b16 data.img_size=343 train.batch_size=16
uv run main.py --config-name=wandb_aug wandb.run_name=rs384_b16 data.img_size=384 train.batch_size=16
uv run main.py --config-name=wandb_aug wandb.run_name=rs480_b16 data.img_size=480 train.batch_size=16

uv run main.py --config-name=wandb_aug wandb.run_name=rs64_b10 data.img_size=64 train.batch_size=10
uv run main.py --config-name=wandb_aug wandb.run_name=rs128_b10 data.img_size=128 train.batch_size=10
uv run main.py --config-name=wandb_aug wandb.run_name=rs224_b10 data.img_size=224 train.batch_size=10
uv run main.py --config-name=wandb_aug wandb.run_name=rs343_b10 data.img_size=343 train.batch_size=10
uv run main.py --config-name=wandb_aug wandb.run_name=rs384_b10 data.img_size=384 train.batch_size=10
uv run main.py --config-name=wandb_aug wandb.run_name=rs480_b10 data.img_size=480 train.batch_size=10


# 학습률
uv run main.py --config-name=wandb_aug wandb.run_name=lr_1e-3 train.lr=1e-3
uv run main.py --config-name=wandb_aug wandb.run_name=lr_1e-4 train.lr=1e-4
uv run main.py --config-name=wandb_aug wandb.run_name=lr_1e-5 train.lr=1e-5

# label_smoothing
uv run main.py --config-name=wandb_aug wandb.run_name=label_smoothing_false train.label_smoothing.enabled=false
uv run main.py --config-name=wandb_aug wandb.run_name=label_smoothing_0.1 train.label_smoothing.smoothing=0.1
uv run main.py --config-name=wandb_aug wandb.run_name=label_smoothing_0.2 train.label_smoothing.smoothing=0.2
uv run main.py --config-name=wandb_aug wandb.run_name=label_smoothing_0.3 train.label_smoothing.smoothing=0.3
uv run main.py --config-name=wandb_aug wandb.run_name=label_smoothing_0.4 train.label_smoothing.smoothing=0.4
uv run main.py --config-name=wandb_aug wandb.run_name=label_smoothing_0.5 train.label_smoothing.smoothing=0.5

# mixed_precision
uv run main.py --config-name=wandb_aug wandb.run_name=mixed_precision_false train.mixed_precision.enabled=false
uv run main.py --config-name=wandb_aug wandb.run_name=mixed_precision_true train.mixed_precision.enabled=true

# 모델 종류별 비교
uv run main.py --config-name=wandb_aug wandb.run_name=model_resnet18 model.name=resnet18
uv run main.py --config-name=wandb_aug wandb.run_name=model_resnet34 model.name=resnet34
uv run main.py --config-name=wandb_aug wandb.run_name=model_efnv2s model.name=efficientnetv2_s
uv run main.py --config-name=wandb_aug wandb.run_name=model_efnv2m model.name=efficientnetv2_m
uv run main.py --config-name=wandb_aug wandb.run_name=model_efnv2l model.name=efficientnetv2_l
uv run main.py --config-name=wandb_aug wandb.run_name=model_efnv2xl model.name=efficientnetv2_xl
uv run main.py --config-name=wandb_aug wandb.run_name=model_tf_efnv2s model.name=tf_efficientnetv2_s
uv run main.py --config-name=wandb_aug wandb.run_name=model_tf_efnv2m model.name=tf_efficientnetv2_m
uv run main.py --config-name=wandb_aug wandb.run_name=model_tf_efnv2l model.name=tf_efficientnetv2_l
uv run main.py --config-name=wandb_aug wandb.run_name=model_tf_efnv2xl model.name=tf_efficientnetv2_xl
uv run main.py --config-name=wandb_aug wandb.run_name=model_rw_t model.name=efficientnetv2_rw_t
uv run main.py --config-name=wandb_aug wandb.run_name=model_rw_s model.name=efficientnetv2_rw_s
uv run main.py --config-name=wandb_aug wandb.run_name=model_rw_m model.name=efficientnetv2_rw_m

# 데이터 증강 테스트
uv run main.py --config-name=wandb_aug wandb.run_name=aug_none augment.train_aug_count=0 augment.valid_aug_count=0
uv run main.py --config-name=wandb_aug wandb.run_name=aug_1 augment.train_aug_count=1 augment.valid_aug_count=1
uv run main.py --config-name=wandb_aug wandb.run_name=aug_3 augment.train_aug_count=3 augment.valid_aug_count=3
uv run main.py --config-name=wandb_aug wandb.run_name=aug_5 augment.train_aug_count=5 augment.valid_aug_count=5
uv run main.py --config-name=wandb_aug wandb.run_name=aug_7 augment.train_aug_count=7 augment.valid_aug_count=7
uv run main.py --config-name=wandb_aug wandb.run_name=aug_9 augment.train_aug_count=9 augment.valid_aug_count=9
uv run main.py --config-name=wandb_aug wandb.run_name=aug_11 augment.train_aug_count=10 augment.valid_aug_count=10

# TTA 테스트
uv run main.py --config-name=wandb_aug wandb.run_name=tta_false augment.test_tta_enabled=false
uv run main.py --config-name=wandb_aug wandb.run_name=tta_true augment.test_tta_enabled=true
