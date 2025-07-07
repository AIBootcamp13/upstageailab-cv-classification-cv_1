# 이미지 사이즈 비교
# 실험결과 : 이미지 사이즈가 클수록 성능이 좋아진다. 배치 사이즈가 작을 수록 성능이 좋아진다.
# uv run main.py --config-name=wandb_aug wandb.run_name=rs64_b32 data.img_size=64 training.batch_size=32
# uv run main.py --config-name=wandb_aug wandb.run_name=rs128_b32 data.img_size=128 training.batch_size=32
# uv run main.py --config-name=wandb_aug wandb.run_name=rs224_b32 data.img_size=224 training.batch_size=32
# uv run main.py --config-name=wandb_aug wandb.run_name=rs343_b32 data.img_size=343 training.batch_size=32
# uv run main.py --config-name=wandb_aug wandb.run_name=rs384_b32 data.img_size=384 training.batch_size=32
# uv run main.py --config-name=wandb_aug wandb.run_name=rs480_b32 data.img_size=480 training.batch_size=32

# uv run main.py --config-name=wandb_aug wandb.run_name=rs64_b16 data.img_size=64 training.batch_size=16
# uv run main.py --config-name=wandb_aug wandb.run_name=rs128_b16 data.img_size=128 training.batch_size=16
# uv run main.py --config-name=wandb_aug wandb.run_name=rs224_b16 data.img_size=224 training.batch_size=16
# uv run main.py --config-name=wandb_aug wandb.run_name=rs343_b16 data.img_size=343 training.batch_size=16
# uv run main.py --config-name=wandb_aug wandb.run_name=rs384_b16 data.img_size=384 training.batch_size=16
# uv run main.py --config-name=wandb_aug wandb.run_name=rs480_b16 data.img_size=480 training.batch_size=16

# uv run main.py --config-name=wandb_aug wandb.run_name=rs64_b10 data.img_size=64 training.batch_size=10
# uv run main.py --config-name=wandb_aug wandb.run_name=rs128_b10 data.img_size=128 training.batch_size=10
# uv run main.py --config-name=wandb_aug wandb.run_name=rs224_b10 data.img_size=224 training.batch_size=10
# uv run main.py --config-name=wandb_aug wandb.run_name=rs343_b10 data.img_size=343 training.batch_size=10
# uv run main.py --config-name=wandb_aug wandb.run_name=rs384_b10 data.img_size=384 training.batch_size=10
# uv run main.py --config-name=wandb_aug wandb.run_name=rs480_b10 data.img_size=480 training.batch_size=10


# # 학습률
# uv run main.py --config-name=wandb_aug wandb.run_name=lr_1e-3 training.lr=1e-3
# uv run main.py --config-name=wandb_aug wandb.run_name=lr_1e-4 training.lr=1e-4
# uv run main.py --config-name=wandb_aug wandb.run_name=lr_1e-5 training.lr=1e-5

# # 배치 사이즈
# uv run main.py --config-name=wandb_aug wandb.run_name=batch_size_32 training.batch_size=32
# uv run main.py --config-name=wandb_aug wandb.run_name=batch_size_64 training.batch_size=64
# uv run main.py --config-name=wandb_aug wandb.run_name=batch_size_128 training.batch_size=128

# # label_smoothing
# uv run main.py --config-name=wandb_aug wandb.run_name=label_smoothing_false training.label_smoothing.enabled=false
# uv run main.py --config-name=wandb_aug wandb.run_name=label_smoothing_0.1 training.label_smoothing.smoothing=0.1
# uv run main.py --config-name=wandb_aug wandb.run_name=label_smoothing_0.2 training.label_smoothing.smoothing=0.2
# uv run main.py --config-name=wandb_aug wandb.run_name=label_smoothing_0.3 training.label_smoothing.smoothing=0.3
# uv run main.py --config-name=wandb_aug wandb.run_name=label_smoothing_0.4 training.label_smoothing.smoothing=0.4
# uv run main.py --config-name=wandb_aug wandb.run_name=label_smoothing_0.5 training.label_smoothing.smoothing=0.5

# # mixed_precision
# uv run main.py --config-name=wandb_aug wandb.run_name=mixed_precision_false training.mixed_precision.enabled=false
# uv run main.py --config-name=wandb_aug wandb.run_name=mixed_precision_true training.mixed_precision.enabled=true

# # 증강 강도 비교
uv run main.py --config-name=wandb_aug wandb.run_name=intensity_0.5 augment.intensity=0.5 train.epochs=5
uv run main.py --config-name=wandb_aug wandb.run_name=intensity_1.0 augment.intensity=1.0 train.epochs=5
uv run main.py --config-name=wandb_aug wandb.run_name=intensity_2.0 augment.intensity=2.0 train.epochs=5
uv run main.py --config-name=wandb_aug wandb.run_name=intensity_5.0 augment.intensity=5.0 train.epochs=5

# # 모델 종류별 비교
# uv run main.py --config-name=wandb_aug wandb.run_name=model_resnet18 model.name=resnet18
# uv run main.py --config-name=wandb_aug wandb.run_name=model_resnet34 model.name=resnet34
# uv run main.py --config-name=wandb_aug wandb.run_name=model_efnv2s model.name=efficientnetv2_s
# uv run main.py --config-name=wandb_aug wandb.run_name=model_efnv2m model.name=efficientnetv2_m
# uv run main.py --config-name=wandb_aug wandb.run_name=model_efnv2l model.name=efficientnetv2_l
# uv run main.py --config-name=wandb_aug wandb.run_name=model_efnv2xl model.name=efficientnetv2_xl
# uv run main.py --config-name=wandb_aug wandb.run_name=model_tf_efnv2s model.name=tf_efficientnetv2_s
# uv run main.py --config-name=wandb_aug wandb.run_name=model_tf_efnv2m model.name=tf_efficientnetv2_m
# uv run main.py --config-name=wandb_aug wandb.run_name=model_tf_efnv2l model.name=tf_efficientnetv2_l
# uv run main.py --config-name=wandb_aug wandb.run_name=model_tf_efnv2xl model.name=tf_efficientnetv2_xl
# uv run main.py --config-name=wandb_aug wandb.run_name=model_rw_t model.name=efficientnetv2_rw_t
# uv run main.py --config-name=wandb_aug wandb.run_name=model_rw_s model.name=efficientnetv2_rw_s
# uv run main.py --config-name=wandb_aug wandb.run_name=model_rw_m model.name=efficientnetv2_rw_m
