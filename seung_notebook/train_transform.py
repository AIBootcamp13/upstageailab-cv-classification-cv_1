import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

# Custom motion blur 함수
def apply_motion_blur(img, degree=10, angle=45):
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    kernel = np.diag(np.ones(degree))
    kernel = cv2.warpAffine(kernel, M, (degree, degree))
    kernel = kernel / degree
    blurred = cv2.filter2D(img, -1, kernel)
    return blurred

# Albumentations custom transform 정의
class MotionBlurTransform(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(MotionBlurTransform, self).__init__(always_apply, p)

    def apply(self, img, **params):
        return apply_motion_blur(img, degree=np.random.randint(5, 20), angle=np.random.randint(-45, 45))

# Train transform pipeline 정의
train_transform = A.Compose([
    A.Resize(256, 256),  # 필요에 따라 img_size에 맞게 수정
    A.RandomRotate90(p=0.5),
    A.Rotate(limit=30, p=0.5),  # ±30도 기울기
    A.OneOf([
        A.MotionBlur(p=0.3),
        MotionBlurTransform(p=0.3),
        A.GaussianBlur(p=0.3),
    ], p=0.5),
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        # A.JpegCompression(quality_lower=70, quality_upper=100, p=0.5),
    ], p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.OneOf([
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5),
        A.GridDropout(ratio=0.5, p=0.5),
    ], p=0.4),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])