import os
import pydicom  # DICOM 파일 읽기용 라이브러리
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms  # 이미지 변환
import scipy.ndimage  # 보간법 사용
from skimage.draw import disk

# 원형 마스크 생성: 스캐너 외부 영역의 잡음 제거
def create_circle_mask(shape=(512, 512), radius=253):
    mask = np.zeros(shape, dtype=np.uint16)
    rr, cc = disk((shape[0] // 2, shape[1] // 2), radius)
    mask[rr, cc] = 1
    return mask, 1 - mask  # 원 내부와 외부의 반전된 마스크 반환

# HU 정규화 함수
def normalize_16bit_dicom_images(cta_image, HU_window=np.array([-1000., 400.])):  # -300~200임 Breast의 경우
    th_cta_image = (cta_image - HU_window[0]) / (HU_window[1] - HU_window[0])
    th_cta_image = np.clip(th_cta_image, 0, 1)  # 0~1 범위로 클리핑
    return (th_cta_image * 255).astype('uint16')

# CT 이미지 해상도 재조정 함수 (1mm 해상도로)
def resample(image, spacing, new_spacing=[1.0, 0.8, 0.8]):
    spacing = np.array(spacing, dtype=np.float32)
    resize_factor = spacing / np.array(new_spacing)
    new_shape = np.round(image.shape * resize_factor).astype(int)

    print(f"Resampling from {image.shape} to {new_shape} with factor {resize_factor}")

    # 이미지 보간 (order=1: 선형 보간)
    resampled_image = scipy.ndimage.zoom(image, resize_factor, order=1, mode='nearest')
    return resampled_image

# DICOM 파일 로드 및 전처리
# DICOM 파일 로드 및 전처리
def load_16bit_dicom_images(path):
    slices = [pydicom.dcmread(os.path.join(path, s)) for s in os.listdir(path) if s.endswith('.dcm')]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    images = np.stack([s.pixel_array for s in slices]).astype(np.int16)  # images를 int16으로 변환

    # HU 변환 수행
    for i, s in enumerate(slices):
        images[i] = images[i] * s.RescaleSlope + s.RescaleIntercept

    # PixelSpacing과 SliceThickness 읽기
    try:
        pixel_spacing = [float(sp) for sp in slices[0].PixelSpacing]
    except (AttributeError, ValueError, TypeError) as e:
        print(f"Error reading PixelSpacing: {e}. Defaulting to [1.0, 1.0].")
        pixel_spacing = [1.0, 1.0]

    try:
        slice_thickness = float(slices[0].SliceThickness)
    except (AttributeError, ValueError, TypeError) as e:
        print(f"Error reading SliceThickness: {e}. Defaulting to 1.0.")
        slice_thickness = 1.0

    spacing = [slice_thickness] + pixel_spacing

    # 원형 마스크 생성 및 적용
    mask, mask_inv = create_circle_mask(images.shape[1:])
    for i in range(images.shape[0]):
        images[i] = images[i] * mask - 2000 * mask_inv  # 2000과 곱할 때 int16에서 처리

    # HU 정규화 및 클리핑
    normalized_images = normalize_16bit_dicom_images(images)

    # 이미지를 uint8로 변환 (0~255 범위로 클리핑)
    normalized_images = np.clip(normalized_images, 0, 255).astype(np.uint16)

    return normalized_images, spacing


# CT 데이터셋 정의 (레이블 없이 특징만 반환)
class NSCLCDataset(Dataset):
    def __init__(self, annotations, transform=None, target_depth=270):
        self.annotations = annotations
        self.transform = transform
        self.target_depth = target_depth

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        dir_path = self.annotations.iloc[idx]['File Location']
        sample_id = self.annotations.iloc[idx]['Case ID']

        images, spacing = load_16bit_dicom_images(dir_path)
        print(f"Loaded images shape: {images.shape}, Spacing: {spacing}")

        # 해상도 재조정
        resampled_images = resample(images, spacing)
        print(f"Resampled images shape: {resampled_images.shape}")

        # 슬라이스 개수 맞춤
        if resampled_images.shape[0] != self.target_depth:
            resampled_images = scipy.ndimage.zoom(
                resampled_images, (self.target_depth / resampled_images.shape[0], 1, 1), order=2
            )
        print(f"resampled_images_shape : {resampled_images.shape}")
        
        images_tensor = torch.from_numpy(resampled_images).unsqueeze(0).float()

        # 레이블을 반환하지 않음 (특징만 반환)
        return images_tensor, sample_id

# DataLoader 반환 함수
def get_data_loaders(annotations_file, batch_size, num_workers=0):
    transform = transforms.Compose([transforms.ToTensor()])
    annotations = pd.read_csv(annotations_file)

    dataset = NSCLCDataset(annotations=annotations, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader
