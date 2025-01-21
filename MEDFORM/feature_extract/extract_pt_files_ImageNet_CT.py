import torch
import torch.nn as nn
from torchvision import models
from dataloader import get_data_loaders  # dataloader.py 파일로부터 가져오기
import pandas as pd

# CUDA 설정
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 데이터 로더 설정
annotations_file = '/home/alleun1110/Digital_bio/TANGLE_renew/feature_extract/NSCLC_annotations.csv'
dataloader = get_data_loaders(annotations_file, batch_size=1, num_workers=16)

# ResNet50 기반 feature 추출기 정의
class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, out_dim, simclr_check_point_path=None):
        super(ResNet50FeatureExtractor, self).__init__()
        resnet = models.resnet50(pretrained=True)  # 사전 학습된 ResNet50 모델 불러오기
        # resnet = models.resnet50(pretrained=False)  # 사전 학습된 가중치 사용 안 함
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1채널 입력
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # 마지막 두 레이어 제외
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive Average Pooling
        self.fc = nn.Linear(2048, out_dim)  # 최종 FC 레이어

        if simclr_check_point_path:
            self.load_simclr_weights(simclr_check_point_path)

    def load_simclr_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.features.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # 평탄화
        x = self.fc(x)
        return x

# 모델 초기화 및 평가 모드 설정
feature_size = 2048
# model = ResNet50FeatureExtractor(out_dim=feature_size, 
    # simclr_check_point_path='/mnt/2021_NIA_data/projects/Digital_Bio/simclr_model/checkpoint_0010.pth.tar').to(device)
model = ResNet50FeatureExtractor(feature_size).to(device)  # 커스텀 모델 정의
model.eval()

# 결과 저장을 위한 DataFrame 생성
# df = pd.DataFrame(columns=['case_id', 'slide_id', 'label'])

# 특징 추출 및 저장
with torch.no_grad():
    for i, (images, sample_id) in enumerate(dataloader):  # label을 받지 않음
        print(f"images shape : {images.shape}")
        # 빈 텐서 생성 (이미지 개수, 2048차원 feature 공간)
        features = torch.empty(images.size(2), feature_size).to(device)

        for idx in range(images.size(2)):
            image = images[:, :, idx,:, :].to(device)  # 이미지 로드 및 전송
            print(image.shape)
            feature = model(image)  # 특징 추출
            features[idx] = feature.flatten()  # 평탄화 후 저장

        # 파일 경로 및 저장
        file_path = f'/home/alleun1110/Digital_bio/TANGLE_renew/data/NSCLC_Radiogenomics_fixed/{sample_id[0]}.pt'
        torch.save(features, file_path)
        print(f"Saved features for {sample_id[0]}")



