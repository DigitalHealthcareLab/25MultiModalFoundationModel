# 필요한 라이브러리 임포트
import os  # 파일 시스템 작업을 위한 모듈
from tqdm import tqdm  # 진행 표시를 위한 모듈
import numpy as np  # 배열 및 수치 연산을 위한 모듈

import torch  # PyTorch 라이브러리
from torch.utils.data import DataLoader  # PyTorch에서 데이터를 배치로 로드하기 위한 모듈

# SlideDataset 클래스와 기타 유틸리티 함수들 임포트
from core.dataset.dataset import SlideDataset  # 슬라이드 데이터를 로드하는 데이터셋 클래스
from core.utils.learning import collate_slide, save_pkl, smooth_rank_measure  # 슬라이드 배치 생성, 결과 저장, 순위 측정을 위한 함수들

# 사용 가능한 장치(GPU 또는 CPU)를 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 추론 루프 함수 정의
def inference_loop(ssl_model, val_dataloader):
    """
    모델을 평가 모드로 설정하고, 주어진 데이터로부터 WSI 임베딩을 추출하여 순위를 계산하는 함수.

    Args:
        ssl_model (nn.Module): 훈련된 모델
        val_dataloader (DataLoader): 추론할 데이터를 제공하는 DataLoader

    Returns:
        results_dict (dict): 추출된 임베딩과 슬라이드 ID가 포함된 딕셔너리
        rank (float): 추출된 임베딩에 대한 순위 측정 값
    """
    # 모델을 평가 모드로 설정 (그라디언트 계산 비활성화)
    ssl_model.eval()  # 모델을 평가 모드로 설정
    ssl_model.to(DEVICE)  # 모델을 설정된 디바이스로 이동 (GPU/CPU)

    all_embeds = []  # 모든 WSI 임베딩을 저장할 리스트
    all_slide_ids = []  # 모든 슬라이드 ID를 저장할 리스트

    # 그라디언트 계산을 하지 않도록 설정 (메모리 절약)
    with torch.no_grad():
        # 데이터 로더로부터 배치씩 데이터를 가져와서 추론
        for inputs, slide_id in tqdm(val_dataloader):
            inputs = inputs.to(DEVICE)  # 입력 데이터를 설정된 디바이스로 이동
            # 모델에서 특징(WSI 임베딩)을 추출
            wsi_embed = ssl_model.get_features(inputs)
            wsi_embed = wsi_embed.float().detach().cpu().numpy()  # 텐서를 numpy 배열로 변환하여 저장
            all_embeds.extend(wsi_embed)  # 임베딩을 리스트에 추가
            all_slide_ids.extend(slide_id)  # 슬라이드 ID를 리스트에 추가

    all_embeds = np.array(all_embeds)  # 임베딩 리스트를 numpy 배열로 변환
    all_embeds_tensor = torch.Tensor(np.array(all_embeds))  # numpy 배열을 다시 텐서로 변환 (순위 측정을 위해)

    # 임베딩에 대한 순위를 계산 (유사도 기반)
    rank = smooth_rank_measure(all_embeds_tensor)  # 임베딩 간의 순위를 계산하는 함수 호출
    results_dict = {
        "embeds": all_embeds,  # 임베딩 저장
        "slide_ids": all_slide_ids  # 슬라이드 ID 저장
    }

    return results_dict, rank  # 결과 딕셔너리와 순위 반환

# WSI 임베딩을 추출하고 파일에 저장하는 함수 정의
def extract_wsi_embs_and_save(ssl_model, features_path, save_fname):
    """
    주어진 경로에서 데이터를 로드하여 모델을 통해 WSI 임베딩을 추출하고 결과를 저장하는 함수.

    Args:
        ssl_model (nn.Module): 훈련된 모델
        features_path (str): WSI 특성(패치 임베딩)이 저장된 디렉토리 경로
        save_fname (str): 결과를 저장할 파일 이름 (예: pickle 파일 경로)

    Returns:
        results_dict (dict): 추출된 WSI 임베딩과 슬라이드 ID를 포함하는 딕셔너리
    """
    # 주어진 경로에서 슬라이드 데이터셋을 로드
    test_dataset = SlideDataset(features_path=features_path, extension='.pt') # 기본 설정이 h5로 되어있어서, .pt 파일로 지정 필요함.
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_slide)  # DataLoader 설정

    if len(test_dataloader) == 0:
        raise ValueError("Test DataLoader is empty. Check your dataset or features path.")
    
    # 추론 루프를 실행하여 WSI 임베딩과 순위를 추출
    results_dict, val_rank = inference_loop(ssl_model, test_dataloader)  # WSI 임베딩과 순위 반환
    print("Rank = {}".format(val_rank))  # 추출된 순위 출력

    # 결과를 지정된 파일에 저장
    save_pkl(save_fname, results_dict)  # pickle 파일로 저장

    return results_dict  # 결과 반환
