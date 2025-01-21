# --> General imports
import os  # OS 관련 작업을 위한 모듈
import numpy as np  # 행렬 및 배열 연산을 위한 모듈
from tqdm import tqdm  # 진행 상황을 시각적으로 표시해주는 모듈
import json  # JSON 파일 읽기/쓰기를 위한 모듈

# --> Torch imports 
import torch  # PyTorch 기본 모듈
from torch.utils.data import DataLoader  # 데이터 로딩 및 미니배치 생성을 위한 모듈
import time  # 시간 측정을 위한 모듈
import torch.nn as nn  # 신경망 구축을 위한 모듈
import torch.optim as optim  # 최적화 기법을 제공하는 모듈
from torch.optim.lr_scheduler import CosineAnnealingLR  # 학습률 스케줄링(코사인 감소) 제공
from torch.optim.lr_scheduler import LinearLR  # 학습률 선형 증가 제공

# --> internal imports 
from core.models.mmssl import MMSSL  # 내부 정의된 MMSSL 모델 클래스
from core.dataset.dataset import TangleDataset  # TangleDataset 데이터셋 클래스
from core.loss.tangle_loss import InfoNCE, apply_random_mask, init_intra_wsi_loss_function  # 손실 함수와 데이터 변환 관련 함수
from core.utils.learning import smooth_rank_measure, collate_tangle, set_seed  # 학습 유틸리티 함수들
from core.utils.process_args import process_args  # 명령줄 인자 처리 유틸리티 함수

import pdb  # 디버깅을 위한 Python Debugger

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU 사용 가능 시 GPU, 그렇지 않으면 CPU 선택


# 학습 루프 정의
def train_loop(args, loss_fn_interMod, loss_fn_rnaRecon, loss_fn_intraMod, ssl_model, epoch, dataloader, optimizer, scheduler_warmup, scheduler):
        
    ssl_model.train()  # 모델을 학습 모드로 전환
    ssl_model.to(DEVICE)  # 모델을 선택된 장치(GPU/CPU)에 할당

    ep_loss, ep_recon_loss, ep_inter_loss, ep_intra_loss = 0., 0., 0., 0.  # 에폭 별 손실 초기화
    fb_time = 0.  # Forward-Backward 시간 측정 초기화
    all_embeds = []  # 모든 임베딩을 저장할 리스트
    
    for b_idx, (patch_emb, rna_seq, patch_emb_aug, avg_patch_emb) in enumerate(dataloader):
        
        losses = []  # 각 배치에 대한 손실 초기화
        s_fb = time.time()  # Forward-Backward 시작 시간 기록

        # intra-modality 손실을 위한 전처리 (--method intra일 경우에만 사용됨)
        if args["intra_modality_wsi"]:
            if args["intra_modality_mode_wsi"] == "contrast_token_views":
                patch_emb = torch.cat((patch_emb, patch_emb_aug))  # 증강된 패치 임베딩을 기존 임베딩에 결합
            elif args["intra_modality_mode_wsi"] in ["reconstruct_masked_emb", "reconstruct_masked_emb+contrast_avg_emb"]:
                patch_emb_mask = apply_random_mask(patch_embeddings=patch_emb, percentage=args['mask_percentage'])  # 랜덤 마스킹 적용
                patch_emb = torch.cat((patch_emb, patch_emb_mask))  # 마스크 적용 후 임베딩 결합

        # 데이터를 선택된 장치로 이동
        patch_emb = patch_emb.to(DEVICE)
        rna_seq = rna_seq.to(DEVICE) if rna_seq is not None else rna_seq  # RNA 시퀀스를 장치로 이동
        if args["intra_modality_mode_wsi"] in ["contrast_avg_emb", "reconstruct_avg_emb", "reconstruct_masked_emb+contrast_avg_emb"]:
            avg_patch_emb = avg_patch_emb.cuda()  # 평균 패치 임베딩을 GPU로 이동
                
        # forward pass 및 손실 계산
        if args["intra_modality_wsi"]:
            wsi_emb, rna_emb, rna_reconstruction = ssl_model(patch_emb, None)  # RNA 입력 없이 모델 호출
        else:
            wsi_emb, rna_emb, rna_reconstruction = ssl_model(patch_emb, rna_seq)  # RNA 입력 포함하여 모델 호출
        
        # intra modality 손실 (WSI <-> WSI)
        if rna_emb is None and rna_reconstruction is None:  # RNA 관련 출력이 없는 경우
            if args["intra_modality_mode_wsi"] == "contrast_token_views":
                split_idx = int(patch_emb.shape[0]/2)  # 임베딩 절반으로 분할
                losses.append(loss_fn_intraMod(query=wsi_emb[:split_idx], positive_key=wsi_emb[split_idx:], symmetric=args["symmetric_cl"]))  # 대칭적인 InfoNCE 손실 계산
            elif args["intra_modality_mode_wsi"] == "contrast_avg_emb":
                losses.append(loss_fn_intraMod(query=wsi_emb, positive_key=avg_patch_emb, symmetric=args["symmetric_cl"]))  # 평균 임베딩과의 InfoNCE 손실
            elif args["intra_modality_mode_wsi"] == "reconstruct_avg_emb":
                losses.append(loss_fn_intraMod(wsi_emb, avg_patch_emb))  # 평균 임베딩 재구성 손실
            elif args["intra_modality_mode_wsi"] == "reconstruct_masked_emb":
                split_idx = int(patch_emb.shape[0]/2)
                losses.append(loss_fn_intraMod(wsi_emb[split_idx:], wsi_emb[:split_idx]))  # 마스킹된 WSI 임베딩 재구성 손실
            elif args["intra_modality_mode_wsi"] == "reconstruct_masked_emb+contrast_avg_emb":
                split_idx = int(patch_emb.shape[0]/2)
                losses.append(loss_fn_intraMod(wsi_emb[split_idx:], wsi_emb[:split_idx]))  # 마스킹된 임베딩 재구성 손실
                losses.append(loss_fn_intraMod(query=wsi_emb[:split_idx], positive_key=avg_patch_emb, symmetric=args["symmetric_cl"]))  # 평균 임베딩과의 InfoNCE 손실
            else:
                raise ValueError("Invalid intra_modality_mode_wsi.")  # 잘못된 설정 예외 발생
            ep_intra_loss += losses[-1].item()  # 에폭 별 intra 손실 누적
            
        # inter modality 손실 (WSI <-> RNA)
        if rna_emb is not None:
            losses.append(loss_fn_interMod(query=wsi_emb, positive_key=rna_emb, symmetric=args["symmetric_cl"]))  # 대칭 InfoNCE 손실
            ep_inter_loss += losses[-1].item()  # 에폭 별 inter 손실 누적(가장 최근에 loss의 리스트에 저장된 loss값을 더해줌)
            
        # intra modality 손실 (RNA <-> RNA)
        if rna_reconstruction is not None:
            losses.append(loss_fn_rnaRecon(rna_reconstruction, rna_seq))  # RNA 재구성 손실
            ep_recon_loss += losses[-1].item()  # 에폭 별 RNA 재구성 손실 누적
            
        loss = sum(losses)  # 전체 손실 계산
        optimizer.zero_grad()  # 기존의 기울기 초기화
        loss.backward()  # 역전파 수행
        optimizer.step()  # 가중치 갱신
        
        e_fb = time.time()  # Forward-Backward 종료 시간 기록
        fb_time += e_fb - s_fb  # Forward-Backward 소요 시간 누적

        if epoch <= args["warmup_epochs"]:  # 워밍업 기간 동안
            scheduler_warmup.step()  # 워밍업 스케줄러 갱신
        else:
            scheduler.step()  # 코사인 학습률 스케줄러 갱신
            
        if (b_idx % 3) == 0:  # 매 3번째 배치마다
            print(f"Loss for batch: {b_idx} = {loss}")  # 배치 손실 출력
            
        ep_loss += loss.item()  # 에폭 손실 누적
        
        # 학습 임베딩 저장 (순위 계산을 위해) - 순위 : WSI-RNA 데이터 유사한 임베딩 TOP-k개
        ssl_model.eval()  # 모델을 평가 모드로 전환
        with torch.no_grad():  # 역전파 비활성화
            wsi_emb_to_store, _, _ = ssl_model(patch_emb)  # WSI 임베딩 계산
            all_embeds.extend(wsi_emb_to_store.detach().cpu().numpy())  # 임베딩을 리스트에 저장
        ssl_model.train()  # 모델을 다시 학습 모드로 전환
    
    # 순위 측정
    all_embeds_tensor = torch.Tensor(np.array(all_embeds))  # 임베딩을 텐서로 변환
    rank = smooth_rank_measure(all_embeds_tensor)  # 순위 측정 함수 호출
        
    return ep_loss, rank  # 에폭 손실과 순위 반환


# 검증 루프 정의
def val_loop(ssl_model, val_dataloader):
    
    ssl_model.eval()  # 모델을 평가 모드로 전환
    ssl_model.to(DEVICE)  # 모델을 선택된 장치로 이동
    
    all_embeds = []  # 모든 임베딩 저장 리스트 초기화
    all_labels = []  # 모든 라벨 저장 리스트 초기화
    
    with torch.no_grad():  # 역전파 비활성화
        
        for inputs, labels in tqdm(val_dataloader):  # 검증 데이터 반복
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)  # 데이터를 장치로 이동
            wsi_embed, _, _ = ssl_model(inputs)  # 모델을 사용하여 임베딩 계산
            wsi_embed = wsi_embed.detach().cpu().numpy()  # 임베딩을 CPU로 이동 및 Numpy 배열로 변환
            all_embeds.extend(wsi_embed)  # 임베딩 저장
            all_labels.append(labels.item())  # 라벨 저장
            
    all_embeds = np.array(all_embeds)  # 임베딩 배열 생성
    all_labels = np.array(all_labels)  # 라벨 배열 생성
    
    all_embeds_tensor = torch.Tensor(np.array(all_embeds))  # 텐서 변환
    rank = smooth_rank_measure(all_embeds_tensor)  # 순위 측정
    results_dict = {"embeds": all_embeds, "labels": all_labels}  # 결과 딕셔너리 생성
    
    return results_dict, rank  # 결과 딕셔너리와 순위 반환


# 딕셔너리를 JSON 파일로 저장
def write_dict_to_config_file(config_dict, json_file_path):
    """
    딕셔너리를 구성 파일로 작성.
    Args:
        config_dict (dict): 구성 파일에 작성할 딕셔너리.
        config_file_path (str): 구성 파일 경로.
    Returns:
        None
    """
    with open(json_file_path, 'w') as jsonfile:  # 파일 열기
        json.dump(config_dict, jsonfile, indent=4)  # JSON 형식으로 파일 저장


if __name__ == "__main__":
    
    # 인자 설정 및 시드 설정
    args = process_args()  # 명령줄 인자 처리
    args = vars(args)  # 인자를 딕셔너리로 변환
    set_seed(args["seed"])  # 시드 설정
    
    # 손실 계산을 위한 파라미터 설정
    RNA_RECONSTRUCTION = True if args["method"] == 'tanglerec' else False  # RNA 재구성 사용 여부
    INTRA_MODALITY = True if args["method"] == 'intra' else False  # Intra-modality 사용 여부
    # STOPPING_CRITERIA = 'train_rank' if args["method"] in ['tangle', 'intra'] else 'fixed'  # 중단 기준 설정
    STOPPING_CRITERIA = 'train_rank' if args["method"] in ['intra'] else 'fixed'  # 중단 기준 설정 (원래 위의 기준인데, 모델 저장 loss 기준으로 그냥 마지막 에폭에 저장하도록 해봄)
    # N_TOKENS_RNA = 4908 if args["study"] == 'nsclc' else 4999  # RNA 토큰 수 결정
    N_TOKENS_RNA =4908 if args["study"] == 'nsclc' else 25  # RNA 토큰 수 결정 (ACRIN Breast tabular data의 feature 수 151개임, NSCLC V1은 18, NSCLC V2는 49, NSCLC_Tabnet은 64임)
    

    args["rna_reconstruction"] = RNA_RECONSTRUCTION  # RNA 재구성 설정 저장
    args["intra_modality_wsi"] = INTRA_MODALITY  # WSI Intra-modality 설정 저장
    args["rna_token_dim"] = N_TOKENS_RNA  # RNA 토큰 차원 설정 저장

    # 경로 설정
    ROOT_SAVE_DIR = "/home/alleun1110/Digital_bio/TANGLE_renew/results/{}_checkpoints_and_embeddings".format(args["study"])  # 결과 저장 디렉토리
    EXP_CODE = "{}_{}_{}_T_stage_clinical_RNA_lr{}_epochs{}_bs{}_tokensize{}_temperature{}_uni".format(
        "Synthetic_TabNet_pretrained",
        args["method"],  # 방법
        args["study"],  # 연구
        args["learning_rate"],  # 학습률
        args["epochs"],  # 에폭 수
        args["batch_size"],  # 배치 크기
        args["n_tokens"],  # 토큰 수
        args["temperature"]  # 온도 파라미터,
    )
    RESULTS_SAVE_PATH = os.path.join(ROOT_SAVE_DIR, EXP_CODE)  # 결과 저장 경로 생성
    os.makedirs(RESULTS_SAVE_PATH, exist_ok=True)  # 경로가 없으면 생성
    write_dict_to_config_file(args, os.path.join(RESULTS_SAVE_PATH, "config.json"))  # 설정을 JSON으로 저장

    print()
    print(f"Running experiment {EXP_CODE}...")  # 실험 시작 알림
    print()
    
    # 로그 디렉토리 생성
    log_dir = os.path.join(ROOT_SAVE_DIR, 'logs', EXP_CODE)  # 로그 저장 경로
    os.makedirs(log_dir, exist_ok=True)  # 디렉토리 생성
    
    # Tangle 데이터셋 생성
    print("* Setup dataset...")  # 데이터셋 설정 알림
    dataset = TangleDataset(
        feats_dir="/home/alleun1110/Digital_bio/TANGLE_renew/data/Colorectal/CT_data/representation_learning".format(args["study"]),  # 특징 파일 경로
        rna_dir='/home/alleun1110/Digital_bio/TANGLE_renew/data/Colorectal/clinical_data'.format(args["study"]),  # RNA 파일 경로
        sampling_strategy=args["sampling_strategy"],  # 샘플링 전략
        n_tokens=args["n_tokens"]  # 토큰 수
    )
    
    # 데이터로더 설정
    print("* Setup dataloader...")  # 데이터로더 설정 알림
    dataloader = DataLoader(
        dataset,  # 데이터셋
        batch_size=args["batch_size"],  # 배치 크기
        shuffle=True,  # 데이터 셔플
        collate_fn=collate_tangle  # 데이터 수집 함수
    )

    # 모델 설정
    print("* Setup model...")  # 모델 설정 알림
    
    ssl_model = MMSSL(config=args, n_tokens_rna=N_TOKENS_RNA).to(DEVICE)  # MMSSL 모델 생성 및 장치 할당
    
    if len(args["gpu_devices"]) > 1:  # 여러 GPU 사용 시
        print(f"* Using {torch.cuda.device_count()} GPUs.")  # GPU 사용 알림
        ssl_model = nn.DataParallel(ssl_model, device_ids=args["gpu_devices"])  # 모델 병렬 처리
    ssl_model.to("cuda:0")  # 모델을 첫 번째 GPU로 할당
    
    # 옵티마이저 설정
    print("* Setup optimizer...")  # 옵티마이저 설정 알림
    optimizer = optim.AdamW(ssl_model.parameters(), lr=args["learning_rate"])  # AdamW 옵티마이저 생성
    
    # 학습률 스케줄러 설정
    print("* Setup schedulers...")  # 스케줄러 설정 알림
    T_max = (args["epochs"] - args["warmup_epochs"]) * len(dataloader) if args["warmup"] else args["epochs"] * len(dataloader)  # T_max 계산
    scheduler = CosineAnnealingLR(
        optimizer,  # 옵티마이저
        T_max=T_max,  # 최대 반복
        eta_min=args["end_learning_rate"]  # 최소 학습률
    )
    
    if args["warmup"]:  # 워밍업 설정이 있을 때
        scheduler_warmup = LinearLR(
            optimizer,  # 옵티마이저
            start_factor=0.00001,  # 초기 학습률 비율
            total_iters=args["warmup_epochs"] * len(dataloader)  # 총 반복 수
        )
    else:
        scheduler_warmup = None  # 워밍업 스케줄러 미사용
    
    # 손실 함수 설정
    print("* Setup losses...")  # 손실 함수 설정 알림
    loss_fn_interMod = InfoNCE(temperature=args["temperature"])  # InfoNCE 손실 함수 생성
    loss_fn_rnaRecon = nn.MSELoss()  # MSE 손실 함수 생성
    loss_fn_intraMod = init_intra_wsi_loss_function(args)  # intra-modality 손실 함수 초기화

    # 주요 학습 루프 시작
    best_rank = 0.  # 최고 순위 초기화
    for epoch in range(args["epochs"]):  # 에폭 루프
        
        print()
        print(f"Training for epoch {epoch}...")  # 에폭 시작 알림
        print()
        
        # 학습 단계
        start = time.time()  # 시작 시간 기록
        ep_loss, train_rank = train_loop(args, loss_fn_interMod, loss_fn_rnaRecon, loss_fn_intraMod, ssl_model, epoch, dataloader, optimizer, scheduler_warmup, scheduler)  # 학습 루프 호출
        end = time.time()  # 종료 시간 기록

        print()
        print(f"Done with epoch {epoch}")  # 에폭 종료 알림
        print(f"Total loss = {ep_loss}")  # 총 손실 출력
        print(f"Train rank = {train_rank}")  # 학습 순위 출력
        print("Total time = {:.3f} seconds".format(end-start))  # 에폭 시간 출력

        # 중단 기준에 따른 모델 저장
        if STOPPING_CRITERIA == 'train_rank':  # 순위 기준일 때
            if train_rank > best_rank:  # 더 나은 순위일 경우
                print('Better rank: {} --> {}. Saving model'.format(best_rank, train_rank))  # 모델 저장 알림
                best_rank = train_rank  # 최고 순위 갱신
                torch.save(ssl_model.state_dict(), os.path.join(RESULTS_SAVE_PATH, "model_pretrained.pt"))  # 모델 상태 저장
        else:  # 고정된 에폭일 때
            torch.save(ssl_model.state_dict(), os.path.join(RESULTS_SAVE_PATH, "model_pretrained.pt"), pickle_protocol=4)  # 모델 상태 저장
        print()
    
    print()
    print("Done")  # 전체 학습 종료 알림
    print()
