
"""
python extract_slide_embeddings_from_checkpoint.py --pretrained results/brca_checkpoints_and_embeddings/tangle_brca_lr0.0001_epochs100_bs64_tokensize2048_temperature0.01/
python extract_slide_embeddings_from_checkpoint.py --pretrained results/brca_checkpoints_and_embeddings/intra_brca_lr0.0001_epochs100_bs64_tokensize2048_temperature0.01/
python extract_slide_embeddings_from_checkpoint.py --pretrained results/brca_checkpoints_and_embeddings/tanglerec_brca_lr0.0001_epochs100_bs64_tokensize2048_temperature0.01/
python extract_slide_embeddings_from_checkpoint.py --pretrained results/pancancer_checkpoints_and_embeddings/tangle_pancancer/
"""

# 필요한 라이브러리 및 모듈을 임포트
import os  # 파일 및 디렉토리 작업을 위한 라이브러리
import json  # JSON 파일을 읽고 쓰기 위한 라이브러리
from collections import OrderedDict  # 순서가 있는 딕셔너리를 위한 라이브러리
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import torch  # PyTorch 라이브러리

# MMSSL 모델과 관련된 함수들 임포트
from core.models.mmssl import MMSSL  
from core.downstream.downstream import extract_wsi_embs_and_save  # WSI 임베딩을 추출하고 저장하는 함수
from core.utils.process_args import process_args  # 명령줄 인자를 처리하는 함수

import pdb  # 디버깅을 위한 Python Debugger

# 현재 사용할 디바이스 설정 (GPU가 있으면 GPU, 없으면 CPU 사용)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 다운스트림 작업에서 사용할 데이터 경로 설정
DOWNSTREAM_TASKS_CONFIG = {
    # "bcnb": "./data/brca/uni_features/bcnb_features",  # BCNB 데이터 경로 예시 (주석 처리됨)
    # "ACRIN_Breast": "/home/alleun1110/Digital_bio/TANGLE_renew/data/ACRIN_Breast/pt_files_SimCLR_PET-CT_pretrained_few_shot"
    # "NSCLC_Radiogenomics_fixed" : "/home/alleun1110/Digital_bio/TANGLE_renew/data/NSCLC_Radiogenomics_fixed/pt_files_SimCLR_CT_pretrained_few_shot"
    "Colorectal" : "/home/alleun1110/Digital_bio/TANGLE_renew/data/Colorectal/CT_data/few_shot_learning"
}

# 명령줄 인자와 모델 설정을 결합하여 최종적으로 사용할 인자들 설정
def set_args(args, config_from_model):
    # 실험 코드 생성 (예: "tangle_brca_lr0.0001_epochs100_bs64_tokensize2048_temperature0.01")
    exp_code = os.path.split(os.path.normpath(args['pretrained']))[-1]
    args['study'] = exp_code.split('_')[0]  # study는 EXP_CODE에서 첫 번째 부분
    # 모델 설정 값을 명시적으로 할당
    for key in ['wsi_encoder', 'activation', 'method', 'n_heads', 'hidden_dim', 'rna_encoder', 'embedding_dim', 'rna_token_dim']:
        args[key] = config_from_model[key]  # config_from_model에서 가져온 값으로 업데이트

    # 'tanglerec' 또는 'intra'에 따라 'rna_reconstruction'과 'intra_modality_wsi' 설정
    args["rna_reconstruction"] = True if args["method"] == 'tanglerec' else False
    args["intra_modality_wsi"] = True if args["method"] == 'intra' else False 
    return args  # 수정된 args 반환

# 모델의 config.json 파일을 읽어오기 위한 함수
def read_config(path_to_config):
    with open(os.path.join(path_to_config, 'config.json')) as json_file:  # config.json 파일을 읽기
        data = json.load(json_file)  # JSON 파일을 파이썬 딕셔너리로 로드
        return data  # 딕셔너리 형태로 반환

# 모델의 상태를 복원하는 함수 (module 관련 처리 포함)
def restore_model(model, state_dict):
    sd = list(state_dict.keys())  # state_dict의 키 목록
    contains_module = any('module' in entry for entry in sd)  # state_dict에 'module'이 포함되어 있는지 확인
    
    if not contains_module:  # 만약 'module'이 없다면
        model.load_state_dict(state_dict, strict=True)  # 직접 모델에 state_dict를 로드
    else:  # 'module'이 포함된 경우
        new_state_dict = OrderedDict()  # 새롭게 OrderedDict 생성
        for k, v in state_dict.items():  # 기존 state_dict에서 key와 value를 꺼내서
            name = k[7:]  # 'module.' 부분을 제거
            new_state_dict[name] = v  # 새 state_dict에 저장
        model.load_state_dict(new_state_dict, strict=True)  # 새 state_dict를 모델에 로드

    return model  # 모델 반환

if __name__ == "__main__":  # 메인 실행 부분
    
    # 명령줄 인자 처리
    args = process_args()  # process_args()로 인자를 처리
    args = vars(args)  # 딕셔너리 형태로 변환
    assert args['pretrained'] is not None, "Must provide a path to a pretrained dir. Usage: --pretrained SOME_PATH/EXP_CODE/"  # pretrained 경로가 있어야 함

    # pretrained 모델의 config 파일 읽기
    config_from_model = read_config(args['pretrained'])
    args = set_args(args, config_from_model)  # 모델의 설정을 인자에 적용

    # 모델 설정
    print("* Setup model...")
    model = MMSSL(
        config=args,  # 인자로 받은 설정을 모델에 전달
        n_tokens_rna=int(args["rna_token_dim"]),  # RNA 토큰 수 설정
    ).to(DEVICE)  # 모델을 설정한 디바이스(GPU/CPU)로 이동

    # 모델의 총 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    print("* Total number of parameters = {}".format(total_params))  # 총 파라미터 수 출력
        
    # pretrained 모델 로드
    print("* Loading model from {}...".format(args['pretrained']))
    model = restore_model(model, torch.load(os.path.join(args["pretrained"], 'model_pretrained.pt')))  # model.pt에서 모델 상태 로드
    # model = restore_model(model, '/home/alleun1110/Digital_bio/TANGLE_renew/results/ACRIN_Breast_checkpoints_and_embeddings/tangle_ACRIN_Breast_lr0.0001_epochs100_bs64_tokensize150_temperature0.01_uni/model.pt')
    # state_dict = torch.load('/home/alleun1110/Digital_bio/TANGLE_renew/results/NSCLC_Radiogenomics_checkpoints_and_embeddings/tangle_NSCLC_Radiogenomics_lr0.0001_epochs100_bs64_tokensize112_temperature0.01_uni/model_ImageNet_pretrained.pt', map_location=DEVICE)
    # model = restore_model(model, state_dict)

    # 다운스트림 작업에 대한 WSI 임베딩을 추출하고 저장
    for key, val in DOWNSTREAM_TASKS_CONFIG.items():
        print('Extracting slide embeddings in :', key)  # 작업별로 임베딩 추출 시작

        # WSI 임베딩을 추출하여 지정된 위치에 저장
        _ = extract_wsi_embs_and_save(
            ssl_model=model,  # 모델을 전달
            features_path=val,  # 기능 경로
            save_fname=os.path.join(args["pretrained"], "{}_results_dict_pretrained.pkl".format(key)),  # 저장 파일명 지정
        )
