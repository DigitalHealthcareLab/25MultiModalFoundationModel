import pandas as pd
import torch
import os

# 파일 경로 설정
file_path = "/home/alleun1110/Digital_bio/TANGLE_renew/dataset_csv/Colorectal_cancer/CRC_bilobar_clinical_representation_learning.csv"
pt_output_dir = "/home/alleun1110/Digital_bio/TANGLE_renew/data/Colorectal/clinical_data"

# 데이터 로드
df = pd.read_csv(file_path)

# 출력 디렉토리 생성
os.makedirs(pt_output_dir, exist_ok=True)

# 각 slide_id에 대해 .pt 파일 생성 및 저장
for idx, row in df.iterrows():
    slide_id = str(row['case_id']).split('.')[0]  # slide_id를 문자열로 변환 후 ".0" 제거
    row_data = row.drop('case_id')  # slide_id 컬럼 제외

    # NaN 값 처리 및 숫자형 변환
    row_data = pd.to_numeric(row_data, errors='coerce').fillna(0)  # NaN은 0으로 대체

    # 텐서 변환
    tensor_data = torch.tensor(row_data.values, dtype=torch.float32)
    pt_file_path = os.path.join(pt_output_dir, f"{slide_id}.pt")  # slide_id를 파일명으로 저장
    torch.save(tensor_data, pt_file_path)

    print(f"Saved {slide_id} as .pt file at {pt_file_path}")
