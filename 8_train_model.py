import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import pandas as pd
import numpy as np
import os
import cv2
import time
from tqdm import tqdm
import warnings

# 경고 메시지 무시 설정
warnings.filterwarnings("ignore")

# ----------------------------------------------------
# 0. 환경 설정 (Configuration)
# ----------------------------------------------------
# 데이터 경로 설정
DATA_DIR = "../data/Arm/masked_output"
CSV_FILE = os.path.join(DATA_DIR, "masked_joint_labels.csv")
# Min/Max 통계 파일 경로 설정 (새로 추가)
STATS_PATH = os.path.join(DATA_DIR, "joint_stats.csv")

# 학습 설정
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

# 모델 저장 경로
MODEL_SAVE_DIR = "models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
CHECKPOINT_INTERVAL = 5 # 5 epoch마다 모델 저장

# CUDA 설정
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"✅ 사용 장치: {DEVICE}")

# ----------------------------------------------------
# A. 데이터셋 통계 생성 함수
# ----------------------------------------------------
def create_joint_stats(csv_path, stats_path, label_cols=['J1', 'J2', 'J3', 'J4', 'J5', 'J6']):
    """
    관절 각도 데이터의 Min/Max 통계를 계산하고 CSV 파일로 저장합니다.
    """
    if os.path.exists(stats_path):
        print(f"✔️ 통계 파일이 이미 존재합니다: {stats_path}")
        return

    print(f"⏳ 통계 파일을 생성 중입니다...")
    try:
        df = pd.read_csv(csv_path)
        stats = df[label_cols].agg(['min', 'max']).T
        stats.columns = ['Min', 'Max'] # 열 이름을 'Min', 'Max'로 설정
        
        # 파일 저장
        stats.to_csv(stats_path)
        print(f"🎉 통계 파일 생성 완료: {stats_path}")
    except Exception as e:
        print(f"❌ 통계 파일 생성 중 오류 발생: {e}")
        exit()


# ----------------------------------------------------
# 1. 커스텀 데이터셋 클래스 (MyCobotDataset) - 정규화 로직 수정
# ----------------------------------------------------
class MyCobotDataset(Dataset):
    """
    이미지와 6개의 관절 각도(J1-J6)를 로드하고 정규화하는 PyTorch Dataset 클래스
    (데이터 통계 기반 정규화 적용)
    """
    def __init__(self, csv_file, root_dir, stats_path, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.label_cols = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
        
        # Min/Max 통계 로드
        self.stats = pd.read_csv(stats_path, index_col=0)
        self.joint_min = torch.from_numpy(self.stats['Min'].values.astype(np.float32))
        self.joint_max = torch.from_numpy(self.stats['Max'].values.astype(np.float32))
        self.joint_range = self.joint_max - self.joint_min # range 미리 계산

    def __len__(self):
        return len(self.data_frame)

    def normalize_joints(self, joint_tensor):
        """Min-Max Scaling to [-1, 1]"""
        # 공식: y = (X_raw - Min) / (Max - Min) * 2 - 1
        # tensor.numpy()를 사용하지 않고 바로 tensor 연산 수행
        return 2.0 * ((joint_tensor - self.joint_min) / self.joint_range) - 1.0

    @staticmethod
    def denormalize_joints(joint_tensor_norm, joint_min, joint_max):
        """Denormalization from [-1, 1] to original range"""
        # 공식: X_raw = (X_norm + 1) / 2 * (Max - Min) + Min
        joint_range = joint_max - joint_min
        return (joint_tensor_norm + 1.0) / 2.0 * joint_range + joint_min
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. 이미지 로드
        # 첫 번째 열이 Image_File이라고 가정
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0]) 
        image = cv2.imread(img_name)
        
        if image is None:
            # 파일을 찾지 못하면 텐서와 플래그 반환
            print(f"⚠️ 이미지 파일을 찾을 수 없습니다: {img_name}. 0 텐서 반환.")
            # ResNet의 입력 크기에 맞게 빈 텐서 반환
            return torch.zeros(3, 224, 224), torch.zeros(len(self.label_cols)), True

        # OpenCV는 BGR이므로 RGB로 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Joint Angle 로드 및 정규화
        angles = self.data_frame.iloc[idx][self.label_cols].values.astype(np.float32)
        angles_tensor = torch.from_numpy(angles)
        
        # 데이터셋 통계를 사용한 정규화 적용
        normalized_angles = self.normalize_joints(angles_tensor)
        
        # 3. 이미지 변환 (Transformation)
        if self.transform:
            image = self.transform(image)
        
        # 세 번째 반환 값(True/False)은 유효하지 않은 데이터 처리를 위한 플래그 (학습 코드의 단순화를 위해 False 고정)
        return image, normalized_angles, False


# ----------------------------------------------------
# 2. 이미지 전처리 및 데이터 로더 (Preprocessing & DataLoader)
# ----------------------------------------------------
# ResNet 모델에 적합한 표준 전처리 (224x224 리사이즈 및 ImageNet 표준 정규화)
transform = transforms.Compose([
    transforms.ToPILImage(), # numpy array를 PIL Image로 변환
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # ImageNet 평균 및 표준편차 사용
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ----------------------------------------------------
# 3. 모델 정의 (ResNet-50 Regression Model)
# ----------------------------------------------------
# 사전 학습된 ResNet-50 로드
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# 최종 Fully Connected Layer를 회귀 문제에 맞게 수정
# 입력 특징 수 (모델에 따라 다름, ResNet-50은 2048)
num_ftrs = model.fc.in_features 

# 출력: Joint 1 ~ Joint 6 (총 6개)
model.fc = nn.Linear(num_ftrs, 6) 

# 모델을 GPU로 이동
model = model.to(DEVICE)


# ----------------------------------------------------
# 4. 학습 설정 및 함수 (Training Setup)
# ----------------------------------------------------
# 회귀 문제이므로 평균 제곱 오차(MSE)를 손실 함수로 사용
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_model():
    # ----------------------------------------------------
    # 데이터셋 통계 로드 및 DataLoader 생성
    # ----------------------------------------------------
    # A-1. Min/Max 통계 파일 확인 및 생성
    create_joint_stats(CSV_FILE, STATS_PATH)
    
    # A-2. Dataset 로드 시 통계 파일 경로 전달
    full_dataset = MyCobotDataset(csv_file=CSV_FILE, root_dir=DATA_DIR, stats_path=STATS_PATH, transform=transform)
    
    # 정규화/역정규화에 사용할 Min/Max 값 저장 (GPU로 전송)
    # 학습 루프 외부에서 한 번만 설정
    J_MIN_TENSOR = full_dataset.joint_min.to(DEVICE)
    J_MAX_TENSOR = full_dataset.joint_max.to(DEVICE)
    
    # A-3. 유효하지 않은 데이터 필터링 (불필요한 에러 방지)
    valid_indices = [i for i in range(len(full_dataset)) if not full_dataset[i][2]]
    valid_dataset = torch.utils.data.Subset(full_dataset, valid_indices)

    if len(valid_dataset) == 0:
        print("Error: 유효한 데이터 샘플이 없습니다. 이미지 경로, CSV, STATS 파일을 확인해 주세요.")
        return

    dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print(f"Total Valid Samples: {len(valid_dataset)}")
    
    # ----------------------------------------------------
    # 학습 루프 시작
    # ----------------------------------------------------
    best_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        # 훈련 단계
        model.train()
        running_loss_norm = 0.0
        
        # tqdm을 사용하여 진행률 표시
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", unit="batch")
        
        for inputs, labels_norm, _ in pbar:
            inputs = inputs.to(DEVICE)
            # labels_norm은 이미 Dataset에서 정규화되어 [-1, 1] 범위
            labels_norm = labels_norm.to(DEVICE).float()
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + Backward + Optimize
            outputs_norm = model(inputs)
            
            # Loss 계산은 정규화된 값 [-1, 1]으로 수행
            loss_norm = criterion(outputs_norm, labels_norm) 
            
            loss_norm.backward()
            optimizer.step()

            running_loss_norm += loss_norm.item() * inputs.size(0)
            pbar.set_postfix({'Loss (Norm)': loss_norm.item()})

        epoch_loss_norm = running_loss_norm / len(valid_dataset)
        
        # 역정규화된 Loss 계산 (실제 각도 오차 RMSE 추정)
        # 역정규화된 출력 및 라벨을 계산
        outputs_denorm = full_dataset.denormalize_joints(outputs_norm, J_MIN_TENSOR, J_MAX_TENSOR)
        labels_denorm = full_dataset.denormalize_joints(labels_norm, J_MIN_TENSOR, J_MAX_TENSOR)
        
        # 실제 각도 단위로 MSE Loss 계산
        loss_denorm = criterion(outputs_denorm, labels_denorm).item() 
        avg_angle_error = np.sqrt(loss_denorm / 6) # 6개 관절 평균 각도 오차 (RMSE)
        
        print(f"\n[Epoch {epoch}/{EPOCHS}] Average Loss (Norm): {epoch_loss_norm:.6f} | Avg. Joint Error (RMSE): {avg_angle_error:.2f} deg")
        
        # 5. 모델 저장 로직
        
        # 5-1. Best Model 저장 (정규화된 Loss 기준)
        if epoch_loss_norm < best_loss:
            best_loss = epoch_loss_norm
            best_model_path = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"✨ Best Model 저장됨! Loss (Norm): {best_loss:.6f}")

        # 5-2. Checkpoint (5 Epoch)마다 저장
        if epoch % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"checkpoint_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Checkpoint 저장됨: {checkpoint_path}")

    print("\n\n🎉 학습 완료!")
    print(f"최종 Best Loss (Norm): {best_loss:.6f}")


if __name__ == "__main__":
    train_model()