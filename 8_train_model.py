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

# ----------------------------------------------------
# 0. 환경 설정 (Configuration)
# ----------------------------------------------------
# 데이터 경로 설정
# DATA_DIR = "./mycobot_labeling_data"
DATA_DIR = "../data/Arm/masked_output"
CSV_FILE = os.path.join(DATA_DIR, "joint_labels.csv")

# 학습 설정
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

# 모델 저장 경로
MODEL_SAVE_DIR = "models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
CHECKPOINT_INTERVAL = 5 # 5 epoch마다 모델 저장

# CUDA 설정
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"✅ 사용 장치: {DEVICE}")

# myCobot 320 관절 제한 (Normalization을 위한 Min/Max)
JOINT_MIN = np.array([-165.0, -165.0, -165.0, -165.0, -165.0, -175.0])
JOINT_MAX = np.array([ 165.0,  165.0,  165.0,  165.0,  165.0,  175.0])
JOINT_RANGE = JOINT_MAX - JOINT_MIN


# ----------------------------------------------------
# 1. 커스텀 데이터셋 클래스 (MyCobotDataset)
# ----------------------------------------------------
class MyCobotDataset(Dataset):
    """
    이미지와 6개의 관절 각도(J1-J6)를 로드하고 정규화하는 PyTorch Dataset 클래스
    """
    def __init__(self, csv_file, root_dir, transform=None):
        """
        :param csv_file: 이미지 파일명과 Joint Angle이 포함된 CSV 파일 경로
        :param root_dir: 이미지 파일이 저장된 폴더 경로
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.label_cols = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. 이미지 로드
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0]) # 첫 번째 열이 Image_File이라고 가정
        image = cv2.imread(img_name)
        if image is None:
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {img_name}")
            
        # OpenCV는 BGR이므로 RGB로 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Joint Angle 로드 및 정규화
        angles = self.data_frame.iloc[idx][self.label_cols].values.astype(np.float32)
        
        # 정규화 (Normalization): [Min, Max] -> [-1, 1]
        # 공식: y = (Angle - Min) / (Max - Min) * 2 - 1
        normalized_angles = (angles - JOINT_MIN) / JOINT_RANGE * 2 - 1
        
        normalized_angles = torch.from_numpy(normalized_angles)

        # 3. 이미지 변환 (Transformation)
        if self.transform:
            image = self.transform(image)

        return image, normalized_angles


# ----------------------------------------------------
# 2. 이미지 전처리 및 데이터 로더 (Preprocessing & DataLoader)
# ----------------------------------------------------
# ResNet 모델에 적합한 표준 전처리 (224x224 리사이즈 및 ImageNet 표준 정규화)
# 참고: 이미 크롭된 이미지를 사용하므로, 크롭은 생략하고 리사이즈만 적용합니다.
transform = transforms.Compose([
    transforms.ToPILImage(), # numpy array를 PIL Image로 변환
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # ImageNet 평균 및 표준편차 사용
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = MyCobotDataset(csv_file=CSV_FILE, root_dir=DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

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
    best_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        # 훈련 단계
        model.train()
        running_loss = 0.0
        
        # tqdm을 사용하여 진행률 표시
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", unit="batch")
        
        for inputs, labels in pbar:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).float()
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + Backward + Optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            pbar.set_postfix({'Loss': loss.item()})

        epoch_loss = running_loss / len(dataloader.dataset)
        
        print(f"\n[Epoch {epoch}/{EPOCHS}] Average Loss: {epoch_loss:.6f}")
        
        # 5. 모델 저장 로직
        
        # 5-1. Best Model 저장
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_path = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"✨ Best Model 저장됨! Loss: {best_loss:.6f}")

        # 5-2. Checkpoint (5 Epoch)마다 저장
        if epoch % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"checkpoint_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Checkpoint 저장됨: {checkpoint_path}")

    print("\n\n🎉 학습 완료!")
    print(f"최종 Best Loss: {best_loss:.6f}")


if __name__ == "__main__":
    train_model()