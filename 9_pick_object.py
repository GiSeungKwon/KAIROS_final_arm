import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import os
import cv2
import time
from typing import List, Tuple

# ----------------------------------------------------
# 0. 환경 및 상수 설정 (Configuration)
#    (학습 코드와 동일하게 유지)
# ----------------------------------------------------
# 모델 저장 경로 (best_model.pth 파일이 이 경로에 있어야 함)
MODEL_SAVE_DIR = "models"
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.pth") # 최적 모델 사용 가정

# myCobot 320 관절 제한 (Normalization을 위한 Min/Max)
JOINT_MIN = np.array([-165.0, -165.0, -165.0, -165.0, -165.0, -175.0], dtype=np.float32)
JOINT_MAX = np.array([ 165.0,  165.0,  165.0,  165.0,  165.0,  175.0], dtype=np.float32)
JOINT_RANGE = JOINT_MAX - JOINT_MIN
JOINT_MIN_TENSOR = torch.from_numpy(JOINT_MIN).to(torch.device("cpu"))
JOINT_RANGE_TENSOR = torch.from_numpy(JOINT_RANGE).to(torch.device("cpu"))

# CUDA 설정
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"✅ 사용 장치: {DEVICE}")

# ----------------------------------------------------
# 1. 모델 정의 및 로드 (Model Definition and Loading)
#    (학습 코드의 3번 섹션과 동일해야 함)
# ----------------------------------------------------

def load_inference_model(model_path: str, device: torch.device) -> nn.Module:
    """사전 학습된 ResNet-50 모델 구조를 정의하고 학습된 가중치를 로드합니다."""
    
    # 1. 모델 구조 정의 (ResNet-50)
    model = models.resnet50(weights=None) # 가중치는 나중에 로드하므로 None
    
    # 최종 Fully Connected Layer 수정
    num_ftrs = model.fc.in_features 
    model.fc = nn.Linear(num_ftrs, 6) # 출력: Joint 1 ~ Joint 6
    
    # 2. 모델 가중치 로드
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {model_path}")
        
    try:
        # 가중치만 로드 (state_dict)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ 모델 가중치 로드 성공: {model_path}")
    except Exception as e:
        print(f"❌ 모델 로드 중 오류 발생: {e}")
        # 오류가 발생하면 초기화된 모델 반환
        return model.to(device)

    # 3. 추론 모드로 설정
    model.eval() 
    
    # 4. 장치로 이동
    return model.to(device)


# ----------------------------------------------------
# 2. 전처리 함수 (Preprocessing Function)
#    (학습 코드의 2번 섹션 transforms.Compose와 동일해야 함)
# ----------------------------------------------------

def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
    """
    OpenCV BGR 이미지를 학습 시와 동일하게 전처리합니다.
    (BGR -> RGB, Resize 224x224, ToTensor, Normalize)
    """
    # 1. BGR -> RGB 변환
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # 2. 학습 시 사용한 transforms.Compose 정의
    # ToPILImage는 numpy array/tensor를 입력받을 수 있습니다.
    inference_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # ImageNet 평균 및 표준편차 사용
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. 변환 적용 및 배치 차원 추가
    # output: (C, H, W) 텐서
    tensor = inference_transform(image_rgb)
    
    # output: (1, C, H, W) - 배치 차원 추가 (모델 입력 형식)
    return tensor.unsqueeze(0)


# ----------------------------------------------------
# 3. 역정규화 함수 (Denormalization Function)
# ----------------------------------------------------

def denormalize_angles(normalized_angles: torch.Tensor) -> np.ndarray:
    """
    예측된 정규화된 관절 각도 [-1, 1]를 실제 각도 [Min, Max]로 역변환합니다.
    학습 시 정규화 공식: y = (Angle - Min) / (Max - Min) * 2 - 1
    역변환 공식: Angle = ((y + 1) / 2) * (Max - Min) + Min
    """
    # 1. 텐서를 numpy로 변환 (GPU에 있다면 CPU로 이동 후)
    angles_numpy = normalized_angles.detach().cpu().numpy().flatten()
    
    # 2. 역정규화 계산
    # y = angles_numpy
    # Angle = ((y + 1) / 2) * JOINT_RANGE + JOINT_MIN
    
    # NumPy를 사용하여 계산
    denorm_angles = ((angles_numpy + 1.0) / 2.0) * JOINT_RANGE + JOINT_MIN
    
    # myCobot 관절 제한 범위 내로 클리핑 (선택 사항이나 안전을 위해 권장)
    denorm_angles = np.clip(denorm_angles, JOINT_MIN, JOINT_MAX)
    
    return denorm_angles # [J1, J2, J3, J4, J5, J6] 실수 배열


# ----------------------------------------------------
# 4. 추론 실행 함수 (Main Inference Function)
# ----------------------------------------------------

def run_inference(image_path: str, model: nn.Module) -> Tuple[np.ndarray, float]:
    """
    단일 이미지에 대해 로봇 팔 관절 각도를 추론하고 역정규화합니다.
    """
    
    # 1. 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

    print(f"➡️ 이미지 로드 성공: {image_path}")
    
    # 2. 이미지 전처리
    input_tensor = preprocess_image(image)
    
    # 3. 모델 추론
    with torch.no_grad(): # 메모리 사용 절감 및 계산 속도 향상을 위해 필수
        input_tensor = input_tensor.to(DEVICE)
        start_time = time.time()
        
        # 모델 포워드 패스 -> 정규화된 각도 [-1, 1] 예측
        normalized_output = model(input_tensor)
        
        end_time = time.time()
        inference_time = end_time - start_time

    # 4. 역정규화 (실제 각도 [Degrees]로 변환)
    predicted_angles = denormalize_angles(normalized_output)

    print(f"✅ 추론 완료 (시간: {inference_time:.4f}s)")
    print(f"   - 예측된 정규화된 각도 (J1~J6): {normalized_output.cpu().numpy().flatten()}")
    print(f"   - **최종 예측 관절 각도 (Degrees)**: {predicted_angles}")
    
    return predicted_angles, inference_time


# ----------------------------------------------------
# 5. 실행 예제 (Example Execution)
# ----------------------------------------------------

if __name__ == "__main__":
    
    # 실제 컨베이어 벨트에서 물체를 인식한 카메라 이미지를 가정
    # TODO: 테스트용 이미지 파일 경로로 변경하세요.
    # 예: CONVEYOR_IMAGE_PATH = "./test_images/object_on_belt.jpg"
    CONVEYOR_IMAGE_PATH = "../data/Arm/masked_output/00001_mask.jpg" 
    
    try:
        # 1. 모델 로드
        inference_model = load_inference_model(MODEL_PATH, DEVICE)
        
        # 2. 추론 실행
        predicted_joints, latency = run_inference(CONVEYOR_IMAGE_PATH, inference_model)
        
        print("\n--- 로봇 제어 시스템 전송 값 ---")
        print(f"Joint 1: {predicted_joints[0]:.2f}°")
        print(f"Joint 2: {predicted_joints[1]:.2f}°")
        print(f"Joint 3: {predicted_joints[2]:.2f}°")
        print(f"Joint 4: {predicted_joints[3]:.2f}°")
        print(f"Joint 5: {predicted_joints[4]:.2f}°")
        print(f"Joint 6: {predicted_joints[5]:.2f}°")
        print("-------------------------------")

        # 실제 로봇 팔 제어 코드에서는 이 predicted_joints 배열을
        # 로봇 제어 인터페이스(예: myCobot API)로 전송하여 로봇을 해당 자세로 이동시킵니다.
        # 예: mycobot.send_angles(list(predicted_joints), speed)
        
    except FileNotFoundError as e:
        print(f"\n❌ 오류 발생: {e}")
        print("💡 학습 코드를 먼저 실행하여 'models/best_model.pth' 파일을 생성하거나, 'CONVEYOR_IMAGE_PATH' 경로를 확인해주세요.")