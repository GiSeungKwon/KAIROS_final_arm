import torch
import torch.nn as nn
from torchvision import models, transforms
import pandas as pd
import numpy as np
import os
import cv2
import time
import sys
from pymycobot.mycobot import MyCobot

# ----------------------------------------------------
# 0. 환경 설정 및 상수 (Configuration & Constants)
# ----------------------------------------------------
# A. 파일 경로 및 설정
DATA_DIR = "../data/Arm/masked_output" # stats.csv 경로를 포함
STATS_PATH = os.path.join(DATA_DIR, "joint_stats.csv")
MODEL_SAVE_PATH = "models/checkpoint_epoch_15.pth" 
CAMERA_INDEX = 0 # 로봇 팔에 연결된 카메라의 인덱스

# B. ROI 및 HSV 설정 (학습 시와 동일하게)
# 학습 전처리 코드에 제공된 값으로 수정 (단, 원래 코드에는 V_LOW=0, V_HIGH=255)
# **주의: 학습 시 사용된 정확한 HSV 범위를 적용해야 합니다.**
# (제공된 전처리 코드는 V_LOW=0, V_HIGH=255를 사용했으므로 이 값으로 설정합니다.)
H_LOW, S_LOW, V_LOW = 0, 0, 0
H_HIGH, S_HIGH, V_HIGH = 179, 255, 240
HSV_LOW = np.array([H_LOW, S_LOW, V_LOW])
HSV_HIGH = np.array([H_HIGH, S_HIGH, V_HIGH])

# 요청하신 ROI 설정 적용
ROI_START = (30, 30) # (x_min, y_min)
ROI_END = (430, 430) # (x_max, y_max) 
TARGET_IMAGE_SIZE = (224, 224) # ResNet 입력 크기

# C. MyCobot 제어 설정 (사용자 코드 참조)
PORT = "COM3"
BAUD = 115200
MOVEMENT_SPEED = 30
INTERMEDIATE_POSE_ANGLES = [-17.2, 30.49, 4.48, 53.08, -90.87, -85.86]
SEQUENTIAL_MOVE_DELAY = 1 
GRIPPER_OPEN_VALUE = 55 
GRIPPER_SPEED = 20

# D. PyTorch 설정
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_JOINTS = 6


# ----------------------------------------------------
# 1. 모델 및 유틸리티 함수 정의
# ----------------------------------------------------

# A. ResNet 모델 정의 (학습 코드와 동일)
class JointPredictor(nn.Module):
    def __init__(self, num_joints=6):
        super(JointPredictor, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_joints) 

    def forward(self, x):
        return self.resnet(x)

# B. 정규화/역정규화 유틸리티 (학습 코드와 동일)
def denormalize_joints(joint_tensor_norm, joint_min, joint_max):
    """Denormalization from [-1, 1] to original range"""
    joint_range = joint_max - joint_min
    return (joint_tensor_norm + 1.0) / 2.0 * joint_range + joint_min

# C. 학습 시 사용된 이미지 전처리 파이프라인
transform = transforms.Compose([
    transforms.ToPILImage(), # numpy array를 PIL Image로 변환
    transforms.Resize(TARGET_IMAGE_SIZE),
    transforms.ToTensor(),
    # ImageNet 평균 및 표준편차 사용
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# D. ROI 및 HSV 전처리 함수 (사용자 코드 기반)
def apply_roi_and_hsv_masking(image, roi_start, roi_end, hsv_low, hsv_high):
    """
    원본 이미지에 ROI 마스크를 적용하고, ROI 영역 내에서 HSV 마스킹을 적용하여 
    최종 바이너리 마스크 이미지를 생성합니다.
    """
    # 1. ROI 적용: ROI 외부를 검은색으로 만들 마스크 생성
    # 학습된 마스크 이미지가 '배경 검은색, 물체 흰색' 바이너리였으므로 이를 따릅니다.
    x_min, y_min = roi_start
    x_max, y_max = roi_end
    
    masked_image_roi = image.copy()
    
    # 2. HSV 변환 및 마스킹
    # ROI 외부를 검은색으로 설정 (검은색 픽셀은 HSV 변환 후에도 마스크에 포함되지 않음)
    # ROI 외부를 먼저 검은색(0, 0, 0)으로 만듭니다.
    masked_image_roi[:y_min, :] = 0 # 상단
    masked_image_roi[y_max:, :] = 0 # 하단
    masked_image_roi[:, :x_min] = 0 # 왼쪽
    masked_image_roi[:, x_max:] = 0 # 오른쪽
    
    hsv = cv2.cvtColor(masked_image_roi, cv2.COLOR_BGR2HSV)
    
    # HSV 범위에 따라 마스크 생성
    hsv_mask = cv2.inRange(hsv, hsv_low, hsv_high)
    
    # 3. 최종 바이너리 이미지 생성
    final_binary_image = np.zeros_like(image)
    
    # 마스크 영역 (물체)만 흰색 (255, 255, 255)으로 채움
    final_binary_image[hsv_mask > 0] = [255, 255, 255]

    return final_binary_image

# ----------------------------------------------------
# 2. 메인 추론 및 제어 루프
# ----------------------------------------------------
def main():
    # 1. 모델 로드 및 통계 로드
    try:
        model = JointPredictor(num_joints=NUM_JOINTS).to(DEVICE)
        state_dict = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
        
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict['resnet.' + k] = v 
        
        model.load_state_dict(new_state_dict) 
        model.eval()
        
        print(f"✅ 모델 로드 성공: {MODEL_SAVE_PATH}")
    except FileNotFoundError:
        print(f"❌ 오류: 모델 파일 또는 통계 파일을 찾을 수 없습니다. 학습을 먼저 수행하세요.")
        return
    
    try:
        stats = pd.read_csv(STATS_PATH, index_col=0)
        J_MIN_TENSOR = torch.from_numpy(stats['Min'].values.astype(np.float32)).to(DEVICE)
        J_MAX_TENSOR = torch.from_numpy(stats['Max'].values.astype(np.float32)).to(DEVICE)
        print(f"✅ 통계 로드 성공: {STATS_PATH}")
    except Exception as e:
        print(f"❌ 오류: 통계 파일 로드 실패. joint_stats.csv를 확인하세요. 오류: {e}")
        return

    # 2. MyCobot 연결
    try:
        mc = MyCobot(PORT, BAUD)
        mc.power_on() 
        mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED) # 그리퍼 열기
        print(f"🤖 MyCobot 연결 성공: {PORT}. 준비 완료.")
    except Exception as e:
        print(f"❌ MyCobot 연결 실패 ({PORT}): {e}")
        sys.exit(1)

    # 3. 카메라 연결
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"❌ 카메라 인덱스 {CAMERA_INDEX}를 열 수 없습니다.")
        mc.close()
        sys.exit(1)
    
    # 전처리 이미지 표시 토글 상태
    show_processed_image = False 
    processed_window_open = False

    print("\n--- 🧠 MyCobot 실시간 추론 도구 ---")
    print("   [i] : 현재 프레임으로 Joint Angle 추론 및 로봇 이동")
    print("   [p] : **모델 입력 전** 전처리된 이미지 토글") # 요청하신 기능
    print("   [q] : 프로그램 종료")
    print("---------------------------------------")
    
    with torch.no_grad(): # 추론 시에는 gradient 계산 불필요
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            # 1. ROI 영역 시각화
            cv2.rectangle(frame, ROI_START, ROI_END, (0, 0, 255), 2)
            
            # 2. 전처리 이미지 생성
            processed_image = apply_roi_and_hsv_masking(frame, ROI_START, ROI_END, HSV_LOW, HSV_HIGH)
            
            # 3. 'p' 키를 눌렀을 때 전처리 이미지 표시
            if show_processed_image:
                cv2.putText(processed_image, "Processed (p:toggle)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Processed Image (Model Input)', processed_image)
                processed_window_open = True # 윈도우가 열림
                
            # 📌 수정된 로직: 윈도우가 열려있는데 (processed_window_open=True) 표시가 꺼졌을 때 (show_processed_image=False) 닫는다.
            elif processed_window_open: 
                cv2.destroyWindow('Processed Image (Model Input)')
                processed_window_open = False # 윈도우 닫힘 상태로 변경

            # 4. 실시간 카메라 프레임 표시
            cv2.imshow('MyCobot Live Camera & Inference Tool', frame)

            # -----------------------------------------
            # 키 입력 처리
            # -----------------------------------------
            key = cv2.waitKey(1) & 0xFF

            # [q]: 프로그램 종료
            if key == ord('q'):
                print("\n👋 프로그램을 종료합니다.")
                break
            
            # [p]: 전처리 이미지 표시 토글 (요청 사항)
            elif key == ord('p'):
                show_processed_image = not show_processed_image
                print(f"\n💡 전처리 이미지 표시: {'ON' if show_processed_image else 'OFF'}")

            # [i]: 추론 및 로봇 이동
            elif key == ord('i'):
                print("\n🧠 Joint Angle 추론 시작...")
                
                # a) 전처리 이미지 -> PyTorch 텐서 변환
                # OpenCV (BGR/numpy) -> RGB/numpy -> PIL -> Tensor/Normalize
                try:
                    input_tensor = transform(processed_image) 
                    input_tensor = input_tensor.unsqueeze(0).to(DEVICE) # Batch 차원 추가 및 GPU 전송
                except Exception as e:
                    print(f"❌ 이미지 변환 중 오류 발생: {e}")
                    continue

                # b) 모델 추론 (정규화된 각도 출력)
                outputs_norm = model(input_tensor)
                
                # c) 역정규화 (실제 각도 복원)
                outputs_denorm = denormalize_joints(outputs_norm, J_MIN_TENSOR, J_MAX_TENSOR)
                
                # d) 결과 출력 및 로봇 제어
                predicted_angles = outputs_denorm.cpu().squeeze(0).numpy().tolist()
                
                # 소수점 한 자리로 반올림하여 제어 정확도 유지 및 출력 가독성 개선
                predicted_angles = [round(a, 1) for a in predicted_angles] 

                print(f"✅ 추론된 Joint Angles: {predicted_angles}")
                
                # 로봇 팔 이동 (안전을 위해 경유지를 경유)
                print(f"⚙️ 로봇 이동 시작 (경유지 경유 후 최종지 {predicted_angles}로 이동)")
                
                # 1. 그리퍼 열기
                mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
                
                # 2. 중간 경유 자세로 이동 (안전성 확보)
                mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
                time.sleep(SEQUENTIAL_MOVE_DELAY)
                
                # 3. 최종 추론된 자세로 이동
                mc.send_angles(predicted_angles, MOVEMENT_SPEED)
                time.sleep(SEQUENTIAL_MOVE_DELAY * 2) # 최종 이동 후 충분히 기다림
                print("✅ 로봇 이동 완료.")
            
            # [0, 1, 2, g, h] 키는 mycobot_labeling.py의 기능과 동일하게 작동
            # [0]: 모든 Joint 각도를 0으로 이동
            elif key == ord('0'):
                mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
                time.sleep(SEQUENTIAL_MOVE_DELAY)
                mc.send_angles([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], MOVEMENT_SPEED)
                print("✅ ZERO_POSE 이동 완료.")
                
            # [1]: CONVEYOR_CAPTURE_POSE 이동
            elif key == ord('1'):
                mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
                mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
                time.sleep(SEQUENTIAL_MOVE_DELAY)
                mc.send_angles([0, 0, 90, 0, -90, -90], MOVEMENT_SPEED)
                print("✅ CONVEYOR_CAPTURE_POSE 이동 완료.")
                
            # [2]: ROBOTARM_CAPTURE_POSE 이동
            elif key == ord('2'):
                mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
                mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
                time.sleep(SEQUENTIAL_MOVE_DELAY)
                mc.send_angles([0, 0, 90, 0, -90, 90], MOVEMENT_SPEED)
                print("✅ ROBOTARM_CAPTURE_POSE 이동 완료.")

            # [g]: 그리퍼 닫기
            elif key == ord('g'):
                mc.set_gripper_value(25, GRIPPER_SPEED) # GRIPPER_CLOSED_VALUE는 25 (참조 코드 기준)
                time.sleep(1) 
                print("✅ 그리퍼 닫힘 완료.")
                
            # [h]: 그리퍼 열기
            elif key == ord('h'):
                mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
                time.sleep(1) 
                print("✅ 그리퍼 열림 완료.")

    # 종료 정리 작업
    cap.release()
    cv2.destroyAllWindows()
    try:
        mc.close()
    except Exception:
        pass
        
if __name__ == "__main__":
    main()