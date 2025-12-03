import torch
from torchvision import models, transforms
import cv2
import numpy as np
import os
import sys
import time
import torch.nn as nn
# MyCobot 제어용 라이브러리
from pymycobot.mycobot import MyCobot

# ----------------------------------------------------
# 1. 설정 변수 (Configuration)
# ----------------------------------------------------
# MyCobot 설정
PORT = "COM3"
BAUD = 115200
CAMERA_INDEX = 0

MOVEMENT_SPEED = 30 
GRIPPER_SPEED = 50
GRIPPER_CLOSED_VALUE = 25 # 닫힘 값 (사용자 지정)
GRIPPER_OPEN_VALUE = 55   # 열림 값 (사용자 지정)
MOVE_DELAY = 3 # 로봇 이동 후 대기 시간 (초)

# 추론 설정
MODEL_PATH = "models/checkpoint_epoch_5.pth"
# MODEL_PATH = "models/best_model.pth"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 🌟 학습에 사용된 ROI 설정 (시작: (0, 100), 끝: (440, 380))
X_START, Y_START = 0, 140
X_END, Y_END = 440, 420

# MyCobot 320 관절 제한 (Denormalization에 필요)
JOINT_MIN = np.array([-165.0, -165.0, -165.0, -165.0, -165.0, -175.0], dtype=np.float32)
JOINT_MAX = np.array([ 165.0,  165.0,  165.0,  165.0,  165.0,  175.0], dtype=np.float32)
JOINT_RANGE = JOINT_MAX - JOINT_MIN

GRIPPER_ACTION_DELAY = 1

# 🌟 로봇 프리셋 자세 (사용자 지정)
ZERO_POSE_ANGLES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
CONVEYOR_CAPTURE_POSE = [0, 0, 90, 0, -90, -90]
ROBOTARM_CAPTURE_POSE = [0, 0, 90, 0, -90, 90]
INTERMEDIATE_POSE_ANGLES = [-17.2, 30.49, 4.48, 53.08, -90.87, -85.86]

# ----------------------------------------------------
# 2. Denormalization 함수
# ----------------------------------------------------
def denormalize_angles(normalized_angles_tensor):
    """정규화된 [-1, 1] 값을 실제 Joint Angle [Min, Max] 범위로 복원"""
    normalized_angles = normalized_angles_tensor.cpu().numpy().flatten()
    angles = ((normalized_angles + 1) / 2) * JOINT_RANGE + JOINT_MIN
    return angles.tolist()

# ----------------------------------------------------
# 3. 모델 및 전처리 설정
# ----------------------------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_inference_model():
    """학습된 ResNet-50 모델 가중치 불러오기"""
    model = models.resnet50(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 6)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval() # 추론 모드 설정
        print(f"✅ 모델 로드 성공: {MODEL_PATH}")
        return model
    except FileNotFoundError:
        print(f"❌ 오류: 모델 파일 ({MODEL_PATH})을 찾을 수 없습니다. 학습을 먼저 진행하세요.")
        sys.exit(1)


# ----------------------------------------------------
# 4. 메인 제어 루프 (Main Control Loop)
# ----------------------------------------------------
def main():
    # 1. 모델 로드
    model = load_inference_model()

    # 2. MyCobot 연결
    try:
        mc = MyCobot(PORT, BAUD)
        mc.power_on() 
        mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
        time.sleep(1)
        print(f"\n🤖 MyCobot 연결 성공: {PORT}. 그리퍼 열림.")
    except Exception as e:
        print(f"\n❌ MyCobot 연결 실패 ({PORT}): {e}")
        sys.exit(1)

    # 3. 카메라 연결
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        mc.close()
        sys.exit(1)
        
    predicted_angles = None
    
    print("\n--- 🕹️ 실시간 Pick & Place 제어 ---")
    print("  [p] 키: 경유 후 모델 예측 위치로 이동 및 Pick 작업 수행")
    print("  [0] 키: ZERO_POSE_ANGLES로 이동 (홈 자세)")
    print("  [1] 키: CONVEYOR_CAPTURE_POSE로 이동 (컨베이어 관찰 자세)")
    print("  [2] 키: ROBOTARM_CAPTURE_POSE로 이동 (로봇팔 관찰 자세)")
    print("  [3] 키: INTERMEDIATE_POSE_ANGLES로 이동 (안전 경유 자세)")
    print(f"  [g] : 그리퍼 닫기 (위치: {GRIPPER_CLOSED_VALUE})") 
    print(f"  [h] : 그리퍼 열기 (위치: {GRIPPER_OPEN_VALUE})") 
    print("  [q] 키: 프로그램 종료")
    print("---------------------------------------")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        # 4. ROI 크롭 적용
        cropped_frame = frame[Y_START:Y_END, X_START:X_END]
        
        # 5. 추론 및 예측
        with torch.no_grad():
            input_tensor = transform(cropped_frame)
            input_batch = input_tensor.unsqueeze(0).to(DEVICE)
            output = model(input_batch)
            predicted_angles = denormalize_angles(output)

        # 6. 디스플레이 오버레이
        display_frame = frame.copy()
        
        # ROI 영역 표시
        cv2.rectangle(display_frame, (X_START, Y_START), (X_END, Y_END), (0, 0, 255), 2)
        
        # 예측된 각도 정보 표시
        angle_text = f"Pred: J1:{predicted_angles[0]:.1f} J2:{predicted_angles[1]:.1f} J3:{predicted_angles[2]:.1f} J4:{predicted_angles[3]:.1f} J5:{predicted_angles[4]:.1f} J6:{predicted_angles[5]:.1f}"
        cv2.putText(display_frame, angle_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 키 가이드 표시
        cv2.putText(display_frame, "Press 'p' to Pick | 0-3 for Poses | 'q' to Quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow('MyCobot Real-time Control', display_frame)

        # 7. 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        
        # ----------------------------------------
        # 사전 정의된 자세로 이동
        # ----------------------------------------
        if key == ord('0'):
            print(f"\n🏠 0번 키 입력: ZERO_POSE_ANGLES({ZERO_POSE_ANGLES})로 이동합니다.")
            mc.send_angles(ZERO_POSE_ANGLES, MOVEMENT_SPEED)
            time.sleep(MOVE_DELAY)
            
        elif key == ord('1'):
            print(f"\n📸 1번 키 입력: CONVEYOR_CAPTURE_POSE({CONVEYOR_CAPTURE_POSE})로 이동합니다.")
            mc.send_angles(CONVEYOR_CAPTURE_POSE, MOVEMENT_SPEED)
            time.sleep(MOVE_DELAY)
            
        elif key == ord('2'):
            print(f"\n🤖 2번 키 입력: ROBOTARM_CAPTURE_POSE({ROBOTARM_CAPTURE_POSE})로 이동합니다.")
            mc.send_angles(ROBOTARM_CAPTURE_POSE, MOVEMENT_SPEED)
            time.sleep(MOVE_DELAY)
            
        elif key == ord('3'):
            print(f"\n🚧 3번 키 입력: INTERMEDIATE_POSE_ANGLES({INTERMEDIATE_POSE_ANGLES})로 이동합니다.")
            mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
            time.sleep(MOVE_DELAY)
            
        # ----------------------------------------
        # Pick 작업 수행 (P: 경유지 포함)
        # ----------------------------------------
        elif key == ord('p'):
            if predicted_angles:
                print("\n⚙️ Pick 작업 시작: 경유 자세를 거쳐 예측된 위치로 이동합니다...")
                
                # 1. 그리퍼 열기
                mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
                time.sleep(0.5)
                
                # 2. 중간 경유 자세로 이동 (안전성 확보)
                print(f"-> 중간 경유 자세로 이동: {INTERMEDIATE_POSE_ANGLES}")
                mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
                time.sleep(2) 
                
                # 3. 최종 예측된 위치로 이동 (물체 집는 자세)
                print(f"-> 최종 예측 위치로 이동: {predicted_angles}")
                mc.send_angles(predicted_angles, MOVEMENT_SPEED)
                time.sleep(3) 
                
                # 4. 그리퍼 닫기 (Pick)
                print("-> 물체 집기 (그리퍼 닫기)")
                mc.set_gripper_value(GRIPPER_CLOSED_VALUE, GRIPPER_SPEED)
                time.sleep(1)
                
                # 5. 물체 들고 다시 중간 경유 자세로 복귀 (안전한 복귀)
                print("-> 물체 들고 중간 경유 자세로 복귀...")
                mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
                time.sleep(2)
                
                print("-> Pick 완료. 다음 물체를 대기 중...")
        # [g]: 그리퍼 닫기 (물건 집기) - set_gripper_value 사용
        elif key == ord('g'):
            print("\n✊ 그리퍼 닫는 중...")
            # GRIPPER_CLOSED_VALUE 위치로 이동
            mc.set_gripper_value(GRIPPER_CLOSED_VALUE, GRIPPER_SPEED) 
            time.sleep(GRIPPER_ACTION_DELAY) # 그리퍼 작동을 기다립니다.
            print(f"✅ 그리퍼 닫힘 완료 (위치: {GRIPPER_CLOSED_VALUE}).")
            
        # [h]: 그리퍼 열기 - set_gripper_value 사용
        elif key == ord('h'):
            print("\n👐 그리퍼 여는 중...")
            # GRIPPER_OPEN_VALUE 위치로 이동
            mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
            time.sleep(GRIPPER_ACTION_DELAY) # 그리퍼 작동을 기다립니다.
            print(f"✅ 그리퍼 열림 완료 (위치: {GRIPPER_OPEN_VALUE}).")
        # ----------------------------------------
        # 프로그램 종료 (Q)
        # ----------------------------------------
        elif key == ord('q'):
            print("\n👋 프로그램 종료.")
            break
            
    # 종료 정리 작업
    cap.release()
    cv2.destroyAllWindows()
    mc.close()

if __name__ == "__main__":
    main()