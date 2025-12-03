import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import os
import cv2
import time
import sys
from pymycobot.mycobot import MyCobot

# ----------------------------------------------------
# 0. 환경 설정 및 상수 (Configuration & Constants)
# ----------------------------------------------------
# 훈련 시 설정과 동일하게 유지
MODEL_PATH = os.path.join("models", "best_model.pth") # 저장된 Best Model 경로
PORT = "COM3"
BAUD = 115200
CAMERA_INDEX = 0

# 훈련 시 사용된 관절 제한 (Normalization을 위한 Min/Max)
# 전역 변수로 선언된 것을 활용하여 일관성 유지
JOINT_MIN = np.array([-165.0, -165.0, -165.0, -165.0, -165.0, -175.0], dtype=np.float32)
JOINT_MAX = np.array([ 165.0, 165.0, 165.0, 165.0, 165.0, 175.0], dtype=np.float32)
JOINT_RANGE = JOINT_MAX - JOINT_MIN

# ROI 영역 (훈련 시와 동일)
ROI_START = (30, 30) # (x_min, y_min)
ROI_END = (430, 430) # (x_max, y_max)

# 로봇 동작 설정
MOVEMENT_SPEED = 30 
GRIPPER_SPEED = 20 
SEQUENTIAL_MOVE_DELAY = 1 
GRIPPER_ACTION_DELAY = 1 

# 로봇 자세 설정
INTERMEDIATE_POSE_ANGLES = [-17.2, 30.49, 4.48, 53.08, -90.87, -85.86] # 경유 자세
CONVEYOR_CAPTURE_POSE = [0, 0, 90, 0, -90, -90] # 1 키 자세
ROBOTARM_CAPTURE_POSE = [0, 0, 90, 0, -90, 90] # 2 키 자세
ZERO_POSE_ANGLES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
GRIPPER_OPEN_VALUE = 55 
GRIPPER_CLOSED_VALUE = 25 

# CUDA 설정
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"✅ 사용 장치: {DEVICE}")

# ResNet 표준 정규화 상수
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ----------------------------------------------------
# 1. 모델 정의 및 로드 (Model Definition and Loading)
# ----------------------------------------------------
def load_model(model_path):
    """ResNet-50 모델을 정의하고 사전 학습된 가중치를 로드합니다."""
    # 1. 모델 구조 정의 (훈련 코드와 동일해야 함)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    # 출력: Joint 1 ~ Joint 6 (총 6개)
    model.fc = nn.Linear(num_ftrs, 6)

    # 2. 가중치 로드
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval() # 추론 모드 설정
        print(f"✅ 모델 로드 성공: {model_path}")
        return model
    except FileNotFoundError:
        print(f"\n❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        print("훈련을 먼저 진행하여 모델 파일을 생성하세요.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 모델 로드 중 오류 발생: {e}")
        sys.exit(1)

# ----------------------------------------------------
# 2. 이미지 전처리 및 역정규화 (Preprocessing & Denormalization)
# ----------------------------------------------------
def preprocess_image(image, roi_start, roi_end):
    """
    OpenCV 이미지를 훈련 시 사용한 PyTorch 전처리(Resize, ToTensor, Normalize)와 동일하게 변환합니다.
    :param image: 원본 OpenCV BGR 이미지
    :param roi_start: ROI 시작점 (x_min, y_min)
    :param roi_end: ROI 끝점 (x_max, y_max)
    :return: 전처리된 PyTorch Tensor (batch=1)
    """
    
    # 1. ROI 크롭 (훈련 데이터셋이 masked_output에서 왔으므로 크롭 수행)
    x_min, y_min = roi_start
    x_max, y_max = roi_end
    
    # OpenCV 이미지는 (H, W, C)
    # 이미지 배열 인덱싱은 [y_min:y_max, x_min:x_max] 순서
    cropped_image = image[y_min:y_max, x_min:x_max]

    # 2. BGR -> RGB 변환 (훈련 코드의 MyCobotDataset과 동일)
    image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    
    # 3. PyTorch Transforms 적용
    # ToPILImage() -> Resize(224, 224) -> ToTensor() -> Normalize()
    transform = transforms.Compose([
        transforms.ToPILImage(), # numpy array를 PIL Image로 변환
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    # 
    
    input_tensor = transform(image_rgb)
    
    # 배치 차원 추가 (1, C, H, W)
    input_batch = input_tensor.unsqueeze(0).to(DEVICE)
    return input_batch, image_rgb # 전처리 후의 RGB 이미지도 반환하여 화면에 표시

def denormalize_angles(normalized_angles):
    """
    모델 출력값([-1, 1])을 실제 관절 각도(Joint Angle, [Min, Max])로 역정규화합니다.
    :param normalized_angles: 모델이 예측한 정규화된 각도 (Shape: (6,))
    :return: 실제 Joint Angle (Deg, Shape: (6,))
    """
    # 공식: Angle = (Normalized_Angle + 1) / 2 * (Max - Min) + Min
    # Angle = (Normalized_Angle + 1) / 2 * JOINT_RANGE + JOINT_MIN
    
    normalized_angles_np = normalized_angles.cpu().detach().numpy()
    
    # [-1, 1] -> [0, 1]
    y_prime = (normalized_angles_np + 1.0) / 2.0
    
    # [0, 1] -> [Min, Max]
    actual_angles = y_prime * JOINT_RANGE + JOINT_MIN
    
    # MyCobot은 각도를 소수점 첫째 자리까지만 받으므로 반올림
    actual_angles = np.round(actual_angles, 1)
    
    return actual_angles.tolist()

# ----------------------------------------------------
# 3. 추론 및 로봇 제어 함수 (Inference and Robot Control)
# ----------------------------------------------------
def infer_and_move(mc, model, inputs):
    """
    모델 추론을 수행하고, 로봇을 추론된 자세로 이동시킵니다.
    """
    try:
        # 모델 추론
        with torch.no_grad():
            outputs = model(inputs)
        
        # 출력: (1, 6) -> (6,)
        normalized_angles = outputs.squeeze(0) 
        
        # 역정규화
        target_angles = denormalize_angles(normalized_angles)
        
        print("\n✨ 추론 결과:")
        print(f"  > 정규화된 예측값: {normalized_angles.tolist()}")
        print(f"  > 최종 관절 각도(Deg): {target_angles}")
        
        # 로봇 이동 로직
        # 1. 경유 자세로 이동 (안전 경로 확보)
        mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
        time.sleep(SEQUENTIAL_MOVE_DELAY)
        
        # 2. 목표 자세(추론 결과)로 이동
        mc.send_angles(target_angles, MOVEMENT_SPEED)
        print("✅ 로봇 팔 목표 자세로 이동 완료.")
        
        return target_angles
        
    except Exception as e:
        print(f"\n❌ 추론 및 로봇 이동 중 오류 발생: {e}")
        return None

# ----------------------------------------------------
# 4. 메인 루프 (Main Loop)
# ----------------------------------------------------
def main():
    # 1. MyCobot 및 모델 로드
    model = load_model(MODEL_PATH)
    try:
        mc = MyCobot(PORT, BAUD)
        mc.power_on() 
        mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
        print(f"\n🤖 MyCobot 연결 성공: {PORT}. 그리퍼 Open.")

    except Exception as e:
        print(f"\n❌ MyCobot 연결 실패 ({PORT}): {e}")
        print("프로그램을 종료합니다.")
        sys.exit(1)

    # 2. 카메라 연결
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"\n❌ 카메라 인덱스 {CAMERA_INDEX}를 열 수 없습니다.")
        mc.close()
        sys.exit(1)
        
    last_inferred_angles = None # 마지막으로 추론된 각도 저장

    # 안내 메시지 출력
    print("\n--- 🚀 MyCobot 추론 및 제어 도구 사용법 ---")
    print("  [p] : 현재 ROI 이미지를 캡처/전처리하고, 모델에 **추론(Inference)**을 요청합니다.")
    print("  [e] : 마지막으로 추론된 관절 각도로 로봇 팔을 **이동(Execute)**시킵니다.")
    print("  [0] : 모든 관절을 [0, 0, 0, 0, 0, 0] 자세로 이동")
    print("  [1] : (경유지 경유 후) CONVEYOR_CAPTURE_POSE 이동 및 고정")
    print("  [2] : ROBOTARM_CAPTURE_POSE 이동 및 고정")
    print("  [g] : 그리퍼 닫기") 
    print("  [h] : 그리퍼 열기") 
    print("  [q] : 프로그램 종료")
    print("---------------------------------------")


    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 수신 실패. 카메라 연결을 확인하세요.", end='\r')
            time.sleep(0.1)
            continue
        
        # 1. 라이브 화면에 ROI 표시
        cv2.rectangle(frame, ROI_START, ROI_END, (0, 255, 0), 2) # 초록색 (Green)
        cv2.putText(frame, "ROI (Region of Interest)", (ROI_START[0], ROI_START[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 상태 표시
        if last_inferred_angles:
            status_text = f"STATUS: Inferenced. Press 'e' to MOVE to {last_inferred_angles[0]}..."
            color = (0, 0, 255) # 빨간색: 이동 대기 중
        else:
            status_text = "STATUS: Ready. Press 'p' to Infer."
            color = (255, 255, 255) # 흰색: 추론 대기 중

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow('MyCobot Live Camera', frame)

        key = cv2.waitKey(1) & 0xFF

        # [q]: 프로그램 종료
        if key == ord('q'):
            print("\n👋 end...")
            break
        
        # [p]: 추론(Inference) 실행 및 전처리 이미지 표시
        elif key == ord('p'):
            print("\n🔍 'p' 키 입력: 추론 시작.")
            # 1. 이미지 전처리
            input_batch, preprocessed_rgb_img = preprocess_image(frame, ROI_START, ROI_END)

            # 2. 전처리 완료된 이미지 표시 (224x224)
            # RGB -> BGR 변환하여 OpenCV로 표시
            display_img = cv2.cvtColor(preprocessed_rgb_img, cv2.COLOR_RGB2BGR)
            # 표시를 위해 크기 조정 (선택 사항, 원본 크기 224x224)
            display_img = cv2.resize(display_img, (300, 300), interpolation=cv2.INTER_NEAREST) 
            cv2.imshow('Preprocessed Input (224x224)', display_img)
            
            print("🖼️ 전처리 이미지 표시 완료.")

            # 3. 모델 추론
            inferred_angles = infer_and_move(mc, model, input_batch)
            if inferred_angles:
                # 'e' 키를 누를 때 이동할 수 있도록 저장
                last_inferred_angles = inferred_angles 

        # [e]: 마지막 추론 자세로 이동 (실제 로봇 동작)
        # 'p' 키를 눌러 추론을 완료한 후, 'e' 키를 눌러 로봇을 해당 위치로 이동.
        elif key == ord('e'):
            if last_inferred_angles:
                print(f"\n🚀 'e' 키 입력: 마지막 추론 위치({last_inferred_angles})로 이동 시작.")
                
                # 1. 경유 자세로 이동 
                mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
                time.sleep(SEQUENTIAL_MOVE_DELAY)

                # 2. 목표 자세로 이동
                mc.send_angles(last_inferred_angles, MOVEMENT_SPEED)
                print("✅ 로봇 팔 목표 자세로 이동 완료.")
                
            else:
                print("\n⚠️ 먼저 'p' 키를 눌러 관절 각도를 추론(Infer)해야 합니다.")

        # [0], [1], [2], [g], [h] 키 처리 (로봇 제어)
        elif key == ord('0'):
            print(f"\n⚙️ ZERO_POSE 이동 시작.")
            mc.send_angles(CONVEYOR_CAPTURE_POSE, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            mc.send_angles(ZERO_POSE_ANGLES, MOVEMENT_SPEED)
            print("✅ ZERO_POSE 이동 완료.")
        
        elif key == ord('1'):
            print(f"\n🏠 CONVEYOR_CAPTURE_POSE 이동 시작.")
            mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
            mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            mc.send_angles(CONVEYOR_CAPTURE_POSE, MOVEMENT_SPEED)
            time.sleep(5)
            print("✅ CONVEYOR_CAPTURE_POSE 이동 완료.")

        elif key == ord('2'):
            print(f"\n🏠 ROBOTARM_CAPTURE_POSE 이동 시작.")
            mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
            mc.send_angles(ROBOTARM_CAPTURE_POSE, MOVEMENT_SPEED)
            time.sleep(5)
            print("✅ ROBOTARM_CAPTURE_POSE 이동 완료.")

        elif key == ord('g'):
            print("\n✊ 그리퍼 닫는 중...")
            mc.set_gripper_value(GRIPPER_CLOSED_VALUE, GRIPPER_SPEED) 
            time.sleep(GRIPPER_ACTION_DELAY)
            print(f"✅ 그리퍼 닫힘 완료 (위치: {GRIPPER_CLOSED_VALUE}).")
            
        elif key == ord('h'):
            print("\n👐 그리퍼 여는 중...")
            mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
            time.sleep(GRIPPER_ACTION_DELAY)
            print(f"✅ 그리퍼 열림 완료 (위치: {GRIPPER_OPEN_VALUE}).")


    # 종료 정리 작업
    cap.release()
    cv2.destroyAllWindows()
    try:
        mc.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()