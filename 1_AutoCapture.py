import cv2
import time
import os
import sys
import csv
import numpy as np
from pymycobot.mycobot import MyCobot

# ----------------------------------------------------
# 1. 설정 변수 (Configuration Variables)
# ----------------------------------------------------
# 사용 환경에 맞게 COM 포트 및 카메라 인덱스를 설정합니다.
PORT = "COM3"
BAUD = 115200
CAMERA_INDEX = 0    # 로봇 팔에 연결된 카메라의 인덱스 (보통 0, 1, 2 중 하나)

roi_start = (30, 30) 
roi_end = (430, 430)

# 로봇 동작 속도 설정
MOVEMENT_SPEED = 30 # 로봇 팔 관절 이동 속도 (0-100 범위)
GRIPPER_SPEED = 20  # 그리퍼 작동 속도 (0-100 범위)
SEQUENTIAL_MOVE_DELAY = 1 # 순차 이동 (경유지-목적지) 간의 대기 시간 (초)

# ⚠️ 그리퍼 작동 후 대기 시간 (wait=True를 대체합니다)
GRIPPER_ACTION_DELAY = 1 

CONVEYOR_CAPTURE_POSE = [0, 0, 90, 0, -90, -90]
ROBOTARM_CAPTURE_POSE = [0, 0, 90, 0, -90, 90]

# 중간 경유 자세 (새로 추가된 경유 지점)
INTERMEDIATE_POSE_ANGLES = [-17.2, 30.49, 4.48, 53.08, -90.87, -85.86]

# 🌟 모든 관절이 0도인 기준 자세
ZERO_POSE_ANGLES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# 🌟 그리퍼 위치 제어 값 (0-100 범위, 정밀 제어용)
GRIPPER_OPEN_VALUE = 55    # 최대로 엶.
GRIPPER_CLOSED_VALUE = 25   # 최대로 닫음.

# 파일 저장 경로 및 이름
DATA_DIR = "mycobot_labeling_data"
CSV_FILE = os.path.join(DATA_DIR, "joint_labels.csv")

# ----------------------------------------------------
# 2. 전역 상태 관리 (Global State Management)
# ----------------------------------------------------
# 마지막으로 'c'를 눌러 저장된 이미지 파일 경로
last_captured_image_path = None

# 이미지 파일 카운터
image_counter = 0
# 마지막으로 'j' 키를 눌러 성공적으로 기록된 Joint Angles (이동용)
last_recorded_angles = None 

def init_csv_file(csv_path):
    """CSV 파일을 초기화하고 헤더를 작성합니다."""
    if not os.path.exists(csv_path):
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # J1 ~ J6까지의 각도 외에 타임스탬프와 파일명을 헤더에 추가합니다.
            writer.writerow(["Image_File", "Timestamp", "J1", "J2", "J3", "J4", "J5", "J6"])
        print(f"✅ 데이터 저장 경로 및 CSV 파일 생성됨: {csv_path}")

def save_label_data(angles):
    """현재 Joint 값을 마지막 캡처된 이미지에 대한 라벨로 CSV에 저장합니다."""
    global last_captured_image_path, last_recorded_angles
    
    if not last_captured_image_path:
        print("\n⚠️ 'j' 키를 누르기 전에 'c' 키를 눌러 캡처된 이미지가 없습니다. 먼저 사진을 찍으세요.")
        return

    # Joint 값 리스트에 파일명과 타임스탬프를 추가합니다.
    row_data = [
        os.path.basename(last_captured_image_path),
        time.strftime("%Y%m%d_%H%M%S"),
    ] + angles
    
    try:
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
        
        print(f"\n✅ 라벨링 성공: {os.path.basename(last_captured_image_path)}에 Joint 값 매핑 완료.")
        
        # 🌟 성공적으로 기록된 각도를 이동을 위해 저장합니다.
        last_recorded_angles = angles 
        print(f"\n[DEBUG] last_recorded_angles에 저장된 값: {last_recorded_angles}")

        # 라벨링이 완료된 후 매핑 상태 초기화 (다음 'c' 키 입력을 기다림)
        last_captured_image_path = None 
    except Exception as e:
        print(f"\n❌ CSV 파일 저장 중 오류 발생: {e}")

# ----------------------------------------------------
# 3. 메인 로직 및 키 이벤트 처리 (Main Logic and Key Events)
# ----------------------------------------------------
def main():
    global image_counter, last_captured_image_path, last_recorded_angles

    # CSV 파일 초기화
    init_csv_file(CSV_FILE)

    # 1. MyCobot 연결
    try:
        mc = MyCobot(PORT, BAUD)
        mc.power_on() # 초기에는 로봇 팔 고정 (파워 공급)
        print(f"\n🤖 MyCobot 연결 성공: {PORT}. 초기 상태: 파워 ON (고정됨)")

        # 초기에는 그리퍼를 최대로 열어 둡니다. (set_gripper_value 사용)
        mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
        
    except Exception as e:
        print(f"\n❌ MyCobot 연결 실패 ({PORT}): {e}")
        print("프로그램을 종료합니다.")
        sys.exit(1)

    # 2. 카메라 연결
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW) # Windows 환경에서 CAP_DSHOW 권장
    if not cap.isOpened():
        print(f"\n❌ 카메라 인덱스 {CAMERA_INDEX}를 열 수 없습니다. 카메라 연결 상태를 확인하세요.")
        mc.close()
        sys.exit(1)
    
    # 안내 메시지 출력
    print("\n--- 🕹️ MyCobot 라벨링 도구 사용법 ---")
    print("  [0] : 모든 관절을 [0, 0, 0, 0, 0, 0] 자세로 이동") # 새로운 기능 안내
    print("  [1] : (경유지 경유 후) CONVEYOR_CAPTURE_POSE 이동 및 고정")
    print("  [2] : ROBOTARM_CAPTURE_POSE 이동 및 고정")
    print("  [s] : RECORD 시작 (서보 모터 잠금 해제, Teaching Mode 활성화)")
    print("  [e] : RECORD 종료 (서보 모터 잠금, 파워 공급)")
    print("  [c] : 상자 이미지 캡처 (파일 저장)")
    print("  [j] : 현재 Joint 값 읽고, 마지막 캡처 이미지에 라벨링 (CSV 저장)")
    print("  [r] : (경유지 경유 후) 마지막으로 기록된 Joint 값으로 이동")
    print(f"  [g] : 그리퍼 닫기 (위치: {GRIPPER_CLOSED_VALUE})") 
    print(f"  [h] : 그리퍼 열기 (위치: {GRIPPER_OPEN_VALUE})") 
    print("  [q] : 프로그램 종료")
    print("---------------------------------------")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 수신 실패. 카메라 연결을 확인하세요.", end='\r')
            time.sleep(0.1)
            continue
        
        cv2.rectangle(
            frame, 
            roi_start, # 시작점 (x_start, y_start)
            roi_end, # 끝점 (x_end, y_end)
            (0, 0, 255),  # 빨간색 (Red)
            2             # 굵기
        )
        
        # 화면에 현재 상태 표시
        if last_captured_image_path:
            status_text = "STATUS: C-CAP. 'j' key --> let labeling."
            color = (0, 0, 255) # 빨간색
        else:
            status_text = "STATUS: Ready. 's' key --> start Teaching."
            color = (255, 255, 255) # 흰색

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('MyCobot Live Camera & Labeling Tool', frame)

        # -----------------------------------------
        # 키 입력 처리
        # -----------------------------------------
        key = cv2.waitKey(1) & 0xFF

        # [q]: 프로그램 종료
        if key == ord('q'):
            print("\n👋 end...")
            break
        
        # [0]: 모든 Joint 각도를 0으로 이동 (새로 추가된 기능)
        elif key == ord('0'):
            print(f"\n⚙️ ZERO_POSE 이동 시작: 모든 관절을 {ZERO_POSE_ANGLES}로 이동합니다.")
            
            mc.send_angles(CONVEYOR_CAPTURE_POSE, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)

            mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            
            mc.send_angles(ZERO_POSE_ANGLES, MOVEMENT_SPEED)
            print("✅ ZERO_POSE 이동 완료.")
            
        # [s]: RECORD 시작 (서보 모터 잠금 해제/Teaching Mode)
        elif key == ord('s'):
            print("\n▶️ RECORD start: 서보 잠금 해제. 로봇 팔을 움직여 픽업 자세를 잡아주세요.")
            mc.release_all_servos() # 서보 모터 잠금 해제
            
        # [e]: RECORD 종료 (서보 모터 잠금/파워 공급)
        elif key == ord('e'):
            print("\n⏸️ RECORD exit: 현재 위치에 고정.")
            mc.power_on() # 서보 모터에 다시 파워 공급 (고정)
            # mc.set_gripper_value(GRIPPER_CLOSED_VALUE, GRIPPER_SPEED)
            
        # [1]: 사전 정의된 CONVEYOR_CAPTURE_POSE 이동 (경유지 포함)
        elif key == ord('1'):
            print(f"\n🏠 CONVEYOR_CAPTURE_POSE 이동 시작: 경유지 경유 후 최종지 {CONVEYOR_CAPTURE_POSE}로 이동합니다.")
            
            mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
            # 1. 중간 경유 자세로 이동
            mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            
            # 2. 최종 HOME_POSE로 이동
            mc.send_angles(CONVEYOR_CAPTURE_POSE, MOVEMENT_SPEED)
            time.sleep(5)
            print("✅ CONVEYOR_CAPTURE_POSE 이동 완료.")

        # [2]: 사전 정의된 ROBOTARM_CAPTURE_POSE 이동 (경유지 포함)
        elif key == ord('2'):
            print(f"\n🏠 ROBOTARM_CAPTURE_POSE 이동 시작: 경유지 경유 후 최종지 {ROBOTARM_CAPTURE_POSE}로 이동합니다.")
            
            mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
            mc.send_angles(ROBOTARM_CAPTURE_POSE, MOVEMENT_SPEED)
            time.sleep(5)
            print("✅ ROBOTARM_CAPTURE_POSE 이동 완료.")
            
        # [c]: 상자 이미지 캡처 및 저장
        elif key == ord('c'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            image_counter += 1
            filename = f"image_{image_counter:04d}_{timestamp}.jpg"
            save_path = os.path.join(DATA_DIR, filename)
            
            # 현재 프레임을 저장
            cv2.imwrite(save_path, frame)
            
            # 마지막 캡처된 이미지 경로를 전역 변수에 저장 (라벨링 대기 상태)
            last_captured_image_path = save_path
            
            print(f"\n📸 이미지 캡처 완료: {save_path} (라벨링 대기 중...)")
            
        # [j]: 현재 Joint 값 읽고, 마지막 캡처 이미지에 라벨링
        elif key == ord('j'):
            current_angles = mc.get_angles()
            
            # 🌟 수정된 부분: 반환 값이 리스트이고, 오류 코드 [-1]이 아닐 때만 처리
            # mc.get_angles()는 에러 시 [-1] 또는 None을 반환할 수 있음
            if isinstance(current_angles, list) and not all(c == -1 for c in current_angles): 
                print(f"\n📐 현재 Joint 값: {current_angles}")
                
                # 라벨링 함수 호출 (c 키로 찍은 사진과 매핑)
                save_label_data(current_angles)
            else:
                # ⚠️ 추가 수정: mc.get_angles()가 [-1] (리스트)가 아닌 -1 (정수)을 반환할 수도 있음.
                # 코드의 현재 오류 메시지를 보건대, 정수 -1을 반환하는 경우가 발생했으므로, 
                # 더 넓은 범위의 오류 처리가 필요합니다.

                # 🚀 강력한 오류 처리 로직
                if current_angles is None or (isinstance(current_angles, list) and any(c == -1 for c in current_angles)) or current_angles == -1:
                    print("\n❌ Joint 값을 읽을 수 없습니다. 로봇 연결 상태나 M5 스택 상태를 확인하세요. (에러 코드: -1)")
                else:
                    # 예상치 못한 경우 (예: 정수 -1)에 대한 처리
                    print(f"\n❌ Joint 값을 읽을 수 없습니다. 예상치 못한 반환 값: {current_angles}")
        elif key == ord('r'):
            if last_recorded_angles:
                print(f"\n↩️ 마지막 기록 위치로 이동 시작: 경유지 경유 후 최종지 {last_recorded_angles}로 이동합니다.")
                
                # 1. 중간 경유 자세로 이동
                mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
                time.sleep(SEQUENTIAL_MOVE_DELAY)
                
                mc.send_angles(last_recorded_angles, MOVEMENT_SPEED)
                print("✅ 마지막 기록 위치 이동 완료.")
            else:
                print("\n⚠️ 마지막으로 기록된 Joint Angles가 없습니다. 'j' 키를 눌러 라벨링을 먼저 완료하세요.")
        
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

    # 종료 정리 작업
    cap.release()
    cv2.destroyAllWindows()
    try:
        mc.close()
    except Exception:
        pass # 이미 닫혀있을 경우 오류 무시

if __name__ == "__main__":
    main()
