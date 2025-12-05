import cv2
import time
import os
import sys
import csv
import numpy as np
# MyCobot 320 라이브러리 임포트
from pymycobot import MyCobot320

# ----------------------------------------------------
# 1. 설정 변수 (Configuration Variables)
# ----------------------------------------------------
PORT = "COM3"          # 로봇 팔 통신 포트
BAUD = 115200          # 통신 속도
CAMERA_INDEX = 0       # OpenCV 카메라 인덱스

roi_start = (30, 30)   # 관심 영역(ROI) 시작점
roi_end = (430, 430)   # 관심 영역(ROI) 끝점

MOVEMENT_SPEED = 30    # 관절 이동 속도
GRIPPER_SPEED = 20     # 그리퍼 작동 속도
SEQUENTIAL_MOVE_DELAY = 1 # 이동 간 대기 시간

GRIPPER_ACTION_DELAY = 1 # 그리퍼 작동 후 대기 시간

CONVEYOR_CAPTURE_POSE = [0, 0, 50, 40, -90, -90] # 컨베이어벨트 캡처 자세
ROBOTARM_CAPTURE_POSE = [0, 0, 50, 40, -90, 90] # 로봇 팔 위 캡처 자세

TEST_PICK_POSE = [-90, 30, 90, -30, -90, -90] # 테스트 픽 자세
TMP_PICK_POSE = [-90, 20, 90, -20, -90, -90] # 테스트 tmp 픽 자세
TEST_PLACE_POSE = [30, 21.79, 68.11, -0.7, -80.41, -65.56] # 테스트 플레이스 자세

INTERMEDIATE_POSE_ANGLES = [-17.2, 30.49, 4.48, 53.08, -90.87, -85.86] # 충돌 방지 경유 자세

INTERMEDIATE_POSE_ANGLES2 = [25.2, 30.49, 4.48, 53.08, -90.87, -85.86] # 다른 경유 자세

ZERO_POSE_ANGLES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 모든 관절 0도 자세

GRIPPER_OPEN_VALUE = 55 # 그리퍼 열림 위치
GRIPPER_CLOSED_VALUE = 25 # 그리퍼 닫힘 위치

DATA_DIR = "mycobot_labeling_data" # 데이터 저장 디렉토리
CSV_FILE = os.path.join(DATA_DIR, "joint_labels.csv") # 라벨 데이터 파일

last_captured_image_path = None # 마지막 캡처된 이미지 경로 (라벨링 대기)

image_counter = 0              # 이미지 파일명 카운터
last_recorded_angles = None    # 마지막으로 저장된 Joint Angles (r 키 이동용) 

# ----------------------------------------------------
# 2. 전역 상태 관리 및 CSV 함수
# ----------------------------------------------------
def init_csv_file(csv_path):
    # CSV 파일과 헤더 초기화
    if not os.path.exists(csv_path):
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Image_File", "Timestamp", "J1", "J2", "J3", "J4", "J5", "J6"])
        print(f"✅ 데이터 저장 경로 및 CSV 파일 생성됨: {csv_path}")

def save_label_data(angles):
    # 현재 Joint 값을 마지막 캡처된 이미지에 대한 라벨로 CSV에 저장
    global last_captured_image_path, last_recorded_angles
    
    if not last_captured_image_path:
        print("\n⚠️ 'j' 키를 누르기 전에 'c' 키를 눌러 캡처된 이미지가 없습니다. 먼저 사진을 찍으세요.")
        return

    row_data = [
        os.path.basename(last_captured_image_path),
        time.strftime("%Y%m%d_%H%M%S"),
    ] + angles
    
    try:
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
        
        print(f"\n✅ 라벨링 성공: {os.path.basename(last_captured_image_path)}에 Joint 값 매핑 완료.")
        
        last_recorded_angles = angles # 이동을 위해 현재 각도 저장
        print(f"\n[DEBUG] last_recorded_angles에 저장된 값: {last_recorded_angles}")

        last_captured_image_path = None # 매핑 후 초기화
    except Exception as e:
        print(f"\n❌ CSV 파일 저장 중 오류 발생: {e}")

# ----------------------------------------------------
# 3. 메인 로직 및 키 이벤트 처리
# ----------------------------------------------------
def main():
    global image_counter, last_captured_image_path, last_recorded_angles

    init_csv_file(CSV_FILE)

    # MyCobot 연결 및 그리퍼 초기화
    try:
        mc = MyCobot320(PORT, BAUD)
        mc.power_on()
        print(f"\n🤖 MyCobot 연결 성공: {PORT}. 초기 상태: 파워 ON (고정됨)")

        print("-> MyCobot320: 전기 그리퍼 초기화 시작")
        mc.set_gripper_mode(0)
        mc.init_electric_gripper()
        time.sleep(2)
        mc.set_electric_gripper(0)
        mc.set_gripper_value(55, 20, 1) 
        time.sleep(2)
        print("-> MyCobot320: 전기 그리퍼 초기화 완료 (55 위치로 이동).")

        mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
        
    except Exception as e:
        print(f"\n❌ MyCobot 연결 실패 ({PORT}): {e}")
        print("프로그램을 종료합니다.")
        sys.exit(1)

    # 카메라 연결
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"\n❌ 카메라 인덱스 {CAMERA_INDEX}를 열 수 없습니다. 카메라 연결 상태를 확인하세요.")
        mc.close()
        sys.exit(1)
    
    # 사용법 안내 출력
    print("\n--- 🕹️ MyCobot 라벨링 도구 사용법 ---")
    print("  [0] : 모든 관절을 [0, 0, 0, 0, 0, 0] 자세로 이동")
    print("  [1] : (경유지 경유 후) CONVEYOR_CAPTURE_POSE 이동 및 고정")
    print("  [2] : ROBOTARM_CAPTURE_POSE 이동 및 고정")
    print("  [s] : RECORD 시작 (서보 모터 잠금 해제, Teaching Mode 활성화)")
    print("  [e] : RECORD 종료 (서보 모터 잠금, 파워 공급)")
    print("  [c] : 상자 이미지 캡처 (파일 저장)")
    print("  [j] : 현재 Joint 값 읽고, 마지막 캡처 이미지에 라벨링 (CSV 저장)")
    print("  [r] : (경유지 경유 후) 마지막으로 기록된 Joint 값으로 이동")
    print(f"  [g] : 그리퍼 닫기 (위치: {GRIPPER_CLOSED_VALUE})") 
    print(f"  [h] : 그리퍼 열기 (위치: {GRIPPER_OPEN_VALUE})") 
    print("  [q] : 프로그램 종료")
    print("---------------------------------------")

    while True:
        # 비디오 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("프레임 수신 실패. 카메라 연결을 확인하세요.", end='\r')
            time.sleep(0.1)
            continue
        
        # 관심 영역(ROI) 표시 (빨간색 사각형)
        cv2.rectangle(
            frame, 
            roi_start,
            roi_end,
            (0, 0, 255),
            2
        )
        
        # 현재 상태 표시
        if last_captured_image_path:
            status_text = "STATUS: C-CAP. 'j' key --> let labeling."
            color = (0, 0, 255)
        else:
            status_text = "STATUS: Ready. 's' key --> start Teaching."
            color = (255, 255, 255)

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('MyCobot Live Camera & Labeling Tool', frame)

        key = cv2.waitKey(1) & 0xFF

        # [q]: 프로그램 종료
        if key == ord('q'):
            print("\n👋 end...")
            break
        
        # [0]: 0도 자세로 이동
        elif key == ord('0'):
            print(f"\n⚙️ ZERO_POSE 이동 시작: 모든 관절을 {ZERO_POSE_ANGLES}로 이동합니다.")
            
            mc.send_angles(CONVEYOR_CAPTURE_POSE, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)

            mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            
            mc.send_angles(ZERO_POSE_ANGLES, MOVEMENT_SPEED)
            print("✅ ZERO_POSE 이동 완료.")
            
        # [s]: 티칭 모드 시작 (서보 잠금 해제)
        elif key == ord('s'):
            print("\n▶️ RECORD start: 서보 잠금 해제. 로봇 팔을 움직여 픽업 자세를 잡아주세요.")
            mc.release_all_servos()
            
        # [e]: 티칭 모드 종료 (서보 잠금)
        elif key == ord('e'):
            print("\n⏸️ RECORD exit: 현재 위치에 고정.")
            mc.power_on()

        # [t]: 픽앤플레이스 테스트 동작 시퀀스
        elif key == ord('t'):
            print(f"\n🏠 test.")
            
            mc.set_gripper_value(80, 50)
            time.sleep(3)
            mc.send_angles([0, 0, 0, 0, 0, 0], 50)
            time.sleep(3)
            mc.send_angles([-17.2, 30.49, 4.48, 53.08, -90.87, -85.86], 50)
            time.sleep(3)
            mc.send_angles([-27.15, 7.55, 118.56, -31.72, -84.99, -119.35], 50)
            time.sleep(3)
            mc.set_gripper_value(25, 50) # 픽 (그리퍼 닫기)
            time.sleep(3)
            mc.send_angles([-17.2, 30.49, 4.48, 53.08, -90.87, -85.86], 50) # 경유
            time.sleep(3)
            mc.send_angles([25.2, 30.49, 4.48, 53.08, -90.87, -85.86], 50) # 경유
            time.sleep(3)
            mc.send_angles([30, 21.79, 68.11, -0.7, -80.41, -65.56], 50) # 플레이스 자세
            time.sleep(3)
            mc.set_gripper_value(80, 50) # 플레이스 (그리퍼 열기)
            time.sleep(3)
            mc.send_angles([-17.2, 30.49, 4.48, 53.08, -90.87, -85.86], 50) # 경유
            time.sleep(3)
            mc.send_angles([0, 0, 0, 0, 0, 0], 50)
            time.sleep(3)

        # [5]: TEST_PICK_POSE로 이동 (경유지 포함)
        elif key == ord('5'):
            print(f"\n🏠 TEST_PICK_POSE 이동 시작: 경유지 경유 후 최종지 {TEST_PICK_POSE}로 이동합니다.")
            
            mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
            mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)

            mc.send_angles(TMP_PICK_POSE, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)

            mc.send_angles(TEST_PICK_POSE, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
        
        # [6]: TEST_PLACE_POSE로 이동 (경유지 포함)
        elif key == ord('6'):
            print(f"\n🏠 TEST_PLACE_POSE 이동 시작: 경유지 경유 후 최종지 {TEST_PLACE_POSE}로 이동합니다.")
            
            mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            
            mc.send_angles(TEST_PLACE_POSE, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)

            mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
            time.sleep(5)
            print("✅ TEST_PLACE_POSE 이동 완료.")

        # [1]: CONVEYOR_CAPTURE_POSE로 이동 (경유지 포함)
        elif key == ord('1'):
            print(f"\n🏠 CONVEYOR_CAPTURE_POSE 이동 시작: 경유지 경유 후 최종지 {CONVEYOR_CAPTURE_POSE}로 이동합니다.")
            
            mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
            mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            
            mc.send_angles(CONVEYOR_CAPTURE_POSE, MOVEMENT_SPEED)
            time.sleep(5)
            print("✅ CONVEYOR_CAPTURE_POSE 이동 완료.")

        # [2]: ROBOTARM_CAPTURE_POSE로 이동
        elif key == ord('2'):
            print(f"\n🏠 ROBOTARM_CAPTURE_POSE 이동 시작: 경유지 경유 후 최종지 {ROBOTARM_CAPTURE_POSE}로 이동합니다.")
            
            mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
            mc.send_angles(ROBOTARM_CAPTURE_POSE, MOVEMENT_SPEED)
            time.sleep(5)
            print("✅ ROBOTARM_CAPTURE_POSE 이동 완료.")
            
        # [c]: 현재 카메라 프레임 캡처 및 저장
        elif key == ord('c'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            image_counter += 1
            filename = f"image_{image_counter:04d}_{timestamp}.jpg"
            save_path = os.path.join(DATA_DIR, filename)
            
            cv2.imwrite(save_path, frame)
            
            last_captured_image_path = save_path # 라벨링 대기 상태 설정
            
            print(f"\n📸 이미지 캡처 완료: {save_path} (라벨링 대기 중...)")
            
        # [j]: 현재 Joint 값 읽고, 마지막 이미지에 라벨링 (CSV 저장)
        elif key == ord('j'):
            current_angles = mc.get_angles()
            
            if isinstance(current_angles, list) and not all(c == -1 for c in current_angles): 
                print(f"\n📐 현재 Joint 값: {current_angles}")
                
                save_label_data(current_angles) # 라벨링 함수 호출
            else:
                if current_angles is None or (isinstance(current_angles, list) and any(c == -1 for c in current_angles)) or current_angles == -1:
                    print("\n❌ Joint 값을 읽을 수 없습니다. 로봇 연결 상태나 M5 스택 상태를 확인하세요. (에러 코드: -1)")
                else:
                    print(f"\n❌ Joint 값을 읽을 수 없습니다. 예상치 못한 반환 값: {current_angles}")
        # [r]: 마지막 기록된 자세로 이동 (경유지 포함)
        elif key == ord('r'):
            if last_recorded_angles:
                print(f"\n↩️ 마지막 기록 위치로 이동 시작: 경유지 경유 후 최종지 {last_recorded_angles}로 이동합니다.")
                
                mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
                time.sleep(SEQUENTIAL_MOVE_DELAY)
                
                mc.send_angles(last_recorded_angles, MOVEMENT_SPEED)
                print("✅ 마지막 기록 위치 이동 완료.")
            else:
                print("\n⚠️ 마지막으로 기록된 Joint Angles가 없습니다. 'j' 키를 눌러 라벨링을 먼저 완료하세요.")
        
        # [g]: 그리퍼 닫기
        elif key == ord('g'):
            print("\n✊ 그리퍼 닫는 중...")
            mc.set_gripper_value(GRIPPER_CLOSED_VALUE, GRIPPER_SPEED) 
            time.sleep(GRIPPER_ACTION_DELAY)
            print(f"✅ 그리퍼 닫힘 완료 (위치: {GRIPPER_CLOSED_VALUE}).")
            
        # [h]: 그리퍼 열기
        elif key == ord('h'):
            print("\n👐 그리퍼 여는 중...")
            mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
            time.sleep(GRIPPER_ACTION_DELAY)
            print(f"✅ 그리퍼 열림 완료 (위치: {GRIPPER_OPEN_VALUE}).")

    # 종료 시 자원 해제
    cap.release()
    cv2.destroyAllWindows()
    try:
        mc.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()