import os
import sys
import time
import csv
import cv2
from pymycobot import MyCobot320
from typing import List

# ===============================================
# ⚙️ MyCobot 및 시스템 설정
# ===============================================

# ⚠️ myCobot 연결 포트를 실제 환경에 맞게 변경해주세요.
PORT = "COM3" 
BAUD = 115200

# 카메라 설정 (일반적으로 0 또는 1)
CAMERA_INDEX = 0 
CAPTURE_WIDTH = 800
CAPTURE_HEIGHT = 600

# 로봇 이동 설정
MOVEMENT_SPEED = 70 
SEQUENTIAL_MOVE_DELAY = 1.5 # 로봇 이동 후 자세 안정화 대기 시간 (초)

# 데이터 저장 경로 설정
DATA_DIR = "data"
CSV_FILE_PATH = os.path.join(DATA_DIR, "cordinate.csv")

# ===============================================
# 📐 정의된 로봇 자세 (Joint Angles)
# ===============================================
CONVEYOR_CAPTURE_POSE: List[float] = [0, 0, 90, 0, -90, -90]
INTERMEDIATE_POSE_ANGLES: List[float] = [-17.2, 30.49, 4.48, 53.08, -90.87, -85.86]
ZERO_POSE_ANGLES: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# 현재 Rz 값을 조정하기 위한 전역 변수
current_rz_angle = CONVEYOR_CAPTURE_POSE[5] 

# 📌 가장 최근에 캡처된 이미지 파일 이름을 저장하는 변수
latest_captured_filename: str = ""

# ===============================================
# 🛠️ 로봇 제어 및 데이터 기록 함수
# ===============================================

def safe_send_angles(mc: MyCobot320, angles: List[float], speed: int = MOVEMENT_SPEED):
    """중간 자세를 거쳐 안전하게 목표 각도로 이동합니다."""
    print(f"\n🚀 중간 자세({INTERMEDIATE_POSE_ANGLES})로 이동...")
    mc.send_angles(INTERMEDIATE_POSE_ANGLES, speed)
    time.sleep(SEQUENTIAL_MOVE_DELAY)
    
    print(f"🚀 목표 자세({angles})로 이동 시작...")
    mc.send_angles(angles, speed)
    time.sleep(SEQUENTIAL_MOVE_DELAY)
    print("✅ 이동 완료.")

def capture_image(cap: cv2.VideoCapture) -> str:
    """
    P 키에 대응. 현재 화면을 캡처하고 파일 이름을 반환합니다. (Rz 기록 X)
    """
    global DATA_DIR
    
    ret, frame = cap.read()
    if not ret:
        print("\n❌ 카메라 프레임을 읽을 수 없습니다.")
        return ""

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"image_{timestamp}.jpg"
    image_save_path = os.path.join(DATA_DIR, filename)

    try:
        cv2.imwrite(image_save_path, frame)
        print(f"\n📸 이미지 캡처 완료 (P 키): {image_save_path} ({frame.shape[1]}x{frame.shape[0]})")
        return filename
    except Exception as e:
        print(f"\n❌ 이미지 저장 오류: {e}")
        return ""

def record_pose_to_csv(filename: str, rz_angle: float) -> bool:
    """
    J 키에 대응. 가장 최근 캡처된 이미지 이름과 현재 Rz 값을 CSV에 기록합니다.
    기록 성공 시 True 반환.
    """
    global CSV_FILE_PATH
    
    if not filename:
        print("\n⚠️ Rz 좌표 기록 실패: 먼저 'P' 키를 눌러 이미지를 캡처해야 합니다.")
        return False

    # CSV 파일에 데이터 기록
    try:
        is_new_file = not os.path.exists(CSV_FILE_PATH)
        with open(CSV_FILE_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            if is_new_file:
                writer.writerow(['Timestamp', 'Image_Filename', 'Rz_Angle_J6']) # 헤더
            
            timestamp = filename.split('_')[1].split('.')[0]
            writer.writerow([timestamp, filename, f"{rz_angle:.2f}"])
        
        print(f"✅ Rz 좌표 기록 완료 (J 키): 파일명={filename}, Rz={rz_angle:.2f} --> {CSV_FILE_PATH}")
        return True
    except Exception as e:
        print(f"\n❌ CSV 파일 기록 오류: {e}")
        return False

def adjust_rz_angle(mc: MyCobot320, adjustment: float):
    """
    로봇 팔의 Rz (Joint 6) 값을 조정하고 로봇을 이동시킵니다.
    """
    global current_rz_angle, MOVEMENT_SPEED
    
    current_angles = mc.get_angles()
    if isinstance(current_angles, list) and not all(c == -1 for c in current_angles): 
        current_angles[5] += adjustment
        
        # 새로운 Rz 값 저장
        current_rz_angle = current_angles[5]
        
        # 로봇 이동
        mc.send_angles(current_angles, MOVEMENT_SPEED - 10) 
        time.sleep(0.1)
        print(f"\n➡️ Rz 조정 완료: Rz(J6) **{current_rz_angle:.2f}** 도 (조정량: {adjustment:+.1f})")
        
    else:
        print("\n❌ Joint 값을 읽을 수 없어 Rz 조정 및 이동을 수행할 수 없습니다. 로봇 연결/파워 상태를 확인하세요.")


# ===============================================
# 🚀 메인 실행 함수
# ===============================================

def main():
    global current_rz_angle, latest_captured_filename
    
    # 1. 데이터 저장 폴더 생성
    os.makedirs(DATA_DIR, exist_ok=True)

    # 2. MyCobot 연결
    try:
        mc = MyCobot320(PORT, BAUD)
        mc.power_on()
        print(f"\n🤖 MyCobot 연결 성공: {PORT}. 초기 상태: 파워 ON (서보 잠금)")
        
    except Exception as e:
        print(f"\n❌ MyCobot 연결 실패 ({PORT}): {e}")
        sys.exit(1)

    # 3. 카메라 연결 및 해상도 설정
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"\n❌ 카메라 인덱스 {CAMERA_INDEX}를 열 수 없습니다.")
        mc.close()
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ 카메라 연결 완료. 실제 해상도: **{actual_width} x {actual_height}** 픽셀.")
    
    # 4. 초기 자세 이동
    try:
        safe_send_angles(mc, CONVEYOR_CAPTURE_POSE) 
        current_rz_angle = CONVEYOR_CAPTURE_POSE[5]
    except Exception as e:
        print(f"❌ 초기 자세 이동 중 오류 발생: {e}")

    # 최종 키 가이드
    print("\n--- 🔑 MyCobot 키보드 제어 가이드 (데이터 수집) ---")
    print(" [q]: 프로그램 종료")
    print(" [p]: **1단계: 이미지 캡처** (./data/image_[...].jpg 저장)")
    print(" [j]: **2단계: Rz 매핑 & CSV 기록** (기록 후 P를 다시 눌러야 함)")
    print("---------------------------------")
    print(" [E/T]: Rz ±20° | [D/G]: Rz ±10° | [C/B]: Rz ±5°") # E: +20, T: -20, D: +10, G: -10, C: +5, B: -5
    print("---------------------------------")
    print(" [0]: 0° 자세 | [1]: 중간 자세 | [2]: 캡처 자세 복귀")
    print(f" 현재 Rz: **{current_rz_angle:.2f}** 도, 최근 이미지: **{latest_captured_filename if latest_captured_filename else '없음'}**")
    print("---------------------------------")
    

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        
        # 현재 상태 및 안내를 화면에 표시
        display_frame = frame.copy()
        
        # 1. Rz 및 최근 이미지 상태
        cv2.putText(display_frame, f"Rz (J6): {current_rz_angle:.2f} deg", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Last Captured: {latest_captured_filename if latest_captured_filename else 'NONE (P required)'}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 2. Rz 조절 키 안내
        cv2.putText(display_frame, "Rz Adjust: [E/T] +/-20 | [D/G] +/-10 | [C/B] +/-5", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # 3. P/J 상태 안내
        if latest_captured_filename:
             # P를 눌러 이미지가 저장되었으나, 아직 J를 누르지 않은 상태
            status_text = "READY! Press [J] to record pose."
            status_color = (0, 0, 255) # 빨간색
        else:
            # J를 눌러 초기화되었거나, 아직 P를 누르지 않은 상태
            status_text = "Wait for P... Press [P] to capture image."
            status_color = (255, 165, 0) # 주황색/파란색
            
        cv2.putText(display_frame, status_text, 
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        cv2.imshow('MyCobot Rz Control & Data Collection', display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
            
        # ------------------------------------------------
        # 📌 Rz 조절 기능 (업데이트)
        # ------------------------------------------------
        elif key == ord('e'):
            adjust_rz_angle(mc, 20.0) # Rz +20°
        elif key == ord('t'):
            adjust_rz_angle(mc, -20.0) # Rz -20°
        elif key == ord('d'):
            adjust_rz_angle(mc, 10.0) # Rz +10°
        elif key == ord('g'):
            adjust_rz_angle(mc, -10.0) # Rz -10°
        elif key == ord('c'):
            adjust_rz_angle(mc, 5.0) # Rz +5°
        elif key == ord('b'):
            adjust_rz_angle(mc, -5.0) # Rz -5°
        
        # ------------------------------------------------
        # 📌 데이터 수집 기능
        # ------------------------------------------------
        elif key == ord('p'):
            # P 키: 이미지 캡처만 수행하고 파일 이름 업데이트
            filename = capture_image(cap)
            if filename:
                latest_captured_filename = filename
            
        elif key == ord('j'):
            # J 키: 기록된 이미지 이름과 현재 Rz 값을 매핑하여 CSV에 저장
            if record_pose_to_csv(latest_captured_filename, current_rz_angle):
                # 기록 성공 시, 중복 저장을 막기 위해 파일 이름 초기화
                latest_captured_filename = "" 
                print("\n** 다음 데이터 수집을 위해 [P] 키를 눌러 새 이미지를 캡처하십시오. **")

        # ------------------------------------------------
        # 📌 자세 이동 기능
        # ------------------------------------------------
        elif key == ord('0'):
            safe_send_angles(mc, ZERO_POSE_ANGLES)
            current_rz_angle = ZERO_POSE_ANGLES[5]
            
        elif key == ord('1'):
            safe_send_angles(mc, INTERMEDIATE_POSE_ANGLES)
            current_rz_angle = INTERMEDIATE_POSE_ANGLES[5]
            
        elif key == ord('2'):
            safe_send_angles(mc, CONVEYOR_CAPTURE_POSE)
            current_rz_angle = CONVEYOR_CAPTURE_POSE[5] 
        
    print("🧹 자원 해제 중: 카메라 및 로봇 연결 종료...")
    cap.release()
    cv2.destroyAllWindows()
    try:
        mc.close()
    except Exception:
        pass
    print("👍 프로그램 종료 완료.")

if __name__ == "__main__":
    main()