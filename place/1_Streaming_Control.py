import cv2
import time
import os
import sys
import csv
import numpy as np
# MyCobot 320 로봇 팔 제어를 위한 라이브러리 임포트
from pymycobot import MyCobot320

# ----------------------------------------------------
# 1. 설정 변수 (Configuration Variables)
# ----------------------------------------------------
PORT = "COM3"          # 📌 로봇 팔 통신 포트 (Windows 환경)
BAUD = 115200          # 📌 로봇 팔 통신 속도
CAMERA_INDEX = 0       # 📌 OpenCV 카메라 인덱스 (기본 카메라)

roi_start = (0, 0)   # 📌 관심 영역(ROI) 시작점 (좌상단)
roi_end = (640, 360)   # 📌 관심 영역(ROI) 끝점 (우하단)

MOVEMENT_SPEED = 30    # 📌 관절 이동 속도 (퍼센트 단위, 느릴수록 안전)
GRIPPER_SPEED = 20     # 📌 그리퍼 작동 속도
SEQUENTIAL_MOVE_DELAY = 1 # 📌 자세 이동 명령 간 대기 시간 (안정성 확보)

GRIPPER_ACTION_DELAY = 1 # 📌 그리퍼 작동 후 대기 시간

# 📌 주요 로봇 자세 (Joint Angles [J1, J2, J3, J4, J5, J6])
CONVEYOR_CAPTURE_POSE = [0, 0, 90, 0, -90, -90] # 컨베이어벨트 캡처 자세
ROBOTARM_CAPTURE_POSE = [0, 0, 10, 80, -90, 90] # 로봇 팔 위 캡처 자세

TEST_PICK_POSE = [-90, 30, 90, -30, -90, -90]   # 테스트 픽업 최종 자세
TMP_PICK_POSE = [-90, 20, 90, -20, -90, -90]     # 픽업 접근 시 중간 자세 (충돌 방지)
TEST_PLACE_POSE = [30, 21.79, 68.11, -0.7, -80.41, -65.56] # 테스트 플레이스 자세

INTERMEDIATE_POSE_ANGLES = [-17.2, 30.49, 4.48, 53.08, -90.87, -85.86] # 📌 주요 충돌 방지 경유 자세 1
INTERMEDIATE_POSE_ANGLES2 = [25.2, 30.49, 4.48, 53.08, -90.87, -85.86] # 다른 경유 자세
ZERO_POSE_ANGLES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 모든 관절 0도 자세

GRIPPER_OPEN_VALUE = 55   # 📌 그리퍼 완전 열림 위치 값
GRIPPER_CLOSED_VALUE = 25 # 📌 그리퍼 완전 닫힘 위치 값

# ----------------------------------------------------
# 3. 메인 로직 및 키 이벤트 처리
# ----------------------------------------------------
def main():
    # MyCobot 연결 및 그리퍼 초기화
    try:
        mc = MyCobot320(PORT, BAUD)
        mc.power_on()
        print(f"\n🤖 MyCobot 연결 성공: {PORT}. 초기 상태: 파워 ON (고정됨)")

        # 📌 그리퍼 초기화 및 기본 열림 설정
        print("-> MyCobot320: 전기 그리퍼 초기화 시작")
        mc.set_gripper_mode(0) # 전기 그리퍼 모드 설정
        mc.init_electric_gripper()
        time.sleep(2)
        mc.set_electric_gripper(0) # 그리퍼 ID 설정
        mc.set_gripper_value(55, 20, 1) # 초기 위치로 이동
        time.sleep(2)
        print("-> MyCobot320: 전기 그리퍼 초기화 완료 (55 위치로 이동).")

        mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
        print(f"-> MyCobot320: 그리퍼 최종 초기화 완료. 위치: **{GRIPPER_OPEN_VALUE} (열림)**.")
        
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
    
    current_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    current_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ 현재 카메라 창 크기 (해상도): {current_width} x {current_height} 픽셀")
    print("\n--- 💡 로봇 제어 가이드 ---")
    print(" [q]: 종료 | [s]: 티칭 시작(서보 잠금 해제) | [e]: 티칭 종료(서보 잠금)")
    print(" [0]: 0도 자세 | [1]: 컨베이어 자세 | [2]: 로봇팔 위 자세")
    print(" [5]: 픽 자세 | [6]: 플레이스 자세")
    print(" [j]: Joint 값 읽기 | [k]: Coordinates 읽기 | [g]: 그리퍼 닫기 | [h]: 그리퍼 열기")
    print("----------------------------")

    while True:
        # 비디오 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("프레임 수신 실패. 카메라 연결을 확인하세요.", end='\r')
            time.sleep(0.1)
            continue
        
        # 관심 영역(ROI) 표시 (빨간색 사각형)
        # 📌 카메라 이미지에서 관심 영역(ROI)을 시각적으로 표시
        cv2.rectangle(
            frame, 
            roi_start,
            roi_end,
            (0, 0, 255), # BGR: 빨간색
            2
        )

        cv2.imshow('MyCobot Live Camera & Labeling Tool', frame)

        # 📌 1ms 동안 키 입력을 대기하고, 입력 시 해당 문자의 ASCII 값 반환
        key = cv2.waitKey(1) & 0xFF

        # [q]: 프로그램 종료
        if key == ord('q'):
            print("\n👋 프로그램 종료 명령 수신. 자원을 해제합니다...")
            break
        
        # [0]: 0도 자세로 이동 (경유지 포함)
        elif key == ord('0'):
            print(f"\n🔄 로봇을 0도 자세 ({ZERO_POSE_ANGLES})로 이동 시작 (경유지 경유)...")
            # 📌 경유지 1로 이동
            mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            # 📌 최종 0도 자세로 이동
            mc.send_angles(ZERO_POSE_ANGLES, MOVEMENT_SPEED)
            print("✅ 0도 자세 이동 완료.")
            
        # [s]: 티칭 모드 시작 (서보 잠금 해제)
        elif key == ord('s'):
            print("\n▶️ **티칭 모드 시작** (모든 서보 잠금 해제, 수동 제어 가능)")
            mc.release_all_servos()
            
        # [e]: 티칭 모드 종료 (서보 잠금)
        elif key == ord('e'):
            print("\n⏸️ **티칭 모드 종료** (모든 서보 잠금, 로봇 움직임 고정)")
            mc.power_on()

        # [1]: CONVEYOR_CAPTURE_POSE로 이동 (경유지 포함)
        elif key == ord('1'):
            print(f"\n🚀 컨베이어 캡처 자세 ({CONVEYOR_CAPTURE_POSE})로 이동 시작...")
            mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
            # 📌 경유지 1로 이동
            mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            
            # 📌 최종 컨베이어 캡처 자세로 이동
            mc.send_angles(CONVEYOR_CAPTURE_POSE, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            print("✅ CONVEYOR_CAPTURE_POSE 이동 완료.")

        # [2]: ROBOTARM_CAPTURE_POSE로 이동 (경유지 포함)
        elif key == ord('2'):
            print(f"\n🚀 로봇팔 위 캡처 자세 ({ROBOTARM_CAPTURE_POSE})로 이동 시작...")
            mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
            
            # 📌 경유지 1로 이동
            mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            
            # 📌 최종 로봇팔 캡처 자세로 이동
            mc.send_angles(ROBOTARM_CAPTURE_POSE, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            print("✅ ROBOTARM_CAPTURE_POSE 이동 완료.")
        
        # [5]: TEST_PICK_POSE로 이동 (경유지 및 중간 자세 포함)
        elif key == ord('5'):
            print(f"\n⬇️ 픽업 자세 ({TEST_PICK_POSE})로 이동 시작...")
            mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
            # 📌 경유지 1로 이동
            mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)

            # 📌 픽업 중간 자세 (TMP_PICK_POSE)로 이동
            mc.send_angles(TMP_PICK_POSE, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)

            # 📌 픽업 최종 자세 (TEST_PICK_POSE)로 이동
            mc.send_angles(TEST_PICK_POSE, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            print("✅ TEST_PICK_POSE 이동 완료.")
        
        # [6]: TEST_PLACE_POSE로 이동 (경유지 포함 및 그리퍼 열기)
        elif key == ord('6'):
            print(f"\n⬆️ 플레이스 자세 ({TEST_PLACE_POSE})로 이동 시작...")
            # 📌 경유지 1로 이동
            mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            
            # 📌 최종 플레이스 자세로 이동
            mc.send_angles(TEST_PLACE_POSE, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)

            # 📌 물체 놓기 위해 그리퍼 열기
            mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
            time.sleep(5)
            print("✅ TEST_PLACE_POSE 이동 및 그리퍼 열림 완료.")

        # [j]: 현재 Joint 값 읽기
        elif key == ord('j'):
            current_angles = mc.get_angles()
            # 📌 로봇으로부터 Joint 값을 읽어오는 함수 호출
            if isinstance(current_angles, list) and not all(c == -1 for c in current_angles): 
                print(f"\n📐 현재 Joint 값 (J1~J6): **{current_angles}**")
            else:
                if current_angles is None or (isinstance(current_angles, list) and any(c == -1 for c in current_angles)) or current_angles == -1:
                    print("\n❌ Joint 값을 읽을 수 없습니다. (에러 코드: -1) 로봇 연결 상태나 M5 스택 상태를 확인하세요.")
                else:
                    print(f"\n❌ Joint 값을 읽을 수 없습니다. 예상치 못한 반환 값: {current_angles}")

		# [k]: 현재 Coordinates (좌표 값) 읽기
        elif key == ord('k'):
            current_coords = mc.get_coords()
            # 📌 로봇으로부터 End-Effector의 좌표 값 (X, Y, Z, Rx, Ry, Rz)을 읽어오는 함수 호출
            if isinstance(current_coords, list) and not all(c == -1 for c in current_coords): 
                print(f"\n🗺️ 현재 Coordinates (X, Y, Z, R, P, Y): **{current_coords}**") 
            else:
                if current_coords is None or (isinstance(current_coords, list) and any(c == -1 for c in current_coords)) or current_coords == -1:
                    print("\n❌ Coordinates 값을 읽을 수 없습니다. (에러 코드: -1) 로봇 연결 상태나 M5 스택 상태를 확인하세요.")
                else:
                    print(f"\n❌ Coordinates 값을 읽을 수 없습니다. 예상치 못한 반환 값: {current_coords}")
        
        # [g]: 그리퍼 닫기
        elif key == ord('g'):
            print("\n✊ 그리퍼 닫는 중...")
            # 📌 그리퍼를 GRIPPER_CLOSED_VALUE로 이동
            mc.set_gripper_value(GRIPPER_CLOSED_VALUE, GRIPPER_SPEED) 
            time.sleep(GRIPPER_ACTION_DELAY)
            print(f"✅ 그리퍼 닫힘 완료 (위치: **{GRIPPER_CLOSED_VALUE}**).")
            
        # [h]: 그리퍼 열기
        elif key == ord('h'):
            print("\n👐 그리퍼 여는 중...")
            # 📌 그리퍼를 GRIPPER_OPEN_VALUE로 이동
            mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
            time.sleep(GRIPPER_ACTION_DELAY)
            print(f"✅ 그리퍼 열림 완료 (위치: **{GRIPPER_OPEN_VALUE}**).")

    # 종료 시 자원 해제
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