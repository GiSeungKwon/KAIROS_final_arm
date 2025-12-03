import cv2
import numpy as np
import os
import sys

# ----------------------------------------------------
# 1. 설정 변수 (Configuration)
# ----------------------------------------------------
# 카메라 설정
CAMERA_INDEX = 0  # 시스템에 연결된 웹캠 인덱스 (일반적으로 0)

# ROI 설정 (학습 코드와 동일하게 유지)
ROI_START = (30, 30) # (x_min, y_min)
ROI_END = (430, 430) # (x_max, y_max)

# OpenCV 창 이름
WINDOW_NAME = "MyCobot Real-time HSV Masking Tool"
# 트랙바 창 이름
TRACKBAR_WINDOW_NAME = "HSV Controls"

# ----------------------------------------------------
# 2. 전역 변수 및 유틸리티 함수
# ----------------------------------------------------
def nothing(x):
    """트랙바 콜백 함수 (아무 작업도 하지 않음)"""
    pass

def apply_roi_and_hsv_masking(image, hsv_low, hsv_high):
    """
    1. 원본 이미지에 ROI 마스크를 적용합니다 (ROI 외부를 검은색으로).
    2. ROI 영역 내에서 HSV 마스킹을 적용하여 물체를 분리하고 바이너리 이미지를 반환합니다.
    """
    x_min, y_min = ROI_START
    x_max, y_max = ROI_END
    
    # ROI 체크
    if x_max <= x_min or y_max <= y_min:
        return np.zeros_like(image)
        
    # 1. ROI 적용: ROI 외부를 검은색(0)으로 설정
    masked_image_roi = image.copy()
    masked_image_roi[0:y_min, :] = 0   # 상단
    masked_image_roi[y_max:, :] = 0    # 하단
    masked_image_roi[:, 0:x_min] = 0   # 왼쪽
    masked_image_roi[:, x_max:] = 0    # 오른쪽
    
    # 2. HSV 변환 및 마스킹
    # OpenCV는 BGR -> HSV
    hsv = cv2.cvtColor(masked_image_roi, cv2.COLOR_BGR2HSV)
    
    # HSV 범위에 따라 마스크 생성
    hsv_mask = cv2.inRange(hsv, hsv_low, hsv_high)
    
    # 3. 최종 바이너리 이미지 생성 (3채널)
    final_binary_image = np.zeros_like(image)
    
    # 마스크 영역 (물체)만 흰색 (255, 255, 255)으로 채움
    final_binary_image[hsv_mask > 0] = [255, 255, 255]

    return final_binary_image

# ----------------------------------------------------
# 3. 메인 실행 루프
# ----------------------------------------------------
def main():
    # 1. 카메라 연결
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"❌ 오류: 카메라 인덱스 {CAMERA_INDEX}를 열 수 없습니다.")
        sys.exit(1)
        
    # 2. 윈도우 생성 및 트랙바 초기 설정
    cv2.namedWindow(WINDOW_NAME)
    cv2.namedWindow(TRACKBAR_WINDOW_NAME)
    
    # HSV 트랙바 (0-179, 0-255, 0-255 범위)
    # 초기값은 모든 색상을 포함하도록 설정 (H: 0~179, S: 0~255, V: 0~255)
    cv2.createTrackbar('H_Low', TRACKBAR_WINDOW_NAME, 0, 179, nothing)
    cv2.createTrackbar('S_Low', TRACKBAR_WINDOW_NAME, 0, 255, nothing)
    cv2.createTrackbar('V_Low', TRACKBAR_WINDOW_NAME, 0, 255, nothing)
    cv2.createTrackbar('H_High', TRACKBAR_WINDOW_NAME, 179, 179, nothing)
    cv2.createTrackbar('S_High', TRACKBAR_WINDOW_NAME, 255, 255, nothing)
    cv2.createTrackbar('V_High', TRACKBAR_WINDOW_NAME, 255, 255, nothing)
    
    print("\n--- 📹 실시간 웹캠 HSV 마스킹 도구 ---")
    print(f"✅ ROI 설정: {ROI_START} ~ {ROI_END}")
    print("🖱️ 트랙바를 조절하여 물체가 흰색으로 가장 잘 분리되는 HSV 범위를 찾으세요.")
    print("   [q] 또는 [ESC] : 프로그램 종료")
    print("---------------------------------------")
    
    while True:
        # 카메라에서 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 경고: 프레임을 읽을 수 없습니다.")
            break
            
        # 트랙바에서 현재 HSV 값 읽기
        h_low = cv2.getTrackbarPos('H_Low', TRACKBAR_WINDOW_NAME)
        s_low = cv2.getTrackbarPos('S_Low', TRACKBAR_WINDOW_NAME)
        v_low = cv2.getTrackbarPos('V_Low', TRACKBAR_WINDOW_NAME)
        h_high = cv2.getTrackbarPos('H_High', TRACKBAR_WINDOW_NAME)
        s_high = cv2.getTrackbarPos('S_High', TRACKBAR_WINDOW_NAME)
        v_high = cv2.getTrackbarPos('V_High', TRACKBAR_WINDOW_NAME)
        
        hsv_low = np.array([h_low, s_low, v_low])
        hsv_high = np.array([h_high, s_high, v_high])
        
        # 1. 원본 이미지에 ROI 시각화
        temp_frame = frame.copy()
        cv2.rectangle(temp_frame, ROI_START, ROI_END, (0, 0, 255), 2)
        cv2.putText(temp_frame, "Original Frame (with ROI)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 2. 전처리 및 마스킹 적용
        processed_image = apply_roi_and_hsv_masking(frame, hsv_low, hsv_high)
        cv2.putText(processed_image, "Processed Output (Mask)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 3. 결과 이미지들을 수평으로 합치기 (시각적 비교 용이)
        combined_image = np.hstack([temp_frame, processed_image])
        
        # 4. 결과 이미지 표시
        cv2.imshow(WINDOW_NAME, combined_image)
        
        # 5. 현재 HSV 값 터미널 출력 (실시간 확인용)
        sys.stdout.write(f"\r🔍 Current HSV Range: [{h_low}, {s_low}, {v_low}] ~ [{h_high}, {s_high}, {v_high}] ")
        sys.stdout.flush()
        
        # 6. 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27: # q 또는 Esc로 종료
            print("\n👋 프로그램을 종료합니다.")
            break
            
    # 종료 정리 작업
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()