import cv2
import numpy as np
import os
import sys

# ----------------------------------------------------
# 1. 설정 변수 (Configuration)
# ----------------------------------------------------
# 데이터 경로 및 ROI 설정
DATA_DIR = "../data/arm/mycobot_labeling_data"
ROI_START = (30, 30) # (x_min, y_min)
ROI_END = (430, 430) # (x_max, y_max)
OUTPUT_DIR = os.path.join(DATA_DIR, "masked_output") # 전처리된 이미지 저장 폴더

# OpenCV 창 이름
WINDOW_NAME = "Image Preprocessing & HSV Masking Tool"

# ----------------------------------------------------
# 2. 전역 변수 및 유틸리티 함수
# ----------------------------------------------------
def get_image_list():
    """DATA_DIR에서 .jpg 이미지 파일 목록을 가져옵니다."""
    if not os.path.exists(DATA_DIR):
        print(f"❌ 오류: 데이터 폴더를 찾을 수 없습니다: {DATA_DIR}")
        sys.exit(1)
    
    # jpg 파일만 필터링
    return [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.jpg', '.jpeg'))]

def nothing(x):
    """트랙바 콜백 함수 (아무 작업도 하지 않음)"""
    pass

def apply_roi_and_hsv_masking(image, hsv_low, hsv_high):
    """
    1. 원본 이미지에 ROI 마스크를 적용합니다 (ROI 외부를 흰색으로).
    2. ROI 영역 내에서 HSV 마스킹을 적용하여 물체를 분리합니다.
    """
    # 1. ROI 적용: ROI 외부를 흰색으로 만들 마스크 생성
    # mask_roi = np.zeros_like(image)
    mask_roi = np.ones_like(image)
    
    # ROI 영역에 픽셀 복사 (원본 이미지)
    x_min, y_min = ROI_START
    x_max, y_max = ROI_END
    
    if x_max <= x_min or y_max <= y_min:
         print("❌ 오류: 유효하지 않은 ROI 설정입니다.")
         return image # 오류 시 원본 반환
         
    # ROI 영역의 원본 픽셀만 살림
    masked_image_roi = image.copy()
    masked_image_roi[0:y_min, :] = 0  # 상단
    masked_image_roi[y_max:, :] = 0   # 하단
    masked_image_roi[:, 0:x_min] = 0  # 왼쪽
    masked_image_roi[:, x_max:] = 0   # 오른쪽
    
    # 2. HSV 변환 및 마스킹 (ROI 영역 내부만 처리 가능)
    hsv = cv2.cvtColor(masked_image_roi, cv2.COLOR_BGR2HSV)
    
    # HSV 범위에 따라 마스크 생성
    hsv_mask = cv2.inRange(hsv, hsv_low, hsv_high)
    
    # 3. 최종 바이너리 이미지 생성
    # hsv_mask는 단일 채널(흑백) 마스크입니다.
    # 이를 3채널 RGB 이미지로 변환하여 학습 입력 이미지와 동일하게 만듭니다.
    final_binary_image = np.zeros_like(image)
    
    # 마스크 영역 (물체)만 흰색 (255, 255, 255)으로 채움
    final_binary_image[hsv_mask > 0] = [255, 255, 255]

    return final_binary_image

# ----------------------------------------------------
# 3. 메인 실행 루프
# ----------------------------------------------------
def main():
    image_filenames = get_image_list()
    if not image_filenames:
        print("⚠️ 경고: 학습할 이미지가 DATA_DIR에 없습니다.")
        return

    # 출력 폴더 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 윈도우 생성 및 트랙바 초기 설정
    cv2.namedWindow(WINDOW_NAME)
    
    # HSV 트랙바 (0-179, 0-255, 0-255 범위)
    cv2.createTrackbar('H_Low', WINDOW_NAME, 0, 179, nothing)
    cv2.createTrackbar('S_Low', WINDOW_NAME, 0, 255, nothing)
    cv2.createTrackbar('V_Low', WINDOW_NAME, 0, 255, nothing)
    cv2.createTrackbar('H_High', WINDOW_NAME, 179, 179, nothing)
    cv2.createTrackbar('S_High', WINDOW_NAME, 255, 255, nothing)
    cv2.createTrackbar('V_High', WINDOW_NAME, 255, 255, nothing)
    
    img_index = 0
    print("\n--- 🎨 이미지 전처리 도구 ---")
    print(f"✅ ROI 설정: {ROI_START} ~ {ROI_END}")
    print("🖱️ 트랙바를 조절하여 물체가 흰색으로 가장 잘 분리되는 HSV 범위를 찾으세요.")
    print("   [d] : 다음 이미지로 이동")
    print("   [s] : 현재 HSV 범위로 모든 이미지를 처리하고 저장")
    print("   [q] : 종료")
    print("----------------------------")
    
    while True:
        current_filename = image_filenames[img_index]
        current_path = os.path.join(DATA_DIR, current_filename)
        
        # 이미지 로드 (BGR 포맷)
        image = cv2.imread(current_path)
        if image is None:
            print(f"❌ 이미지 로드 실패: {current_path}. 건너뜁니다.")
            img_index = (img_index + 1) % len(image_filenames)
            continue
            
        # 1. 트랙바에서 현재 HSV 값 읽기
        h_low = cv2.getTrackbarPos('H_Low', WINDOW_NAME)
        s_low = cv2.getTrackbarPos('S_Low', WINDOW_NAME)
        v_low = cv2.getTrackbarPos('V_Low', WINDOW_NAME)
        h_high = cv2.getTrackbarPos('H_High', WINDOW_NAME)
        s_high = cv2.getTrackbarPos('S_High', WINDOW_NAME)
        v_high = cv2.getTrackbarPos('V_High', WINDOW_NAME)
        
        hsv_low = np.array([h_low, s_low, v_low])
        hsv_high = np.array([h_high, s_high, v_high])
        
        # 2. 전처리 및 마스킹 적용
        processed_image = apply_roi_and_hsv_masking(image, hsv_low, hsv_high)
        
        # 3. 이미지 정보 오버레이
        info_text = f"Image {img_index+1}/{len(image_filenames)}: {current_filename}"
        cv2.putText(processed_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 4. 결과 이미지 표시
        cv2.imshow(WINDOW_NAME, processed_image)
        
        # 5. 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27: # q 또는 Esc로 종료
            print("\n👋 프로그램을 종료합니다.")
            break
            
        elif key == ord('a'): # a로 이전 이미지
            img_index = (img_index - 1) % len(image_filenames)
            print(f"🖼️ 이전 이미지: {image_filenames[img_index]}")

        elif key == ord('d'): # d로 다음 이미지
            img_index = (img_index + 1) % len(image_filenames)
            print(f"🖼️ 다음 이미지: {image_filenames[img_index]}")
            
        elif key == ord('s'): # s로 전체 저장
            print(f"\n💾 현재 HSV 범위 ({hsv_low} ~ {hsv_high})로 모든 이미지를 처리하고 저장합니다...")
            
            total_count = len(image_filenames)
            for i, filename in enumerate(image_filenames):
                print(f"   -> 처리 중 ({i+1}/{total_count}): {filename}", end='\r')
                full_path = os.path.join(DATA_DIR, filename)
                img_to_process = cv2.imread(full_path)
                
                # 전처리 적용
                final_output = apply_roi_and_hsv_masking(img_to_process, hsv_low, hsv_high)
                
                # 파일명 변경 (예: original.jpg -> masked_original.png)
                base, _ = os.path.splitext(filename)
                save_path = os.path.join(OUTPUT_DIR, f"masked_{base}.png")
                
                cv2.imwrite(save_path, final_output)
            
            print("\n✅ 모든 이미지 처리가 완료되었습니다. (저장 위치: ./mycobot_labeling_data/masked_output)")
            break # 저장 완료 후 종료
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()