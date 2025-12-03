import os
import sys
import csv
import cv2
import numpy as np

# ----------------------------------------------------
# 1. 설정 변수 (Configuration)
# ----------------------------------------------------
# CSV 파일과 이미지 파일이 위치한 폴더
DATA_DIR = "mycobot_labeling_data"
CSV_FILE = os.path.join(DATA_DIR, "joint_labels.csv")
WINDOW_NAME = "ROI Selection Tool - 'a':이전, 'd':다음, 'q':종료"

# ----------------------------------------------------
# 2. 전역 상태 변수 (Global State)
# ----------------------------------------------------
# 이미지 파일 리스트 (CSV에서 로드)
image_filenames = []
img_index = 0
drawing = False # 마우스 드래그 중인지 여부

# 현재 드래그 중인 임시 시작/끝 좌표
temp_p1 = (0, 0)
temp_p2 = (0, 0)

# 모든 이미지에 적용될 최종 ROI 좌표
# (x_start, y_start), (x_end, y_end) 형식으로 저장됩니다.
final_roi_p1 = (30, 30) 
final_roi_p2 = (430, 430)

# ----------------------------------------------------
# 3. 마우스 콜백 함수 (Mouse Callback Function)
# ----------------------------------------------------
def draw_roi(event, x, y, flags, param):
    """마우스 이벤트를 처리하고 ROI 좌표를 업데이트합니다."""
    global drawing, temp_p1, temp_p2, final_roi_p1, final_roi_p2

    if event == cv2.EVENT_LBUTTONDOWN:
        # 왼쪽 마우스 버튼 클릭: 드래그 시작
        drawing = True
        temp_p1 = (x, y)
        temp_p2 = (x, y) 

    elif event == cv2.EVENT_MOUSEMOVE:
        # 마우스 이동: 현재 끝점 업데이트
        if drawing:
            temp_p2 = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        # 왼쪽 마우스 버튼 떼기: 드래그 종료 및 최종 ROI 저장
        drawing = False
        temp_p2 = (x, y)
        
        # 드래그를 수행했다면 (시작점과 끝점이 다를 때)
        if temp_p1 != temp_p2:
            # ROI 좌표를 정렬하여 (x_min, y_min)과 (x_max, y_max)를 만듭니다.
            x_start = min(temp_p1[0], temp_p2[0])
            y_start = min(temp_p1[1], temp_p2[1])
            x_end = max(temp_p1[0], temp_p2[0])
            y_end = max(temp_p1[1], temp_p2[1])
            
            final_roi_p1 = (x_start, y_start)
            final_roi_p2 = (x_end, y_end)
            
            print(f"\n✅ ROI 설정 완료: 시작 좌표: {final_roi_p1}, 끝 좌표: {final_roi_p2}")
        else:
            # 클릭만 했을 경우 임시 좌표 초기화
            temp_p1, temp_p2 = (0, 0), (0, 0)

# ----------------------------------------------------
# 4. 메인 로직 (Main Logic)
# ----------------------------------------------------
def main():
    global img_index, drawing, temp_p1, temp_p2, final_roi_p1, final_roi_p2, image_filenames

    # 1. 이미지 파일 리스트 로드
    try:
        with open(CSV_FILE, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            # CSV 파일 헤더에서 'Image_File' 인덱스를 찾습니다.
            img_file_index = header.index("Image_File")
            for row in reader:
                image_filenames.append(row[img_file_index])
    except FileNotFoundError:
        print(f"❌ 오류: CSV 파일이 경로에 없습니다. DATA_DIR={DATA_DIR} 확인: {CSV_FILE}")
        sys.exit(1)
    except ValueError:
        print("❌ 오류: CSV 파일에 'Image_File' 헤더가 없습니다.")
        sys.exit(1)

    if not image_filenames:
        print("⚠️ 경고: CSV 파일에서 이미지 파일 목록을 찾을 수 없습니다.")
        sys.exit(0)

    # 2. 윈도우 생성 및 마우스 콜백 설정
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, draw_roi)
    
    print("\n--- 🖱️ ROI 선택 도구 사용법 ---")
    print("  [d] 키: 다음 이미지 표시")
    print("  [a] 키: 이전 이미지 표시")
    print("  [마우스 드래그]: ROI 영역 지정 (빨간색)")
    print("  [q] 또는 [Esc] 키: 프로그램 종료")
    print("---------------------------------------")
    print(f"✅ 총 {len(image_filenames)}개의 이미지를 로드했습니다.")


    while True:
        # 이미지 파일 로드
        current_filename = image_filenames[img_index]
        current_path = os.path.join(DATA_DIR, current_filename)
        current_image = cv2.imread(current_path)

        if current_image is None:
            print(f"❌ 이미지를 로드할 수 없습니다: {current_path}. 다음 이미지로 건너뜁니다.")
            img_index = (img_index + 1) % len(image_filenames)
            continue
        
        # 렌더링을 위한 이미지 복사본
        display_image = current_image.copy()

        # ----------------------------------------------------
        # 렌더링: ROI 그리기 및 정보 표시
        # ----------------------------------------------------
        p1_to_draw = None
        p2_to_draw = None
        
        if drawing:
            # 1. 드래그 중: 현재 임시 좌표 사용
            p1_to_draw = temp_p1
            p2_to_draw = temp_p2
            status_text = f"DRAGGING | Start: {temp_p1} | End: {temp_p2}"
            
        elif final_roi_p1 is not None:
            # 2. 드래그 완료/이동 후: 최종 저장된 좌표 사용
            p1_to_draw = final_roi_p1
            p2_to_draw = final_roi_p2
            status_text = f"ROI: Start {final_roi_p1} | End {final_roi_p2}"
        
        else:
            # 3. ROI 미지정 상태
            status_text = "ROI를 마우스로 드래그하여 지정하세요."

        # 빨간색 사각형 그리기 (ROI가 지정되었을 때)
        if p1_to_draw is not None and p2_to_draw is not None:
            cv2.rectangle(display_image, p1_to_draw, p2_to_draw, (0, 0, 255), 2) 

        # 텍스트 정보 오버레이
        img_info_text = f"Image {img_index + 1}/{len(image_filenames)}: {current_filename}"
        
        # 검은색 배경 상자
        # cv2.rectangle(display_image, (0, 0), (display_image.shape[1], 70), (0, 0, 0), -1) 
        
        # 이미지 정보 (흰색)
        cv2.putText(display_image, img_info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # ROI 상태/좌표 정보 (흰색)
        cv2.putText(display_image, status_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 이미지 표시
        cv2.imshow(WINDOW_NAME, display_image)
        
        # ----------------------------------------------------
        # 키 입력 처리
        # ----------------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        # [q] 또는 [Esc]: 종료
        if key == ord('q') or key == 27:
            print("\n👋 프로그램 종료.")
            break
        
        # [a]: 이전 이미지
        elif key == ord('a'):
            img_index = (img_index - 1 + len(image_filenames)) % len(image_filenames)
            drawing = False
            temp_p1, temp_p2 = (0, 0), (0, 0)
            print(f"\n🖼️ 이전 이미지로 이동. ({img_index + 1}/{len(image_filenames)})")
            
        # [d]: 다음 이미지
        elif key == ord('d'):
            img_index = (img_index + 1) % len(image_filenames)
            drawing = False
            temp_p1, temp_p2 = (0, 0), (0, 0)
            print(f"\n🖼️ 다음 이미지로 이동. ({img_index + 1}/{len(image_filenames)})")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()