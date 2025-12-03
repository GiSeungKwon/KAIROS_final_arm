import os
import cv2
import numpy as np

# ----------------------------------------------------
# 1. 설정 변수 (Configuration)
# ----------------------------------------------------
# 이 스크립트가 'arm/bright.py'에 있다고 가정하고 상대 경로를 설정합니다.
# 입력 폴더 (크롭된 이미지들이 있는 곳)
INPUT_DIR = "cropped" 

# 출력 폴더 (밝기 증강된 이미지들을 저장할 곳)
OUTPUT_DIR = "cropped" 

# 밝기 증강 계수 (Brightness Factors)
# 1.0 미만: 어둡게, 1.0 초과: 밝게
BRIGHTNESS_FACTORS = [0.8, 0.9, 1.1, 1.2]

# ----------------------------------------------------
# 2. 메인 증강 로직 (Main Augmentation Logic)
# ----------------------------------------------------
def apply_brightness_augmentation():
    """
    INPUT_DIR의 모든 이미지에 대해 밝기 증강을 적용하고 OUTPUT_DIR에 저장합니다.
    """
    
    # 1. 출력 폴더 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✅ 출력 폴더 준비 완료: {OUTPUT_DIR}")

    # 2. 이미지 파일 목록 로드 (JPG 파일만)
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"⚠️ 경고: {INPUT_DIR} 폴더에서 JPG 이미지를 찾을 수 없습니다. 경로를 확인하세요.")
        return
        
    print(f"✅ 총 {len(image_files)}개의 원본 이미지와 {len(BRIGHTNESS_FACTORS)}가지 증강을 적용합니다. (총 {len(image_files) * len(BRIGHTNESS_FACTORS)}개)")

    total_count = 0
    
    # 3. 이미지 반복 및 증강 적용
    for i, filename in enumerate(image_files):
        # 원본 이미지 로드
        image_path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"❌ 오류: 이미지 로드 실패 - {image_path}")
            continue

        # 파일 이름 분리 (확장자 제거)
        base_name, ext = os.path.splitext(filename)

        # 4. 밝기 증강 계수 반복 적용
        for factor in BRIGHTNESS_FACTORS:
            # 밝기 증강 적용 (alpha=factor, beta=0)
            # cv2.convertScaleAbs는 픽셀 값을 [0, 255] 범위로 자동 클리핑(Clamping) 처리합니다.
            augmented_img = cv2.convertScaleAbs(img, alpha=factor, beta=0)

            # 새 파일 이름 생성: '원본이름_bright_1.1.jpg'
            # 소수점 오류 방지를 위해 factor를 문자열로 변환하고 점(.)을 밑줄(_)로 대체
            factor_str = str(factor).replace('.', '_')
            new_filename = f"{base_name}_bright_{factor_str}.jpg"
            save_path = os.path.join(OUTPUT_DIR, new_filename)

            # 증강된 이미지 저장
            cv2.imwrite(save_path, augmented_img)
            total_count += 1
            
            # 진행 상황 출력
            print(f"처리 중: {total_count} / {len(image_files) * len(BRIGHTNESS_FACTORS)} - {new_filename}", end='\r')


    print(f"\n\n🎉 밝기 증강 완료! 총 {total_count}개의 새로운 이미지가 {OUTPUT_DIR}에 저장되었습니다.")

if __name__ == "__main__":
    apply_brightness_augmentation()