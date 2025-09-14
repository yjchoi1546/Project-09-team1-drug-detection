import os
import cv2
import numpy as np
from ultralytics import FastSAM

def process_images_with_fastsam(image_names: list, image_dir: str, result_dir: str):
    # 결과 디렉토리가 없으면 생성합니다.
    os.makedirs(result_dir, exist_ok=True)

    # FastSAM 모델 로드 (FastSAM-s.pt는 사용자 코드에 따름)
    model = FastSAM("C:\Potenup\Drug-Detection-Chatbot\modeling\segment\models\FastSAM-s.pt")
    
    # 이미지 리스트를 순회하며 처리
    for source_path in image_names:
        if not os.path.exists(image_dir + source_path):
            print(f"Error: {image_dir + source_path} 경로의 파일을 찾을 수 없습니다.")
            continue

        # 1. 이미지 불러오기
        original_image = cv2.imread(image_dir + source_path)
        if original_image is None:
            print(f"Error: {image_dir + source_path} 파일을 불러올 수 없습니다.")
            continue
            
        # 2. BGR을 RGB로 변환하여 모델에 입력
        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # 3. FastSAM으로 Segmentation 수행
        results = model(image_rgb, device="cpu", retina_masks=True, conf=0.8, iou=0.8)
        
        # 'everything' 프롬프트로 모든 객체 마스크 추출
        masks = results[0].masks.data
        
        # 4. 마스크 통합 및 배경 제거
        if len(masks) == 0:
            print(f"Info: {source_path}에서 객체가 탐지되지 않아 마스크를 생성하지 않습니다.")
        else:
            # 모든 객체의 마스크를 통합
            combined_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
            for mask_tensor in masks:
                mask_array = mask_tensor.cpu().numpy().astype(np.uint8)
                combined_mask = cv2.bitwise_or(combined_mask, mask_array)

            # 배경 제거 및 투명 배경 PNG 저장
            image_rgba = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)
            image_rgba[:, :, 3] = combined_mask * 255

            # 결과 이미지 저장
            output_path = result_dir + ''.join(source_path.split('.')[:-1]) + '.png'
            cv2.imwrite(output_path, image_rgba)
            print(f"Image with transparent background saved to {output_path}")

# 사용 예시
if __name__ == "__main__":
    # 처리할 이미지 경로 리스트를 정의합니다.
    # 'test1.jpg' 대신 실제 이미지 파일명들을 넣어주세요.
    image_base_path = "C:\Potenup\Drug-Detection-Chatbot\modeling\segment\images/original/"
    result_base_path = "C:\Potenup\Drug-Detection-Chatbot\modeling\segment\images\\results/"
    image_list = ["test1.jpg", "test2.jpg", "test3.jpg"]
    
    process_images_with_fastsam(image_list, image_base_path, result_base_path)