# Contest images: Pre-process 16bit-tiff images
import os
import cv2
import numpy as np
import yaml
from osgeo import gdal
from utils.CD_utils import load_config

def crop_image(image, crop_size, stride):
    """
    이미지에서 일정한 크기로 슬라이딩 윈도우를 사용해 crop합니다.
    image: 입력 이미지 (numpy 배열)
    crop_size: 잘라낼 이미지 크기 (가로, 세로)
    stride: 슬라이딩 윈도우의 보폭 (겹쳐지는 정도 조절)
    """
    crops = []
    h, w, _ = image.shape
    crop_h, crop_w = crop_size

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Adjust x and y to ensure the crop includes the edge
            if y + crop_h > h:
                y = h - crop_h
            if x + crop_w > w:
                x = w - crop_w
            
            crop = image[y:y + crop_h, x:x + crop_w]
            crops.append(crop)
            
            # Break if the crop reaches the edge
            if x == w - crop_w:
                break
        if y == h - crop_h:
            break

    return crops
# 상위 및 하위 2% 클리핑
def clip_and_normalize(band):
    lower_limit, upper_limit = np.percentile(band, 2), np.percentile(band, 98)
    band = np.clip(band, lower_limit, upper_limit)
    normalized_band = ((band - lower_limit) / (upper_limit - lower_limit) * 255).astype(np.uint8)
    return normalized_band

def process_satellite_image(input_tif_path, output_dir, crop_size=(512, 512), stride=256):
    # TIF 파일 열기
    dataset = gdal.Open(input_tif_path)
    if dataset is None:
        raise FileNotFoundError(f"파일을 열 수 없습니다: {input_tif_path}")
    
    # RGB 밴드 읽기 (Band 1: Blue, Band 2: Green, Band 3: Red)
    band1 = dataset.GetRasterBand(1).ReadAsArray()  # Blue
    band2 = dataset.GetRasterBand(2).ReadAsArray()  # Green
    band3 = dataset.GetRasterBand(3).ReadAsArray()  # Red
    
    bands = [band1, band2, band3]
    band_names = ["Blue", "Green", "Red"]

    # 16비트 이미지를 8비트로 정규화
    blue_8bit = clip_and_normalize(band1)
    green_8bit = clip_and_normalize(band2)
    red_8bit = clip_and_normalize(band3)


    # RGB 이미지 결합
    image_rgb = cv2.merge((blue_8bit, green_8bit, red_8bit))
    

    # 이미지 crop
    crops = crop_image(image_rgb, crop_size, stride)
    
    
    
    # 잘라낸 이미지 저장
    for idx, crop in enumerate(crops):
        output_path = os.path.join(output_dir, f"crop_{idx}.jpg")
        cv2.imwrite(output_path, crop)
    
    print(f"총 {len(crops)}개의 이미지가 저장되었습니다.")

if __name__ == "__main__":
    # Load configuration
    config_path = "./config/satellite-config.yaml"
    config = load_config(config_path)    

    basic_res = config["basic_res"]
    times = config["times"]
    step = config["step"]

    initial_tif_path = config["initial_tif_path"]
    new_tif_path = config["new_tif_path"]
    output_dir_base = config["output_dir_base"]

    for time in times:
        crop_size = (basic_res * time, basic_res * time)
        stride = crop_size[0] // step
                        
        initial_output_dir = os.path.join(output_dir_base, f"data_x{basic_res*time}", "initial_images")
        os.makedirs(initial_output_dir, exist_ok=True)
        # Process image
        process_satellite_image(initial_tif_path, initial_output_dir, crop_size, stride)

        new_output_dir = os.path.join(output_dir_base, f"data_x{basic_res*time}", "new_images")
        os.makedirs(new_output_dir, exist_ok=True)

        # Process image
        process_satellite_image(new_tif_path, new_output_dir, crop_size, stride)
