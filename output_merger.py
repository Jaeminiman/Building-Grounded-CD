import numpy as np
import cv2
import os
from utils.CD_utils import load_config
from natsort import natsorted
from osgeo import gdal, osr

def save_as_geotiff(image_array, output_path, original_shape, geotransform, projection):
    """
    Saves a NumPy array as a GeoTIFF file.
    
    image_array: The binary image data as a NumPy array.
    output_path: Path to save the GeoTIFF file.
    original_shape: The shape (height, width) of the original image.
    geotransform: Geotransform tuple for geolocation data.
    projection: Projection in WKT format.
    """
    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(output_path, original_shape[1], original_shape[0], 1, gdal.GDT_Byte)
    out_raster.SetGeoTransform(geotransform)
    out_raster.SetProjection(projection)
    
    out_band = out_raster.GetRasterBand(1)
    out_band.WriteArray(image_array)
    out_band.FlushCache()
    out_band.SetNoDataValue(0)  # Set no-data value as 0

    out_raster = None  # Save and close file
    print(f"GeoTIFF saved at {output_path}")

def merge_crops_to_image(crops, merged_image, weight_mask, crop_size, stride):
    """
    Merges cropped images back into a single original-size image.
    crops: Cropped image array (as a list).
    crop_size: Size of the cropped images (width, height).
    stride: Sliding window step.
    """
    h, w = merged_image.shape
    crop_h, crop_w = crop_size    

    crop_idx = 0
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Adjust x and y to ensure the crop includes the edge
            if y + crop_h > h:
                y = h - crop_h
            if x + crop_w > w:
                x = w - crop_w

            # Merge the cropped images and average overlapping areas
            merged_image[y:y + crop_h, x:x + crop_w] += crops[crop_idx] 
            weight_mask[y:y + crop_h, x:x + crop_w] += 1
            crop_idx += 1    

            # Break if the crop reaches the edge
            if x == w - crop_w:
                break
        if y == h - crop_h:
            break

    return merged_image, weight_mask



if __name__ == "__main__":
    CD_config_path = "./config/CD-config.yaml"
    satel_config_path = "./config/satellite-config.yaml"

    CD_config = load_config(CD_config_path)
    satel_config = load_config(satel_config_path)

    crop_dir_base = CD_config["output_dir_base"]
    basic_res = satel_config["basic_res"]
    times = satel_config["times"]
    step = satel_config["step"]
    pred_thr = satel_config["pred_thr"]

    original_shape = satel_config["original_shape"] 

    result_dir = satel_config["final_result_dir"]
    os.makedirs(result_dir, exist_ok=True)

    merged_image = np.zeros(original_shape, dtype=np.uint8)
    weight_mask = np.zeros(original_shape, dtype=np.uint8) # Mask to store weights for overlapping areas

    for time in times:        
        data_scale = f"data_x{basic_res*time}"        
        crop_dir = os.path.join(crop_dir_base, data_scale)                
        

        crop_size = (basic_res * time, basic_res * time)
        stride = crop_size[0] // step

        crop_files = natsorted([os.path.join(crop_dir, f) for f in os.listdir(crop_dir) if f.endswith(".jpg")])
        crops = np.array([cv2.imread(crop_file, cv2.IMREAD_GRAYSCALE) for crop_file in crop_files])

        # To avoid jpg extraction information loss 
        crops = np.where(crops > 127, 1, 0).astype(np.uint8)

   
        # Merge crops to restore the original image
        merged_image, weight_mask = merge_crops_to_image(crops, merged_image, weight_mask, crop_size, stride)
    
    

    # Average overlapping areas
    merged_image = merged_image.astype(np.float32) / np.maximum(weight_mask, 1)  # Divide by weight_mask, using 1 where it's 0
    
    # jpg
    # Set values greater than pred_thr to 255, and the rest to 0    
    result_image = np.where(merged_image > pred_thr, 255, 0).astype(np.uint8)
    result_image_path = os.path.join(result_dir, 'merged_image.jpg')
    cv2.imwrite(result_image_path, result_image)

    # tiff
    # Set values greater than pred_thr to 255, and the rest to 0    
    binary_image = np.where(merged_image > pred_thr, 1, 0).astype(np.uint8)
    result_geotiff_path = os.path.join(result_dir, 'merged_image.tif')    
    geotransform = (0, 1, 0, 0, 0, -1)  # Example geotransform (top left x, pixel width, rotation, top left y, rotation, pixel height)
    projection = osr.SpatialReference()
    projection.ImportFromEPSG(4326)  # Example EPSG (WGS84)
    projection_wkt = projection.ExportToWkt()

    # Save the binary image as GeoTIFF
    save_as_geotiff(binary_image, result_geotiff_path, original_shape, geotransform, projection_wkt)
