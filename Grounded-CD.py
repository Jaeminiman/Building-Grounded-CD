import os
import cv2
import numpy as np
import logging
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from utils.image_pair import ImagePair
from utils.object_detection_utils import *
from utils.CD_utils import *
import yaml
from mmdet.apis import DetInferencer
from typing import List, Tuple


# Configure logging
logging.basicConfig(level=logging.INFO)


class Grounded_CD:
    #########################################################################################################
    # setting        
    #########################################################################################################
    def __init__(self, config_path: str):
        """Initialize the Grounded_CD class with configuration file."""
        self.conf = load_config(config_path)
        self._setup()

    def _setup(self) -> None:        
        """Setup function to initialize devices and models."""
        self.device = self.conf['device']
        # Setup SAM model
        self._setup_sam()

        # Setup Grounding Dino model
        self._setup_object_detection()

    def _setup_sam(self) -> None:
        """Function to set up SAM model."""
        sam_checkpoint = self.conf['sam_checkpoint']
        model_type = self.conf['model_type']        
        self.res_embed = self.conf['res_embed']
        self.res_sam = self.conf['res_sam']        
        self.threshold_cossim = self.conf['threshold_cos_sim']
        
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        self.predictor1 = SamPredictor(sam) 
        self.predictor2 = SamPredictor(sam) 

    def _setup_object_detection(self) -> None:
        """Function to set up object detection model."""
        # Grounding Dino config
        gd_conf = self.conf['grounding_dino_config']
        gd_checkpoint = self.conf['grounding_dino_checkpoint']
        self.gd_texts = self.conf['grounding_dino_texts']
        self.gd_pred_thr = self.conf['grounding_dino_pred_score_thr']
        self.gd_iou_thr = self.conf['grounding_dino_iou_thr']
        self.gd_save_dir = self.conf['grounding_dino_save_dir']
        os.makedirs(self.gd_save_dir, exist_ok=True)
        
        # Initialize the DetInferencer
        self.inferencer = DetInferencer(
            model=gd_conf,
            weights=gd_checkpoint,
            device=self.device,    
        )

        # Input image directories & sorting image paths
        self.image_dir_base = self.conf['image_dir_base']                

        # For saving CD results
        self.output_dir_base = self.conf['output_dir_base']
        os.makedirs(self.output_dir_base, exist_ok=True) 
    
    def set_data_scale(self, data_scale: str) -> None:

        initial_image_dir = os.path.join(self.image_dir_base, data_scale, "initial_images")
        new_image_dir = os.path.join(self.image_dir_base, data_scale, "new_images")        
        self.new_paths = sorted([os.path.join(new_image_dir, f) for f in os.listdir(new_image_dir) if (f.endswith('.png') or f.endswith('.jpg'))])
        self.initial_paths = sorted([os.path.join(initial_image_dir, f) for f in os.listdir(initial_image_dir) if (f.endswith('.png') or f.endswith('.jpg'))])
        self.output_dir = os.path.join(self.output_dir_base, data_scale)
        os.makedirs(self.output_dir, exist_ok=True) 

    #########################################################################################################
    # object detection        
    #########################################################################################################
    def _extract_boxes(self, gd_output: str, width: int, height: int) -> List[List[float]]:
        """Function to extract bounding boxes from Grounding Dino output."""
        boxes = []
        with open(gd_output, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # Extract bounding box information
                cls, conf, x_center_r, y_center_r, box_width_r, box_height_r = map(float, line.split(' '))
                cls = int(cls)

                x1 = int((x_center_r - box_width_r / 2) * width)
                x2 = int((x_center_r + box_width_r / 2) * width)
                y1 = int((y_center_r - box_height_r / 2) * height)
                y2 = int((y_center_r + box_height_r / 2) * height)
                
                boxes.append([cls, x1, y1, x2, y2])
        return boxes

    def _debut_detection(self, debut_boxes: List[List[int]], cossim_debut: List[float], width: int, height: int) -> Tuple[List[np.ndarray], List[List[int]]]:
        """Function to detect newly appearing objects (debut detection)."""
        debut_masks = []
        pop_idx = []
        for idx, cos_sim in enumerate(cossim_debut):
            if cos_sim > self.threshold_cossim:        
                pop_idx.append(idx)
                continue

            cls, x1, y1, x2, y2 = debut_boxes[idx]
            x1, y1, x2, y2 = raw_to_sam_scale([x1, y1, x2, y2], width, height, self.res_sam)
            
            input_box = np.array([x1, y1, x2, y2])

            debut_mask, scores, logits = self.predictor2.predict(    
                box=input_box[None, :],
                multimask_output=True    
            )
            debut_masks.append(debut_mask)    

        for idx in reversed(pop_idx):
            debut_boxes.pop(idx)
        
        return debut_masks, debut_boxes

    def _retirement_detection(self, retirement_boxes: List[List[int]], cossim_retirement: List[float], width: int, height: int) -> Tuple[List[np.ndarray], List[List[int]]]:
        """Function to detect objects that have disappeared (retirement detection)."""
        retirement_masks = []
        pop_idx = []
        for idx, cos_sim in enumerate(cossim_retirement):    
            if cos_sim > self.threshold_cossim:
                pop_idx.append(idx)
                continue

            cls, x1, y1, x2, y2 = retirement_boxes[idx]
            x1, y1, x2, y2 = raw_to_sam_scale([x1, y1, x2, y2], width, height, self.res_sam)
            
            input_box = np.array([x1, y1, x2, y2])

            masks_retirement, scores, logits = self.predictor1.predict(    
                box=input_box[None, :],
                multimask_output=True
            )     
            retirement_masks.append(masks_retirement) 

        for idx in reversed(pop_idx):
            retirement_boxes.pop(idx)
        
        return retirement_masks, retirement_boxes

    def _grounding_dino(self, image: np.ndarray, image_path: str, phase: str) -> str:
        """Function to run inference using Grounding Dino."""
        texts = self.gd_texts            
        pred_score_thr = self.gd_pred_thr
        iou_threshold = self.gd_iou_thr
        save_dir = self.gd_save_dir
        save_image_path = os.path.join(save_dir, phase, os.path.basename(image_path))

        # Use the detector to do inference
                

        result = self.inferencer(image, texts=texts, pred_score_thr=pred_score_thr)
        labels = np.array(result['predictions'][0]['labels'])
        scores = np.array(result['predictions'][0]['scores'])
        bboxes = np.array(result['predictions'][0]['bboxes'])

        # Post processing of bounding boxes
        bboxes, scores, labels = filter_bboxes_by_score(bboxes, scores, labels, score_threshold = pred_score_thr)

        # Set the IoU threshold for NMS
        bboxes, scores, labels = class_wise_nms(bboxes, scores, labels, iou_threshold=iou_threshold)

        # Remove too large boxes
        bboxes, scores, labels = remove_large_containing_boxes(bboxes, scores, labels)


        # Save the label file (YOLO format)
        save_label_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.txt")
        logging.info(save_label_path)

        # visualization of object detection
        image_canvas = image.copy()
        color = [0, 255, 0]

        with open(save_label_path, "w") as label_file:
            for i, box in enumerate(bboxes):
                conf = scores[i]  # Confidence score
                class_id = labels[i]  # Class ID
                
                # Extract bounding box information                    
                x1, y1, x2, y2 = map(int, box)  # Pixel coordinates
                x1_f, y1_f, x2_f, y2_f = box  # Float coordinates

                # Exception 1: Skip false prediction if region is black
                if np.all(image[y1, x1] == 0) or np.all(image[y2 - 1, x2 - 1] == 0):
                    continue
                
                # Calculate image area
                image_area = image.shape[0] * image.shape[1]

                # Calculate bounding box area
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height

                # Exception 2: Skip if box area is greater than 90% of the image area
                box_ratio = box_area / image_area                    
                if box_ratio >= 0.9:
                    continue
                
                # YOLO style object detection output
                x_center = (x1_f + x2_f) / 2 / image.shape[1]
                y_center = (y1_f + y2_f) / 2 / image.shape[0]
                width = box_width / image.shape[1]
                height = box_height / image.shape[0]
                
                txt_line = f'{class_id} {conf:.4f} {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}\n'
                label_file.write(txt_line)

                # 바운딩 박스 그리기
                cv2.rectangle(image_canvas, (x1, y1), (x2, y2), color, 1)
                # 라벨 그리기
                cv2.putText(image_canvas, f"building ({scores[i]:2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imwrite(save_image_path, image_canvas)

        return save_label_path

    #########################################################################################################
    # main        
    #########################################################################################################
    def _process(self, image1_path: str, image2_path: str) -> None:
        """Function to process an image pair."""
        # Get image pair
        image_pair = ImagePair(image1_path, image2_path)
        image1_rectified, image2_rectified, homography_12 = image_pair.rectify()

        initial_gd_output = self._grounding_dino(image1_rectified, image1_path, "initial")
        new_gd_output = self._grounding_dino(image2_rectified, image2_path, "new")

        # Resize images for embedding
        image1_sam = cv2.resize(image1_rectified, (self.res_sam, self.res_sam)) 
        image2_sam = cv2.resize(image2_rectified, (self.res_sam, self.res_sam))

        # Set images to predictors to generate embeddings
        self.predictor1.set_image(image1_sam)
        self.predictor2.set_image(image2_sam)
        
        embed1 = self.predictor1.features.squeeze().cpu().numpy()
        embed2 = self.predictor2.features.squeeze().cpu().numpy()

        height, width, _ = image1_rectified.shape
        
        debut_boxes = self._extract_boxes(new_gd_output, width, height)
        retirement_boxes = self._extract_boxes(initial_gd_output, width, height)

        cossim_debut, cossim_map_debut = box_cosine_similarity(embed1, embed2, debut_boxes, width, height, self.res_embed)
        cossim_retirement, cossim_map_retirement = box_cosine_similarity(embed1, embed2, retirement_boxes, width, height, self.res_embed)

        debut_masks, debut_boxes = self._debut_detection(debut_boxes, cossim_debut, width, height)
        retirement_masks, retirement_boxes = self._retirement_detection(retirement_boxes, cossim_retirement, width, height)

        debut_mask_images = map(lambda masks: cv2.resize(masks[0].astype(np.uint8), (width, height)), debut_masks)
        retirement_mask_images = map(lambda masks: cv2.resize(masks[0].astype(np.uint8), (width, height)), retirement_masks)

        debut_change_mask = np.zeros((height, width), dtype=np.uint8)
        retirement_change_mask = np.zeros((height, width), dtype=np.uint8)

        for mask_image in debut_mask_images:    
            debut_change_mask = np.bitwise_or(debut_change_mask, mask_image)
        for mask_image in retirement_mask_images:
            retirement_change_mask = np.bitwise_or(retirement_change_mask, mask_image)
            
        retirement_change_mask_warped = cv2.warpPerspective(retirement_change_mask, homography_12, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=0) 

        change_mask = np.bitwise_or(debut_change_mask, retirement_change_mask_warped)
        
        # Save masks for visualization
        change_mask = np.where(change_mask >= 1, 255, 0).astype(np.uint8)        
        
        cv2.imwrite(os.path.join(self.output_dir, os.path.basename(image1_path)), change_mask)
    
    def process_all(self) -> None:
        """Function to iterate through all image files and process them."""
        for idx, (image1_path, image2_path) in enumerate(zip(self.initial_paths, self.new_paths)):            
            logging.info(f"Processing: {idx}/{len(self.new_paths)}")

            self._process(image1_path, image2_path)

    
def main() -> None:
    """Main function to start the Grounded_CD processing."""
    CD_config_path = "./config/CD-config.yaml"
    satel_config_path = "./config/satellite-config.yaml"

    grounded_CD = Grounded_CD(CD_config_path)

    satel_config = load_config(satel_config_path)
    basic_res = satel_config["basic_res"]
    times = satel_config["times"]

    for time in times:
        data_scale = f"data_x{basic_res*time}"        
        grounded_CD.set_data_scale(data_scale)

        grounded_CD.process_all()
    

if __name__ == "__main__":
    main()
