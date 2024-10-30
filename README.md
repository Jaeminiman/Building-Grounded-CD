# Grounded-CD: Grounded Building Change Detection

Grounded-CD leverages **Grounding DINO** and **Segment Anything Model (SAM)** for effective building change detection from satellite images.

## Overview

- **Algorithm Name**: Grounded-CD (Grounded SAM-based Change Detection)
- **Platform**: Ubuntu 22.04
- **CUDA Version**: 11.8

## Project Structure

```plaintext
grounded-cd/
├── config/
│   ├── CD-config.yaml
│   └── satellite-config.yaml
├── data/
│   └── raw_data/
│       ├── K3A_20151028.tif
│       └── K3A_20231006.tif
├── outputs/  # Result files
├── utils/
│   ├── CD_utils.py
│   ├── image_pair.py
│   └── object_detection_utils.py
├── satelliteImageProcess.py  # Execution file 1
├── Grounded-CD.py            # Execution file 2
└── output_merger.py          # Execution file 3
```
- **Checkpoint download**:
```bash
  # Check point 다운 커맨드를 실행하여 다운로드하고 check point에 넣어주세요.
  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  wget https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_all/grounding_dino_swin-l_pretrain_all-56d69e78.pth

  # CD-config.yaml에서 경로를 수정해주세요.
  grounding_dino_config: "~/mmdetection/configs/mm_grounding_dino/grounding_dino_swin-l_pretrain_all.py" # git clone을 수행한 mmdetection의 소스코드 파일 경로로 설정
```
## Setup and Execution Steps

### 1. Environment Setup

#### GDAL Environment
Create a GDAL environment and install dependencies:

```bash
conda create -n gdal_env python=3.13
conda activate gdal_env
pip install altgraph==0.17.4 geojson==3.1.0 natsort==8.4.0 \
    opencv-python==4.10.0.84 osmnx==1.9.3 \
    pyyaml==6.0.2 tqdm==4.66.5
```

#### Grounded-CD Environment
Create another environment for Grounded-CD:  


```bash
conda create -n grounded-cd python=3.8
conda activate grounded-cd
```

1. **Install MMDetection**
   
   Follow the [MMDetection documentation](https://mmdetection.readthedocs.io/en/latest/get_started.html):

   ```bash
   pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
   pip install -U openmim
   mim install mmengine
   pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
   
   git clone https://github.com/open-mmlab/mmdetection.git
   cd mmdetection
   pip install -v -e .
   ```

2. **Install MMGroundingDino**

   ```bash
   pip install -r requirements/multimodal.txt
   pip install emoji ddd-dataset
   pip install git+https://github.com/lvis-dataset/lvis-api.git

   python3
   > import nltk
   > nltk.download('punkt', download_dir='~/nltk_data')
   > nltk.download('punkt_tab')
   > nltk.download('averaged_perceptron_tagger', download_dir='~/nltk_data')
   > nltk.download('averaged_perceptron_tagger_eng')
   ```

3. **Install SAM**

   Follow the [SAM GitHub repository](https://github.com/facebookresearch/segment-anything):

   ```bash
   pip install git+https://github.com/facebookresearch/segment-anything.git
   pip install opencv-python pycocotools matplotlib onnxruntime onnx
   ```

### 2. Satellite Image Preprocessing

Activate the GDAL environment and run the preprocessing script:

```bash
conda activate gdal_env
python satelliteImageProcess.py
```

- **Dependencies**: `./config/satellite-config.yaml`
- **Notes**: 
  - Possible GDAL-related errors may occur. If so, follow these steps:
    1. **Install GDAL System Dependencies**
       ```bash
       sudo apt-get update
       sudo apt-get install -y gdal-bin libgdal-dev
       ```
    2. **Install GDAL Python Binding**
       ```bash
       export CPLUS_INCLUDE_PATH=/usr/include/gdal
       export C_INCLUDE_PATH=/usr/include/gdal
       conda create -n gdal_env
       pip install gdal
       ```
    3. **Alternative Installation**
       ```bash
       conda install -c conda-forge gdal libgdal
       ```

### 3. Execute Grounded-CD

Activate the Grounded-CD environment and run the script:

```bash
conda activate grounded-cd
python Grounded-CD.py
```

- **Dependencies**:
  - `./config/CD-config.yaml`
  - `./config/satellite-config.yaml`
- **Notes**: 
  - Use a separate conda environment to avoid GDAL version conflicts.
  - Please check if you corrected the path of `grounding_dino_config` in `./config/CD-config.yaml`

### 4. Merge Results and Convert to TIF

Activate the GDAL environment and run the output merger script:

```bash
conda activate gdal_env
python output_merger.py
```

- **Dependencies**:
  - `./config/CD-config.yaml`
  - `./config/satellite-config.yaml`

## Final Results

### Algorithm Inference Results

<div style="display: flex; justify-content: space-between;">
  <div style="flex: 1; text-align: center;">
    <img src="https://github.com/user-attachments/assets/e139332a-4f05-4343-bab0-57e752964c0d" alt="Prediction Result" width="30%">
    <p>Prediction</p>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="https://github.com/user-attachments/assets/d92c01e7-de44-4873-9240-01fcd5890e75" alt="Ground Truth Result" width="30%">
    <p>Ground Truth</p>
  </div>
</div>

---




