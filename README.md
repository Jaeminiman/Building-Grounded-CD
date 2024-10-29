# Building-Grounded-CD
Grounding dino + SAM based Building Change Detection 

- 알고리즘명: Grounded-CD (Grounded SAM-based Change Detection)
- 구현 환경
    - Ubuntu 22.04
    - Cuda 11.8
- 프로젝트 구조
    - grounded-cd
        - config
            - CD-config.yaml
            - satellite-config.yaml
        - data
            - raw_data
                - K3A_20151028.tif
                - K3A_20231006.tif
        - outputs // 결과 파일
        - utils
            - CD_utils.py
            - image_pair.py
            - object_detection_utils.py
        - [satelliteImageProcess.py](http://satelliteImageProcess.py) // 실행파일 1
        - [Grounded-CD.py](http://Grounded-CD.py) // 실행파일 2
        - output_merger.py // 실행파일 3
- 실행 순서
    1. 환경 세팅
        - gdal 환경 생성
            
            ```bash
            conda create -n gdal_env python=3.13
            conda activate gdal_env
            pip install altgraph==0.17.4 geojson==3.1.0 natsort==8.4.0 \
            opencv-python==4.10.0.84 osmnx==1.9.3 \
            pyyaml==6.0.2 tqdm==4.66.5
            ```
            
        - grounded-cd 환경 생성
            
            ```bash
            
            conda create -n grounded-cd python=3.8 # SAM/mmdetection 버전 일치를 위함
            conda activate grounded-cd
            
            1. mmdetection 설치
            # https://mmdetection.readthedocs.io/en/latest/get_started.html
            pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
            pip install -U openmim
            mim install mmengine
            pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
            
            git clone https://github.com/open-mmlab/mmdetection.git
            cd mmdetection
            pip install -v -e .
            
            2. mmgroundingdino 설치
            # https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/usage.md
            # cd mmdetection
            pip install -r requirements/multimodal.txt
            pip install emoji ddd-dataset
            pip install git+https://github.com/lvis-dataset/lvis-api.git
            
            python3
            > import nltk
            > nltk.download('punkt', download_dir='~/nltk_data')
            > nltk.download('punkt_tab')
            > nltk.download('averaged_perceptron_tagger', download_dir='~/nltk_data')
            > nltk.download('averaged_perceptron_tagger_eng')
            
            3. SAM 설치
            # https://github.com/facebookresearch/segment-anything
            pip install git+https://github.com/facebookresearch/segment-anything.git
            pip install opencv-python pycocotools matplotlib onnxruntime onnx
            ```
            
    2. 위성 영상(.tif) 전처리
        
        ```bash
        conda activate gdal_env
        python satelliteImageProcess.py
        ```
        
        - 의존 파일
            - .config/satellite- config.yaml
        - 주의 사항
            - gdal 관련 라이브러리 오류 발생 가능
                1. GDAL System Dependencies 설치
                    - sudo apt-get update
                    - sudo apt-get install -y gdal-bin libgdal-dev
                2. GDAL Python Binding 설치
                    - export CPLUS_INCLUDE_PATH=/usr/include/gdal
                    - export C_INCLUDE_PATH=/usr/include/gdal
                    - conda create -n gdal_env
                    - pip install gdal
                3. (안되는 경우) 
                    - conda install -c conda-forge gdal libgdal
    3. Grounded-CD 실행
        
        ```jsx
        conda activate grounded-cd
        python Grounded-CD.py
        ```
        
        - 의존 파일
            - ./config/CD-config.yaml
            - ./config/satellite- config.yaml
        - 주의 사항
            - gdal과 라이브러리 버전 충돌 발생 가능하므로 다른 conda 환경에서 수행
    4. 추정 결과 정합 및 tif 변환
        
        ```bash
        conda activate gdal_env
        python output_merger.py
        ```
        
        - 의존 파일
            - ./config/CD-config.yaml
            - ./config/satellite- config.yaml
- 최종 결과

![알고리즘 추정 결과](https://prod-files-secure.s3.us-west-2.amazonaws.com/5e95ede8-38f4-4487-bb35-63d551c37ae7/73ed00bd-2d37-41c5-ba3b-fba8a08017cf/merged_image.jpg)

알고리즘 추정 결과

![Ground truth](https://prod-files-secure.s3.us-west-2.amazonaws.com/5e95ede8-38f4-4487-bb35-63d551c37ae7/8d1620b0-3488-40df-9771-281528bed99a/validation.jpg)

Ground truth
