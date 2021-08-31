Bone Fracture Detection :smile:
=====


# 1. Preprocessing 


- `Segmetator` : 

| Test Case | Segmentator | mAP(50:95) |
|-----------|-------------|------------|
| 1         | Mask RCNN   | 0.934      |
| 2         | Yolact      | 0.940      |
| 3         | SOLO        | 0.911      |
| 4         | Deeplabv3   | x.xxx      |


# 2. Detector


## 2.1 Results - Large Dataset

| Test Case | Segmentator | Detector    | Clip | Window Size | mAP(before) | mAP(after) |
|-----------|-------------|-------------|------|-------------|-------------|------------|
| 1         | Yolact 0.94 | Faster RCNN | 5    | 7x7         | 0.677       |  0.70      |
| 2         | Yolact 0.94 | EfficientDet| 5    | 7x7         | 0.637       |  0.6xx     |
| 3         | Yolact 0.94 | Yolov5      | 5    | 7x7         | 0.788       |  0.7xx    |


## 2.2 Data Augmentation

### 2.2.1 EfficientDet 

- Resize (default size = 512)
- Normalize image with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Horizontal flipping (default p = 0.5) 
- Propose in the original paper: scale jittering with the range of scale is [1.0, 2.0]

### 2.1.2 Yolov5

- Photometric Distortion:
    - image HSV-Hue augmentation
    - image HSV-Saturation augmentation
    - image HSV-Value augmentation
- Geometric Distortion: 
    - image rotation 
    - image translation
    - image scale
    - image sheer
    - image perspective
    - image flip up-down, left-right
- Mosaic Data Augmentation
- MixUp

You can get more information about the hyperparameters [here](https://github.com/ultralytics/yolov5/issues/607).

