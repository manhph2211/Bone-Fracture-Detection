Bone Fracture Detection :smile:
=====

# 0. Introduction

# 1. Methods

- `Clip = 5` and `window_size=(7,7)` 

<img src="https://github.com/manhph2211/Bone-Fracture-Detection/blob/main/imgrm/5_7.png" width="400" height="200">

# 2. Result


## 2.1 Segmentator and Detector 😢

- `Segmetator` : 

| Test Case | Segmentator | mAP(50:95) |
|-----------|-------------|------------|
| 1         | Mask RCNN   | 0.934      |
| 2         | Yolact       | 0.940      |
| 3         | SOLO        | 0.911      |
| 4         | Deeplavv3   | x.xxx      |

- `Detector for the small dataset (original from @thay): `

| Test Case | Segmentator | mAP(50)    |
|-----------|-------------|------------|
| 1         | Faster RCNN | 0.7        |
| 2         | Yolov5      | 0.61       |
| 3         | EfficentDet | 0.657      |

## 2.2 E2E 😢

- `FASTER RCNN - SMALL DATASET + ROBOFLOW`

| Test Case | Segmentator | Detector    | Clip | Window Size | mAP(before) | mAP(after) |
|-----------|-------------|-------------|------|-------------|-------------|------------|
| 1         | Mask RCNN 0.934| Faster RCNN | 5    | 7x7      | 0.633       | 0.523      |
| 2         | Mask RCNN 0.934| Faster RCNN | 5    |10x10    | 0.633       | 0.618      |
| 3         | Mask RCNN 0.934| Faster RCNN | 150  | 100x100  | 0.633       | 0.520      |
| 4         | Mask RCNN 0.934|    Faster RCNN | 5      | 10x10|              |            |
| 5         |             |             |      |             |             |            |



- `Yolov5 - SMALL DATASET `

| Test Case | Segmentator | Detector    | Clip | Window Size | mAP(before) | mAP(after) |
|-----------|-------------|-------------|------|-------------|-------------|------------|
| 1         | Yolact 0.94 | Yolov5  | 5    | 7x7      | 0.61       |  0.549     |
| 2         | Yolact 0.94 | Yolov5  | 5    | 7x7      | 0.610       | x          |
| 3         |             |             |      |             |             |            |
| 4         |             |             |      |             |             |            |
| 5         |             |             |      |             |             |            |
| 6         |             |             |      |             |             |            |
| 7         |             |             |      |             |             |            |
| 8         |             |             |      |             |             |            |
| 9         |             |             |      |             |             |            |
| 10        |             |             |      |             |             |            |

- `Efficientdet - SMALL DATASET + ROBOFLOW`

| Test Case | Segmentator | Detector    | Clip | Window Size | mAP(before) | mAP(after) |
|-----------|-------------|-------------|------|-------------|-------------|------------|
| 1         | Deeplabv3   | Efficientdet| 5    | 7x7         | 0.657       | 0.645      |
| 2         | Deeplabv3   | Efficientdet| 7    | 4x4         | 0.657       | 0.598      |
| 3         | Deeplabv3   | Efficientdet| 5    | 4           | 0.657       | 0.536      |
| 4         |             |             |      |             |             |            |
| 6         |             |             |      |             |             |            |
| 7         |             |             |      |             |             |            |
| 8         |             |             |      |             |             |            |
| 9         |             |             |      |             |             |            |
| 10        |             |             |      |             |             |            |
