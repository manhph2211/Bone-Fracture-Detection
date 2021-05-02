Bone Fracture Detection :smile:
=====

# 0. Introduction

# 1. Methods

 `Clip = 5` and `window_size=(7,7)` 

<img src="https://github.com/manhph2211/Bone-Fracture-Detection/blob/main/imgrm/5_7.png" width="400" height="200">

# 2. Result

- `FASTER RCNN - SMALL DATASET + ROBOFLOW`
| Test Case | Segmentator | Detector    | Clip | Window Size | mAP(before) | mAP(after) |
|-----------|-------------|-------------|------|-------------|-------------|------------|
| 1         | Mask RCNN 0.934| Faster RCNN | 5    | 7x7      | 0.633       | 0.523      |
| 3         | Mask RCNN 0.934| Faster RCNN | 150  | 40x40    | 0.633       | 0.430      |
| 4         | Mask RCNN 0.934| Faster RCNN | 150  | 100x100  | 0.633       | 0.520      |
| 5         |             |             |      |             |             |            |
| 6         |             |             |      |             |             |            |
| 7         |             |             |      |             |             |            |
| 8         |             |             |      |             |             |            |
| 9         |             |             |      |             |             |            |
| 10        |             |             |      |             |             |            |


- `Yolov5 - SMALL DATASET + ROBOFLOW`
| Test Case | Segmentator | Detector    | Clip | Window Size | mAP(before) | mAP(after) |
|-----------|-------------|-------------|------|-------------|-------------|------------|
| 1         | Yolax 0.94  | Yolov5  | 5    | 7x7      | 0.633       | 0.523      |
| 2         | Yolax 0.94  | Yolov5  | 5    | 7x7      | 0.61        | ?      |
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
| 1         | Deeplabv3   | Efficientdet  | 5    | 7x7         | 0.657       | 0.645      |
| 2         | Deeplabv3   | Efficientdet  | 7    | 4x4         | 0.657       | 0.598      |
| 3         | Deeplabv3   | Efficientdet | 5    | 4           | 0.657       | 0.536      |
| 4         |             |             |      |             |             |            |
| 6         |             |             |      |             |             |            |
| 7         |             |             |      |             |             |            |
| 8         |             |             |      |             |             |            |
| 9         |             |             |      |             |             |            |
| 10        |             |             |      |             |             |            |
