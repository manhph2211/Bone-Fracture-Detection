Bone Fracture Detection :smile:
=====

# 0. Introduction

# 1. Methods

`Clip = 5` and `window_size=(7,7)` 

<img src="https://github.com/manhph2211/Bone-Fracture-Detection/blob/main/imgrm/5_7.png" width="400" height="200">

# 2. Result

| Test Case | Segmentator | Detector    | Clip | Window Size | mAP(before) | mAP(after) |
|-----------|-------------|-------------|------|-------------|-------------|------------|
| 1         | Mask RCNN 0.934| Faster RCNN | 5    | 7x7      | 0.7         | 0.523      |
| 2         | Mask RCNN 0.934| Faster RCNN | 4    | 20x20    | 0.7         | 0.678      |
| 3         | Mask RCNN 0.934| Faster RCNN | 150  | 40x40    | 0.7         | 0.430      |
| 4         | Mask RCNN 0.934| Faster RCNN | 150  | 100x100  | 0.7         | 0.520      |
| 5         |             |             |      |             |             |            |
| 6         |             |             |      |             |             |            |
| 7         |             |             |      |             |             |            |
| 8         |             |             |      |             |             |            |
| 9         |             |             |      |             |             |            |
| 10        |             |             |      |             |             |            |
