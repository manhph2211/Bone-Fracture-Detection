Bone Fracture Detection :smile:
=====

`Clip = 5` and `window_size=(7,7)` 

<img src="https://github.com/manhph2211/Bone-Fracture-Detection/blob/main/imgrm/5_7.png" width="600" height="300">

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
| 2         | Yolact 0.94 | EfficientDet| 5    | 7x7         | 0.637       |  0.565     |
| 3         | Yolact 0.94 | Yolov5      | 5    | 7x7         | 0.788       |  0.723     |


## 2.2 Data Augmentation

### 2.2.1 Faster RCNN - MMdetection
 
```

img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]


```

### 2.2.2 EfficientDet 

- Resize (default size = 512)
- Normalize image with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Horizontal flipping (default p = 0.5) 
- Propose in the original paper: scale jittering with the range of scale is [1.0, 2.0]

### 2.1.3 Yolov5

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

