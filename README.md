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
| 4         | Deeplavv3   | x.xxx      |


# 2. Detector


## 2.1 Data Augmentation


### 2.1.1 Faster RCNN - MMdetection

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

## 2.2 Results - Large Dataset

| Test Case | Segmentator | Detector    | Clip | Window Size | mAP(before) | mAP(after) |
|-----------|-------------|-------------|------|-------------|-------------|------------|
| 1         | Yolact 0.94 | Faster RCNN | 5    | 7x7         | 0.677       |  0.70      |
| 2         | Yolact 0.94 | EfficientDet| 5    | 7x7         | x.xxx       |  x         |
| 3         | Yolact 0.94 | Yolov5      | 5    | 7x7         | x.xxx       |  x         |
|
