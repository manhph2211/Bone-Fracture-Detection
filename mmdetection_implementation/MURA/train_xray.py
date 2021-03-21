# 2. Some Config

_base_ = '/content/gdrive/MyDrive/Data Augmentation-XrayImg/mmdetection/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('fracture',)
data = dict(
    train=dict(
        img_prefix='/content/gdrive/MyDrive/Data Augmentation-XrayImg/original/dataDetect_train/',
        classes=classes,
        ann_file='/content/gdrive/MyDrive/Data Augmentation-XrayImg/original/dataDetect_train/train_data_coco.json'),
    val=dict(
        img_prefix='/content/gdrive/MyDrive/Data Augmentation-XrayImg/original/dataDetect_test',
        classes=classes,
        ann_file='/content/gdrive/MyDrive/Data Augmentation-XrayImg/original/dataDetect_test/test_data_coco.json'),
   )

load_from = '/content/gdrive/MyDrive/Data Augmentation-XrayImg/mmdetection/checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
