

_base_ = '/content/gdrive/MyDrive/Data Augmentation-XrayImg/mmdetection/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('bone',)
data = dict(
    train=dict(
        img_prefix='/content/gdrive/MyDrive/Data Augmentation-XrayImg/XRAYBoneDataset/train',
        classes=classes,
        ann_file='/content/gdrive/MyDrive/Data Augmentation-XrayImg/XRAYBoneDataset/annotations/instances_train.json'),
    val=dict(
        img_prefix='/content/gdrive/MyDrive/Data Augmentation-XrayImg/XRAYBoneDataset/val',
        classes=classes,
        ann_file='/content/gdrive/MyDrive/Data Augmentation-XrayImg/XRAYBoneDataset/annotations/instances_val.json'),
    test=dict(
        img_prefix='/content/gdrive/MyDrive/Data Augmentation-XrayImg/XRAYBoneDataset/test',
        classes=classes,
        ann_file='/content/gdrive/MyDrive/Data Augmentation-XrayImg/XRAYBoneDataset/annotations/instances_test.json'))

optimizer = dict(
  type='SGD',
  lr = 2e-4,
  momentum=0.9,
  weight_decay=0.0001
)
runner = dict(type='EpochBasedRunner', max_epochs=1)

load_from = '/content/gdrive/MyDrive/Data Augmentation-XrayImg/mmdetection/checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
