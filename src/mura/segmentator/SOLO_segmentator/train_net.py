# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.utils.events import EventStorage
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
    SemSegEvaluator
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger

from adet.data.dataset_mapper import DatasetMapperWithBasis
from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer
from adet.evaluation import TextEvaluator
from detectron2.data.datasets import register_coco_instances



class Trainer(DefaultTrainer):
    """
    This is the same Trainer except that we rewrite the
    `build_train_loader`/`resume_or_load` method.
    """
    def resume_or_load(self, resume=True):
        if not isinstance(self.checkpointer, AdetCheckpointer):
            # support loading a few other backbones
            self.checkpointer = AdetCheckpointer(
                self.model,
                self.cfg.OUTPUT_DIR,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
        super().resume_or_load(resume=resume)

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger("adet.trainer")
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
            self.after_train()

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        mapper = DatasetMapperWithBasis(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        # evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        #     evaluator_list.append(
        #         SemSegEvaluator(
        #             dataset_name,
        #             distributed=True,
        #             num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
        #             ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
        #             output_dir=output_folder,
        #         )
        #     )
        
        # return SemSegEvaluator(
        #             dataset_name,
        #             distributed=True,
        #             num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
        #             ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
        #             output_dir=output_folder,
        #         )
        # if evaluator_type in ["coco", "coco_panoptic_seg"]:
        #     evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        # if evaluator_type == "coco_panoptic_seg":
        #     evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # if evaluator_type == "pascal_voc":
        #     return PascalVOCDetectionEvaluator(dataset_name)
        # if evaluator_type == "lvis":
        #     return LVISEvaluator(dataset_name, cfg, True, output_folder)
        # if evaluator_type == "text":
        #     return TextEvaluator(dataset_name, cfg, True, output_folder)
        # if len(evaluator_list) == 0:
        #     raise NotImplementedError(
        #         "no Evaluator for the dataset {} with the type {}".format(
        #             dataset_name, evaluator_type
        #         )
        #     )
        # if len(evaluator_list) == 1:
        #     return evaluator_list[0]
        # return DatasetEvaluators(evaluator_list)
        return COCOEvaluator(dataset_name, ('bbox', 'segm'), False, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("adet.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    
    cfg = get_cfg()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet")

    # cfg.merge_from_file('/content/drive/MyDrive/Solo/AdelaiDet/configs/SOLOv2/R50_3x.yaml')
    cfg.DATASETS.TRAIN = ("train_data",)
    cfg.DATASETS.TEST = ("val_data",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.OUTPUT_DIR = 'training_dir/SOLOv2_R50_3x'
    # cfg.merge_from_list(["MODEL.WEIGHTS",'/content/drive/MyDrive/Solo/pretrained/SOLOv2_R50_3x.pth'])
    cfg.MODEL.WEIGHTS = '/content/drive/MyDrive/Solo/AdelaiDet/training_dir/SOLOv2_R50_3x/model_final.pth'  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 3   # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2

    return cfg


def main(args):
    cfg = setup(args)    

    if args.eval_only:

        model = Trainer.build_model(cfg)
        AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model) # d2 defaults.py
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    register_coco_instances("train_data", {}, "/content/drive/MyDrive/Solo/AdelaiDet/datasets/xray/annotations/instances_train.json", "/content/drive/MyDrive/Solo/AdelaiDet/datasets/xray/train")
    register_coco_instances("val_data", {}, "/content/drive/MyDrive/Solo/AdelaiDet/datasets/xray/annotations/instances_val.json", "/content/drive/MyDrive/Solo/AdelaiDet/datasets/xray/val")
    register_coco_instances("test_data", {}, "/content/drive/MyDrive/Solo/AdelaiDet/datasets/xray/annotations/instances_test.json", "/content/drive/MyDrive/Solo/AdelaiDet/datasets/xray/test")
    MetadataCatalog.get("train_data").thing_classes = ["_background_", "bone"]
    MetadataCatalog.get("val_data").thing_classes = ["_background_", "bone"]
    MetadataCatalog.get("test_data").thing_classes = ["_background_", "bone"]

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
