
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()
_C.EXP_NAME = ''
_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.DATA_DIR = ''
_C.DISTRIBUTED = False
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0
_C.VERBOSE = True
_C.DIST_BACKEND = 'nccl'
_C.MULTIPROCESSING_DISTRIBUTED = False
_C.WITHOUT_EVAL = False
_C.WITH_DATA_AUG = False
# FP16 training params
_C.FP16 = CN()
_C.FP16.ENABLED = False
_C.FP16.STATIC_LOSS_SCALE = 1.0
_C.FP16.DYNAMIC_LOSS_SCALE = True

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'pose_hrnet'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.HRNET_PRETRAINED = ''
_C.MODEL.NUM_JOINTS = 21
_C.MODEL.TAG_PER_JOINT = True
_C.MODEL.TARGET_TYPE = 'gaussian'
_C.MODEL.INPUT_SIZE = [256, 256]  # width * height, ex: 256 * 256
_C.MODEL.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
_C.MODEL.SIGMA = 2
_C.MODEL.EXTRA = CN(new_allowed=True)
_C.MODEL.SYNC_BN = False

# for RPN
_C.MODEL.RESNET_PRETRAINED = True
_C.MODEL.RESNET_LAYERS = 50
_C.MODEL.RESNET_FIXED_BLOCKS = 3
_C.MODEL.N_ANCHORS_H = [2,6,10,14]
_C.MODEL.N_ANCHORS_W = [2,6,10,14]
_C.MODEL.ANCHOR_STRIDE = [4,4]
_C.MODEL.SIMILARITY_METRIC = 'IoU'
_C.MODEL.POSITIVE_THRESHOLD = 0.15
_C.MODEL.KL_SIGMA = 0.02
_C.MODEL.TRAINABLE_SOFTMAX = False
# for multiview_pose_hrnet
_C.MODEL.AGGRE = True
# for poseAggr
_C.MODEL.DILATION_RATES = [3, 6, 12, 18, 24]
_C.MODEL.USE_WARPING_TRAIN = True
_C.MODEL.USE_WARPING_TEST = True


# For loss functions
_C.LOSS = CN()
_C.LOSS.USE_OHKM = False
_C.LOSS.TOPK = 8
_C.LOSS.USE_TARGET_WEIGHT = True
_C.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False

_C.LOSS.WITH_HEATMAP_LOSS = False
_C.LOSS.HEATMAP_LOSS_FACTOR = 1.0

_C.LOSS.WITH_POSE2D_LOSS = False
_C.LOSS.POSE2D_LOSS_FACTOR = 1.0

_C.LOSS.WITH_POSE3D_LOSS = False
_C.LOSS.POSE3D_LOSS_FACTOR = 1.0

_C.LOSS.WITH_KL_LOSS = True
_C.LOSS.KL_LOSS_WEIGHT = 1.0

_C.LOSS.WITH_TIME_CONSISTENCY_LOSS = False
_C.LOSS.TIME_CONSISTENCY_LOSS_FACTOR = 1.0

_C.LOSS.WITH_BONE_LOSS = False
_C.LOSS.BONE_LOSS_FACTOR = 1.0

_C.LOSS.WITH_JOINTANGLE_LOSS = False
_C.LOSS.JOINTANGLE_LOSS_FACTOR = 1.0

_C.LOSS.WITH_SURROUDING_ANCHOR_LOSS = False
_C.LOSS.SURROUDING_ANCHOR_LOSS_FACTOR = 1.0

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ORIGINAL_SIZE = [512,334]
_C.DATASET.DATASET_NAME = ''
_C.DATASET.DATA_DIR = ''
_C.DATASET.BACKGROUND_DIR = ''
_C.DATASET.DATASET = []
_C.DATASET.TEST_DATASET = []
_C.DATASET.TRAIN_SET = 'training'
_C.DATASET.VAL_SET = 'val'
_C.DATASET.TEST_SET = 'evaluation'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.HYBRID_JOINTS_TYPE = ''
_C.DATASET.SELECT_DATA = False
_C.DATASET.NUM_VIEWS = 4
_C.DATASET.SEQ_IDX = [-2,-1,0,1,2]
_C.DATASET.STRIDE = 2
_C.DATASET.NUM_JOINTS = 21
_C.DATASET.INPUT_SIZE = 256
_C.DATASET.OUTPUT_SIZE = [64]
# training data augmentation
_C.DATASET.MAX_ROTATION = 30
_C.DATASET.MIN_SCALE = 0.75
_C.DATASET.MAX_SCALE = 1.25
_C.DATASET.SCALE_TYPE = 'short'
_C.DATASET.MAX_TRANSLATE = 40
_C.DATASET.FLIP = False
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 30
_C.DATASET.PROB_HALF_BODY = 0.0
_C.DATASET.NUM_JOINTS_HALF_BODY = 8
_C.DATASET.COLOR_RGB = False

# heatmap generation
_C.DATASET.SIGMA = 2
_C.DATASET.SCALE_AWARE_SIGMA = False
_C.DATASET.BASE_SIZE = 256.0
_C.DATASET.BASE_SIGMA = 2.0
_C.DATASET.INT_SIGMA = False

# Temporal information
_C.DATASET.N_FRAMES = 1
_C.DATASET.FRAME_STRIDE = 1
_C.DATASET.SAMPLE_STRIDE = 10


# train
_C.TRAIN = CN()
_C.TRAIN.TRUNCATED = True
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [3, 6]
_C.TRAIN.LR = 0.001
_C.TRAIN.LR_SCHEDULE = 'multi_step'
_C.TRAIN.WARMUP_EPOCHS = 10
_C.TRAIN.PROCESS_FEATURE_LR = 0.001
_C.TRAIN.VOLUME_NET_LR = 0.001

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.IMAGES_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.IMAGES_PER_GPU = 32
# Test Model Epoch
_C.TEST.FLIP_TEST = False
_C.TEST.POST_PROCESS = False
_C.TEST.SHIFT_HEATMAP = False

_C.TEST.USE_GT_BBOX = False

# nms
_C.TEST.IMAGE_THRE = 0.1
_C.TEST.NMS_THRE = 0.6
_C.TEST.SOFT_NMS = False
_C.TEST.OKS_THRE = 0.5
_C.TEST.IN_VIS_THRE = 0.0
_C.TEST.COCO_BBOX_FILE = ''
_C.TEST.BBOX_THRE = 1.0
_C.TEST.MODEL_FILE = ''

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    # if not os.path.exists(cfg.DATASET.ROOT):
    #     cfg.DATASET.ROOT = os.path.join(
    #         cfg.DATA_DIR, cfg.DATASET.ROOT
    #     )

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

