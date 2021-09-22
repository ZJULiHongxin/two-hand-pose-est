# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from cv2 import normalize

import torch.utils.data
import torchvision.transforms as transforms
from torchvision.transforms.transforms import Normalize
from .InterHandDataset import InterHandDataset

trans = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])]
                )
def build_dataset(cfg, is_train):

    if is_train:
        subset = cfg.DATASET.TRAIN_SET # 'training'
    else:
        subset = cfg.DATASET.VAL_SET  # 'evaluation'

    # pack all datasets in a dict for the convenience of joint dataset training

    dataset = eval(cfg.DATASET.DATASET_NAME)(
                                        cfg=cfg,
                                        transforms=trans,
                                        mode = subset
                                        )

    return dataset


def make_dataloader(cfg, is_train=True, distributed=False):
    if is_train:
        images_per_gpu = cfg.TRAIN.IMAGES_PER_GPU
        shuffle = True
    else:
        images_per_gpu = cfg.TEST.IMAGES_PER_GPU
        shuffle = False
    
    dataset = build_dataset(cfg, is_train)

    if is_train and distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset
        )
        shuffle = False
        images_per_batch = images_per_gpu
    else:
        images_per_batch = images_per_gpu * len(cfg.GPUS)
        train_sampler = None
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=images_per_batch,
        shuffle=shuffle,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        sampler=train_sampler
    )

    return data_loader


def make_test_dataloader(cfg):
    transforms = build_transforms(cfg, is_train=False)

    if cfg.DATASET.SCALE_AWARE_SIGMA:
        _HeatmapGenerator = ScaleAwareHeatmapGenerator
    else:
        _HeatmapGenerator = HeatmapGenerator

    heatmap_generator = [
        _HeatmapGenerator(
            output_size, cfg.DATASET.NUM_JOINTS, cfg.DATASET.SIGMA
        ) for output_size in cfg.DATASET.OUTPUT_SIZE
    ]

    dataset = eval(cfg.DATASET.DATASET)(
        cfg,
        cfg.DATASET.TEST_SET,
        heatmap_generator[0],
        transforms
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return data_loader, dataset
# def build_dataset(cfg, is_train):
#     transforms = build_transforms(cfg, is_train)

#     if cfg.DATASET.SCALE_AWARE_SIGMA:
#         _HeatmapGenerator = ScaleAwareHeatmapGenerator
#     else:
#         _HeatmapGenerator = HeatmapGenerator

#     heatmap_generator = [
#         _HeatmapGenerator(
#             output_size, cfg.DATASET.NUM_JOINTS * cfg.DATASET.N_FRAMES, cfg.DATASET.SIGMA
#         ) for output_size in cfg.DATASET.OUTPUT_SIZE #[64]
#     ]

#     dataset_name = cfg.DATASET.TRAIN_SET if is_train else cfg.DATASET.TEST_SET

#     dataset = eval(cfg.DATASET.DATASET)(
#         cfg,
#         dataset_name,
#         heatmap_generator[0],
#         transforms
#     )

#     return dataset


# def make_dataloader(cfg, is_train=True, distributed=False):
#     if is_train:
#         images_per_gpu = cfg.TRAIN.IMAGES_PER_GPU
#         shuffle = True
#     else:
#         images_per_gpu = cfg.TEST.IMAGES_PER_GPU
#         shuffle = False
    

#     dataset = build_dataset(cfg, is_train)

#     if is_train and distributed:
#         train_sampler = torch.utils.data.distributed.DistributedSampler(
#             dataset
#         )
#         shuffle = False
#         images_per_batch = images_per_gpu
#     else:
#         images_per_batch = images_per_gpu * len(cfg.GPUS)
#         train_sampler = None

#     data_loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=images_per_batch,
#         shuffle=shuffle,
#         num_workers=cfg.WORKERS,
#         pin_memory=cfg.PIN_MEMORY,
#         sampler=train_sampler
#     )

#     return data_loader


# def make_test_dataloader(cfg):
#     transforms = build_transforms(cfg, is_train=False)

#     if cfg.DATASET.SCALE_AWARE_SIGMA:
#         _HeatmapGenerator = ScaleAwareHeatmapGenerator
#     else:
#         _HeatmapGenerator = HeatmapGenerator

#     heatmap_generator = [
#         _HeatmapGenerator(
#             output_size, cfg.DATASET.NUM_JOINTS, cfg.DATASET.SIGMA
#         ) for output_size in cfg.DATASET.OUTPUT_SIZE
#     ]

#     dataset = eval(cfg.DATASET.DATASET)(
#         cfg,
#         cfg.DATASET.TEST_SET,
#         heatmap_generator[0],
#         transforms
#     )

#     data_loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=1,
#         shuffle=False,
#         num_workers=0,
#         pin_memory=False
#     )

#     return data_loader, dataset
