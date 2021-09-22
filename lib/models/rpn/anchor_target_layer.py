from __future__ import absolute_import
from numpy.core.fromnumeric import argmax
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr
import cv2
import matplotlib.pyplot as plt
from .generate_anchors import generate_anchor_poses
from .bbox_transform import clip_boxes, anchors_transform_batch, pose_OKS_batch, pose_IoU_batch
from utils.vis import plot_hand

ref_pose = np.array([
    [213.33335876464844,124.50563049316406], # palm
    [190.504638671875,115.11840057373047],  # thumb
    [169.9791717529297,101.77180480957031],
    [146.72341918945312,96.25749206542969],
    [128.86770629882812,87.2344970703125], # thumb tip
    [150.34292602539062,101.61070251464844], # index
    [119.29926300048828,98.73982238769531],
    [100.03463745117188,99.74459838867188],
    [82.62400817871094,101.2509536743164],
    [148.91049194335938,112.71517181396484], # middle
    [114.37303161621094,113.20121002197266],
    [91.90096282958984,116.49812316894531],
    [74.75020599365234,119.37875366210938],
    [149.59658813476562,124.09295654296875], # ring
    [119.72419738769531,126.36898040771484],
    [99.59107208251953,129.40196228027344],
    [82.82524108886719,131.584228515625],
    [154.55911254882812,135.07681274414062], # pinky
    [133.8833770751953,140.85983276367188],
    [120.45906066894531,145.40306091308594],
    [106.21541595458984,150.072265625],
    ])

class _AnchorTargetLayer(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """
    def __init__(self, scales, orientations, cfg):
        super(_AnchorTargetLayer, self).__init__()
        self.cfg = cfg
        self.similarity_metric = self.cfg.MODEL.SIMILARITY_METRIC
        self._scales = scales

        self._anchors = torch.from_numpy(generate_anchor_poses(ref_pose, scales=scales, orientations=orientations)).float() # n_anchors x 21 x 2
        self._num_anchors = self._anchors.size(0)

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = 0  # default is 0
        self.first_flag = False

        self.accu_left_lables = 0
        self.accu_right_lables = 0
        self.accu_neg_lables = 0

    def forward(self, pose_gt, hand_type, featmap_shape):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors

        # pose_gt: b x 42 x 3 [u,v,vis]
        # hand_type: b x 2
        pose_gt = pose_gt.cpu()
        batch_size = pose_gt.size(0)
        
        # implemented only once
        if self.first_flag == False:
            self.first_flag = True
            feat_height, feat_width = featmap_shape[2:]

            width_ratio, height_ratio = self.cfg.MODEL.INPUT_SIZE[1] / feat_width,  self.cfg.MODEL.INPUT_SIZE[0] / feat_height
            feat_x_stride, feat_y_stride = self.cfg.MODEL.ANCHOR_STRIDE[1], self.cfg.MODEL.ANCHOR_STRIDE[0]
            shift_x = np.arange(0, feat_width, feat_x_stride) * width_ratio
            shift_y = np.arange(0, feat_height, feat_y_stride) * height_ratio
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel())).transpose())
            
            #     an example of shifts
            #     u  v
            #   [[0, 0],
            #    [4, 0],
            #    [8, 0],
            #    [0, 4],
            #    [4, 4],
            #    [8, 4],
            #    [0, 8],
            #    [4, 8],
            #    [8, 8]])
            shifts = shifts.contiguous().type_as(self._anchors)# Feat_H*Feat_W x 2

            A = self._num_anchors
            K = shifts.size(0)

            self._anchors = self._anchors.type_as(pose_gt) # move to specific gpu.
            all_anchors = self._anchors.view(1, *self._anchors.shape) + shifts.view(K, 1, 1, 2) # K x A x 21 x 2

            # f = plt.figure()
            # ax1 =f.add_subplot(1,1,1)
            # for k in range(K):
            #     plot_hand(ax1, all_anchors[k,0].numpy(), order='uv')
            # plot_hand(ax1,pose_gt[0,0:21,0:2].numpy(), order='uv')
            # plot_hand(ax1,pose_gt[0,21:42,0:2].numpy(), order='uv')
            # plt.show()
            
            all_anchors = all_anchors.view(-1,*all_anchors.shape[2:]) # K*A x 21 x 2

            self.total_anchors = all_anchors.shape[0] # K * A

            # During training, we ignore all cross-boundary anchors so they do not contribute to the loss
            keep = ((all_anchors[:, :, 0] >= -self._allowed_border) &
                    (all_anchors[:, :, 1] >= -self._allowed_border) &
                    (all_anchors[:, :, 0] < self.cfg.MODEL.INPUT_SIZE[0] + self._allowed_border) &
                    (all_anchors[:, :, 1] < self.cfg.MODEL.INPUT_SIZE[1] + self._allowed_border)) # K * A x 21

            # keep those anchors with all joints located inside the image
            
            self.inds_inside = keep.sum(1) == 21 # a boolean tensor of size K * A

            # keep only inside anchors
            self.anchors = all_anchors[self.inds_inside, :] # <K*A x 21 x 2
            self.all_anchors = all_anchors
            self.n_valid_anchors = self.anchors.shape[0]

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = pose_gt.new(batch_size, self.n_valid_anchors, 3).fill_(0) # b x <K*A x 3
        pose_targets = pose_gt.new(batch_size, self.n_valid_anchors, 21, 3).fill_(0) # b x <K*A x 21 x 3 [u,v,vis]

        if self.similarity_metric == 'OKS':
            Sim = pose_OKS_batch(self.anchors, pose_gt.view(batch_size, 2, 21, 3)) # b x <(K*A) x 2 (<(K*A) means there are less than K*A anchors, and 2 the two hands)
        elif self.similarity_metric == 'IoU':
            Sim = pose_IoU_batch(self.anchors, pose_gt.view(batch_size, 2, 21, 3))

        print('Sim metric:',Sim.max())
        #overlaps = bbox_overlaps_batch(anchors, gt_boxes)
        idx_left, idx_right, idx_negative = [], [], []

        for b in range(batch_size):
            for anchor_id in range(self.n_valid_anchors):
                argmax_sim = torch.argmax(Sim[b, anchor_id])
                max_sim = Sim[b, anchor_id, argmax_sim]
                # negative: [0,1,0], left_hand: [1,0,0], right_hand: [0,0,1]
                # if there is a right hand in the img and the anchor is similar to it
                if argmax_sim == 0 and max_sim > self.cfg.MODEL.POSITIVE_THRESHOLD and hand_type[b, 0] == 1:
                    labels[b, anchor_id, 2] = 1
                    pose_targets[b, anchor_id] = pose_gt[b, 0:21]
                    idx_right.append(anchor_id)
                # if there is a left hand in the img and the anchor is similar to it
                elif argmax_sim == 1 and max_sim > self.cfg.MODEL.POSITIVE_THRESHOLD and hand_type[b, 1] == 1:
                    labels[b, anchor_id, 0] = 1
                    pose_targets[b, anchor_id] = pose_gt[b, 21:42]
                    idx_left.append(anchor_id)
                else:
                    labels[b, anchor_id, 1] = 1
                    idx_negative.append(anchor_id)

        self.accu_left_lables += len(idx_left)
        self.accu_right_lables += len(idx_right)
        self.accu_neg_lables += len(idx_negative)

        print('[Anchor Label Statistics] left:{}, right:{}, neg:{}'.format(self.accu_left_lables, self.accu_right_lables, self.accu_neg_lables))
        # balance the number of the three categories

#         for i in range(batch_size):
#             # subsample positive labels if we have too many
#             if sum_fg[i] > num_fg:
#                 fg_inds = torch.nonzero(labels[i] == 1).view(-1)
#                 # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
#                 # See https://github.com/pytorch/pytorch/issues/1868 for more details.
#                 # use numpy instead.
#                 #rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
#                 rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_boxes).long()
#                 disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
#                 labels[i][disable_inds] = -1

# #           num_bg = cfg.TRAIN.RPN_BATCHSIZE - sum_fg[i]
#             num_bg = cfg.TRAIN.RPN_BATCHSIZE - torch.sum((labels == 1).int(), 1)[i]

#             # subsample negative labels if we have too many
#             if sum_bg[i] > num_bg:
#                 bg_inds = torch.nonzero(labels[i] == 0).view(-1)
#                 #rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes).long()

#                 rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_boxes).long()
#                 disable_inds = bg_inds[rand_num[:bg_inds.size(0)-num_bg]]
#                 labels[i][disable_inds] = -1

        # offset = torch.arange(0, batch_size)*gt_boxes.size(1)

        # argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)
        # bbox_targets = _compute_targets_batch(anchors, gt_boxes.view(-1,5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5))

        # # use a single value instead of 4 values for easy index.
        # bbox_inside_weights[labels==1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]

        # if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
        #     num_examples = torch.sum(labels[i] >= 0)
        #     positive_weights = 1.0 / num_examples.item()
        #     negative_weights = 1.0 / num_examples.item()
        # else:
        #     assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
        #             (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))

        # bbox_outside_weights[labels == 1] = positive_weights
        # bbox_outside_weights[labels == 0] = negative_weights

        labels = _unmap(labels, self.total_anchors, self.inds_inside, batch_size, fill=-1) # b x K*A x 3
        offsets = anchors_transform_batch(self.anchors, pose_targets[:,:,:,0:2]) # # b x <K*A x 42
        offsets = _unmap(offsets, self.total_anchors, self.inds_inside, batch_size, fill=0) #  b x K*A x 42
        # bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, batch_size, fill=0)
        # bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0)
        # bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0)

        #outputs = []

        #labels = labels.view(batch_size, height, width, A, 3).permute(0,3,1,2,4).contiguous() # b x n_anchors x feat_H x feat_W x 3
        #offsets = offsets.view(batch_size, height, width, A, 42).permute(0,3,1,2,4).contiguous() # b x n_anchors x feat_H x feat_W x 42
        # labels = labels.view(batch_size, 1, A * height, width)
        #outputs.append(labels)

        # bbox_targets = bbox_targets.view(batch_size, height, width, A*4).permute(0,3,1,2).contiguous()
        # outputs.append(bbox_targets)

        # anchors_count = bbox_inside_weights.size(1)
        # bbox_inside_weights = bbox_inside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)

        # bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, height, width, 4*A)\
        #                     .permute(0,3,1,2).contiguous()

        # outputs.append(bbox_inside_weights)

        # bbox_outside_weights = bbox_outside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)
        # bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, height, width, 4*A)\
        #                     .permute(0,3,1,2).contiguous()
        # outputs.append(bbox_outside_weights)

        return self.all_anchors, self.inds_inside, labels, offsets

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) 
    data:   - labels: b x <K*A x 3
            - 
    """
    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
        ret[:, inds,:] = data
    return ret


def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])
