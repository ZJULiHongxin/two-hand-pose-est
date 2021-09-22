from __future__ import absolute_import
from re import template
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from math import log2
from utils.vis import plot_hand
#from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from .nms_utils import nms_pose

import numpy as np
import math
import pdb
import time


class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din, cfg):
        super(_RPN, self).__init__()
        self.cfg = cfg
        self.din = din  # get the channel num of input feature map, e.g., 512
        self.anchor_scales = cfg.MODEL.ANCHOR_SCALES # a list of scales
        self.anchor_orientations = cfg.MODEL.ANCHOR_ORIENTATIONS
        
        self.n_anchors = len(self.anchor_scales) * len(self.anchor_orientations)

        self.downsample = [nn.AvgPool2d(kernel_size=2, stride=2) for i in range(int(log2(self.cfg.MODEL.ANCHOR_STRIDE[0])))]
        # define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv2d(self.din, 1024, 3, 1, 1, bias=True)

        # define offset estiamtion layer
        self.offset_channels = self.n_anchors * 42 # [u,v] of 21 joints
        self.RPN_offset = nn.Conv2d(1024, self.offset_channels, 1, 1, 0)

        # define offset confidence layer
        self.conf_channels = self.n_anchors * 42
        self.RPN_confidence = nn.Conv2d(1024, self.conf_channels, 1, 1, 0) # training. To avoid gradient exploding, our network predicts α = log(σ2) instead of σ

        # define hand classification layer
        self.handednss_channels = self.n_anchors * 3
        self.RPN_handedness = nn.Conv2d(1024, self.handednss_channels, 1, 1, 0) # the second dim 3 represents a 3d one-hot vector

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.anchor_scales, self.anchor_orientations, cfg)

        # self.rpn_loss_cls = 0
        # self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat,  pose_gt, hand_type):
        # base_feat: b x c x H x W
        # pose_gt: b x 42 x 3
        # hand_type

        device = base_feat.device
        batch_size = base_feat.size(0)
        H, W = base_feat.shape[-2:]
        # return feature map after convrelu layer
        rpn_conv1 = self.RPN_Conv(base_feat) # b x 1024 x H x W
        for i in range(len(self.downsample)):
            rpn_conv1 = self.downsample[i](rpn_conv1)
        rpn_conv1 = F.relu(rpn_conv1, inplace=True)

        # get offsets relative to anchors
        rpn_offset = self.RPN_offset(rpn_conv1).view(batch_size, 21, 2, -1).permute(0, 3, 1, 2) # b x (42*n_anchors) x H x W -> b x total_anchors x 21 x 2
        # get vaiance for each est offset
        rpn_variance =  self.RPN_confidence(rpn_conv1).view(batch_size, 21, 2, -1).permute(0, 3, 1, 2) # b x (42*n_anchors) x H x W -> b x total_anchors x 21 x 2
        # get handedness for each anchor
        rpn_handedness = self.RPN_handedness(rpn_conv1).view(batch_size, 3, -1).permute(0,2,1) # b x (3*n_anchors) x H x W -> b x total_anchors x 3
        rpn_handedness = F.softmax(rpn_handedness, dim=2)

        assert base_feat.isnan().sum() == 0
        assert rpn_handedness.isnan().sum() == 0
        assert rpn_variance.isnan().sum() == 0
        assert rpn_offset.isnan().sum() == 0
        # anchors: total_anchors x 21 x 2
        # inds_inside: denotes those anchors inside the image, size: total_anchors
        # labels: b x total_anchors x 3
        # offsets: b x total_anchors x 42
        # idx_left, idx_right, idx_negative: [N]
        # total_anchors: the total number of anchor poses = n_anchors_per_px x feat_H x feat_W
        # n_anchors: the number of anchors per pixel of the feature map
        anchors, inds_inside, labels, offsets = self.RPN_anchor_target(pose_gt, hand_type, base_feat.shape) # b x feat_H x feat_W x n_anchors x 3; b x n_anchors x feat_H x feat_W x 42
        anchors = anchors.to(device)
        inds_inside = inds_inside.to(device)
        anchor_W = anchors[:,:,0].max(dim=1).values - anchors[:,:,0].min(dim=1).values # total_anchors
        anchor_H = anchors[:,:,1].max(dim=1).values - anchors[:,:,1].min(dim=1).values

        #         1 x total_anchors x 21 x 2     +        b x total_anchors x 21 x 2       1 x total_anchors x 1 x 2
        pose_pred = anchors.unsqueeze(0) + rpn_offset * torch.stack((anchor_W, anchor_H), 1).unsqueeze(0).unsqueeze(2) # b x total_anchors x 21 x 2

        left_pose_pred_lst, left_score_lst, left_var_lst = [], [], []
        right_pose_pred_lst, right_score_lst, right_var_lst = [], [], []

        return_dict = {'KL_loss':0., 'cls_loss':0.}
        
        for b in range(batch_size):
        # pose KL-Loss
        # During training, we ignore all cross-boundary anchors so they do not contribute to the loss
            left_pred_idx = (rpn_handedness[b,:,0] > rpn_handedness[b,:,1]) & \
                            (rpn_handedness[b,:,0] > rpn_handedness[b,:,2])
            print(rpn_handedness[b],left_pred_idx) # the initial values of rpn_handedness are around 1/3
            left_keep = left_pred_idx & inds_inside    # (total_anchors,)

            left_pose_pred = pose_pred[b,left_keep] # n_left_anchors x 21 x 2
            
            left_var = torch.exp(rpn_variance[b,left_keep])    # n_left_anchors x 21 x 2
            left_scores = rpn_handedness[b,left_keep,0] # n_left_anchors

            right_pred_idx = (rpn_handedness[b,:,2] > rpn_handedness[b,:,0]) & \
                             (rpn_handedness[b,:,2] > rpn_handedness[b,:,1])
            right_keep = right_pred_idx & inds_inside   # (total_anchors,)
            
            right_pose_pred = pose_pred[b,right_keep] # n_left_anchors x 21 x 2
            right_var = torch.exp(rpn_variance[b,right_keep])    # n_left_anchors x 21 x 2
            right_scores = rpn_handedness[b,right_keep,2] # n_left_anchors

            print('left', left_keep.sum().item(),left_pred_idx.sum().item(), inds_inside.sum().item())
            print('right', right_keep.sum().item(),right_pred_idx.sum().item())

            # f = plt.figure()
            # ax1 =f.add_subplot(1,3,1)
            # for k in range(pose_pred.shape[1]):
            #     plot_hand(ax1, pose_pred[b,k].detach().cpu().numpy(), order='uv')
            
            # ax2 =f.add_subplot(1,2,1)
            # for k in range(left_pose_pred.shape[0]):
            #     plot_hand(ax2, left_pose_pred[k].detach().cpu().numpy(), order='uv')
            # ax2.set_title('left')
            # ax3 =f.add_subplot(1,2,2)
            # for k in range(right_pose_pred.shape[0]):
            #     plot_hand(ax3, right_pose_pred[k].detach().cpu().numpy(), order='uv')
            # ax3.set_title('right')
            # plt.show()

            

            # pose_pred_final: 21 x 2
            # score: 1
            # var: 21 x 2
            if left_keep.sum() > 0:
                #print(left_pose_pred.shape, left_scores.shape, left_var.shape)
                left_pose_pred_final, left_score, left_var = nms_pose(self.cfg, left_pose_pred, left_scores, left_var)
            else:
                left_pose_pred_final = torch.zeros(21,2).type_as(left_pose_pred)
                left_score = torch.zeros(1).type_as(left_pose_pred)
                left_var = torch.zeros(21,2).type_as(left_pose_pred)
            
            if right_keep.sum() > 0:
                #print(right_pose_pred.shape, right_scores.shape, right_var.shape)
                right_pose_pred_final, right_score, right_var = nms_pose(self.cfg, right_pose_pred, right_scores, right_var)
            else:
                right_pose_pred_final = torch.zeros(21,2).type_as(left_pose_pred)
                right_score = torch.zeros(1).type_as(left_pose_pred)
                right_var = torch.zeros(21,2).type_as(left_pose_pred)
            
            left_pose_pred_lst.append(left_pose_pred_final)
            left_score_lst.append(left_score)
            left_var_lst.append(left_var)
            right_pose_pred_lst.append(right_pose_pred_final)
            right_score_lst.append(right_score)
            right_var_lst.append(right_var)

            
            # calcualte KL loss
            # 对于标签为左手，RPN估计预设姿态都为右手的情况，
            #
            smooth_loss = torch.nn.SmoothL1Loss(reduction='none')

            input_area = self.cfg.MODEL.INPUT_SIZE[0] ** 2
            KL_loss_weight = self.cfg.LOSS.KL_LOSS_WEIGHT #, self.cfg.LOSS.CLS_LOSS_WEIGHT

            h_type = hand_type[b]
            KL_loss = torch.tensor(0.).to(device)
            #if h_type[1] == 1 and left_keep.sum() > 0:
            
            if left_keep.sum() > 0:
                left_anchor_scale = 2.0 - 1.0 * (left_pose_pred_final[:,0].max()-left_pose_pred_final[:,0].min()) * \
                                                (left_pose_pred_final[:,1].max()-left_pose_pred_final[:,1].min()) / input_area
                
                # print(left_score, left_anchor_scale, left_var.shape, smooth_loss(target=pose_gt[b,21:42,0:2], input=left_pose_pred_final).shape)
                # print(torch.exp(-left_var) * smooth_loss(target=pose_gt[b,21:42,0:2], input=left_pose_pred_final))
                if h_type[1] == 1:
                    target_pose = pose_gt[b,21:42,0:2]
                else:
                    target_pose = pose_gt[b,0:21,0:2]
                
                KL_loss = KL_loss + left_score * left_anchor_scale * \
                (torch.exp(-left_var) * smooth_loss(target=target_pose, input=left_pose_pred_final) + 0.5 * left_var)
                
                print('left KLLOSS', KL_loss.sum())
            # if h_type[0] == 1 and right_keep.sum() > 0:
            
            if right_keep.sum() > 0:
                right_anchor_scale = 2.0 - 1.0 * (right_pose_pred_final[:,0].max()-right_pose_pred_final[:,0].min()) * \
                                                (right_pose_pred_final[:,1].max()-right_pose_pred_final[:,1].min()) / input_area
                
                # print(right_score, right_anchor_scale,right_var.shape, smooth_loss(target=pose_gt[b,0:21,0:2], input=right_pose_pred_final).shape)
                # print(torch.exp(-right_var) * smooth_loss(target=pose_gt[b,0:21,0:2], input=right_pose_pred_final))
                if h_type[0] == 1:
                    target_pose = pose_gt[b,0:21,0:2]
                else:
                    target_pose = pose_gt[b,21:42,0:2]
                
                KL_loss = KL_loss + right_score * right_anchor_scale * \
                (torch.exp(-right_var) * smooth_loss(target=target_pose, input=right_pose_pred_final) + 0.5 * right_var) 
                print('right KLLOSS', KL_loss.sum)
            return_dict['KL_loss'] += KL_loss_weight * KL_loss.sum()

            # handedness cross-entropy loss
            n_neg_samples = max(left_keep.sum().item() + right_keep.sum().item(), 10)
            idx_neg = torch.nonzero(labels[b,:,1] == 1).squeeze()
            random_neg_idx = torch.randperm(idx_neg.shape[0])[0:n_neg_samples]
            keep = []
            if left_keep.shape[0] > 0:
                keep.append(torch.nonzero(left_keep).squeeze().cpu())
            if right_keep.shape[0] > 0:
                keep.append(torch.nonzero(right_keep).squeeze().cpu())
            if random_neg_idx.shape[0] > 0:
                keep.append(random_neg_idx)
            
            print(left_keep.shape,  torch.nonzero(left_keep).squeeze().cpu().shape, right_keep.shape, torch.nonzero(right_keep).squeeze().cpu().shape, random_neg_idx.shape)

            keep = torch.cat(keep)
            print('keep',keep.shape)

            return_dict['cls_loss'] += F.cross_entropy(rpn_handedness[b,keep], torch.argmax(labels[b,keep], dim=1).to(device)) / keep.sum().to(device)

        return_dict.update({
            'handedness_pred': rpn_handedness,
            'left_pose_pred': torch.stack(left_pose_pred_lst),
            'left_score': torch.stack(left_score_lst),
            'left_var': torch.stack(left_var_lst),
            'right_pose_pre': torch.stack(right_pose_pred_lst),
            'right_score': torch.stack(right_score_lst),
            'right_var': torch.stack(right_var_lst)
        })

        return return_dict
            #rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            # compute bbox regression loss
            #rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            #rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            #rpn_bbox_targets = Variable(rpn_bbox_targets)

            # self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
            #                                                rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])

        #return rois, self.rpn_loss_cls, self.rpn_loss_box
