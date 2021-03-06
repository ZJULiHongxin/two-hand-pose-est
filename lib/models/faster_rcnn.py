import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np

from .rpn.rpn import _RPN
# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_crop.modules.roi_crop import _RoICrop
# from model.roi_align.modules.roi_align import RoIAlignAvg
# from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb

from utils.utils import random_pose
#from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class FasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, cfg):
        super().__init__()
        self.num_joints = cfg.DATASET.NUM_JOINTS
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model, cfg)
        #self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        #self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        #self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        #self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        #self.RCNN_roi_crop = _RoICrop()

    def forward(self, imgs, pose_gt=None, hand_type=None):
        # imgs: b x 3 x H x W
        # pose_gt: b x 42 x 3 [u,v,vis]
        # hand_type: b
        if pose_gt is None:
            pose_gt = random_pose
        if hand_type is None:
            hand_type = torch.ones(1,2)
        # feed image data to base model to obtain base feature map

        base_feat = self.RCNN_base(imgs) # see resnetpy#line248 for info of RCNN_base
        assert imgs.isnan().sum() == 0
        # feed base feature map to RPN to obtain rois

        # 'pose_pred': b x total_anchors x 21 x 2
        # 'var': b x total_anchors x 21 x 2
        # 'handedness': b x total_anchors x 3

        return self.RCNN_rpn(base_feat, pose_gt, hand_type)

        # if it is training phase, then use ground truth bboxes for refining
        # if self.training:
        #     #roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
        #     rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

        #     rois_label = Variable(rois_label.view(-1).long())
        #     rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
        #     rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
        #     rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        # else:
        #     rois_label = None
        #     rois_target = None
        #     rois_inside_ws = None
        #     rois_outside_ws = None
        #     rpn_loss_cls = 0
        #     rpn_loss_bbox = 0

        # rois = Variable(rois)
        # # do roi pooling based on predicted rois

        # if cfg.POOLING_MODE == 'crop':
        #     # pdb.set_trace()
        #     # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
        #     grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
        #     grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
        #     pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
        #     if cfg.CROP_RESIZE_WITH_MAX_POOL:
        #         pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        # elif cfg.POOLING_MODE == 'align':
        #     pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        # elif cfg.POOLING_MODE == 'pool':
        #     pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # # feed pooled features to top model
        # pooled_feat = self._head_to_tail(pooled_feat)

        # # compute bbox offset
        # bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        # if self.training and not self.class_agnostic:
        #     # select the corresponding columns according to roi labels
        #     bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
        #     bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
        #     bbox_pred = bbox_pred_select.squeeze(1)

        # # compute object classification probability
        # cls_score = self.RCNN_cls_score(pooled_feat)
        # cls_prob = F.softmax(cls_score, 1)

        # RCNN_loss_cls = 0
        # RCNN_loss_bbox = 0

        # if self.training:
        #     # classification loss
        #     RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

        #     # bounding box regression L1 loss
        #     RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        # cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        # bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        #return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_offset, 0, 0.01, self.cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_confidence, 0, 0.01, self.cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_handedness, 0, 0.01, self.cfg.TRAIN.TRUNCATED)
        #normal_init(self.RCNN_confidence, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #normal_init(self.RCNN_offset_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        #normal_init(self.RCNN_handedness, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
