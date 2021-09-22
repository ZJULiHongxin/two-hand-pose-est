import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from .A2JPoseNet import RegressionModel, ClassificationModel, generate_anchors, shift
from .pose_hrnet import get_pose_net

class HRNetA2JPoseNet(nn.Module):
    def __init__(self, cfg):
        super(HRNetA2JPoseNet, self).__init__()
        res_layers = cfg.MODEL.RESNET_LAYERS
        self.Backbone = get_pose_net(cfg, is_train=True) # 1 channel depth only, resnet50 
        channels = sum(cfg.MODEL.EXTRA.STAGE4.NUM_CHANNELS)
        self.downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        n_anchors_per_px = len(cfg.MODEL.N_ANCHORS_H) * len(cfg.MODEL.N_ANCHORS_W)
        self.regressionModel = RegressionModel(channels, num_anchors=n_anchors_per_px, num_classes=cfg.DATASET.NUM_JOINTS * 2)
        self.classificationModel = ClassificationModel(channels, num_anchors=n_anchors_per_px, num_classes=cfg.DATASET.NUM_JOINTS * 2)

        self.wh = torch.tensor(cfg.MODEL.INPUT_SIZE).view(1,1,1,-1).float()
        anchors = generate_anchors(P_h=cfg.MODEL.N_ANCHORS_H, P_w=cfg.MODEL.N_ANCHORS_W)
        self.all_anchors = torch.from_numpy(shift(shape=[16,16], stride=16, anchors=anchors)).float() # (w*h*A, 2)

        self.trainable_temp = torch.nn.parameter.Parameter(torch.tensor(5.0), requires_grad=cfg.MODEL.TRAINABLE_SOFTMAX) if cfg.MODEL.TRAINABLE_SOFTMAX else 1.0

    def forward(self, x): 
        output_feat, inter_feat = self.Backbone(x)
        inter_feat = self.downsample(inter_feat)
        output_feat = self.downsample(output_feat)

        # cls: b x w*h*n_anchors x n_joints
        # reg: B x w*h*n_anchors x n_joints x 2

        classification  = self.classificationModel(inter_feat)
        relative_regression = torch.tanh(self.regressionModel(output_feat))
        #print(self.wh)
        #print('---relative_regression---\n',relative_regression)
        reg = relative_regression * self.wh.to(relative_regression.device)

        reg_weight = F.softmax(classification * self.trainable_temp,dim=1)
        reg_weight_xy = torch.unsqueeze(reg_weight,3).expand(reg_weight.shape[0], reg_weight.shape[1],reg_weight.shape[2],2)#b x (w*h*A) x n_joints x 2         
        #print('---regression---\n',reg)
        #print('---reg_weight_xy---',reg_weight_xy)
        pose_pred = (reg_weight_xy * (reg + self.all_anchors.unsqueeze(1).unsqueeze(0).to(x.device))).sum(1) # b x n_joints x 2
        
        surrounding_anchors_pred = (reg_weight_xy * self.all_anchors.unsqueeze(1).unsqueeze(0).to(x.device)).sum(1)
        return pose_pred, surrounding_anchors_pred, classification, reg, self.trainable_temp

