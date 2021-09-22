import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from .A2JPoseNet import ResNetBackBone

class Baseline(nn.Module):
    def __init__(self, cfg):
        super(Baseline, self).__init__()
        res_layers = cfg.MODEL.RESNET_LAYERS
        self.num_joints = cfg.DATASET.NUM_JOINTS * 2
        self.Backbone = ResNetBackBone(res_layers) # 1 channel depth only, resnet50 
        if res_layers == 50:
            reg_in_channels, cls_in_channels = 2048, 1024
        elif res_layers == 34:
            reg_in_channels, cls_in_channels = 512, 256

        MLP_in_channels = reg_in_channels * 16 * 16

        self.MLP = nn.Sequential(
            nn.Flatten(),
            nn.Linear(MLP_in_channels, 640),
            nn.BatchNorm1d(640),
            nn.ReLU(inplace=True),
            # nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            # nn.ReLU(inplace=True),
            nn.Linear(640, self.num_joints * 2),
        )
    
    def forward(self, x):
        interfeat, feat = self.Backbone(x)
        two_hand_pose = self.MLP(feat)
        return  two_hand_pose.view(-1, self.num_joints, 2)