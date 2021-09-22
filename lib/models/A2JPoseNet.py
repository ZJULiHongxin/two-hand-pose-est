import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from .resnet import *

def generate_anchors(P_h=None, P_w=None):
    if P_h is None:
        P_h = np.array([2,6,10,14])

    if P_w is None:
        P_w = np.array([2,6,10,14])

    num_anchors = len(P_h) * len(P_h)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 2))
    k = 0
    for i in range(len(P_w)):
        for j in range(len(P_h)):
            anchors[k,1] = P_w[j]
            anchors[k,0] = P_h[i]
            k += 1  
    return anchors          

def shift(shape, stride, anchors):
    # generally, shape = [16,16], stride=16
    shift_h = np.arange(0, shape[0]) * stride
    shift_w = np.arange(0, shape[1]) * stride

    shift_h, shift_w = np.meshgrid(shift_h, shift_w)
    shifts = np.vstack((shift_h.ravel(), shift_w.ravel())).transpose()

    # add A anchors (1, A, 2) to
    # cell K shifts (K, 1, 2) to get
    # shift anchors (K, A, 2)
    # reshape to (K*A, 2) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 2))

    return all_anchors

class DepthRegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=16, num_classes=15, feature_size=256):
        super(DepthRegressionModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(feature_size)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchors*num_classes, kernel_size=3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.act4(out)
        out = self.output(out)

        # out is B x C x W x H, with C = 3*num_anchors
        out1 = out.permute(0, 3, 2, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        return out2.contiguous().view(out2.shape[0], -1, self.num_classes)

class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=16, num_classes=15, feature_size=256):
        super(RegressionModel, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(feature_size)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchors*num_classes*2, kernel_size=3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.act4(out)
        out = self.output(out)

        # out is B x C x W x H, with C = 3*num_anchors
        out1 = out.permute(0, 3, 2, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes, 2)
        return out2.contiguous().view(out2.shape[0], -1, self.num_classes, 2)

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=16, num_classes=15, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(feature_size)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchors*num_classes, kernel_size=3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.act4(out)
        out = self.output(out)
    
        # out is B x C x W x H, with C = n_classes * n_anchors
        out1 = out.permute(0, 3, 2, 1) # b x w x h x c
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes) # b x w x h x n_anchors x n_joints
        return out2.contiguous().view(x.shape[0], -1, self.num_classes) # b x w*h*n_anchors x n_joints

class ResNetBackBone(nn.Module):
    def __init__(self,num_layers):
        super(ResNetBackBone, self).__init__()
        
        modelPreTrain50 = eval('resnet'+str(num_layers)+'(pretrained=True)')
        self.model = modelPreTrain50
        
    def forward(self, x): 
        n, c, h, w = x.size()  # x: [B, 1, H ,W]
        
        x = x[:,0:1,:,:]  # depth
        x = x.expand(n,3,h,w)
              
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x1 = self.model.layer1(x)
        x2 = self.model.layer2(x1)
        x3 = self.model.layer3(x2)
        x4 = self.model.layer4(x3)
        
        return x3,x4  

class A2JPoseNet(nn.Module):
    def __init__(self, cfg):
        super(A2JPoseNet, self).__init__()
        res_layers = cfg.MODEL.RESNET_LAYERS
        self.Backbone = ResNetBackBone(res_layers) # 1 channel depth only, resnet50 
        if res_layers == 50:
            reg_in_channels, cls_in_channels = 2048, 1024
        elif res_layers == 34:
            reg_in_channels, cls_in_channels = 512, 256
        n_anchors_per_px = len(cfg.MODEL.N_ANCHORS_H) * len(cfg.MODEL.N_ANCHORS_W)
        self.regressionModel = RegressionModel(reg_in_channels, num_anchors=n_anchors_per_px, num_classes=cfg.DATASET.NUM_JOINTS * 2)
        self.classificationModel = ClassificationModel(cls_in_channels, num_anchors=n_anchors_per_px, num_classes=cfg.DATASET.NUM_JOINTS * 2)

        self.wh = torch.tensor(cfg.MODEL.INPUT_SIZE).view(1,1,1,-1).float()
        anchors = generate_anchors(P_h=cfg.MODEL.N_ANCHORS_H, P_w=cfg.MODEL.N_ANCHORS_W)
        self.all_anchors = torch.from_numpy(shift(shape=[16,16], stride=16, anchors=anchors)).float() # (w*h*A, 2)

        self.trainable_temp = torch.nn.parameter.Parameter(torch.tensor(5.0), requires_grad=cfg.MODEL.TRAINABLE_SOFTMAX) if cfg.MODEL.TRAINABLE_SOFTMAX else 1.0

    def forward(self, x): 
        x3,x4 = self.Backbone(x)

        # cls: b x w*h*n_anchors x n_joints
        # reg: B x w*h*n_anchors x n_joints x 2

        classification  = self.classificationModel(x3)
        relative_regression = torch.tanh(self.regressionModel(x4))
        #print(self.wh)
        #print('---relative_regression---\n',relative_regression)
        reg = relative_regression * self.wh.to(relative_regression.device)

        reg_weight = F.softmax(classification * self.trainable_temp,dim=1)
        reg_weight_xy = torch.unsqueeze(reg_weight,3).expand(reg_weight.shape[0], reg_weight.shape[1],reg_weight.shape[2],2)#b x (w*h*A) x n_joints x 2         
        #print('---regression---\n',reg)
        #print('---reg_weight_xy---',reg_weight_xy)
        pose_pred = (reg_weight_xy * (reg + self.all_anchors.unsqueeze(1).unsqueeze(0).to(x.device))).sum(1) # b x n_joints x 2
        
        surrounding_anchors_pred = (reg_weight_xy * self.all_anchors.unsqueeze(1).unsqueeze(0).to(x.device)).sum(1)
        return pose_pred, surrounding_anchors_pred, classification, reg, x4, self.trainable_temp
