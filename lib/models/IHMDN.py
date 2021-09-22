from re import I
import torch
import torch.nn as nn
import torch.nn.functional as F
from .A2JPoseNet import A2JPoseNet

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import add_self_loops, degree

from utils.standard_legends import part_ratio

class HandGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)
        # TODO: design different transform matrices for the different kinds of nodes
        # self.lin_cur_node = torch.nn.Linear(in_channels, out_channels)
        # self.lin_neighboring_node = torch.nn.Linear(in_channels, out_channels)
        # self.lin_global_node = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

class HandGNNBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gcn_layer1 = HandGCNConv(in_channels, out_channels)
        self.bn1 = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.gcn_layer2 = HandGCNConv(out_channels, out_channels)
        self.bn2 = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.reduce = nn.Linear(in_channels, out_channels)
    
    def forward(self, x):
        x1 = self.relu(self.bn1(self.gcn_layer1(x)))
        x1 = self.bn2(self.gcn_layer2(x1))

        output = self.relu(x1 + self.reduce(x))

        return output


class HandGNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        GNN_channels = cfg.MODEL.GNN_CHANNELS
        GNN_layers = [
            HandGCNConv(cfg.MODEL.NODE_FEAT_DIM, GNN_channels[0]),
            BatchNorm(GNN_channels[0]),
            nn.ReLU(inplace=True)]
        
        for i in range(len(GNN_channels) - 1):
            GNN_layers.append(HandGNNBasicBlock(in_channels=GNN_channels[i], out_channels=GNN_channels[i+1]))
        
        self.GNN_layers = nn.ModuleList(GNN_layers)

    def forward(self, x):
        pass

class IHMDN(nn.Module):
    def __init__(self,cfg):
        super(IHMDN, self).__init__()
        self.cfg = cfg
        self.A2J = A2JPoseNet(cfg) # estimate rough keypoint locations
        self.HandGNN = HandGNN(cfg)
        self.edges = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # the global node connects with all 42 joints
        1,1,1,1,1,1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        ]])
    def calc_hand_scale(self, pose):
        # pose: b x 21 x 2

        # the distance between the index MCP and the pinky MCP, and that between the middle tip and the wrist
        hand_w = torch.norm(pose[:,5] - pose[:,17], dim=1)
        hand_h = torch.norm(pose[:,0] - pose[:,9], dim=1) \
        + torch.norm(pose[:,9] - pose[:,10], dim=1) \
        + torch.norm(pose[:,10] - pose[:,11], dim=1) \
        + torch.norm(pose[:,11] - pose[:,12], dim=1)

        return hand_w, hand_h

    def extract_feat(self, pose2d_pred, feat):
        # pose2d_pred: b x 42 x 2
        # feat: b x c x 64 x 64

        right_hand_w, right_hand_h = self.calc_hand_scale(pose2d_pred[:,0:21]) # (b, )
        right_hand_feat = torch.zeros(feat.shape[0], 21, feat.shape[1]).type_as(feat).to(feat.device) # b x 21 x c
        for b in range(feat.shape[0]):
            for k in range(21):
                radius = part_ratio[k] * right_hand_h

                for delta_u in range(-radius, radius):
                    for delta_v in range(-radius, radius):
                        if delta_u**2 + delta_v**2 < radius:
                            right_hand_feat[b,k] += F.grid_sample(input=feat, grid=)

    def forward(self, imgs):
        pose2d_pred, surrounding_anchors_pred, classification, reg, backbone_feat, trainable_temp = self.A2J(imgs)

        joint_features = self.extract_feat(pose2d_pred, backbone_feat) # b x 42 x D
        joint_features_GNN = self.HandGNN(joint_features)


        




