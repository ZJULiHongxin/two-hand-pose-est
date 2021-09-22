# coding: utf-8

from __future__ import division, print_function
from typing import IO

import numpy as np
import torch
from .bbox_transform import calc_OKS, calc_pose_IoU

def nms_pose(cfg, pose_pred, score_pred, var_pred):
        # pose_pred: n_poses x 21 x 2
        # score_pred (a column detached from one-hot matrices): n_poses
        # var_pred: n_poses x 21 x 2

        if pose_pred.shape[0] == 1: # not need of NMS for only one anchor
            return pose_pred[0], score_pred, var_pred[0]
        
        device = pose_pred.device
        #print(score_pred.shape)
        max_idx = torch.argmax(score_pred)
        max_score = score_pred[max_idx]
        max_var = var_pred[max_idx] # 21 x 2
        max_pose = pose_pred[max_idx]
        pose_pred = torch.cat((pose_pred[:max_idx], pose_pred[max_idx+1:]), 0)
        var_pred = torch.cat((var_pred[:max_idx], var_pred[max_idx+1:]), 0)
        #print(max_pose.shape, pose_pred.shape)

        if cfg.MODEL.SIMILARITY_METRIC == 'OKS':
            Sim = calc_OKS(max_pose.unsqueeze(0), pose_pred).squeeze() # n_poses - 1
        elif cfg.MODEL.SIMILARITY_METRIC == 'IoU':
            Sim = calc_pose_IoU(max_pose.unsqueeze(0), pose_pred).squeeze() # n_poses - 1
        
        # var voting
        Sim_mask = Sim > 0.2

        if Sim_mask.sum() == 0: # in case that there are no other anchors overlapping with the one with the max score
            return max_pose, max_score, max_var

        kl_poses = pose_pred[Sim_mask] # remove x invalid poses

        print(Sim_mask.sum(), Sim.shape, kl_poses.shape, max_pose.shape)
        kl_poses = torch.cat((kl_poses, max_pose.unsqueeze(0)), 0) # (n_poses-x) x 21 x 2
        kl_OKS = Sim[Sim_mask] # n_poses-1-x
        kl_var = torch.cat((var_pred[Sim_mask], max_var.unsqueeze(0)), 0) # n_poses-x x 21 x 2
        
        # try KL_SIGMA = 0.02
        pi = torch.exp(-1 * torch.pow((1- kl_OKS), 2) / cfg.MODEL.KL_SIGMA) # n_poses-1-x
        pi = torch.cat((pi, torch.ones(1).to(device)), 0) # n_poses-x (assign 1 to self)
        pi = pi.unsqueeze(1) / kl_var.view(kl_var.shape[0], -1) # (n_poses-x) x 42
        pi = pi / pi.sum(0) # (n_poses-x) x 42
        max_pose = (pi * kl_poses.view(kl_poses.shape[0], -1)).sum(0) # 42

        return max_pose.view(cfg.DATASET.NUM_JOINTS, -1), max_score, max_var # 21 x 2
        # weight = torch.ones_like(OKS)
        # if not cfg.MODEL.USE_SOFTNMS:
        #     weight[OKS > cfg.MODEL.NMS_OKS] = 0
        # else:
        #     weight = torch.exp(-1.0 * (OKS ** 2 / cfg.MODEL.SOFTNMS_SIGMA)) 
        
        # score_pred = score_pred * weight
        # filter_idx = (score_pred >= cfg.MODEL.SCORE_THRESHOLD).nonzero().squeeze(-1)
        # pose_pred = pose_pred[filter_idx]

        #return torch.cat(keep, 0).to(device)


def torch_pose_nms(left_pose_pred, right_pose_pred, left_pose_score, right_pose_score, left_var, right_var):
    """
    left_pose_pred:   n_left x 21 x 2
    right_pose_pred:  n_right x 21 x 2
    left_pose_score: n_left
    right_pose_score: n_right
    left_var: n_left x 21 x 2
    right_var: b_right x 21 x 2
    """

    
    #filter_idx = left_pose_score >= cfg.MODEL.KL_SIGMA
    left_pose_pred_final, left_score, left_var = nms_pose(left_pose_pred, left_pose_score, left_var)
    # if filter_idx.sum() > 0:
    #     left_pose_pred_final, left_score, left_var = nms_pose(left_pose_pred[filter_idx], left_pose_score[filter_idx], left_var[filter_idx])
    # else:
    #     left_pose_pred_final = torch.zeros(21,2).type_as(left_pose_pred)
    #     left_score = torch.zeros(1).type_as(left_pose_pred)
    #     left_var = torch.zeros(21,2).type_as(left_pose_pred)

    # filter_idx = right_pose_score >= cfg.MODEL.KL_SIGMA

    right_pose_pred_final, right_score, right_var = nms_pose(right_pose_pred, right_pose_score, right_var)
    # if filter_idx.sum() > 0:
    #     right_pose_pred_final, right_score, right_var = nms_pose(right_pose_pred[filter_idx], right_pose_score[filter_idx], right_var[filter_idx])
    # else:
    #     right_pose_pred_final = torch.zeros(21,2).type_as(left_pose_pred)
    #     right_score = torch.zeros(1).type_as(left_pose_pred)
    #     right_var = torch.zeros(21,2).type_as(left_pose_pred)

    return left_pose_pred_final, left_score, left_var, right_pose_pred_final, right_score, right_var
    
def torch_nms(cfg, boxes, variance=None):
    def nms_class(clsboxes):
        assert clsboxes.shape[1] == 5 or clsboxes.shape[1] == 9
        keep = []
        while clsboxes.shape[0] > 0:
            maxidx = torch.argmax(clsboxes[:, 4])
            maxbox = clsboxes[maxidx].unsqueeze(0)
            clsboxes = torch.cat((clsboxes[:maxidx], clsboxes[maxidx + 1:]), 0)
            iou = iou_calc3(maxbox[:, :4], clsboxes[:, :4])
            # KL VOTE
            if variance is not None:
                ioumask = iou > 0
                klbox = clsboxes[ioumask]
                klbox = torch.cat((klbox, maxbox), 0)
                kliou = iou[ioumask]
                klvar = klbox[:, -4:]
                pi = torch.exp(-1 * torch.pow((1 - kliou), 2) / cfg.vvsigma)
                pi = torch.cat((pi, torch.ones(1).cuda()), 0).unsqueeze(1)
                pi = pi / klvar
                pi = pi / pi.sum(0)
                maxbox[0, :4] = (pi * klbox[:, :4]).sum(0)
            keep.append(maxbox)

            weight = torch.ones_like(iou)
            if not cfg.soft:
                weight[iou > cfg.nms_iou] = 0
            else:
                weight = torch.exp(-1.0 * (iou ** 2 / cfg.softsigma))
            clsboxes[:, 4] = clsboxes[:, 4] * weight
            filter_idx = (clsboxes[:, 4] >= cfg.score_thres).nonzero().squeeze(-1)
            clsboxes = clsboxes[filter_idx]
        return torch.cat(keep, 0).to(clsboxes.device)

    bbox = boxes[:, :4].view(-1, 4)
    numcls = boxes.shape[1] - 4
    scores = boxes[:, 4:].view(-1, numcls)
    # Picked bounding boxes
    picked_boxes, picked_score, picked_label = [], [], []
    for i in range(numcls):
        filter_idx = (scores[:, i] >= cfg.score_thres).nonzero().squeeze(-1)
        if len(filter_idx) == 0:
            continue
        filter_boxes = bbox[filter_idx]
        filter_scores = scores[:, i][filter_idx].unsqueeze(1)
        if variance is not None:
            filter_variance = variance[filter_idx]
            clsbox = nms_class(torch.cat((filter_boxes, filter_scores, filter_variance), 1))
        else:
            clsbox = nms_class(torch.cat((filter_boxes, filter_scores), 1))
        if clsbox.shape[0] > 0:
            picked_boxes.append(clsbox[:, :4])
            picked_score.append(clsbox[:, 4])
            picked_label.extend([torch.ByteTensor([i]) for _ in range(len(clsbox))])
    
    if len(picked_boxes) == 0:
        return None, None, None
    else:
        return torch.cat(picked_boxes), torch.cat(picked_score), torch.cat(picked_label)
