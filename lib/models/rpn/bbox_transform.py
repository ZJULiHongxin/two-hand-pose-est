# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import numpy as np
import pdb

from torch.nn.functional import threshold

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = torch.log(gt_widths / ex_widths)
    targets_dh = torch.log(gt_heights / ex_heights)

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh),1)

    return targets

def bbox_transform_batch(ex_rois, gt_rois):

    if ex_rois.dim() == 2:
        ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
        ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

        targets_dx = (gt_ctr_x - ex_ctr_x.view(1,-1).expand_as(gt_ctr_x)) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y.view(1,-1).expand_as(gt_ctr_y)) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths.view(1,-1).expand_as(gt_widths))
        targets_dh = torch.log(gt_heights / ex_heights.view(1,-1).expand_as(gt_heights))

    elif ex_rois.dim() == 3:
        ex_widths = ex_rois[:, :, 2] - ex_rois[:, :, 0] + 1.0
        ex_heights = ex_rois[:,:, 3] - ex_rois[:,:, 1] + 1.0
        ex_ctr_x = ex_rois[:, :, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, :, 1] + 0.5 * ex_heights

        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths)
        targets_dh = torch.log(gt_heights / ex_heights)
    else:
        raise ValueError('ex_roi input dimension is not correct.')

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh),2)

    return targets

def anchors_transform_batch(anchors, pose_targets):
    """
    relative offset: (x_gt - x_anc) / w_anc
    anchors: <K*A x 21 x 2
    pose_targets: b x <K*A x 21 x 2
    """
    
    anchor_ws = anchors[:,:,0].max(dim=1).values - anchors[:,:,0].min(dim=1).values #  <K*A
    anchor_hs = anchors[:,:,1].max(dim=1).values - anchors[:,:,1].min(dim=1).values

    targets_dx = (pose_targets[:,:,:,0] - anchors[None,:,:,0]) / anchor_ws.unsqueeze(1) # b x <K*A x 21
    targets_dy = (pose_targets[:,:,:,1] - anchors[None,:,:,1]) / anchor_hs.unsqueeze(1) # b x <K*A x 21

    return torch.cat((targets_dx, targets_dy), 2) # b x <K*A x 42


def bbox_transform_inv(boxes, deltas, batch_size):
    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights

    dx = deltas[:, :, 0::4]
    dy = deltas[:, :, 1::4]
    dw = deltas[:, :, 2::4]
    dh = deltas[:, :, 3::4]

    pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    pred_w = torch.exp(dw) * widths.unsqueeze(2)
    pred_h = torch.exp(dh) * heights.unsqueeze(2)

    pred_boxes = deltas.clone()
    # x1
    pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def clip_boxes_batch(boxes, im_shape, batch_size):
    """
    Clip boxes to image boundaries.
    """
    num_rois = boxes.size(1)

    boxes[boxes < 0] = 0
    # batch_x = (im_shape[:,0]-1).view(batch_size, 1).expand(batch_size, num_rois)
    # batch_y = (im_shape[:,1]-1).view(batch_size, 1).expand(batch_size, num_rois)

    batch_x = im_shape[:, 1] - 1
    batch_y = im_shape[:, 0] - 1

    boxes[:,:,0][boxes[:,:,0] > batch_x] = batch_x
    boxes[:,:,1][boxes[:,:,1] > batch_y] = batch_y
    boxes[:,:,2][boxes[:,:,2] > batch_x] = batch_x
    boxes[:,:,3][boxes[:,:,3] > batch_y] = batch_y

    return boxes

def clip_boxes(boxes, im_shape, batch_size):

    for i in range(batch_size):
        boxes[i,:,0::4].clamp_(0, im_shape[i, 1]-1)
        boxes[i,:,1::4].clamp_(0, im_shape[i, 0]-1)
        boxes[i,:,2::4].clamp_(0, im_shape[i, 1]-1)
        boxes[i,:,3::4].clamp_(0, im_shape[i, 0]-1)

    return boxes

def bbox_overlaps(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float (u_tl, v_tl, u_br, v_br)
    gt_boxes: (K, 4) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_area = ((gt_boxes[:,2] - gt_boxes[:,0] + 1) *
                (gt_boxes[:,3] - gt_boxes[:,1] + 1)).view(1, K)

    anchors_area = ((anchors[:,2] - anchors[:,0] + 1) *
                (anchors[:,3] - anchors[:,1] + 1)).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (torch.min(boxes[:,:,2], query_boxes[:,:,2]) -
        torch.max(boxes[:,:,0], query_boxes[:,:,0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:,:,3], query_boxes[:,:,3]) -
        torch.max(boxes[:,:,1], query_boxes[:,:,1]) + 1)
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps

def calc_pose_IoU(anchor, gt):
    # anchor: N x 21 x 2
    # gt: K x 21 x 3 [u,v,vis] or K x 21 x 2 [u,v]
    anchor_boxes = torch.cat(
        (
            anchor[:,:,0].min(dim=1, keepdim=True).values, anchor[:,:,1].min(dim=1, keepdim=True).values,
            anchor[:,:,0].max(dim=1, keepdim=True).values, anchor[:,:,1].max(dim=1, keepdim=True).values
            ), dim=1)
    gt_boxes = torch.cat(
        (
            gt[:,:,0].min(dim=1, keepdim=True).values, gt[:,:,1].min(dim=1, keepdim=True).values,
            gt[:,:,0].max(dim=1, keepdim=True).values, gt[:,:,1].max(dim=1, keepdim=True).values
            ), dim=1)

    IoUs = bbox_overlaps(anchor_boxes, gt_boxes)

    return IoUs

def calc_OKS(anchor, gt):
    # anchor: N x 21 x 2
    # gt: K x 21 x 3 [u,v,vis] or K x 21 x 2 [u,v]
    # OKS = Σi[exp(-di2/2s2κi2)δ(vi>0)] / Σi[δ(vi>0)]

    sigmas = torch.tensor([1.0, 0.6, 0.6, 0.6, 0.4, # palm and thumb from mcp to tip
                            0.6, 0.6, 0.6, 0.4,
                            0.6, 0.6, 0.6, 0.4,
                            0.6, 0.6, 0.6, 0.4,
                            0.6, 0.6, 0.6, 0.4])/10.0 # pinky
    vars = (sigmas[:anchor.shape[1]]*2)**2 # 21
    vars = vars.to(gt.device)
    k = len(sigmas)

    u_gt, v_gt = gt[None, :, :, 0], gt[None, :, :, 1], #u,v: 1 x K x 21   vis: 1 x k x 21
    vis = gt[None, :, :, 2] if gt.shape[2] == 3 else torch.ones(*u_gt.shape).to(gt.device)

    u_anc = anchor[:, None, :, 0]; v_anc = anchor[:, None, :, 1] # N x 1 x 21 

    du = u_gt - u_anc # N x K x 21
    dv = v_gt - v_anc # N x K x 21
    gt_area = (u_gt.max(dim=2).values - u_gt.min(dim=2).values) * (v_gt.max(dim=2).values - v_gt.min(dim=2).values) # 1 x K

    numerator = (du**2 + dv**2) # N x K x 21  the square dist between the joints of each pair of pose
    denominator = 2 * gt_area.unsqueeze(2) * vars[None, None, :].to(gt.device) # 1 x K x 21

    e = numerator / denominator # N x K x 21

    # threshold = 0.5
    # if threshold is not None:
    #     e = e[:, vis == 1]
    OKS = torch.sum(torch.exp(-e) * vis, dim=2) / vis.sum(dim=2) # N x K

    return OKS

def pose_IoU_batch(anchors, pose_gt):
    """
    anchors: (N, 21, 2) ndarray of float  N anchors per img
    pose_gt: (b, K, 21, 2) ndarray of float   K gt poses per img

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    IoUs = torch.stack([calc_pose_IoU(anchors, pose_gt[b]) for b in range(pose_gt.shape[0])]) # b x N x K

    return IoUs

def pose_OKS_batch(anchors, pose_gt):
    """
    anchors: (N, 21, 2) ndarray of float  N anchors per img
    pose_gt: (b, K, 21, 2) ndarray of float   K gt poses per img

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    OKSs = torch.stack([calc_OKS(anchors, pose_gt[b]) for b in range(pose_gt.shape[0])]) # b x N x K
    
    return OKSs

def bbox_overlaps_batch(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (b, K, 5) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    batch_size = gt_boxes.size(0)

    if anchors.dim() == 2:

        N = anchors.size(0)
        K = gt_boxes.size(1)

        anchors = anchors.view(1, N, 4).expand(batch_size, N, 4).contiguous()
        gt_boxes = gt_boxes[:,:,:4].contiguous()

        gt_boxes_x = (gt_boxes[:,:,2] - gt_boxes[:,:,0] + 1)
        gt_boxes_y = (gt_boxes[:,:,3] - gt_boxes[:,:,1] + 1)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

        anchors_boxes_x = (anchors[:,:,2] - anchors[:,:,0] + 1)
        anchors_boxes_y = (anchors[:,:,3] - anchors[:,:,1] + 1)
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

        iw = (torch.min(boxes[:,:,:,2], query_boxes[:,:,:,2]) -
            torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:,:,:,3], query_boxes[:,:,:,3]) -
            torch.max(boxes[:,:,:,1], query_boxes[:,:,:,1]) + 1)
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih)
        overlaps = iw * ih / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)

    elif anchors.dim() == 3:
        N = anchors.size(1)
        K = gt_boxes.size(1)

        if anchors.size(2) == 4:
            anchors = anchors[:,:,:4].contiguous()
        else:
            anchors = anchors[:,:,1:5].contiguous()

        gt_boxes = gt_boxes[:,:,:4].contiguous()

        gt_boxes_x = (gt_boxes[:,:,2] - gt_boxes[:,:,0] + 1)
        gt_boxes_y = (gt_boxes[:,:,3] - gt_boxes[:,:,1] + 1)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

        anchors_boxes_x = (anchors[:,:,2] - anchors[:,:,0] + 1)
        anchors_boxes_y = (anchors[:,:,3] - anchors[:,:,1] + 1)
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

        iw = (torch.min(boxes[:,:,:,2], query_boxes[:,:,:,2]) -
            torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:,:,:,3], query_boxes[:,:,:,3]) -
            torch.max(boxes[:,:,:,1], query_boxes[:,:,:,1]) + 1)
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih)
        
        # Intersection (iw * ih) divided by Union (ua)
        overlaps = iw * ih / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)
    else:
        raise ValueError('anchors input dimension is not correct.')

    return overlaps

if __name__ == '__main__':
    p1 = torch.randint(1,10,(1,3,2)).float()
    gt = p1#torch.randint(1,10,(1,3,2)).float()

    print(p1)
    print(gt)
    print(calc_OKS(p1, gt))