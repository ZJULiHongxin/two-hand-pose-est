# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os
import cv2
import numpy as np
from numpy.core.numeric import count_nonzero
import torch
import matplotlib.pyplot as plt
from torch.utils.data import dataset
# from utils.transforms import flip_back, scale_pose3d, scale_pose2d
# from utils.vis import save_debug_images
# from utils.heatmap_decoding import get_final_preds


def train_helper(epoch, i, args, config, master, ret, model, optimizer, dataset_name, train_loader, writer_dict, logger, output_dir, tb_log_dir,
                pose3d_gt=None, recorder=None, fp16=False, device=None):
    end = time.time()
    #
    imgs, heatmaps_gt, pose2d_gt, visibility = ret['imgs'], ret['heatmaps'], ret['pose2d'], ret['visibility']
    if config.MODEL.NAME == 'CPM':
        # each element of heatmap_lst is of size b x 22 x 32 x 32
        heatmap_lst = model(imgs.cuda(device), center_map=ret['centermaps'].cuda(device))
        heatmaps_pred = heatmap_lst[-1]#torch.cat((heatmap_lst), dim=1) # b x 22*6 x 32 x 32
        heatmaps_gt = heatmaps_gt.repeat((1,1,1,1))
        pose2d_pred = get_final_preds(heatmaps_pred, use_softmax=config.MODEL.HEATMAP_SOFTMAX) # batch_size x 21 x 2
    elif 'Aggr' in config.MODEL.NAME:
        # imgs: b x (4*seq_len) x 3 x 256 x 256
        n_batches, seq_len = imgs.shape[0], imgs.shape[1] // 4

        pose2d_gt = torch.cat(
            [pose2d_gt[b,4*(seq_len//2):4*(seq_len//2+1)] for b in range(n_batches)],
            dim=0)
        heatmaps_gt = torch.cat(
            [heatmaps_gt[b,4*(seq_len//2):4*(seq_len//2+1)] for b in range(n_batches)],
            dim=0)
        visibility = torch.cat(
            [visibility[b,4*(seq_len//2):4*(seq_len//2+1)] for b in range(n_batches)],
            dim=0)

        imgs = torch.cat(
            [imgs[b,4*j:4*(j+1)] for j in range(seq_len) for b in range(n_batches)],
            dim=0) # (b*4*5) x 3 x 256 x 256

        heatmaps_pred, temperature = model(imgs.cuda(device))
        pose2d_pred = get_final_preds(heatmaps_pred, use_softmax=config.MODEL.HEATMAP_SOFTMAX) # batch_size x 21 x 2
    elif config.MODEL.NAME == 'pose_hrnet_transformer':
        # imgs: b(1) x (4*seq_len) x 3 x 256 x 256
        n_batches, seq_len = imgs.shape[0], imgs.shape[1] // 4
        idx_lst = torch.tensor([4 * i for i in range(seq_len)])
        imgs = torch.stack(
            [imgs[b, idx_lst + cam_idx] for b in range(n_batches) for cam_idx in range(4)]
        ) # (b*4) x seq_len x 3 x 256 x 256
        pose2d_pred, heatmaps_pred, temperature = model(imgs.cuda(device)) # (b*4) x 21 x 2

        pose2d_gt = pose2d_gt[:,4*(seq_len//2):4*(seq_len//2+1)].contiguous().view(-1, *pose2d_pred.shape[-2:]) # (b*4) x 21 x 2
        heatmaps_gt = heatmaps_gt[:,4*(seq_len//2):4*(seq_len//2+1)].contiguous().view(-1, *heatmaps_gt.shape[-3:]) # (b*4) x 21 x 64 x 64
        visibility = visibility[:,4*(seq_len//2):4*(seq_len//2+1)].view(-1, *visibility.shape[-2:]) # (b*4) x 21
    elif config.MODEL.NAME == 'my_pose_transformer':
        pose2d_pred = model(imgs.cuda(device)) # b x 21 x 2
    elif config.MODEL.NAME == 'pose_hrnet_hamburger':
        heatmaps_pred, temperature = model(imgs.cuda(device))
        pose2d_pred = get_final_preds(heatmaps_pred, use_softmax=config.MODEL.HEATMAP_SOFTMAX) # batch_size x 21 x 2
    else:
        heatmaps_pred, _, temperature = model(imgs.cuda(device))
        pose2d_pred = get_final_preds(heatmaps_pred, use_softmax=config.MODEL.HEATMAP_SOFTMAX) # batch_size x 21 x 2
    
    heatmaps_gt = heatmaps_gt.cuda(device, non_blocking=True) if config.LOSS.WITH_HEATMAP_LOSS else heatmaps_gt
    pose2d_gt = pose2d_gt.cuda(device, non_blocking=True)  if config.LOSS.WITH_POSE2D_LOSS else pose2d_gt # batch_size x 21 x 3
    visibility = visibility.cuda(device, non_blocking=True).squeeze()   if config.LOSS.WITH_POSE2D_LOSS else visibility.squeeze()   # b x 21

    item_dict = {}
    # calculate losses  
    loss_items = config.LOSS
    # the palm position in MHP is not consistent with that in other datasets
    MHP_flag =  'MHP' in dataset_name and len(config.DATASET.DATASET) > 1
    if loss_items.WITH_HEATMAP_LOSS:
        item_dict['heatmaps_pred'] = heatmaps_pred[:,1:] if MHP_flag else heatmaps_pred
        item_dict['heatmaps_gt'] = heatmaps_gt[:,1:] if MHP_flag else heatmaps_gt
    if loss_items.WITH_POSE2D_LOSS or loss_items.WITH_BONE_LOSS or loss_items.WITH_JOINTANGLE_LOSS:
        item_dict['pose2d_pred'] = pose2d_pred[:,1:] if MHP_flag else pose2d_pred
        item_dict['pose2d_gt'] = pose2d_gt[:,1:] if MHP_flag else pose2d_gt
        item_dict['visibility'] = visibility

    loss_dict = recorder.computeLosses(item_dict)

    total_loss = loss_dict['total_loss']
    heatmap_loss = loss_dict['heatmap_loss']
    pose2d_loss = loss_dict['pose2d_loss']
    TC_loss = loss_dict['TC_loss']
    bone_loss = loss_dict['bone_loss']
    jointangle_loss = loss_dict['jointangle_loss']

    # compute gradient and do update step
    optimizer.zero_grad()
    if fp16:
        optimizer.backward(total_loss)
    else:
        total_loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time = time.time() - end

    if i % config.PRINT_FREQ == 0 and master:
        recorder.computeAvgLosses()

        msg = 'Dataset: {0} Epoch: [{1}][{2}/{3}]\t' \
            'Time {batch_time:.3f}s\t' \
            'Speed {speed:.1f} samples/s\t' \
            'TotalLoss {total_loss:.5f} ({total_loss_avg:.5f})'.format(
                dataset_name, epoch, i, len(train_loader),
                batch_time=batch_time,
                speed=imgs.size(0)/batch_time,
                total_loss=total_loss.item(),
                total_loss_avg=recorder.avg_total_loss)
        
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']

        if heatmap_loss is not None:
            msg += '\tHeatmapLoss {heatmap_loss:.5f} ({heatmap_loss_avg:.5f})'.format(
                heatmap_loss = heatmap_loss.item(), heatmap_loss_avg = recorder.avg_heatmap_loss
            )
            writer.add_scalar('train_loss/heatmap_loss', heatmap_loss, global_steps)
        if pose2d_loss:
            msg += '\tPose2DLoss {pose2d_loss:.5f} ({pose2d_loss_avg:.5f})'.format(
                pose2d_loss = pose2d_loss.item(), pose2d_loss_avg = recorder.avg_pose2d_loss
            )
            writer.add_scalar('train_loss/pose2d_loss', pose2d_loss, global_steps)
        if TC_loss:
            msg += '\tTimeConsistencyLoss {TC_loss:.5f} ({TC_loss_avg:.5f})'.format(
                TC_loss = TC_loss.item(), TC_loss_avg = recorder.avg_time_consistency_loss
            )
            writer.add_scalar('train_loss/time_consistency_loss', TC_loss, global_steps)
        if bone_loss:
            msg += '\tBoneLoss {Bone_loss:.5f} ({Bone_loss_avg:.5f})'.format(
                Bone_loss = bone_loss.item(), Bone_loss_avg = recorder.avg_bone_loss
            )
            writer.add_scalar('train_loss/bone_loss', bone_loss, global_steps)
        if jointangle_loss:
            msg += '\tJointangleLoss {Jointangle_loss:.5f} ({Jointangle_loss_avg:.5f})'.format(
                Jointangle_loss= jointangle_loss.item(), Jointangle_loss_avg = recorder.avg_jointangle_loss
            )
            writer.add_scalar('train_loss/jointangle_loss', jointangle_loss, global_steps)
        logger.info(msg)
        
        writer.add_scalar('train_loss/total_loss', total_loss, global_steps)

        if getattr(config.MODEL, 'USE_TEMP_NET', False):
            writer.add_histogram('train_loss/trainable_temperature', temperature, global_step=global_steps)
        elif config.MODEL.TRAINABLE_SOFTMAX == True:
            # when DP is used, temperature is a vector of replicas; when DDP is used, temperature is a scalar
            writer.add_scalar('train_loss/trainable_temperature', temperature[0] if len(temperature.shape) > 0 else temperature, global_steps)

        #prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
        #save_debug_images(config, imgs, heatmaps_gt, heatmaps_pred, prefix)

    writer_dict['train_global_steps'] += 1

def train(config, master, train_loader, model, optimizer, epoch,
          writer_dict, logger, fp16=False, device=None):
    """
    - Brief: Training phase
    - params:
        config
        train_loader:
        model:
        criterion: a dict containing loss items
    """

    writer = writer_dict['writer']
    # switch to train mode
    model.train()

    dataset_name = train_loader.dataset.name
    logger.info('Training on {} dataset [Batch size: {}]\n'.format(train_loader.dataset.name, train_loader.batch_size))

    if 'interhand' in dataset_name.lower() and 'anchorpose' in config.MODEL.NAME.lower():
        for i, ret in enumerate(train_loader):
            end = time.time()
            # imgs: b x 3 x H x W
            # pose2d_gt: b x 42 x 3
            # hand_type: b x 2 ([1,0] for right, [0,1] for left and [1,1] for interacting hands)
            imgs, pose2d_gt = ret['imgs'].cuda(device, non_blocking=True), ret['pose2d_gt'].cuda(device, non_blocking=True)
            hand_type = ret['hand_type']

            assert imgs.isnan().sum() == 0
            output_dict = model(imgs, pose2d_gt, hand_type)

            KL_loss, cls_loss = output_dict['KL_loss'], output_dict['cls_loss']
            total_loss = KL_loss + cls_loss
            
            # compute gradient and do update step
            optimizer.zero_grad()
            if fp16:
                optimizer.backward(total_loss)
            else:
                total_loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time = time.time() - end

            if i % config.PRINT_FREQ == 0 and master:
                msg = 'Dataset: {0} Epoch: [{1}][{2}/{3}]\t' \
                    'Time {batch_time:.3f}s\t' \
                    'Speed {speed:.1f} samples/s\t' \
                    'TotalLoss {total_loss:.5f} '.format(
                        dataset_name, epoch, i+1, len(train_loader),
                        batch_time=batch_time,
                        speed=imgs.size(0)/batch_time,
                        total_loss=total_loss.item(),
                        )
                
                writer = writer_dict['writer']
                writer.add_histogram('handedness', output_dict['handedness_pred'], global_step=None, bins='tensorflow', walltime=None, max_bins=None)
                global_steps = writer_dict['train_global_steps']

                msg += '\tKL_Loss {KL_loss:.5f}'.format(KL_loss = KL_loss.item())
                writer.add_scalar('train_loss/KL_loss', KL_loss, global_steps)

                msg += '\tcls_Loss {cls_loss:.5f}'.format(cls_loss = cls_loss.item())
                writer.add_scalar('train_loss/cls_loss', cls_loss, global_steps)

                logger.info(msg)
                
                writer.add_scalar('train_loss/total_loss', total_loss, global_steps)

            writer_dict['train_global_steps'] += 1
            if config.DEBUG.DEBUG and  i==4: break
    
    elif 'interhand' in dataset_name.lower() and 'a2j' in config.MODEL.NAME.lower():
        smooth_loss = torch.nn.SmoothL1Loss(reduction='mean')
        for i, ret in enumerate(train_loader):
            end = time.time()
            # imgs: b x 3 x H x W
            # pose2d_gt: b x 42 x 3
            # hand_type: b x 2 ([1,0] for right, [0,1] for left and [1,1] for interacting hands)
            imgs, pose2d_gt = ret['imgs'].cuda(device, non_blocking=True), ret['pose2d_gt'].cuda(device, non_blocking=True)
            hand_type = ret['hand_type']
            batch_size = imgs.shape[0]
            # cls: b x w*h*n_anchors x n_joints
            # pose_pred: B x n_joints x 2
            # surrounding_anchors_pred: B x n_joints x 2
            # reg: B x w*h*n_anchors x n_joints x 2
            pose_pred, surrounding_anchors_pred, cls_pred, reg, temperature = model(imgs)

            #print('\tPred\t\t\tgroundtruth')
            #for k in range(42):
            #    print(pose_pred[0,k].tolist(), pose2d_gt[0,k].tolist())

            #input()
            anchor_loss, surrounding_anchor_loss = 0., 0.
            for b in range(imgs.shape[0]):
                if hand_type[b,0] == 1: # right
                    anchor_loss += smooth_loss(target=pose2d_gt[b,0:21,0:2], input=pose_pred[b,0:21])
                    surrounding_anchor_loss += smooth_loss(target=pose2d_gt[b,0:21,0:2], input=surrounding_anchors_pred[b,0:21])
                elif hand_type[b,1] == 1: # left
                    anchor_loss += smooth_loss(target=pose2d_gt[b,21:42,0:2], input=pose_pred[b,21:42])
                    surrounding_anchor_loss += smooth_loss(target=pose2d_gt[b,21:42,0:2], input=surrounding_anchors_pred[b,21:42])

            anchor_loss /= batch_size
            surrounding_anchor_loss /= batch_size
            total_loss =  config.LOSS.POSE2D_LOSS_FACTOR * anchor_loss + config.LOSS.SURROUDING_ANCHOR_LOSS_FACTOR * surrounding_anchor_loss
            
            # TODO: design a loss functon for cls_pred in case that a certain hand is missing

            # compute gradient and do update step
            optimizer.zero_grad()
            if fp16:
                optimizer.backward(total_loss)
            else:
                total_loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time = time.time() - end

            if i % config.PRINT_FREQ == 0 and master:
                msg = 'Dataset: {0} Epoch: [{1}][{2}/{3}]\t' \
                    'Time {batch_time:.3f}s\t' \
                    'Speed {speed:.1f} samples/s\t' \
                    'TotalLoss {total_loss:.5f} '.format(
                        dataset_name, epoch, i+1, len(train_loader),
                        batch_time=batch_time,
                        speed=imgs.size(0)/batch_time,
                        total_loss=total_loss.item(),
                        )
                
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']

                msg += '\tanchor_loss {anchor_loss:.5f}'.format(anchor_loss = anchor_loss.item())
                writer.add_scalar('train_loss/anchor_loss', anchor_loss, global_steps)

                msg += '\tsurrounding_loss {surrounding_anchor_loss:.5f}'.format(surrounding_anchor_loss = surrounding_anchor_loss.item())
                writer.add_scalar('train_loss/surrounding_loss', surrounding_anchor_loss, global_steps)

                writer.add_scalar('train_loss/temperature', temperature, global_steps)
                logger.info(msg)
                
                writer.add_scalar('train_loss/total_loss', total_loss, global_steps)

            writer_dict['train_global_steps'] += 1
            if config.DEBUG.DEBUG and  i==4: break
    
    elif 'interhand' in dataset_name.lower() and 'Baseline' in config.MODEL.NAME:
        smooth_loss = torch.nn.SmoothL1Loss(reduction='mean')
        for i, ret in enumerate(train_loader):
            end = time.time()
            # imgs: b x 3 x H x W
            # pose2d_gt: b x 42 x 3
            # hand_type: b x 2 ([1,0] for right, [0,1] for left and [1,1] for interacting hands)
            imgs, pose2d_gt = ret['imgs'].cuda(device, non_blocking=True), ret['pose2d_gt'].cuda(device, non_blocking=True)
            hand_type = ret['hand_type']
            batch_size = imgs.shape[0]
            # cls: b x w*h*n_anchors x n_joints
            # pose_pred: B x n_joints x 2
            # surrounding_anchors_pred: B x n_joints x 2
            # reg: B x w*h*n_anchors x n_joints x 2
            pose_pred = model(imgs)

            #print('\tPred\t\t\tgroundtruth')
            #for k in range(42):
            #    print(pose_pred[0,k].tolist(), pose2d_gt[0,k].tolist())

            #input()
            pose2d_loss = 0.
            for b in range(imgs.shape[0]):
                if hand_type[b,0] == 1: # right
                    pose2d_loss += smooth_loss(target=pose2d_gt[b,0:21,0:2], input=pose_pred[b,0:21])
                elif hand_type[b,1] == 1: # left
                    pose2d_loss += smooth_loss(target=pose2d_gt[b,21:42,0:2], input=pose_pred[b,21:42])

            pose2d_loss /= batch_size
            total_loss =  config.LOSS.POSE2D_LOSS_FACTOR * pose2d_loss
            
            # TODO: design a loss functon for cls_pred in case that a certain hand is missing

            # compute gradient and do update step
            optimizer.zero_grad()
            if fp16:
                optimizer.backward(total_loss)
            else:
                total_loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time = time.time() - end

            if i % config.PRINT_FREQ == 0 and master:
                msg = 'Dataset: {0} Epoch: [{1}][{2}/{3}]\t' \
                    'Time {batch_time:.3f}s\t' \
                    'Speed {speed:.1f} samples/s\t' \
                    'TotalLoss {total_loss:.5f} '.format(
                        dataset_name, epoch, i+1, len(train_loader),
                        batch_time=batch_time,
                        speed=imgs.size(0)/batch_time,
                        total_loss=total_loss.item(),
                        )
                
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']

                msg += '\tpose2d_loss {pose2d_loss:.5f}'.format(anchor_loss = pose2d_loss.item())
                writer.add_scalar('train_loss/pose2d_loss', pose2d_loss, global_steps)

                logger.info(msg)
                
                writer.add_scalar('train_loss/total_loss', total_loss, global_steps)

            writer_dict['train_global_steps'] += 1
            if config.DEBUG.DEBUG and  i==4: break
     
    else:
        print('[Error in funtion.py] Invalid dataset!')
        exit()


def val_helper(i, config, args, master, ret, model, dataset_name, val_loader, recorder, logger, output_dir, tb_log_dir, device=None):
    end = time.time()
    imgs, heatmaps_gt, pose2d_gt, visibility = ret['imgs'], ret['heatmaps'], ret['pose2d'], ret['visibility']
    # compute output
    if config.MODEL.NAME == 'CPM':
        # each element of heatmap_lst is of size b x 22 x 32 x 32
        heatmap_lst = model(imgs.cuda(device), center_map=ret['centermaps'].cuda(device))
        heatmaps_pred = heatmap_lst[-1]#torch.cat((heatmap_lst), dim=1) # b x 22*6 x 32 x 32
        heatmaps_gt = heatmaps_gt.repeat((1,1,1,1))
        pose2d_pred = get_final_preds(heatmaps_pred, use_softmax=config.MODEL.HEATMAP_SOFTMAX)
    elif 'Aggr' in config.MODEL.NAME:
        # imgs: b x (4*5) x 3 x 256 x 256
        n_batches, seq_len = imgs.shape[0], imgs.shape[1] // 4

        pose2d_gt = torch.cat(
            [pose2d_gt[b,4*(seq_len//2):4*(seq_len//2+1)] for b in range(n_batches)],
            dim=0)
        heatmaps_gt = torch.cat(
            [heatmaps_gt[b,4*(seq_len//2):4*(seq_len//2+1)] for b in range(n_batches)],
            dim=0)
        visibility = torch.cat(
            [visibility[b,4*(seq_len//2):4*(seq_len//2+1)] for b in range(n_batches)],
            dim=0)

        imgs = torch.cat(
            [imgs[b,4*j:4*(j+1)] for j in range(seq_len) for b in range(n_batches)],
            dim=0) # (b*4*5) x 3 x 256 x 256

        heatmaps_pred, temperature = model(imgs.cuda(device))
        pose2d_pred = get_final_preds(heatmaps_pred, use_softmax=config.MODEL.HEATMAP_SOFTMAX)
    elif config.MODEL.NAME == 'pose_hrnet_transformer':
        # imgs: b(1) x (4*seq_len) x 3 x 256 x 256
        n_batches, seq_len = imgs.shape[0], imgs.shape[1] // 4
        idx_lst = torch.tensor([4 * i for i in range(seq_len)])
        imgs = torch.stack(
            [imgs[b, idx_lst + cam_idx] for b in range(n_batches) for cam_idx in range(4)]
        ) # (b*4) x seq_len x 3 x 256 x 256
        pose2d_pred, heatmaps_pred, temperature = model(imgs.cuda(device)) # (b*4) x 21 x 2

        pose2d_gt = pose2d_gt[:,4*(seq_len//2):4*(seq_len//2+1)].contiguous().view(-1, *pose2d_pred.shape[-2:]) # (b*4) x 21 x 2
        heatmaps_gt = heatmaps_gt[:,4*(seq_len//2):4*(seq_len//2+1)].contiguous().view(-1, *heatmaps_gt.shape[-3:]) # (b*4) x 21 x 64 x 64
        visibility = visibility[:,4*(seq_len//2):4*(seq_len//2+1)].contiguous().view(-1, *visibility.shape[-2:]) # (b*4) x 21
    elif config.MODEL.NAME == 'my_pose_transformer':
        pose2d_pred = model(imgs.cuda(device)) # b x 21 x 2
    elif config.MODEL.NAME == 'pose_hrnet_hamburger':
        heatmaps_pred, temperature = model(imgs.cuda(device))
        pose2d_pred = get_final_preds(heatmaps_pred, use_softmax=config.MODEL.HEATMAP_SOFTMAX) # batch_size x 21 x 2
    else:
        heatmaps_pred, _, temperature = model(imgs.cuda(device))
        pose2d_pred = get_final_preds(heatmaps_pred, use_softmax=config.MODEL.HEATMAP_SOFTMAX) # batch_size x 21 x 2
        
    if config.TEST.FLIP_TEST:
        # this part is ugly, because pytorch has not supported negative index
        # input_flipped = model(images[:, :, :, ::-1])
        input_flipped = np.flip(imgs.cpu().numpy(), 3).copy()
        input_flipped = torch.from_numpy(input_flipped).cuda(device)
        heatmaps_pred_flipped = model(input_flipped.cuda(device).float())

        if isinstance(heatmaps_pred_flipped, list):
            heatmaps_pred_flipped = heatmaps_pred_flipped[-1]

        heatmaps_pred_flipped = flip_back(heatmaps_pred_flipped.cpu().numpy(),
                                val_dataset.flip_pairs)
        heatmaps_pred_flipped = torch.from_numpy(heatmaps_pred_flipped.copy()).cuda(device)


        # feature is not aligned, shift flipped heatmap for higher accuracy
        if config.TEST.SHIFT_HEATMAP:
            heatmaps_pred_flipped[:, :, :, 1:] = \
                heatmaps_pred_flipped.clone()[:, :, :, 0:-1]

        heatmaps_pred = (heatmaps_pred + heatmaps_pred_flipped) * 0.5

    heatmaps_gt = heatmaps_gt.cuda(device, non_blocking=True) if config.LOSS.WITH_HEATMAP_LOSS else heatmaps_gt
    pose2d_gt = pose2d_gt.cuda(device, non_blocking=True)  if config.LOSS.WITH_POSE2D_LOSS else pose2d_gt # batch_size x 21 x 3
    visibility = visibility.cuda(device, non_blocking=True).squeeze()   if config.LOSS.WITH_POSE2D_LOSS else visibility.squeeze()   # b x 21

    # calculate losses
    item_dict = {}
    # calculate losses  
    loss_items = config.LOSS
    # the palm position in MHP is not consistent with that in other datasets
    MHP_flag =  'MHP' in dataset_name and len(config.DATASET.DATASET) > 1
    if loss_items.WITH_HEATMAP_LOSS:
        item_dict['heatmaps_pred'] = heatmaps_pred[:,1:] if MHP_flag else heatmaps_pred
        item_dict['heatmaps_gt'] = heatmaps_gt[:,1:] if MHP_flag else heatmaps_gt
    if loss_items.WITH_POSE2D_LOSS or loss_items.WITH_BONE_LOSS or loss_items.WITH_JOINTANGLE_LOSS:
        item_dict['pose2d_pred'] = pose2d_pred[:,1:] if MHP_flag else pose2d_pred
        item_dict['pose2d_gt'] = pose2d_gt[:,1:] if MHP_flag else pose2d_gt
        item_dict['visibility'] = visibility

    loss_dict = recorder.computeLosses(item_dict)

    total_loss=loss_dict['total_loss']
    heatmap_loss=loss_dict['heatmap_loss']
    pose2d_loss = loss_dict['pose2d_loss']
    TC_loss = loss_dict['TC_loss']
    bone_loss = loss_dict['bone_loss']
    jointangle_loss = loss_dict['jointangle_loss']

    if master and i % config.PRINT_FREQ == 0:
        # measure elapsed time
        batch_time = time.time() - end
        recorder.computeAvgLosses()

        msg = 'Dataset: {0} Test: [{1}/{2}]\t' \
        'Time {batch_time:.3f}s\t' \
        'Speed {speed:.1f} samples/s\t' \
        'TotalLoss {total_loss:.5f} ({total_loss_avg:.5f})'.format(
            dataset_name, i, len(val_loader), batch_time=batch_time,
            speed=imgs.size(0)/batch_time,
            total_loss=total_loss.item(),
            total_loss_avg=recorder.avg_total_loss)

        if heatmap_loss is not None:
            msg += '\tHeatmapLoss {heatmap_loss:.5f} ({heatmap_loss_avg:.5f})'.format(
                heatmap_loss = heatmap_loss.item(), heatmap_loss_avg = recorder.avg_heatmap_loss
            )

        if pose2d_loss is not None:
            msg += '\tPose2DLoss {pose2d_loss:.5f} ({pose2d_loss_avg:.5f})'.format(
                pose2d_loss = pose2d_loss.item(), pose2d_loss_avg = recorder.avg_pose2d_loss
            )
            
        if TC_loss is not None:
            msg += '\tTimeConsistencyLoss {TC_loss:.5f} ({TC_loss_avg:.5f})'.format(
                TC_loss = TC_loss.item(), TC_loss_avg = recorder.avg_time_consistency_loss
            )
        if bone_loss is not None:
            msg += '\tBoneLoss {bone_loss:.5f} ({bone_loss_avg:.5f})'.format(
                bone_loss = bone_loss.item(), bone_loss_avg = recorder.avg_bone_loss
            )
        if jointangle_loss is not None:
            msg += '\tJointAngleLoss {jointangle_loss:.5f} ({jointangle_loss_avg:.5f})'.format(
                jointangle_loss = jointangle_loss.item(),jointangle_loss_avg = recorder.avg_jointangle_loss
            )
        
        logger.info(msg)

        # prefix = '{}_{}'.format(
        #     os.path.join(output_dir, 'val'), i
        # )
        #save_debug_images(config, imgs, heatmaps_gt, heatmaps_pred, prefix)

def val_writer_helper(config, writer, recorder, global_steps):
    writer.add_scalar('val_loss/total_loss', recorder.avg_total_loss, global_steps)
    loss_terms = config.LOSS
    if loss_terms.WITH_HEATMAP_LOSS:
        writer.add_scalar('val_loss/heatmap_loss', recorder.avg_heatmap_loss, global_steps)
    if loss_terms.WITH_POSE2D_LOSS:
        writer.add_scalar('val_loss/pose2d_loss', recorder.avg_pose2d_loss, global_steps)
    if loss_terms.WITH_TIME_CONSISTENCY_LOSS:
        writer.add_scalar('val_loss/time_consistency_loss', recorder.avg_time_consistency_loss, global_steps)
    if loss_terms.WITH_BONE_LOSS:
        writer.add_scalar('val_loss/bone_loss', recorder.avg_bone_loss, global_steps)
    if loss_terms.WITH_JOINTANGLE_LOSS:
        writer.add_scalar('val_loss/jointangle_loss', recorder.avg_jointangle_loss, global_steps)

def validate(config, args, master, val_loader, model, writer_dict, logger, device=None):

    writer = writer_dict['writer']
    # switch to evaluate mode
    model.eval()

    dataset_name = val_loader.dataset.name
    logger.info('Validating on {} dataset [Batch size: {}]\n'.format(dataset_name, val_loader.batch_size))

    if dataset_name == 'InterHand' and 'anchorpose' in config.MODEL.NAME.lower():

        avg_total_loss, avg_KL_loss, avg_cls_loss = 0., 0., 0.
        end = time.time()
        for i, ret in enumerate(val_loader):
            # imgs: b x 3 x H x W
            # pose2d_gt: b x 42 x 2
            imgs, pose2d_gt = ret['imgs'].cuda(device, non_blocking=True), ret['pose2d_gt'].cuda(device, non_blocking=True)
            hand_type = ret['hand_type']
            
            output_dict = model(imgs, pose2d_gt, hand_type)

            KL_loss, cls_loss = output_dict['KL_loss'], output_dict['cls_loss']
            avg_KL_loss += KL_loss
            avg_cls_loss += cls_loss
            total_loss = KL_loss + cls_loss
            avg_total_loss += total_loss
            # measure elapsed time
            if master and i % config.PRINT_FREQ == 0:
                # measure elapsed time
                batch_time = time.time() - end

                msg = 'Dataset: {0} Test: [{1}/{2}]\t' \
                'Time {batch_time:.3f}s\t' \
                'Speed {speed:.1f} samples/s\t' \
                'TotalLoss {total_loss:.5f}'.format(
                    dataset_name, i+1, len(val_loader), batch_time=batch_time,
                    speed=imgs.size(0)/batch_time,
                    total_loss=total_loss.item())

                msg += '\tKL_Loss {KL_loss:.5f}'.format(KL_loss = KL_loss.item())
                msg += '\tcls_Loss {cls_loss:.5f}'.format(cls_loss = cls_loss.item())

                logger.info(msg)

            if config.DEBUG.DEBUG and  i==4: break

        global_steps = writer_dict['valid_global_steps']

        if master:
            n = len(val_loader)
            writer.add_scalar('val_loss/total_loss', avg_total_loss / n, global_steps)
            writer.add_scalar('val_loss/KL_Loss', avg_total_loss / n, global_steps)
            writer.add_scalar('val_loss/total_loss', avg_total_loss / n, global_steps)
        
        writer_dict['valid_global_steps'] = global_steps + 1

    elif dataset_name == 'InterHand' and 'A2JPoseNet' in config.MODEL.NAME:
        smooth_loss = torch.nn.SmoothL1Loss(reduction='mean')
        avg_total_loss, avg_anchor_loss, avg_surrounding_anchor_loss = 0., 0., 0.
        
        for i, ret in enumerate(val_loader):
            end = time.time()
            # imgs: b x 3 x H x W
            # pose2d_gt: b x 42 x 3
            # hand_type: b x 2 ([1,0] for right, [0,1] for left and [1,1] for interacting hands)
            imgs, pose2d_gt = ret['imgs'].cuda(device, non_blocking=True), ret['pose2d_gt'].cuda(device, non_blocking=True)
            hand_type = ret['hand_type']
            batch_size = imgs.shape[0]
            # cls: b x w*h*n_anchors x 42
            # pose_pred: B x 42 x 2
            # reg: B x w*h*n_anchors x 42 x 2
            pose_pred, surrounding_anchors_pred, cls_pred, reg, temperature = model(imgs)

            anchor_loss, surrounding_anchor_loss = 0., 0.
            for b in range(imgs.shape[0]):
                if hand_type[b,0] == 1: # right
                    anchor_loss += smooth_loss(target=pose2d_gt[b,0:21,0:2], input=pose_pred[b,0:21])
                    surrounding_anchor_loss += smooth_loss(target=pose2d_gt[b,0:21,0:2], input=surrounding_anchors_pred[b,0:21])
                elif hand_type[b,1] == 1: # left
                    anchor_loss += smooth_loss(target=pose2d_gt[b,21:42,0:2], input=pose_pred[b,21:42])
                    surrounding_anchor_loss += smooth_loss(target=pose2d_gt[b,21:42,0:2], input=surrounding_anchors_pred[b,21:42])
            
            anchor_loss /= batch_size
            surrounding_anchor_loss /= batch_size
            total_loss = config.LOSS.POSE2D_LOSS_FACTOR * anchor_loss + config.LOSS.SURROUDING_ANCHOR_LOSS_FACTOR * surrounding_anchor_loss
            
            avg_total_loss += total_loss
            avg_anchor_loss += anchor_loss
            avg_surrounding_anchor_loss += surrounding_anchor_loss
            
            # measure elapsed time
            if master and i % config.PRINT_FREQ == 0:
                # measure elapsed time
                batch_time = time.time() - end

                msg = 'Dataset: {0} Test: [{1}/{2}]\t' \
                'Time {batch_time:.3f}s\t' \
                'Speed {speed:.1f} samples/s\t' \
                'TotalLoss {total_loss:.5f}'.format(
                    dataset_name, i+1, len(val_loader), batch_time=batch_time,
                    speed=imgs.size(0)/batch_time,
                    total_loss=total_loss.item())

                msg += '\tanchor_loss {anchor_loss:.5f}'.format(anchor_loss = anchor_loss.item())
                msg += '\tsurrounding_loss {surrounding_anchor_loss:.5f}'.format(surrounding_anchor_loss = surrounding_anchor_loss.item())

                logger.info(msg)

            if config.DEBUG.DEBUG and  i==4: break
        
        global_steps = writer_dict['valid_global_steps']
        
        n = len(val_loader)
        avg_total_loss /= n
        if master:
            writer.add_scalar('val_loss/total_loss', avg_total_loss, global_steps)
            writer.add_scalar('val_loss/anchor_loss', avg_anchor_loss / n, global_steps)
            writer.add_scalar('val_loss/surrounding_loss', avg_surrounding_anchor_loss / n, global_steps)
        
        writer_dict['valid_global_steps'] = global_steps + 1
    
    elif dataset_name == 'InterHand' and 'Baseline' in config.MODEL.NAME:
        smooth_loss = torch.nn.SmoothL1Loss(reduction='mean')
        avg_total_loss, avg_pose2d_loss  = 0., 0.
        
        for i, ret in enumerate(val_loader):
            end = time.time()
            # imgs: b x 3 x H x W
            # pose2d_gt: b x 42 x 3
            # hand_type: b x 2 ([1,0] for right, [0,1] for left and [1,1] for interacting hands)
            imgs, pose2d_gt = ret['imgs'].cuda(device, non_blocking=True), ret['pose2d_gt'].cuda(device, non_blocking=True)
            hand_type = ret['hand_type']
            batch_size = imgs.shape[0]
            # cls: b x w*h*n_anchors x 42
            # pose_pred: B x 42 x 2
            # reg: B x w*h*n_anchors x 42 x 2
            pose_pred = model(imgs)

            pose2d_loss = 0.
            for b in range(imgs.shape[0]):
                if hand_type[b,0] == 1: # right
                    pose2d_loss += smooth_loss(target=pose2d_gt[b,0:21,0:2], input=pose_pred[b,0:21])
                elif hand_type[b,1] == 1: # left
                    pose2d_loss += smooth_loss(target=pose2d_gt[b,21:42,0:2], input=pose_pred[b,21:42])
            
            pose2d_loss /= batch_size
            total_loss = config.LOSS.POSE2D_LOSS_FACTOR * pose2d_loss
            
            avg_total_loss += total_loss
            avg_pose2d_loss += pose2d_loss
            
            # measure elapsed time
            if master and i % config.PRINT_FREQ == 0:
                # measure elapsed time
                batch_time = time.time() - end

                msg = 'Dataset: {0} Test: [{1}/{2}]\t' \
                'Time {batch_time:.3f}s\t' \
                'Speed {speed:.1f} samples/s\t' \
                'TotalLoss {total_loss:.5f}'.format(
                    dataset_name, i+1, len(val_loader), batch_time=batch_time,
                    speed=imgs.size(0)/batch_time,
                    total_loss=total_loss.item())

                msg += '\tpose2d_loss {pose2d_loss:.5f}'.format(pose2d_loss = pose2d_loss.item())

                logger.info(msg)

            if config.DEBUG.DEBUG and  i==4: break
        
        global_steps = writer_dict['valid_global_steps']
        
        n = len(val_loader)
        avg_total_loss /= n
        if master:
            writer.add_scalar('val_loss/total_loss', avg_total_loss, global_steps)
            writer.add_scalar('val_loss/anchor_loss', avg_pose2d_loss / n, global_steps)
        
        writer_dict['valid_global_steps'] = global_steps + 1

    else:
        print('[Error in funtion.py] Invalid dataset!')
        exit()

    return avg_total_loss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, config, criterion):
        self.config = config
        self.criterion = criterion

        self.total_loss = 0.
        self.heatmap_loss = 0. if config.LOSS.WITH_HEATMAP_LOSS else None
        self.pose2d_loss = 0. if config.LOSS.WITH_POSE2D_LOSS else None
        self.pose3d_loss = 0. if config.LOSS.WITH_POSE3D_LOSS else None
        self.time_consistency_loss = 0. if config.LOSS.WITH_TIME_CONSISTENCY_LOSS else None
        self.bone_loss = 0. if config.LOSS.WITH_BONE_LOSS else None
        self.jointangle_loss = 0. if config.LOSS.WITH_JOINTANGLE_LOSS else None
        self.n = 0
    
    def computeAvgLosses(self):
        self.avg_total_loss = self.total_loss / self.n

        loss_dict = {'total_loss': self.avg_total_loss}

        if self.config.LOSS.WITH_HEATMAP_LOSS:
            self.avg_heatmap_loss = self.heatmap_loss / self.n
            loss_dict['heatmap_loss'] = self.avg_heatmap_loss

        if self.config.LOSS.WITH_POSE2D_LOSS:
            self.avg_pose2d_loss = self.pose2d_loss / self.n
            loss_dict['pose2d_loss'] = self.avg_pose2d_loss

        if self.config.LOSS.WITH_POSE3D_LOSS:
            self.avg_pose3d_loss = self.pose3d_loss / self.n
            loss_dict['pose3d_loss'] = self.avg_pose3d_loss
        
        if self.config.LOSS.WITH_TIME_CONSISTENCY_LOSS:
            self.avg_time_consistency_loss = self.time_consistency_loss / self.n
            loss_dict['TC_loss'] = self.avg_time_consistency_loss

        if self.config.LOSS.WITH_BONE_LOSS:
            self.avg_bone_loss = self.bone_loss / self.n
            loss_dict['bone_loss'] = self.avg_bone_loss

        if self.config.LOSS.WITH_JOINTANGLE_LOSS:
            self.avg_jointangle_loss = self.jointangle_loss / self.n
            loss_dict['jointangle_loss'] = self.avg_jointangle_loss

        return loss_dict


    def computeLosses(self, item_dict, heatmaps_pred=None, heatmaps_gt=None, pose2d_pred=None, pose2d_gt=None, visibility=None, pose3d_pred=None, pose3d_gt=None, n=1):
        self.n += n
        items = item_dict.keys()
        loss_dict = {
            'heatmap_loss': None,
            'pose2d_loss': None,
            'pose3d_loss': None,
            'TC_loss': None,
            'jointangle_loss': None,
            'bone_loss': None,
            'total_loss': 0
            }

        total_loss = 0
        loss_functions = self.criterion.keys()

        if 'heatmap_loss' in loss_functions and 'heatmap_pred' in items:
            heatmap_loss = self.criterion['heatmap_loss'](item_dict['heatmaps_pred'], item_dict['heatmaps_gt'])
            self.heatmap_loss += heatmap_loss.item()
            total_loss += self.config.LOSS.HEATMAP_LOSS_FACTOR * heatmap_loss
            loss_dict['heatmap_loss'] = heatmap_loss

        if 'pose2d_loss' in loss_functions and 'pose2d_pred' in items:
            pose2d_loss = self.criterion['pose2d_loss'](item_dict['pose2d_pred'][:,:,0:2], item_dict['pose2d_gt'][:,:,0:2], visibility=item_dict['visibility'])
            self.pose2d_loss += pose2d_loss.item()
            total_loss += self.config.LOSS.POSE2D_LOSS_FACTOR * pose2d_loss
            loss_dict['pose2d_loss'] = pose2d_loss

        if 'pose3d_loss' in loss_functions and 'pose3d_pred' in items:
            pose3d_loss = self.criterion['pose3d_loss'](item_dict['pose3d_pred'], item_dict['pose3d_gt'])
            self.pose3d_loss += pose3d_loss.item()
            total_loss += self.config.LOSS.POSE3D_LOSS_FACTOR * pose3d_loss
            loss_dict['pose3d_loss'] = pose3d_loss

        if 'bone_loss' in loss_functions or 'jointangle_loss' in loss_functions:
            pose2d_rel_gt = scale_pose2d(item_dict['pose2d_gt'])
            pose2d_rel_pred = scale_pose2d(item_dict['pose2d_pred'])

            if 'bone_loss' in loss_functions:              
                bone_loss = self.criterion['bone_loss'](pose2d_rel_pred[:,:,0:2], pose2d_rel_gt[:,:,0:2])
                self.bone_loss += bone_loss.item()
                total_loss += self.config.LOSS.BONE_LOSS_FACTOR * bone_loss
                loss_dict['bone_loss'] = bone_loss

            if 'jointangle_loss' in loss_functions:
                # append a z-axis value to each uv coord for the requirement of torch.cross
                zeros = torch.zeros((pose2d_rel_pred.shape[0], pose2d_rel_pred.shape[1], 1), dtype=pose2d_pred.dtype, device=pose2d_pred.device)

                pose2d_rel_pred_with_z = torch.cat(
                    (pose2d_rel_pred[:,:,0:2], zeros),
                    dim=2)

                jointangle_loss = self.criterion['jointangle_loss'](pose2d_rel_pred_with_z)
                self.jointangle_loss += jointangle_loss.item()
                total_loss += self.config.LOSS.JOINTANGLE_LOSS_FACTOR * jointangle_loss
                loss_dict['jointangle_loss'] = jointangle_loss
            
        self.total_loss += total_loss.item()
        loss_dict['total_loss'] = total_loss

        return loss_dict 


def test_sample(img, pose2d):
    """
    img: torch.tensor of size B x H x W x 3
    pose2d: torch.tensor of size B x 21 x 3
    """
    img_np = cv2.cvtColor(img[0].detach().cpu().numpy().transpose(1,2,0), cv2.COLOR_RGB2BGR)
    plt.figure()
    plt.imshow(img_np)
    for i in range(pose2d[0].shape[0]):
        plt.plot(4*pose2d[0][i][0], 4*pose2d[0][i][1],'r*')
    plt.show()
        