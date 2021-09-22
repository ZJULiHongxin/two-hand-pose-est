# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sys
import os
import time
currentUrl = os.path.dirname(__file__)
parentUrl = os.path.abspath(os.path.join(currentUrl, os.pardir))
sys.path.append(parentUrl)

import argparse
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import cv2
from glob import glob
import os.path as osp


from utils.preprocessing import load_img, load_skeleton, get_bbox, process_bbox, augmentation, transform_input_to_output_space, trans_point2d
from utils.transforms import world2cam, cam2pixel, pixel2cam
from utils.vis import vis_keypoints, vis_3d_keypoints, plot_hand
from utils.standard_legends import idx_InterHand
from PIL import Image, ImageDraw
import random
import json
import math
from pycocotools.coco import COCO
import scipy.io as sio




class InterHandDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transforms, mode):
        self.cfg = cfg
        self.name = 'InterHand'
        self.mode = mode # train, test, val
        self.img_path = osp.join(cfg.DATASET.DATA_DIR, 'images') # '../data/InterHand2.6M/images'
        self.annot_path = osp.join(cfg.DATASET.DATA_DIR, 'annotations') # '../data/InterHand2.6M/annotations'
        # if self.mode == 'val':
        #     self.rootnet_output_path = '../data/InterHand2.6M/rootnet_output/rootnet_interhand2.6m_output_val.json'
        # else:
        #     self.rootnet_output_path = '../data/InterHand2.6M/rootnet_output/rootnet_interhand2.6m_output_test.json'
        self.transform = transforms

        self.joint_num = cfg.DATASET.NUM_JOINTS # 21 # single hand
        self.root_joint_idx = {'right': 0, 'left': 21} # Please modify this idx after changing the order of joints
        self.joint_type = {'right': np.arange(0,self.joint_num), 'left': np.arange(self.joint_num,self.joint_num*2)}
        self.skeleton = load_skeleton(osp.join(self.annot_path, 'skeleton.txt'), self.joint_num*2)
        
        self.datalist = []
        self.datalist_sh = []
        self.datalist_ih = []
        self.sequence_names = [] 
        
        # load annotation
        print("Load annotation from  " + osp.join(self.annot_path, self.mode))
        t1 = time.time()
        prefix = 'simple_'
        db = COCO(osp.join(self.annot_path, self.mode, prefix+'InterHand2.6M_' + self.mode + '_data.json'))
        with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_camera.json')) as f:
            cameras = json.load(f)
        with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_joint_3d.json')) as f:
            joints = json.load(f)
        print("Annotation loading spent {}s".format(time.time()-t1))
        # if (self.mode == 'val' or self.mode == 'test') and cfg.trans_test == 'rootnet':
        #     print("Get bbox and root depth from " + self.rootnet_output_path)
        #     rootnet_result = {}
        #     with open(self.rootnet_output_path) as f:
        #         annot = json.load(f)
        #     for i in range(len(annot)):
        #         rootnet_result[str(annot[i]['annot_id'])] = annot[i]
        # else:
        #     print("Get bbox and root depth from groundtruth annotation")
        
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
 
            capture_id = img['capture']
            seq_name = img['seq_name']
            cam = img['camera']
            frame_idx = img['frame_idx']
            img_path = osp.join(self.img_path, self.mode, img['file_name'])
            
            campos, camrot = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
            focal, princpt = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
            # get the groundtruth pose and reorder it
            joint_world = np.array(joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32)[idx_InterHand] # 42 x 3     
            joint_cam = world2cam(joint_world.transpose(1,0), camrot, campos.reshape(3,1)).transpose(1,0)
            joint_img = cam2pixel(joint_cam, focal, princpt) # 42 x 2 [u,v]

            # 1 if a joint is annotated and inside of image. 0 otherwise
            joint_valid = np.array(ann['joint_valid'],dtype=np.float32).reshape(self.joint_num*2)
            # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
            joint_valid[self.joint_type['right']] *= joint_valid[self.root_joint_idx['right']]
            joint_valid[self.joint_type['left']] *= joint_valid[self.root_joint_idx['left']]
            hand_type = ann['hand_type']

            # 1 if hand_type in ('right', 'left') or hand_type == 'interacting' and np.sum(joint_valid) > 30, 0 otherwise
            hand_type_valid = np.array((ann['hand_type_valid']), dtype=np.float32)
            
            # if (self.mode == 'val' or self.mode == 'test') and cfg.trans_test == 'rootnet':
            #     bbox = np.array(rootnet_result[str(aid)]['bbox'],dtype=np.float32)
            #     abs_depth = {'right': rootnet_result[str(aid)]['abs_depth'][0], 'left': rootnet_result[str(aid)]['abs_depth'][1]}
            # else:
            img_width, img_height = img['width'], img['height'] # original image size 344(w) x 512(h)

            bbox = np.array(ann['bbox'],dtype=np.float32) # x,y,w,h
            bbox = process_bbox(bbox, (img_height, img_width))
            abs_depth = {'right': joint_cam[self.root_joint_idx['right'],2], 'left': joint_cam[self.root_joint_idx['left'],2]}

            cam_param = {'focal': focal, 'princpt': princpt}
            joint = {'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid}
            data = {'img_path': img_path, 'seq_name': seq_name, 'cam_param': cam_param, 'bbox': bbox, 'joint': joint, 'hand_type': hand_type, 'hand_type_valid': hand_type_valid, 'abs_depth': abs_depth, 'file_name': img['file_name'], 'capture': capture_id, 'cam': cam, 'frame': frame_idx}
            if hand_type == 'right' or hand_type == 'left':
                self.datalist_sh.append(data)
            else:
                self.datalist_ih.append(data)
            if seq_name not in self.sequence_names:
                self.sequence_names.append(seq_name)

        self.datalist = self.datalist_sh + self.datalist_ih
        print('Number of annotations in single hand sequences: ' + str(len(self.datalist_sh)))
        print('Number of annotations in interacting hand sequences: ' + str(len(self.datalist_ih)))

    def handtype_str2array(self, hand_type):
        if hand_type == 'right':
            return np.array([1,0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0,1], dtype=np.float32)
        elif hand_type == 'interacting':
            return np.array([1,1], dtype=np.float32)
        else:
            assert 0, print('Not supported hand type: ' + hand_type)
    
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        data = self.datalist[idx]
        img_path, bbox, joint, hand_type, hand_type_valid = data['img_path'], data['bbox'], data['joint'], data['hand_type'], data['hand_type_valid']
        joint_cam = joint['cam_coord'].copy()
        joint_img = joint['img_coord'].copy()
        joint_valid = joint['valid'].copy() # 1 if inside the image, o other wise. # 42
        hand_type_vec = self.handtype_str2array(hand_type)
        joint_coord = np.concatenate((joint_img, joint_cam[:,2,None]),1) # 42 x 3 [u,v,z]
        # input(joint_valid)
        # input(joint_coord)
        # image load
        try:
            img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION), cv2.COLOR_BGR2RGB) # 512 x 334 x 3
        except:
            print('[Warning] Invalid image path:', img_path)

        # DEBUG

        # f = plt.figure()
        # ax1 = f.add_subplot(1,1,1)
        # ax1.imshow(img)
        # for k in range(joint_coord.shape[0]):
        #     print('[{:.4f}, {:.4f}, {:.4f}],'.format(*joint_coord[k]))
        # print(hand_type_vec)
        # if hand_type_vec[0] == 1:
        #     plot_hand(ax1, joint_coord[0:21,0:2], vis=joint_valid[0:21], order = 'uv')
        # elif hand_type_vec[1] == 1:
        #     plot_hand(ax1, joint_coord[21:42,0:2], vis=joint_valid[21:42], order = 'uv')
        # ax1.set_title(hand_type)
        # plt.show()

        # augmentation
        img, joint_coord, joint_valid, hand_type_vec, inv_trans = augmentation(img, bbox, joint_coord, joint_valid, hand_type_vec, self.mode, self.joint_type, self.cfg.MODEL.INPUT_SIZE)

        # f1 = plt.figure()
        # ax1 = f1.add_subplot(1,1,1)
        # ax1.imshow(img.astype(int))

        # for k in range(joint_coord.shape[0]):
        #     print('[{:.4f}, {:.4f}, {:.4f}],'.format(*joint_coord[k]))
        # print(joint_coord)
        # if hand_type_vec[0] == 1:
        #     plot_hand(ax1, joint_coord[0:21,0:2], vis=joint_valid[0:21], order = 'uv')
        # elif hand_type_vec[1] == 1:
        #     plot_hand(ax1, joint_coord[21:42,0:2], vis=joint_valid[21:42], order = 'uv')
        # ax1.set_title(hand_type)
        # plt.show()
        
        #rel_root_depth = np.array([joint_coord[self.root_joint_idx['left'],2] - joint_coord[self.root_joint_idx['right'],2]],dtype=np.float32).reshape(1)
        #root_valid = np.array([joint_valid[self.root_joint_idx['right']] * joint_valid[self.root_joint_idx['left']]],dtype=np.float32).reshape(1) if hand_type_vec[0]*hand_type_vec[1] == 1 else np.zeros((1),dtype=np.float32)
        
        # transform to output heatmap space (this line of code is useless for anchor-based estimation)
        #joint_coord, joint_valid, rel_root_depth, root_valid = transform_input_to_output_space(self.cfg, joint_coord, joint_valid, rel_root_depth, root_valid, self.root_joint_idx, self.joint_type)
        img = self.transform(img.astype(np.float32) / 255.)

        # inputs = {'img': img}
        # targets = {'pose2d_gt': joint_coord, 'rel_root_depth': rel_root_depth, 'hand_type': hand_type_vec}
        # meta_info = {'joint_valid': joint_valid, 'root_valid': root_valid, 'hand_type_valid': hand_type_valid, 'inv_trans': inv_trans, 'capture': int(data['capture']), 'cam': int(data['cam']), 'frame': int(data['frame'])}
        
        return  {'imgs': img, 'pose2d_gt': joint_coord, 'joint_valid': joint_valid, 'hand_type': hand_type_vec}
        #return inputs, targets, meta_info

    def evaluate(self, preds):

        print() 
        print('Evaluation start...')

        gts = self.datalist
        preds_joint_coord, preds_rel_root_depth, preds_hand_type, inv_trans = preds['joint_coord'], preds['rel_root_depth'], preds['hand_type'], preds['inv_trans']
        assert len(gts) == len(preds_joint_coord)
        sample_num = len(gts)
        
        mpjpe_sh = [[] for _ in range(self.joint_num*2)]
        mpjpe_ih = [[] for _ in range(self.joint_num*2)]
        mrrpe = []
        acc_hand_cls = 0
        hand_cls_cnt = 0
        for n in range(sample_num):
            data = gts[n]
            bbox, cam_param, joint, gt_hand_type, hand_type_valid = data['bbox'], data['cam_param'], data['joint'], data['hand_type'], data['hand_type_valid']
            focal = cam_param['focal']
            princpt = cam_param['princpt']
            gt_joint_coord = joint['cam_coord']
            joint_valid = joint['valid']
            
            # restore xy coordinates to original image space
            pred_joint_coord_img = preds_joint_coord[n].copy()
            pred_joint_coord_img[:,0] = pred_joint_coord_img[:,0]/cfg.output_hm_shape[2]*cfg.input_img_shape[1]
            pred_joint_coord_img[:,1] = pred_joint_coord_img[:,1]/cfg.output_hm_shape[1]*cfg.input_img_shape[0]
            for j in range(self.joint_num*2):
                pred_joint_coord_img[j,:2] = trans_point2d(pred_joint_coord_img[j,:2],inv_trans[n])
            # restore depth to original camera space
            pred_joint_coord_img[:,2] = (pred_joint_coord_img[:,2]/cfg.output_hm_shape[0] * 2 - 1) * (cfg.bbox_3d_size/2)
 
            # mrrpe
            if gt_hand_type == 'interacting' and joint_valid[self.root_joint_idx['left']] and joint_valid[self.root_joint_idx['right']]:
                pred_rel_root_depth = (preds_rel_root_depth[n]/cfg.output_root_hm_shape * 2 - 1) * (cfg.bbox_3d_size_root/2)

                pred_left_root_img = pred_joint_coord_img[self.root_joint_idx['left']].copy()
                pred_left_root_img[2] += data['abs_depth']['right'] + pred_rel_root_depth
                pred_left_root_cam = pixel2cam(pred_left_root_img[None,:], focal, princpt)[0]

                pred_right_root_img = pred_joint_coord_img[self.root_joint_idx['right']].copy()
                pred_right_root_img[2] += data['abs_depth']['right']
                pred_right_root_cam = pixel2cam(pred_right_root_img[None,:], focal, princpt)[0]
                
                pred_rel_root = pred_left_root_cam - pred_right_root_cam
                gt_rel_root = gt_joint_coord[self.root_joint_idx['left']] - gt_joint_coord[self.root_joint_idx['right']]
                mrrpe.append(float(np.sqrt(np.sum((pred_rel_root - gt_rel_root)**2))))

           
            # add root joint depth
            pred_joint_coord_img[self.joint_type['right'],2] += data['abs_depth']['right']
            pred_joint_coord_img[self.joint_type['left'],2] += data['abs_depth']['left']

            # back project to camera coordinate system
            pred_joint_coord_cam = pixel2cam(pred_joint_coord_img, focal, princpt)

            # root joint alignment
            for h in ('right', 'left'):
                pred_joint_coord_cam[self.joint_type[h]] = pred_joint_coord_cam[self.joint_type[h]] - pred_joint_coord_cam[self.root_joint_idx[h],None,:]
                gt_joint_coord[self.joint_type[h]] = gt_joint_coord[self.joint_type[h]] - gt_joint_coord[self.root_joint_idx[h],None,:]
            
            # mpjpe
            for j in range(self.joint_num*2):
                if joint_valid[j]:
                    if gt_hand_type == 'right' or gt_hand_type == 'left':
                        mpjpe_sh[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j])**2)))
                    else:
                        mpjpe_ih[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j])**2)))

            # handedness accuray
            if hand_type_valid:
                if gt_hand_type == 'right' and preds_hand_type[n][0] > 0.5 and preds_hand_type[n][1] < 0.5:
                    acc_hand_cls += 1
                elif gt_hand_type == 'left' and preds_hand_type[n][0] < 0.5 and preds_hand_type[n][1] > 0.5:
                    acc_hand_cls += 1
                elif gt_hand_type == 'interacting' and preds_hand_type[n][0] > 0.5 and preds_hand_type[n][1] > 0.5:
                    acc_hand_cls += 1
                hand_cls_cnt += 1

            vis = False
            if vis:
                img_path = data['img_path']
                cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                _img = cvimg[:,:,::-1].transpose(2,0,1)
                vis_kps = pred_joint_coord_img.copy()
                vis_valid = joint_valid.copy()
                capture = str(data['capture'])
                cam = str(data['cam'])
                frame = str(data['frame'])
                filename = 'out_' + str(n) + '_' + gt_hand_type + '.jpg'
                vis_keypoints(_img, vis_kps, vis_valid, self.skeleton, filename)

            vis = False
            if vis:
                filename = 'out_' + str(n) + '_3d.jpg'
                vis_3d_keypoints(pred_joint_coord_cam, joint_valid, self.skeleton, filename)
        

        if hand_cls_cnt > 0: print('Handedness accuracy: ' + str(acc_hand_cls / hand_cls_cnt))
        if len(mrrpe) > 0: print('MRRPE: ' + str(sum(mrrpe)/len(mrrpe)))
        print()
 
        tot_err = []
        eval_summary = 'MPJPE for each joint: \n'
        for j in range(self.joint_num*2):
            tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_sh[j]), np.stack(mpjpe_ih[j]))))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % tot_err_j)
            tot_err.append(tot_err_j)
        print(eval_summary)
        print('MPJPE for all hand sequences: %.2f' % (np.mean(tot_err)))
        print()

        eval_summary = 'MPJPE for each joint: \n'
        for j in range(self.joint_num*2):
            mpjpe_sh[j] = np.mean(np.stack(mpjpe_sh[j]))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % mpjpe_sh[j])
        print(eval_summary)
        print('MPJPE for single hand sequences: %.2f' % (np.mean(mpjpe_sh)))
        print()

        eval_summary = 'MPJPE for each joint: \n'
        for j in range(self.joint_num*2):
            mpjpe_ih[j] = np.mean(np.stack(mpjpe_ih[j]))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % mpjpe_ih[j])
        print(eval_summary)
        print('MPJPE for interacting hand sequences: %.2f' % (np.mean(mpjpe_ih)))


if __name__ == "__main__":
    def add_path(path):
        if path not in sys.path:
            sys.path.insert(0, path)

    this_dir = osp.dirname(__file__)

    lib_path = osp.join(this_dir, '..', '..','lib')
    add_path(lib_path)

    mm_path = osp.join(this_dir, '..', 'lib/poseeval/py-motmetrics')
    add_path(mm_path)
    
    from torchvision import transforms
    from config.default import _C as cfg
    from config.default import update_config
    from utils.vis import plot_hand
    
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='../../experiments/exp_test.yaml',
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.cfg = "../../experiments/exp_test.yaml"

    update_config(cfg, args)
    cfg.defrost()
    
    dataset = InterHandDataset(cfg, transforms.ToTensor(), "train")
    batch_generator = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    for itr, (inputs, targets, meta_info) in enumerate(batch_generator):
        img = inputs['imgs'].numpy().squeeze()*255 # 1 x 3 x 256 x 256

        joint_coord = targets['pose2d_gt'].numpy().squeeze() # [42, 3] # u,v pixel, z root-relative discretized depth
        joint_valid = meta_info['joint_valid'].numpy().squeeze() # [42]
        filename = 'result_2d.jpg'

        for k in range(joint_coord.shape[0]):
            print('[{},{}],'.format(joint_coord[k,0],joint_coord[k,1]))

        vis_img = vis_keypoints(img, joint_coord, joint_valid, dataset.skeleton, filename, save_path='.')
        filename = 'result_3d'
        vis_3d_keypoints(joint_coord, joint_valid, dataset.skeleton, filename)
