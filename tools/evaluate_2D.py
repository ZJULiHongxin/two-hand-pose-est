from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import platform
import numpy as np
import time
import os
import torch
import torch.backends.cudnn as cudnn

import _init_paths
from config import cfg
from config import update_config

from utils.utils import get_model_summary
from ptflops import get_model_complexity_info
from fp16_utils.fp16util import network_to_half
from core.loss import BoneLengthLoss, JointAngleLoss, JointsMSELoss
import dataset
from dataset.build import trans
from models import A2JPoseNet
from utils.misc import plot_performance
import matplotlib

if platform.system() == 'Linux':
    matplotlib.use('Agg')
else:
    matplotlib.use('Tkagg')
# python evaluate_2D.py --cfg ../experiments/InterHand/exp_test.yaml --model_path ../output/InterHand/exp_test/model_best.pth.tar --gpu 3 --batch_size 32
def parse_args():
    parser = argparse.ArgumentParser(description='Please specify the mode [training/assessment/predicting]')
    parser.add_argument('--cfg',
                    help='experiment configure file name',
                    required=True,
                    type=str)
    parser.add_argument('opts',
                    help="Modify cfg options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
    parser.add_argument('--gpu',
                        help='gpu id for multiprocessing training',
                        default=-1,
                        type=int)
    parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--is_vis',
                    default=0,
                    type=int)
    parser.add_argument('--batch_size',
                    default=32,
                    type=int)
    parser.add_argument('--model_path', default='', type=str)

    args = parser.parse_args()

    return args



def main():
    args = parse_args()
    
    update_config(cfg, args)
    cfg.defrost()
    cfg.freeze()
    
    file_path = './eval_results'

    if not os.path.exists(file_path):
        os.mkdir(file_path)

    record_prefix = os.path.join(file_path, 'eval2D_results_')
    if args.is_vis:
        result_dir = record_prefix + cfg.EXP_NAME
        mse2d_lst = np.loadtxt(os.path.join(result_dir, 'mse2d_each_joint.txt'))
        PCK2d_lst = np.loadtxt(os.path.join(result_dir, 'PCK2d.txt'))

        plot_performance(PCK2d_lst[1,:], PCK2d_lst[0,:], mse2d_lst)
        exit()

    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model_path = args.model_path
    is_vis = args.is_vis
    
    # FP16 SETTING
    if cfg.FP16.ENABLED:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    if cfg.FP16.STATIC_LOSS_SCALE != 1.0:
        if not cfg.FP16.ENABLED:
            print("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")
    
    model = eval(cfg.MODEL.NAME)(cfg)

    if cfg.FP16.ENABLED:
        model = network_to_half(model)

    if cfg.MODEL.SYNC_BN and not args.distributed:
        print('Warning: Sync BatchNorm is only supported in distributed training.')

    if args.gpu != -1:
        device = torch.device('cuda:'+str(args.gpu))
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device('cpu')
    # load model state
    if model_path:
        print("Loading model:", model_path)
        ckpt = torch.load(model_path, map_location='cpu')
        if 'state_dict' not in ckpt.keys():
            state_dict = ckpt
        else:
            state_dict = ckpt['state_dict']
            print('Model epoch {}'.format(ckpt['epoch']))
        
        for key in list(state_dict.keys()):
            new_key = key.replace("module.", "")
            state_dict[new_key] = state_dict.pop(key)
        
        model.load_state_dict(state_dict, strict=True)
    
    model.to(device)

    model.eval()

    # inference_dataset = eval('dataset.{}'.format(cfg.DATASET.TEST_DATASET[0].replace('_kpt','')))(
    #     cfg.DATA_DIR,
    #     cfg.DATASET.TEST_SET,
    #     transform=transform
    # )
    inference_dataset = eval('dataset.{}'.format(cfg.DATASET.DATASET_NAME))(
        cfg,
        transforms=trans,
        mode='test'
    )

    batch_size = args.batch_size

    if platform.system() == 'Linux':
        main_workers = min(8, batch_size)
    else:
        batch_size = 4
        main_workers = 0
    
    data_loader = torch.utils.data.DataLoader(
        inference_dataset,
        batch_size=batch_size, #48
        shuffle=False,
        num_workers=main_workers, #8
        pin_memory=False
    )

    print('\nEvaluation loader information:\n' + str(data_loader.dataset))
    n_joints = cfg.DATASET.NUM_JOINTS
    th2d_lst = np.array([i for i in range(1,50)])
    PCK2d_lst = np.zeros((len(th2d_lst),))

    # two hands
    mse2d_lst = np.zeros((2*n_joints,))
    visibility_lst = np.zeros((2*n_joints,))

    print('Start evaluating... [Batch size: {}]\n'.format(data_loader.batch_size))
    with torch.no_grad():
        pose2d_mse_loss = JointsMSELoss().to(device)
        infer_time = [0,0]
        start_time = time.time()
        for i, ret in enumerate(data_loader):
            # imgs: b x 3 x H x W
            # pose2d_gt: b x 42 x 3 [u,v,z]
            # hand_type: b x 2 ([1,0] for right, [0,1] for left and [1,1] for interacting hands)
            # pose_valid: b x 42
            imgs, pose2d_gt = ret['imgs'].cuda(device, non_blocking=True), ret['pose2d_gt']
            hand_type, pose_valid = ret['hand_type'], ret['joint_valid'].numpy()

            s1 = time.time()
            batch_size = imgs.shape[0]
            # cls: b x w*h*n_anchors x 42
            # pose_pred: B x 42 x 2
            # reg: B x w*h*n_anchors x 42 x 2
            pose2d_pred, surrounding_anchors_pred, cls_pred, reg, temperature = model(imgs)

            if i+1 >= min(len(data_loader), 20):
                infer_time[0] += 1
                infer_time[1] += time.time() - s1

            # rescale to the original image before DLT
            
            
                # for k in range(21):
                #     print(pose2d_gt[0,k].tolist(), pose2d_pred[0,k].tolist())
                # input()
            # 2D errors

            # import matplotlib.pyplot as plt
            # imgs = cv2.resize(imgs[0].permute(1,2,0).cpu().numpy(), tuple(data_loader.dataset.orig_img_size))
            # for k in range(21):
            #     print(pose2d_gt[0,k],pose2d_pred[0,k],visibility[0,k])
            # for k in range(0,21,5):
            #     fig = plt.figure()
            #     ax1 = fig.add_subplot(131)
            #     ax2 = fig.add_subplot(132)
            #     ax3 = fig.add_subplot(133)
            #     ax1.imshow(cv2.cvtColor(imgs / imgs.max(), cv2.COLOR_BGR2RGB))
            #     plot_hand(ax1, pose2d_gt[0,:,0:2], order='uv')
            #     ax2.imshow(cv2.cvtColor(imgs / imgs.max(), cv2.COLOR_BGR2RGB))
            #     plot_hand(ax2, pose2d_pred[0,:,0:2], order='uv')
            #     ax3.imshow(heatmaps_pred[0,k].cpu().numpy())
            #     plt.show()

            mse_each_joint = np.linalg.norm(pose2d_pred[:,:,0:2].cpu().numpy() - pose2d_gt[:,:,0:2].numpy(), axis=2) * pose_valid # b x 42

            mse2d_lst += mse_each_joint.sum(axis=0)
            visibility_lst += pose_valid.sum(axis=0)

            for th_idx in range(len(th2d_lst)):
                PCK2d_lst[th_idx] += np.sum((mse_each_joint < th2d_lst[th_idx]) * pose_valid)
            
            period = min(len(data_loader), 10)
            if i % (len(data_loader)//period) == 0:
                    print("[Evaluation]{}% finished.".format(period * i // (len(data_loader)//period)))
            #if i == 10:break
        print('Evaluation spent {:.2f} s\tfps: {:.1f} {:.4f}'.format(time.time()-start_time, infer_time[0]/infer_time[1], infer_time[1]/infer_time[0]))

        mse2d_lst /= visibility_lst
        PCK2d_lst /= visibility_lst.sum()

        result_dir = record_prefix+cfg.EXP_NAME
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        
        mse_file, pck_file = os.path.join(result_dir, 'mse2d_each_joint.txt'), os.path.join(result_dir, 'PCK2d.txt')
        print('Saving results to ' + mse_file)
        print('Saving results to ' + pck_file)
        np.savetxt(mse_file, mse2d_lst, fmt='%.4f')
        np.savetxt(pck_file, np.stack((th2d_lst, PCK2d_lst)))

        plot_performance(PCK2d_lst, th2d_lst, mse2d_lst, hand_type='interacting')

main()