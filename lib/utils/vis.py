# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import os.path as osp
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from config import cfg
from PIL import Image, ImageDraw

def get_keypoint_rgb(skeleton):
    rgb_dict= {}
    for joint_id in range(len(skeleton)):
        joint_name = skeleton[joint_id]['name']

        if joint_name.endswith('thumb_null'):
            rgb_dict[joint_name] = (255, 0, 0)
        elif joint_name.endswith('thumb3'):
            rgb_dict[joint_name] = (255, 51, 51)
        elif joint_name.endswith('thumb2'):
            rgb_dict[joint_name] = (255, 102, 102)
        elif joint_name.endswith('thumb1'):
            rgb_dict[joint_name] = (255, 153, 153)
        elif joint_name.endswith('thumb0'):
            rgb_dict[joint_name] = (255, 204, 204)
        elif joint_name.endswith('index_null'):
            rgb_dict[joint_name] = (0, 255, 0)
        elif joint_name.endswith('index3'):
            rgb_dict[joint_name] = (51, 255, 51)
        elif joint_name.endswith('index2'):
            rgb_dict[joint_name] = (102, 255, 102)
        elif joint_name.endswith('index1'):
            rgb_dict[joint_name] = (153, 255, 153)
        elif joint_name.endswith('middle_null'):
            rgb_dict[joint_name] = (255, 128, 0)
        elif joint_name.endswith('middle3'):
            rgb_dict[joint_name] = (255, 153, 51)
        elif joint_name.endswith('middle2'):
            rgb_dict[joint_name] = (255, 178, 102)
        elif joint_name.endswith('middle1'):
            rgb_dict[joint_name] = (255, 204, 153)
        elif joint_name.endswith('ring_null'):
            rgb_dict[joint_name] = (0, 128, 255)
        elif joint_name.endswith('ring3'):
            rgb_dict[joint_name] = (51, 153, 255)
        elif joint_name.endswith('ring2'):
            rgb_dict[joint_name] = (102, 178, 255)
        elif joint_name.endswith('ring1'):
            rgb_dict[joint_name] = (153, 204, 255)
        elif joint_name.endswith('pinky_null'):
            rgb_dict[joint_name] = (255, 0, 255)
        elif joint_name.endswith('pinky3'):
            rgb_dict[joint_name] = (255, 51, 255)
        elif joint_name.endswith('pinky2'):
            rgb_dict[joint_name] = (255, 102, 255)
        elif joint_name.endswith('pinky1'):
            rgb_dict[joint_name] = (255, 153, 255)
        else:
            rgb_dict[joint_name] = (230, 230, 0)
        
    return rgb_dict

def vis_keypoints(img, kps, score, skeleton, filename, score_thr=0.4, line_width=3, circle_rad = 3, save_path=None):
    
    rgb_dict = get_keypoint_rgb(skeleton)
    _img = Image.fromarray(img.transpose(1,2,0).astype('uint8')) 
    draw = ImageDraw.Draw(_img)
    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']
        
        kps_i = (kps[i][0].astype(np.int32), kps[i][1].astype(np.int32))
        kps_pid = (kps[pid][0].astype(np.int32), kps[pid][1].astype(np.int32))

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            draw.line([(kps[i][0], kps[i][1]), (kps[pid][0], kps[pid][1])], fill=rgb_dict[parent_joint_name], width=line_width)
        if score[i] > score_thr:
            draw.ellipse((kps[i][0]-circle_rad, kps[i][1]-circle_rad, kps[i][0]+circle_rad, kps[i][1]+circle_rad), fill=rgb_dict[joint_name])
        if score[pid] > score_thr and pid != -1:
            draw.ellipse((kps[pid][0]-circle_rad, kps[pid][1]-circle_rad, kps[pid][0]+circle_rad, kps[pid][1]+circle_rad), fill=rgb_dict[parent_joint_name])
    _img.show()
    if save_path is None:
        _img.save(osp.join(cfg.vis_dir, filename))
    else:
        _img.save(osp.join(save_path, filename))


def vis_3d_keypoints(kps_3d, score, skeleton, filename, score_thr=0.4, line_width=3, circle_rad=3):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    rgb_dict = get_keypoint_rgb(skeleton)
    
    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']

        x = np.array([kps_3d[i,0], kps_3d[pid,0]])
        y = np.array([kps_3d[i,1], kps_3d[pid,1]])
        z = np.array([kps_3d[i,2], kps_3d[pid,2]])

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            ax.plot(x, z, -y, c = np.array(rgb_dict[parent_joint_name])/255., linewidth = line_width)
        if score[i] > score_thr:
            ax.scatter(kps_3d[i,0], kps_3d[i,2], -kps_3d[i,1], c = np.array(rgb_dict[joint_name]).reshape(1,3)/255., marker='o')
        if score[pid] > score_thr and pid != -1:
            ax.scatter(kps_3d[pid,0], kps_3d[pid,2], -kps_3d[pid,1], c = np.array(rgb_dict[parent_joint_name]).reshape(1,3)/255., marker='o')

    plt.show()

    
    fig.savefig(osp.join('.', filename), dpi=fig.dpi)

def plot_hand(axis, coords_hw, vis=None, color_fixed=None, linewidth='1', order='hw', draw_kp=True):
    """ Plots a hand stick figure into a matplotlib figure. """
    if order == 'uv':
        coords_hw = coords_hw[:, ::-1]

    colors = np.array([[0.4, 0.4, 0.4],
                       [0.4, 0.0, 0.0],
                       [0.6, 0.0, 0.0],
                       [0.8, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.4, 0.4, 0.0],
                       [0.6, 0.6, 0.0],
                       [0.8, 0.8, 0.0],
                       [1.0, 1.0, 0.0],
                       [0.0, 0.4, 0.2],
                       [0.0, 0.6, 0.3],
                       [0.0, 0.8, 0.4],
                       [0.0, 1.0, 0.5],
                       [0.0, 0.2, 0.4],
                       [0.0, 0.3, 0.6],
                       [0.0, 0.4, 0.8],
                       [0.0, 0.5, 1.0],
                       [0.4, 0.0, 0.4],
                       [0.6, 0.0, 0.6],
                       [0.7, 0.0, 0.8],
                       [1.0, 0.0, 1.0]])

    colors = colors[:, ::-1]

    # define connections and colors of the bones
    bones = [((0, 1), colors[1, :]),
             ((1, 2), colors[2, :]),
             ((2, 3), colors[3, :]),
             ((3, 4), colors[4, :]),

             ((0, 5), colors[5, :]),
             ((5, 6), colors[6, :]),
             ((6, 7), colors[7, :]),
             ((7, 8), colors[8, :]),

             ((0, 9), colors[9, :]),
             ((9, 10), colors[10, :]),
             ((10, 11), colors[11, :]),
             ((11, 12), colors[12, :]),

             ((0, 13), colors[13, :]),
             ((13, 14), colors[14, :]),
             ((14, 15), colors[15, :]),
             ((15, 16), colors[16, :]),

             ((0, 17), colors[17, :]),
             ((17, 18), colors[18, :]),
             ((18, 19), colors[19, :]),
             ((19, 20), colors[20, :])]

    if vis is None:
        vis = np.ones_like(coords_hw[:, 0]) == 1.0

    for connection, color in bones:
        if (vis[connection[0]] == False) or (vis[connection[1]] == False):
            continue

        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 1], coords[:, 0], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 1], coords[:, 0], color_fixed, linewidth=linewidth)

    if not draw_kp:
        return

    for i in range(21):
        if vis[i] > 0.5:
            axis.plot(coords_hw[i, 1], coords_hw[i, 0], 'o', color=colors[i, :])
