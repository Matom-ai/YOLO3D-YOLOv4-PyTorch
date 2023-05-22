"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.07.08
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: Testing script
"""

import argparse
import sys
import os
import time

from easydict import EasyDict as edict
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

sys.path.append('../')

import config.kitti_config as cnf
from data_process import kitti_data_utils, kitti_bev_utils
from data_process.kitti_dataloader import create_test_dataloader
from models.model_utils import create_model
from utils.misc import make_folder
from utils.evaluation_utils import post_processing, rescale_boxes, post_processing_v2
from utils.misc import time_synchronized
from utils.visualization_utils import show_image_with_boxes, merge_rgb_to_bev, predictions_to_kitti_format, merge_side


def parse_test_configs():
    parser = argparse.ArgumentParser(description='Demonstration config for YOLO3D Implementation')
    parser.add_argument('--saved_fn', type=str, default='yolo3d_yolov4', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='darknet', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--cfgfile', type=str, default='./config/cfg/yolo3d_yolov4.cfg', metavar='PATH',
                        help='The path for cfgfile (only for darknet)')
    parser.add_argument('--pretrained_path', type=str, default=None, metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--use_giou_loss', action='store_true',
                        help='If true, use GIoU loss during training. If false, use MSE loss for training')

    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=None, type=int,
                        help='GPU index to use.')

    parser.add_argument('--img_size', type=int, default=608,
                        help='the size of input image')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')

    parser.add_argument('--conf_thresh', type=float, default=0.5,
                        help='the threshold for conf')
    parser.add_argument('--nms_thresh', type=float, default=0.5,
                        help='the threshold for conf')

    parser.add_argument('--show_image', action='store_true',
                        help='If true, show the image during demostration')
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_yolo3d_yolov4', metavar='PATH',
                        help='the video filename if the output format is video')
    parser.add_argument('--display_3d', action='store_true',
                        help='If true, return pointcloud alongside images.')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.root_dir = '../'
    #configs.dataset_dir = os.path.join(configs.root_dir, 'dataset', 'kitti')
    #configs.dataset_dir = os.path.join(configs.root_dir, 'dataset', 'zupt')
    configs.dataset_dir = os.path.join(configs.root_dir, 'dataset', 'zupt_car')

    if configs.save_test_output:
        configs.results_dir = os.path.join(configs.root_dir, 'results', configs.saved_fn)
        make_folder(configs.results_dir)

    return configs


if __name__ == '__main__':
    configs = parse_test_configs()
    configs.distributed = False  # For testing

    model = create_model(configs)
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path))

    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)

    out_cap = None

    model.eval()

    test_dataloader = create_test_dataloader(configs)
    with torch.no_grad():
        for batch_idx, (img_paths, imgs_bev) in enumerate(test_dataloader):
            if configs.display_3d:
                imgs_bev, lidarData = imgs_bev
            else:
                lidarData = None
            points_np = lidarData.numpy()[0,:,:3]
            #print(points_np)
            #print(points_np.shape)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_np)

            input_imgs = imgs_bev.to(device=configs.device).float()
            t1 = time_synchronized()
            outputs = model(input_imgs)
            t2 = time_synchronized()
            # Outputs: (batch_size x ... x 12) 12 includes: x,y,z,h,w,l,im,re,conf,cls
            detections = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)

            img_detections = []  # Stores detections for each image index
            img_detections.extend(detections)

            img_bev = imgs_bev.squeeze() * 255
            img_bev = img_bev.permute(1, 2, 0).numpy().astype(np.uint8)
            img_bev = cv2.resize(img_bev, (configs.img_size, configs.img_size))

            img_bev_clean = imgs_bev.squeeze() * 255
            img_bev_clean = img_bev_clean.permute(1, 2, 0).numpy().astype(np.uint8)
            img_bev_clean = cv2.resize(img_bev_clean, (configs.img_size, configs.img_size))
            for detections in img_detections:
                if detections is None:
                    continue
                # Rescale boxes to original image
                for x, y, z, h, w, l, im, re, *_, cls_pred in detections:
                    # Draw rotated box
                    yaw = torch.atan2(im, re)
                    detection_object = kitti_data_utils.Object3d(x, y, z, h, w, l, yaw) # Create the object here
                    _, corners_3d = kitti_data_utils.compute_box_3d(detection_object, None)
                    print(f"x: {x}, y: {y}, z: {z}, w: {w}, h: {h}, l: {l}, yaw: {yaw}")
                    print(corners_3d)
                    lines = [
                        [0, 1],
                        [1, 2],
                        [0, 3],
                        [2, 3],
                        [4, 5],
                        [4, 7],
                        [5, 6],
                        [6, 7],
                        [0, 4],
                        [1, 5],
                        [2, 6],
                        [3, 7],
                    ]
                    line_set = o3d.geometry.LineSet(
                        points=o3d.utility.Vector3dVector(corners_3d),
                        lines=o3d.utility.Vector2iVector(lines),
                    )
                    o3d.visualization.draw_geometries([pcd, line_set])
                    input()
                    kitti_bev_utils.drawRotatedBox(img_bev, x, y, w, l, yaw, cnf.colors[int(cls_pred)])
            o3d.visualization.draw_geometries([pcd])
            raise Exception('')

            img_rgb = cv2.imread(img_paths[0])
            img_rgb_clean = cv2.imread(img_paths[0])
            #img_rgb_clean = cv2.imread(img_file)
            calib = kitti_data_utils.Calibration(img_paths[0].replace(".png", ".txt").replace("image_2", "calib"))
            objects_pred = predictions_to_kitti_format(img_detections, calib, img_rgb.shape, configs.img_size)
            img_rgb = show_image_with_boxes(img_rgb, objects_pred, calib, False)

            img_bev = cv2.flip(cv2.flip(img_bev, 0), 1)
            img_bev_clean = cv2.flip(cv2.flip(img_bev_clean, 0), 1)
            #img_bev_clean = cv2.rotate(img_bev_clean, cv2.ROTATE_180)

            out_img = merge_rgb_to_bev(img_rgb, img_bev, output_width=608)
            out_img_clean = merge_rgb_to_bev(img_rgb_clean, img_bev_clean, output_width=608)
            show_img = merge_side(out_img, out_img_clean)

            print('\tDone testing the {}th sample, time: {:.1f}ms, speed {:.2f}FPS'.format(batch_idx, (t2 - t1) * 1000,
                                                                                           1 / (t2 - t1)))

            if configs.save_test_output:
                if configs.output_format == 'image':
                    img_fn = os.path.basename(img_paths[0])[:-4]
                    #cv2.imwrite(os.path.join(configs.results_dir, '{}.jpg'.format(img_fn)), out_img)
                    cv2.imwrite(os.path.join(configs.results_dir, '{}.jpg'.format(img_fn)), show_img)
                elif configs.output_format == 'video':
                    if out_cap is None:
                        out_cap_h, out_cap_w = out_img.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        out_cap = cv2.VideoWriter(
                            os.path.join(configs.results_dir, '{}.avi'.format(configs.output_video_fn)),
                            fourcc, 30, (out_cap_w, out_cap_h))

                    out_cap.write(out_img)
                else:
                    raise TypeError

            if configs.show_image:
                # VSL: This is very nasty
                print(out_img)
                cv2.imshow('test-img', out_img)
                #plt.imshow(cv2.cvtColor(out_img, cv2.BGR2RGB))
                #matplotlib.use('pyqt5')
                #plt.imshow(out_img)
                #plt.show()
                print('\n[INFO] Press n to see the next sample >>> Press Esc to quit...\n')
                if cv2.waitKey(0) & 0xFF == 27:
                    break
    if out_cap:
        out_cap.release()
    cv2.destroyAllWindows()
