"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.07.31
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: This script for creating the dataloader for training/validation/test phase
"""

import sys

import torch
from torch.utils.data import DataLoader

sys.path.append('../')

from data_process.kitti_dataset import KittiDataset
from data_process.transformation import Compose, OneOf, Random_Rotation, Random_Scaling, Horizontal_Flip, Cutout


def create_train_dataloader(configs):
    """Create dataloader for training"""

    train_lidar_transforms = OneOf([
        Random_Rotation(limit_angle=20., p=1.0),
        Random_Scaling(scaling_range=(0.95, 1.05), p=1.0)
    ], p=0.66)

    train_aug_transforms = Compose([
        Horizontal_Flip(p=configs.hflip_prob),
        Cutout(n_holes=configs.cutout_nholes, ratio=configs.cutout_ratio, fill_value=configs.cutout_fill_value,
               p=configs.cutout_prob)
    ], p=1.)

    train_dataset = KittiDataset(
            configs.dataset_dir,
            mode='train',
            lidar_transforms=train_lidar_transforms,
            aug_transforms=train_aug_transforms,
            multiscale=configs.multiscale_training,
            num_samples=configs.num_samples,
            mosaic=configs.mosaic,
            random_padding=configs.random_padding)
    train_sampler = None
    if configs.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset)
    train_dataloader = DataLoader(
            train_dataset,
            batch_size=configs.batch_size,
            shuffle=(train_sampler is None),
            pin_memory=configs.pin_memory,
            num_workers=configs.num_workers,
            sampler=train_sampler,
            collate_fn=train_dataset.collate_fn)

    return train_dataloader, train_sampler


def create_val_dataloader(configs):
    """Create dataloader for validation"""
    val_sampler = None
    val_dataset = KittiDataset(configs.dataset_dir, mode='val', lidar_transforms=None, aug_transforms=None,
                               multiscale=False, num_samples=configs.num_samples, mosaic=False, random_padding=False,
                               display_3d=configs.display_3d)
    if configs.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False,
                                pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=val_sampler,
                                collate_fn=val_dataset.collate_fn)

    return val_dataloader


def create_test_dataloader(configs):
    """Create dataloader for testing phase"""

    test_dataset = KittiDataset(configs.dataset_dir, mode='test', lidar_transforms=None, aug_transforms=None,
                                multiscale=False, num_samples=configs.num_samples, mosaic=False, random_padding=False,
                                display_3d=configs.display_3d)
    test_sampler = None
    if configs.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False,
                                 pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=test_sampler)

    return test_dataloader


if __name__ == '__main__':
    import argparse
    import os

    import cv2
    import numpy as np
    from easydict import EasyDict as edict

    import data_process.kitti_bev_utils as bev_utils
    from data_process import kitti_data_utils
    from utils.visualization_utils import show_image_with_boxes, merge_rgb_to_bev, invert_target, merge_side
    import config.kitti_config as cnf
    import open3d as o3d

    parser = argparse.ArgumentParser(description='YOLO3D Implementation')

    parser.add_argument('--img_size', type=int, default=608,
                        help='the size of input image')
    parser.add_argument('--hflip_prob', type=float, default=0.,
                        help='The probability of horizontal flip')
    parser.add_argument('--cutout_prob', type=float, default=0.,
                        help='The probability of cutout augmentation')
    parser.add_argument('--cutout_nholes', type=int, default=1,
                        help='The number of cutout area')
    parser.add_argument('--cutout_ratio', type=float, default=0.3,
                        help='The max ratio of the cutout area')
    parser.add_argument('--cutout_fill_value', type=float, default=0.,
                        help='The fill value in the cut out area, default 0. (black)')
    parser.add_argument('--multiscale_training', action='store_true',
                        help='If true, use scaling data for training')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 1)')
    parser.add_argument('--mosaic', action='store_true',
                        help='If true, compose training samples as mosaics')
    parser.add_argument('--random-padding', action='store_true',
                        help='If true, random padding if using mosaic augmentation')
    parser.add_argument('--show-train-data', action='store_true',
                        help='If true, random padding if using mosaic augmentation')
    parser.add_argument('--output-width', type=int, default=608,
                        help='the width of showing output, the height maybe vary')
    parser.add_argument('--save_img', action='store_true',
                        help='If true, save the images')
    parser.add_argument('--display_3d', action='store_true',
                        help='If true, return pointcloud alongside images.')

    configs = edict(vars(parser.parse_args()))
    configs.distributed = False  # For testing
    configs.pin_memory = False
    #configs.dataset_dir = os.path.join('../../', 'dataset', 'kitti')
    #configs.dataset_dir = os.path.join('../../', 'dataset', 'zupt')
    configs.dataset_dir = os.path.join('../../', 'dataset', 'zupt_car')

    if configs.save_img:
        print('saving validation images')
        configs.saved_dir = os.path.join(configs.dataset_dir, 'validation_data')
        if not os.path.isdir(configs.saved_dir):
            os.makedirs(configs.saved_dir)

    if configs.show_train_data:
        dataloader, _ = create_train_dataloader(configs)
        print('len train dataloader: {}'.format(len(dataloader)))
    else:
        dataloader = create_val_dataloader(configs)
        print('len val dataloader: {}'.format(len(dataloader)))

    print('\n\nPress n to see the next sample >>> Press Esc to quit...')

    for batch_i, (img_files, imgs, targets) in enumerate(dataloader):
        if configs.display_3d:
            lidarData = img_files[0][1:][0]
            img_files = img_files[0]
            #img_files, lidarData = img_files
            points_np = lidarData[:,:3]
            #print(points_np)
            #print(points_np.shape)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_np)
            print("mean")
            print(np.mean(points_np[:,0]))
            print(np.mean(points_np[:,1]))
            print(np.mean(points_np[:,2]))
            print("min")
            print(np.min(points_np[:,0]))
            print(np.min(points_np[:,1]))
            print(np.min(points_np[:,2]))
            rotation = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
            #pcd.rotate(rotation, center=(0,0,0))
        else:
            lidarData = None
        if not (configs.mosaic and configs.show_train_data):
            geoms = [pcd]
            img_file = img_files[0]
            print(img_file)
            img_rgb = cv2.imread(img_file)
            img_rgb_clean = cv2.imread(img_file)
            calib = kitti_data_utils.Calibration(img_file.replace(".png", ".txt").replace("image_2", "calib"))
            objects_pred = invert_target(targets[:, 1:], calib, img_rgb.shape, RGB_Map=None)
            img_rgb = show_image_with_boxes(img_rgb, objects_pred, calib, False) #Draws boxes on actual image
            for opred in objects_pred:
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
                opred.translate_ZUPT()
                _, corners_3d = kitti_data_utils.compute_box_3d(opred, None)
                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(corners_3d),
                    lines=o3d.utility.Vector2iVector(lines),
                )
                roll = 0
                pitch = 0
                yaw = opred.ry
                line_set.translate((0, opred.h/2, 0), relative = True)
                rotation = line_set.get_rotation_matrix_from_xyz((roll, pitch, yaw))
                line_set.rotate(rotation, center = opred.t)
                mesh_box = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
                detection_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                detection_sphere.translate(opred.t, relative = False)
                detection_sphere.paint_uniform_color(np.array([0.5,0.5,0]))
                geoms.append(line_set)
                geoms.append(detection_sphere)
                camera_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
                camera_sphere.translate((-1, 0, 4), relative = False)
                spherex = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
                spherex.translate((1, 0, 0), relative = False)
                spherex.paint_uniform_color(np.array([1,0,0]))
                spherey = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
                spherey.translate((0, 1, 0), relative = False)
                spherey.paint_uniform_color(np.array([0,1,0]))
                spherez = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
                spherez.translate((0, 0, 1), relative = False)
                spherez.paint_uniform_color(np.array([0,0,1]))
                #mesh_box = mesh_box.translate((0, 0, 0))
            geoms.append(mesh_box)
            geoms.append(spherex)
            geoms.append(spherey)
            geoms.append(spherez)
            geoms.append(camera_sphere)
            o3d.visualization.draw_geometries(geoms)
            input("Enter to continue...")

        # target has (b, cl, x, y, z, h, w, l, im, re)
        targets[:, 2:8] *= configs.img_size

        img_bev = imgs.squeeze() * 255
        img_bev = img_bev.permute(1, 2, 0).numpy().astype(np.uint8)
        img_bev = cv2.resize(img_bev, (configs.img_size, configs.img_size))

        img_bev_clean = imgs.squeeze() * 255
        img_bev_clean = img_bev_clean.permute(1, 2, 0).numpy().astype(np.uint8)
        img_bev_clean = cv2.resize(img_bev_clean, (configs.img_size, configs.img_size))

        # Draw rotated box
        for c, x, y, z, h, w, l, im, re in targets[:, 1:].numpy():
            yaw = np.arctan2(im, re)
            bev_utils.drawRotatedBox(img_bev, x, y, w, l, yaw, cnf.colors[int(c)])

        img_bev = cv2.rotate(img_bev, cv2.ROTATE_180)
        img_bev_clean = cv2.rotate(img_bev_clean, cv2.ROTATE_180)

        if configs.mosaic and configs.show_train_data:
            if configs.save_img:
                fn = os.path.basename(img_file)
                cv2.imwrite(os.path.join(configs.saved_dir, fn), img_bev)
            else:
                cv2.imshow('mosaic_sample', img_bev)
        else:
            out_img = merge_rgb_to_bev(img_rgb, img_bev, output_width=configs.output_width)
            out_img_clean = merge_rgb_to_bev(img_rgb_clean, img_bev_clean, output_width=configs.output_width)
            show_img = merge_side(out_img, out_img_clean)
            if configs.save_img:
                fn = os.path.basename(img_file)
                #cv2.imwrite(os.path.join(configs.saved_dir, fn), out_img)
                print(configs.saved_dir)
                cv2.imwrite(os.path.join(configs.saved_dir, fn), show_img)
            else:
                #print(out_img)
                #print(out_img*1.0)
                #cv2.imshow(f'single_sample{img_file}', out_img)
                #cv2.imshow(f'single_sample{img_file}', out_img_clean)
                cv2.imshow(f'single_sample{img_file}', show_img)

        if not configs.save_img:
            if cv2.waitKey(0) & 0xff == 27:
                break
