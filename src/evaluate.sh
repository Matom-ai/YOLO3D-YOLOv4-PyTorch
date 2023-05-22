#!/usr/bin/env bash
python evaluate.py \
	--gpu_idx 0 \
  --pretrained_path ../checkpoints/yolo3d_yolov4_im_re/Model_yolo3d_yolov4_im_re_epoch_600.pth \
  --cfgfile ./config/cfg/yolo3d_yolov4.cfg \
	--img_size 608 \
	--conf-thresh 0.5 \
	--nms-thresh 0.5 \
	--iou-thresh 0.5
