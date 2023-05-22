#!/usr/bin/env bash
python test.py \
  --cfgfile ./config/cfg/yolo3d_yolov4.cfg \
  --gpu_idx 0 \
  --pretrained_path ../checkpoints/yolo3d_yolov4_im_re/Model_yolo3d_yolov4_im_re_epoch_600.pth \
  --save_test_output \
  --output_format 'image'\
	--display_3d
