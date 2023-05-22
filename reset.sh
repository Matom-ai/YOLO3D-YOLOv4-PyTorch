echo "Clearnig...";
rm -rf checkpoints/yolo3d_yolov4_im_re/bakup/*;
rm -rf logs/yolo3d_yolov4_im_re/amogus/*;
mv checkpoints/yolo3d_yolov4_im_re/*.pth checkpoints/yolo3d_yolov4_im_re/bakup/;
mv logs/yolo3d_yolov4_im_re/logger_yolo3d_yolov4_im_re.txt logs/yolo3d_yolov4_im_re/amogus/;
mv logs/yolo3d_yolov4_im_re/tensorboard/ logs/yolo3d_yolov4_im_re/amogus/; 
