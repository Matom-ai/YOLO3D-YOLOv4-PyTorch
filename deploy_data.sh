echo "Creating directories...";
mkdir dataset/zupt_car/;
mkdir dataset/zupt_car/ImageSets/;
mkdir dataset/zupt_car/training/;
mkdir dataset/zupt_car/testing/;
mkdir dataset/zupt_car/training/calib/;
mkdir dataset/zupt_car/testing/calib/;
echo "Extracting";
tar xvf pcs.tar -C dataset/zupt_car/;
echo "Copying calibs";
python deploy_calibration.py;
echo "Regenerating imagesets";
ls dataset/zupt_car/training/label_2/ | sed 's/.txt//' > dataset/zupt_car/ImageSets/val.txt;
ls dataset/zupt_car/testing/calib/ | sed 's/.txt//' > dataset/zupt_car/ImageSets/test.txt
