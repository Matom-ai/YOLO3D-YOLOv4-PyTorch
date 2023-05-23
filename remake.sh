echo "Removing olds...";
rm training/image_2/*;
rm training/label_2/*;
rm training/velodyne/*;
rm training/calib/*;
rm testing/image_2/*;
rm testing/velodyne/*;
rm testing/calib/*;
echo "Extracting";
tar xvf pcs.tar;
echo "Copying calibs";
python amogus.py;
echo "Regenerating imagesets";
ls training/label_2/ | sed 's/.txt//' > ImageSets/val.txt;
ls testing/calib/ | sed 's/.txt//' > ImageSets/test.txt
