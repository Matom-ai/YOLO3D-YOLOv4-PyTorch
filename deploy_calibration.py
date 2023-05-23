
import shutil
import os

EXT = 'bin'

subdir = "dataset/zupt_car/training/velodyne/"
tardir = "dataset/zupt_car/training/calib/"
contents = os.listdir(subdir)
for file in contents:
    newfile = file.replace(EXT,"txt")
    print(newfile)
    shutil.copy("calib.txt",tardir + newfile)

subdir = "dataset/zupt_car/testing/velodyne/"
tardir = "dataset/zupt_car/testing/calib/"
contents = os.listdir(subdir)
for file in contents:
    newfile = file.replace(EXT,"txt")
    print(newfile)
    shutil.copy("calib.txt",tardir + newfile)
