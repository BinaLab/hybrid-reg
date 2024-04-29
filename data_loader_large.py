from os.path import *
from os import *
import cv2
import numpy as np
import datetime
import glob

img_ext = '.png'
mat_ext = '.mat'
txt_ext = '.txt'
layer_prfx = 'data' ## 'layer' or 'data'
img_prfx = 'img' ## 'data' or 'image'
# max_labels = 34

"""
train_img_root = '../../Dataset/Snow Radar/2012_main_dv_dry2/train/image'
train_thick_man_root = '../../Dataset/Snow Radar/2012_main_dv_dry2/train/manual_thick_ext'
train_thick_phy_root = '../../Dataset/Snow Radar/2012_main_dv_dry2/train/mar_thick_ext'

test_img_root = '../../Dataset/Snow Radar/2012_main_dv_dry2/test/image'
test_thick_man_root = '../../Dataset/Snow Radar/2012_main_dv_dry2/test/manual_thick_ext'
test_thick_phy_root = '../../Dataset/Snow Radar/2012_main_dv_dry2/test/mar_thick_ext'
"""

train_root = '../../Dataset/SR_Dataset_v1/train_data/'

st = datetime.datetime.now()
### load training images ###
print('loading training images')
traindata = []
# img_files = [join(train_root,file) for file in listdir(train_root) if img_ext in file and img_prfx in file]
img_files = glob.glob(train_root+'img*20km.png')
print('total files = ' + str(len(img_files)))
for file in sorted(img_files):
    img = cv2.imread(file)
    try:
        img_224 = cv2.resize(img, (224,224))
    except:
        print(file + ' error')
        continue
    traindata.append(img_224)

traindata = np.asarray(traindata)
train_mean = np.mean(traindata)
train_std = np.std(traindata)
traindata = (traindata - train_mean) / train_std

print('total time taken to load training images = ' + str(datetime.datetime.now()-st) + ' seconds')
### load training thickness ###
print('loading training thickness')
train_thick_man = []
train_thick_phy = []
thick_files = [join(train_root,file) for file in listdir(train_root) if mat_ext in file]
# for file in sorted(thick_files):
for j in range(len(img_files)):
    # thicks = np.loadtxt(file)
    thicks = np.random.rand(1,34)[0]
    train_thick_man.append(thicks)
train_thick_man = np.asarray(train_thick_man)
train_thick_phy = train_thick_man
train_thick = [train_thick_man, train_thick_phy]

train_thick = np.asarray(train_thick).transpose((1,0,2))

print('traindata shape = ' + str(traindata.shape) + '; train_thick shape = ' + str(train_thick.shape))
### --- test --- ###
# ### load test images ###
# print('loading test images')
# testdata = []
# img_files = [join(test_img_root,file) for file in listdir(test_img_root) if img_ext in file]
# for file in sorted(img_files):
#     img = cv2.imread(file)
#     try:
#         img_224 = cv2.resize(img, (224,224))
#     except:
#         print(file + ' error')
#         continue
#     testdata.append(img_224)

# testdata = np.asarray(testdata)
# test_mean = np.mean(testdata)
# test_std = np.std(testdata)
# testdata = (testdata - test_mean) / test_std
# print(testdata.shape)

# ### load test thickness ###
# print('loading test thickness')

# test_thick_man = []
# thick_files = [join(test_thick_man_root,file) for file in listdir(test_thick_man_root) if txt_ext in file]
# for file in sorted(thick_files):
#     thicks = np.loadtxt(file)
#     test_thick_man.append(thicks)
# test_thick_man = np.asarray(test_thick_man)

# test_thick_phy = []
# thick_files = [join(test_thick_phy_root,file) for file in listdir(test_thick_phy_root) if txt_ext in file]
# for file in sorted(thick_files):
#     thicks = np.loadtxt(file)
#     test_thick_phy.append(thicks)
# test_thick_phy = np.asarray(test_thick_phy)

# test_thick = [test_thick_man, test_thick_phy]