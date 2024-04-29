# -*- coding: utf-8 -*-
"""
Created on Wed May  3 16:04:39 2023

@author: debvrav1
"""
import glob
import cv2
import mat73
import numpy as np
from tqdm import tqdm
from skimage.io import imread

# def find_max_layerIdx(datarr):
#     indices = np.where(np.all(np.isnan(datarr), axis=1))[0] # tuple
#     maxIdx = indices[0] if len(indices) > 0 else 0
#     return maxIdx

# def calc_metrics(rootpath):
#     files = glob.glob(rootpath)
#     p_count = 0
#     p_sum = 0
#     p2_sum = 0
#     max_label = 0
#     for file in tqdm(sorted(files)):
#         matdata = mat73.loadmat(file)
#         img = matdata['Data']
#         rows,cols = img.shape
#         p_count += rows*cols
#         p_sum += np.sum(img)
#         # max_layerIdx = find_max_layerIdx(matdata['layers_vector'])
#         # max_label = max_layerIdx if max_layerIdx > max_label else max_label
        
#     mean = p_sum/p_count
    
#     for file in tqdm(sorted(files)):
#         matdata = mat73.loadmat(file)
#         img = matdata['Data']
#         p2_sum+=np.sum(np.square(img-mean))
    
#     std = np.sqrt(p2_sum/p_count)
        
#     return mean, std, max_label
    
# root = 'G:/My Drive/Research/Dataset/SR_Dataset_v1/train_data/*5km.mat'
# # data_mean, data_std, max_labels = calc_metrics(root)
# print(calc_metrics(root))

a=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]])
for i in range(a.shape[0]):
    print(i)
    print(a)
    if (a[i]==np.array([7,8,9])).all():
        a=np.delete(a,i,axis=0)
        i-=1
    
print(a)