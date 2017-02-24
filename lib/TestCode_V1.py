# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:09:16 2017

@author: sthar
"""

import os
import time
import cv2
import numpy as np
from skimage import io
from skimage.filters import gabor_kernel
from scipy.fftpack import dct
from scipy.signal import fftconvolve
import pickle
import matplotlib.pyplot as plt
############################################# PARAMETER DEFINITION #####################################################

batch_start = 5000
batch_end = 5000

#rgb_dir = '..\dataset\SYNTHIA_RAND_CVPR16\RGB\\'    # Location of folder containing the RGB images of the dataset
SLIC_dir = '..\dataset\SYNTHIA_RAND_CVPR16\SLIC\\'
gt_dir = '..\dataset\SYNTHIA_RAND_CVPR16\GTTXT\\'
feat_dir = '..\dataset\SYNTHIA_RAND_CVPR16\FEAT\\'
label_dir = '..\dataset\SYNTHIA_RAND_CVPR16\LABEL\\'
model_dir= '..\dataset\SYNTHIA_RAND_CVPR16\MODEL\\'

misc = [2,5,7,8,9,10,11]        # Defining all original classes which will be labelled miscellaneous
Overall_Error=0
color={'0':(0,0,0),'1':(132,112,255),'3':(160,160,160),'4':(218,165,32),'6':(0,128,0),'-1':(255,255,255),'10':(145,120,50),'2':(10,150,10)}
########################################################################################################################

list_files_Feat = os.listdir(feat_dir)
list_files_Feat.sort()

list_files_GT= os.listdir(gt_dir)
list_files_GT.sort()

list_files_Label= os.listdir(label_dir)
list_files_Label.sort()

list_files_SLIC = os.listdir(SLIC_dir)
list_files_SLIC.sort()

#######

###LOAD MODEL
model_file=model_dir+'trialmodel.sav'
load_model=pickle.load(open(model_file,'rb'))

################
img=np.zeros([720,960,3],dtype=np.uint8)

for im_no in range(batch_start, batch_end+1):
    ##NOTE!!! FEATURE NUMBERING STARTS FROM 0: SUBTRACT OUT EXTRA
    feat_path = feat_dir + list_files_Feat[im_no-5000]
    label_path = label_dir + list_files_Label[im_no-5000].split(".",1)[0]+".npy"
    TestX = np.load(feat_path)
    TestY = np.load(label_path)
    
    
    gt_path = gt_dir + list_files_GT[im_no].rsplit(".",1)[0] + '.txt'
    gt = np.loadtxt(gt_path)
    
    slic_path = SLIC_dir + list_files_SLIC[im_no].rsplit(".",1)[0] + '.npy'
    segments = np.load(slic_path)
    
    for i in misc:
        gt[gt == i] = 0
        TestY[TestY == i]=0

    PredictY= load_model.predict(TestX)
    
    img_Predict=np.zeros([720,960])

    for (i, segVal) in enumerate(np.unique(segments)):
        spat_cord_seg = np.array(np.where(segments == segVal))
        img_Predict[spat_cord_seg[0,:],spat_cord_seg[1,:]]=PredictY[segVal]
        img[spat_cord_seg[0,:],spat_cord_seg[1,:],:]=color[str(int(PredictY[segVal]))]
    
    Overall_Error=Overall_Error+(np.sum(img_Predict!=gt))
    


print(Overall_Error/(760*920*(batch_end-batch_start+1)))

fig = plt.imshow(img)  
