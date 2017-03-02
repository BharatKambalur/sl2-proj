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
from sklearn.preprocessing import normalize

############################################# PARAMETER DEFINITION #####################################################


batch_start = 6000
batch_end = 6020

error_dir = '..\dataset\SYNTHIA_RAND_CVPR16\ERROR\\'
rgb_dir = '..\dataset\SYNTHIA_RAND_CVPR16\RGB\\'    # Location of folder containing the RGB images of the dataset
SLIC_dir = '..\dataset\SYNTHIA_RAND_CVPR16\SLIC\\'
gt_dir = '..\dataset\SYNTHIA_RAND_CVPR16\GTTXT\\'
feat_dir = '..\dataset\SYNTHIA_RAND_CVPR16\FEAT\\'
label_dir = '..\dataset\SYNTHIA_RAND_CVPR16\LABEL\\'
model_dir= '..\dataset\SYNTHIA_RAND_CVPR16\MODEL\\'
predicted_dir= '..\dataset\SYNTHIA_RAND_CVPR16\PREDICTED\\'

misc = [2,5,7,8,9,11]        # Defining all original classes which will be labelled miscellaneous
orgi = [-1,0,1,3,4,6,10]
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

list_files_RGB = os.listdir(rgb_dir)
list_files_RGB.sort()

#######

###LOAD MODEL
model_name = 'trialmodel.sav'
model_file=model_dir + model_name
load_model=pickle.load(open(model_file,'rb'))

################
img=np.zeros([720,960,3],dtype=np.uint8)

confusion = np.zeros([7,7])
temp_confusion = np.zeros([7,7])
for im_no in range(batch_start, batch_end+1):
    ##NOTE!!! FEATURE NUMBERING STARTS FROM 0: SUBTRACT OUT EXTRA
    gt_path = gt_dir + list_files_GT[im_no].rsplit(".",1)[0] + '.txt'
    feat_path = feat_dir + list_files_RGB[im_no].rsplit(".",1)[0] + '.npy'
    label_path = label_dir + list_files_RGB[im_no].rsplit(".",1)[0] + '.npy'
    slic_path = SLIC_dir + list_files_RGB[im_no].rsplit(".",1)[0] + '.npy'
    
    TestX = np.load(feat_path)
    TestY = np.load(label_path)
    
    gt = np.loadtxt(gt_path)    
    for i in misc:
        gt[gt == i] = 0

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
    
    predicted_path = predicted_dir  +  list_files_RGB[im_no].rsplit(".",1)[0] +'_predicted_output' + '.npy'    
    np.save(predicted_path , PredictY)            
                    
    Overall_Error += (np.sum(img_Predict!=gt))/(720*960)
    
    for ind_gt in range(0,6):
            spat_cord_class = np.array(np.where(gt == orgi[ind_gt]))
            for ind_pred in range(0,6):
                temp_confusion[ind_gt,ind_pred] = np.sum(img_Predict[spat_cord_class[0,:],spat_cord_class[1,:]] == orgi[ind_pred])
                                                                   
    confusion += normalize(temp_confusion, axis=1, norm='l1')          

confusion = confusion/(batch_end-batch_start+1)
Overall_Error=Overall_Error/(batch_end-batch_start+1)
print(Overall_Error/(batch_end-batch_start+1))

error_path = error_dir  +  model_name.rsplit(".",1)[0] +'_overall_error' + '.npy'
confusion_path = error_dir  +  model_name.rsplit(".",1)[0] +'_confusion_error' + '.npy'

np.save(error_path , Overall_Error) 
np.save(confusion_path , confusion) 


plt.imshow(confusion)
fig = plt.imshow(img)  
