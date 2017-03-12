# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 23:19:21 2017

@author: sthar
"""

import os
import time
import cv2
import numpy as np
from skimage import color, io
from skimage.filters import gabor_kernel
from scipy.fftpack import dct
from scipy.signal import fftconvolve

############################################# PARAMETER DEFINITION #####################################################

batch_start = 601 
batch_end = 900

rgb_dir = '..\dataset\CITYSCAPE\RGB\\' # Location of folder containing the RGB images of the dataset
SLIC_dir = '..\dataset\CITYSCAPE\SLIC\\'
gt_dir = '..\dataset\CITYSCAPE\GT\\'
feat_dir = '..\dataset\CITYSCAPE\FEAT\\'
label_dir = '..\dataset\CITYSCAPE\LABEL\\'

#misc = [2,5,7,8,9,10,11]        # Defining all original classes which will be labelled miscellaneous

########################################################################################################################

########################################## MAIN PART OF CODE - START ###################################################

# Generating the required Gabor Kernel (Multiple orientations, sigma and frequency)
kernels = []
for theta in range(6):
    theta = theta / 6. * np.pi
    for sigma in [3,5]:
        for frequency in (0.15, 0.35):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)

print ('Starting Feature Extraction with List no {0} until {1}'.format(batch_start, batch_end))

start_time = time.time()

# Get List of RGB Image files from directory (relative location)
list_files_RGB = os.listdir(rgb_dir)
list_files_RGB.sort()

list_files_GT = os.listdir(gt_dir)
list_files_GT.sort()

for im_no in range(batch_start, batch_end+1):
    rgb_path = rgb_dir + list_files_RGB[im_no]
    slic_path = SLIC_dir + list_files_RGB[im_no].rsplit(".",1)[0] + '.npy'
    gt_path = gt_dir + list_files_GT[im_no].rsplit(".",1)[0] + '.png'

    if os.path.exists(rgb_path):
        im = io.imread(rgb_path)
    else:
        print('RGB Image File Read Error. File Name : {}'.format(list_files_RGB[im_no]))

    if os.path.exists(slic_path):
        segments = np.load(slic_path)
    else:
        print('SLIC File Read Error. File Name : {}'.format(list_files_RGB[im_no].rsplit(".",1)[0] + '.npy'))

    if os.path.exists(gt_path):
        gt = io.imread(gt_path)
    else:
        print('GT File Read Error. File Name : {}'.format(list_files_RGB[im_no].rsplit(".",1)[0] + '.txt'))

    # Declaring empty arrays to store the generated features (BigX) and corresponding label (BigY)
    BigX=[]
    BigY=[]
    print(im_no)
    # Converting image to HSV and Grayscale for further processing
    image_hsv = color.rgb2hsv(im)
    im_g = color.rgb2gray(im)
    im_g2 = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);    # Conversion using cv2 because of datatype mismatch

    # Performing Histogram Equalization for the gray image
    im_g_h=cv2.equalizeHist(im_g2)

    # Converting unused classes to 'misc' class in Ground Truth
    # for i in misc:
    #     gt[gt == i] = 0

    # Defining number of bins for histogram (Used to quantize H_S and V)
    nBins = 10

    # Loop over each segment (SuperPixel)
    for (i, segVal) in enumerate(np.unique(segments)):

        ########################### Code for DCT ################################
        
        mask = np.zeros(im.shape[:2], dtype = "uint8")
        mask[segments == segVal] = 255

        #Create Contours
        ret,thresh = cv2.threshold(mask,127,255,0)
        ret,contours,hierarchy = cv2.findContours(mask, 1, 2)

        #Centroid
        M = cv2.moments(max(contours,key=len))
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])


        #Obtain DCT Patch
        if(cX-3<0):
           cX=3

        if(cY-3<0):
           cY=3

        if(cX+5>2048):
           cX=2042

        if(cY+5>1024):
            cY=1018

        DCT_Patch = (dct(dct(im_g[cY-3:cY+5,cX-3:cX+5], axis=0), axis=1).ravel()).reshape(1,64).ravel()

        Centroid_Patch=np.array([cX,cY])

        ###################### Code for HSV ##############################

        # Mask only the required locations from the image
        spat_cord_seg = np.array(np.where(segments == segVal))

        # Retrieve the HSV values for each pixel position in the segment
        hsv_values_seg = image_hsv[spat_cord_seg[0,:],spat_cord_seg[1,:]]
        h = hsv_values_seg[:,0]
        s = hsv_values_seg[:,1]
        v = hsv_values_seg[:,2]

        # Create a 2D histogram with num of bins nBins = 10, for HUE and SATURATION
        H_S_hist, xedges, yedges = np.histogram2d(h, s, bins=nBins, range = [[0,1],[0,1]])

        # OneD histogram with num of bins nBins = 10
        V_hist, xedges= np.histogram(v, bins = nBins, range =(0,1))

        # Store the feature vectors
        H_S_val= np.resize(H_S_hist,(1,nBins*nBins)).ravel()
        H_S_val= (H_S_val)/(np.sum(H_S_val))
        V_val= V_hist

        ##################### Code for Gabor Features #######################

        # Extracting Patch 'x'
        patch_img = segments == segVal
        patch_img = patch_img.astype(int)

        patch_x, patch_y = np.where(segments == segVal)

        patch_img_g = np.multiply(im_g_h,patch_img)

        # Cropping to only the required region
        x_min_p = min(patch_x)
        x_max_p = max(patch_x)
        y_min_p = min(patch_y)
        y_max_p = max(patch_y)
        req_region = patch_img_g[x_min_p:x_max_p, y_min_p:y_max_p]  # Contains small patch having grayscale info
        bin_patch = patch_img[x_min_p:x_max_p, y_min_p:y_max_p]     # Contains small patch having binary values

        x_vals, y_vals = np.where(bin_patch == 1)
        gab_mean = []
        gab_var = []
        for k_num in range(len(kernels)):
            conv_op = fftconvolve(req_region,np.real(kernels[k_num]),mode='same')
            conv_vals = conv_op[x_vals,y_vals]
            gab_mean.append(np.mean(conv_vals))
            gab_var.append(np.var(conv_vals))
        gab_feat = np.concatenate((gab_mean,gab_var)) # Stores the generated gabor feature vector

        ############################ Code for Area and Aspect Ratio of Blob ##################################

        area = np.sum(bin_patch)
        try:
            aspect_ratio = float(bin_patch.shape[0])/bin_patch.shape[1]
        except ZeroDivisionError:
            aspect_ratio = 0

        ########################## Obtain Class Number for Segment ##############################

        spat_cord_seg_GT = gt[spat_cord_seg[0,:],spat_cord_seg[1,:]]
        unique_val_GT, count_seg_GT = np.unique(spat_cord_seg_GT, return_counts = True)
        class_label_seg = unique_val_GT[np.argmax(count_seg_GT)]

        BigX.append(np.hstack((DCT_Patch,Centroid_Patch,H_S_val,V_val,gab_feat,area,aspect_ratio)))
        BigY.append(class_label_seg)

    # Save the obtained feature vector and the class label vector (which corresponds to each segment)
    feat_path = feat_dir + list_files_RGB[im_no].rsplit(".",1)[0] + '.npy'
    label_path = label_dir + list_files_RGB[im_no].rsplit(".",1)[0] + '.npy'

    np.save(feat_path, np.array(BigX))
    np.save(label_path, np.array(BigY))

end_time = time.time()
print ('{0} Files Processed. Time Taken: {1}'.format(batch_end-batch_start+1, end_time-start_time))