# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 22:50:40 2017

@author: sthar
"""

__author__ = 'Bharat'
from skimage.segmentation import slic
from skimage import io
import numpy as np
import time
import os
import sys

########### PARAMETER DEFINITION #############
rgb_dir = '..\dataset\CITYSCAPE\RGB\\' # Location of folder containing the RGB images of the dataset
SLIC_dir = '..\dataset\CITYSCAPE\SLIC\\'

list_start = 1001
list_end = 1499

numSegments = 2000

sigma = 2  # Sigma value of Gaussian Smoothening Applied before SLIC

############################################

#### Main Part of Program START ###########

print('Starting SLIC with List no {0} until {1}'.format(list_start, list_end))

start_time = time.time()

# Get List of RGB Image files from directory (relative location)
list_files = os.listdir(rgb_dir)
list_files.sort()

for im_no in range(list_start, list_end+1):
    image = io.imread(rgb_dir+list_files[im_no])
    segments = slic(image, n_segments = numSegments, sigma = sigma)
    np.save(SLIC_dir + list_files[im_no].rsplit(".",1)[0] + '.npy',segments)
    print(im_no)

end_time = time.time()

print('{0} Files Processed. Time Taken: {1}'.format(list_end-list_start+1, end_time-start_time))