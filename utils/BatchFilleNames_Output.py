__author__ = 'Bharat'
from skimage.segmentation import slic
from skimage import io
import numpy as np
import time
import os
import sys

########### PARAMETER DEFINITION #############
rgb_dir = '..\dataset\SYNTHIA_RAND_CVPR16\RGB\\' # Location of folder containing the RGB images of the dataset
SLIC_dir = '..\dataset\SYNTHIA_RAND_CVPR16\SLIC\\'

############################################

#### Main Part of Program START ###########

# Get List of RGB Image files from directory (relative location)
list_files = os.listdir(rgb_dir)
list_files.sort()

for block in range(0,14):
    list_start = block * 1000
    list_end = (block+1) * 1000 -1
    print ('{} : {}'.format(list_start, list_files[list_start]))
    if list_end > len(list_files):
        list_end = len(list_files)-1
    print ('{} : {}'.format(list_end, list_files[list_end]))