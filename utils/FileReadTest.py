__author__ = 'Bharat'
from skimage.segmentation import slic
from skimage import io
import numpy as np
import time
import os

########### PARAMETER DEFINITION #############
rgb_dir = '..\dataset\SYNTHIA_RAND_CVPR16\RGB\\' # Location of folder containing the RGB images of the dataset
SLIC_dir = '..\dataset\SYNTHIA_RAND_CVPR16\SLIC\\'

list_start = 1100
list_end = 1399

numSegments = 1000

sigma = 2  # Sigma value of Gaussian Smoothening Applied before SLIC

############################################

#### Main Part of Program START ###########

start_time = time.time()

# Get List of RGB Image files from directory (relative location)
list_files = os.listdir(rgb_dir)
list_files.sort()

for im_no in range(list_start, list_end+1):
    image = io.imread(rgb_dir+list_files[im_no])
    segments = slic(image, n_segments = numSegments, sigma = sigma)
    np.save(SLIC_dir + list_files[im_no].rsplit(".",1)[0] + '.npy',segments)

end_time = time.time()

print ('{0} Files Processed. Time Taken: {1}'.format(list_end-list_start+1, end_time-start_time))

