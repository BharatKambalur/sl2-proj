__author__ = 'Bharat'
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
from skimage import color
import numpy as np
import time
import os

########### PARAMETER DEFINITION #############
rgb_dir = '..\dataset\SYNTHIA_RAND_CVPR16\RGB\\' # Location of folder containing the RGB images of the dataset
SLIC_dir = '..\dataset\SYNTHIA_RAND_CVPR16\SLIC\\'

SLIC_File_Name = 'ap_000_02-11-2015_08-27-21_000068_2_Rand_2.npy'

############################################

#### Main Part of Program START ###########

slic_path = SLIC_dir + SLIC_File_Name
im_path = rgb_dir + SLIC_File_Name.rsplit(".",1)[0] + '.png'

if os.path.exists(im_path):
    image = io.imread(im_path)
if os.path.exists(slic_path):
    segments = np.load(slic_path)
fig = plt.figure('Image with Segments Visualized - %s' % SLIC_File_Name)
ax = fig.add_subplot(1, 1, 1)
plt.imshow(mark_boundaries(image, segments))
plt.show()