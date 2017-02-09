__author__ = 'Bharat'

# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
import cv2
from scipy.signal import convolve2d
import os
from skimage.filters import gabor_kernel

def get_gabor_features(patch_img, bin_patch, kernels):
    # Patch Image - Grayscale patch image
    # bin_patch - Binary patch - Same size as patch image
    # Kernels -> (R X k) X 1 array of 2D ndarrays
    # R - No. of different sizes
    # k - No of different kernels for that size
    x_vals, y_vals = np.where(bin_patch == 1)
    mean_vec = []
    var_vec = []
    for k_num in xrange(len(kernels)):
        conv_op = convolve2d(patch_img,np.real(kernels[k_num]),mode='same')
        #fig = plt.figure("Conv")
        #ax = fig.add_subplot(6,4,k_num+1)
        #ax.imshow(conv_op)
        conv_vals = conv_op[x_vals,y_vals]
        mean_vec.append(np.mean(conv_vals))
        var_vec.append.(np.var(conv_vals))
        # # Calculating Histogram

        # conv_hist, xedges = np.histogram(conv_vals,10,range = (-10,10))
        # fig10 = plt.figure("Conv_Hist")
        # plt.plot(conv_hist)




slic_path = 'SLIC_Dev_Img_1.npy'
if os.path.exists(slic_path):
    segments = np.load(slic_path)

numSegments = segments.max() + 1

im = cv2.imread('Dev_Img_1.png',0)
im_g=cv2.equalizeHist(im)
#im_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);



# Extracting Patch 'x'
patch_no = 512
patch_img = segments == patch_no
patch_img = patch_img.astype(int)

patch_x, patch_y = np.where(patch_img == 1)

patch_img_g = np.multiply(im_g,patch_img)

# Cropping to only the required region
x_min_p = min(patch_x)
x_max_p = max(patch_x)
y_min_p = min(patch_y)
y_max_p = max(patch_y)
req_region = patch_img_g[x_min_p:x_max_p, y_min_p:y_max_p]
bin_patch = patch_img[x_min_p:x_max_p, y_min_p:y_max_p]

# Show the extracted patch
fig2 = plt.figure("Extracted Plot -- %d patch" % (patch_no))
ax2 = fig2.add_subplot(2,1,1)
ax2.imshow(patch_img)
ax3 = fig2.add_subplot(2,1,2)
ax3.imshow(req_region, cmap='gray')

# kernels = []
# for ksize in [5,10,15]:
#     for theta in xrange(0,6):
#         theta = theta / 6. * np.pi
#         for lambd in [0.5,1,1.5]:
#             kernel = cv2.getGaborKernel(ksize=(ksize,ksize),sigma=10,theta=theta,lambd=lambd,gamma=0.02)
#             kernels.append(kernel)

kernels = []
for theta in range(6):
    theta = theta / 6. * np.pi
    for sigma in [3,5]:
        for frequency in (0.15, 0.35):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)

for i in range(len(kernels)):
    fig5 = plt.figure("Kernels")
    ax5 = fig5.add_subplot(6,4,i+1)
    ax5.imshow(np.real(kernels[i]), cmap='gray')

selective_sliding_window(req_region,bin_patch,kernels)
plt.show()