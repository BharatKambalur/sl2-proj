__author__ = 'Bharat'

#%%
# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
import cv2
from scipy.signal import convolve2d, fftconvolve
import os
from skimage.filters import gabor_kernel
import time
#%%


def get_gabor_features(patch_img, bin_patch, kernels):
    # Patch Image - Grayscale patch image
    # bin_patch - Binary patch - Same size as patch image
    # Kernels -> (R X k) X 1 array of 2D ndarrays
    # R - No. of different sizes
    # k - No of different kernels for that size
    x_vals, y_vals = np.where(bin_patch == 1)
    gab_mean = []
    gab_var = []
    for k_num in xrange(len(kernels)):
        conv_op = convolve2d(patch_img,np.real(kernels[k_num]),mode='same')
        #fig = plt.figure("Conv")
        #ax = fig.add_subplot(6,4,k_num+1)
        #ax.imshow(conv_op)
        conv_vals = conv_op[x_vals,y_vals]
        gab_mean.append(np.mean(conv_vals))
        gab_var.append(np.var(conv_vals))
    return gab_mean, gab_var
        # # Calculating Histogram

        # conv_hist, xedges = np.histogram(conv_vals,10,range = (-10,10))
        # fig10 = plt.figure("Conv_Hist")
        # plt.plot(conv_hist)

#%%
########### PARAMETER DEFINITION #############
rgb_dir = '..\dataset\SYNTHIA_RAND_CVPR16\RGB\\' # Location of folder containing the RGB images of the dataset
SLIC_dir = '..\dataset\SYNTHIA_RAND_CVPR16\SLIC\\'
gt_dir = '..\dataset\SYNTHIA_RAND_CVPR16\GTTXT\\'

#%%
#### Main Part of Program START ###########

# Get List of RGB Image files from directory (relative location)
list_files_RGB = os.listdir(rgb_dir)
list_files_RGB.sort()

rgb_path = rgb_dir + list_files_RGB[0]
slic_path = SLIC_dir + list_files_RGB[0].rsplit(".",1)[0] + '.npy'
gt_path = gt_dir + list_files_RGB[0].rsplit(".",1)[0] + '.txt'

if os.path.exists(rgb_path):
    im = io.imread(rgb_path)
else:
    print('RGB Image File Read Error. File Name : {}'.format(list_files_RGB[0]))

if os.path.exists(slic_path):
    segments = np.load(slic_path)
else:
    print('SLIC File Read Error. File Name : {}'.format(list_files_RGB[0].rsplit(".",1)[0] + '.npy'))

if os.path.exists(slic_path):
    gt = np.loadtxt(gt_path)
else:
    print('GT Text File Read Error. File Name : {}'.format(list_files_RGB[0].rsplit(".",1)[0] + '.txt'))

#%%
im_g = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
im_g_h=cv2.equalizeHist(im_g)

kernels = []
for theta in range(6):
    theta = theta / 6. * np.pi
    for sigma in [3,5]:
        for frequency in (0.15, 0.35):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)

start_time = time.time()
for (i, segVal) in enumerate(np.unique(segments)):

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
    req_region = patch_img_g[x_min_p:x_max_p, y_min_p:y_max_p]
    bin_patch = patch_img[x_min_p:x_max_p, y_min_p:y_max_p]

    # # Show the extracted patch
    # fig2 = plt.figure("Extracted Plot -- %d patch" % (patch_no))
    # ax2 = fig2.add_subplot(2,1,1)
    # ax2.imshow(patch_img)
    # ax3 = fig2.add_subplot(2,1,2)
    # ax3.imshow(req_region, cmap='gray')
    #
    # fig10 = plt.figure("Bin Patch")
    # ax10 = fig10.add_subplot(1,1,1)
    # ax10.imshow(bin_patch)

    # kernels = []
    # for ksize in [5,10,15]:
    #     for theta in xrange(0,6):
    #         theta = theta / 6. * np.pi
    #         for lambd in [0.5,1,1.5]:
    #             kernel = cv2.getGaborKernel(ksize=(ksize,ksize),sigma=10,theta=theta,lambd=lambd,gamma=0.02)
    #             kernels.append(kernel)


    # for i in range(len(kernels)):
    #     fig5 = plt.figure("Kernels")
    #     ax5 = fig5.add_subplot(6,4,i+1)
    #     ax5.imshow(np.real(kernels[i]), cmap='gray')

    x_vals, y_vals = np.where(bin_patch == 1)
    gab_mean = []
    gab_var = []
    for k_num in xrange(len(kernels)):
        conv_op = fftconvolve(req_region,np.real(kernels[k_num]),mode='same')
        #fig = plt.figure("Conv")
        #ax = fig.add_subplot(6,4,k_num+1)
        #ax.imshow(conv_op)
        conv_vals = conv_op[x_vals,y_vals]
        gab_mean.append(np.mean(conv_vals))
        gab_var.append(np.var(conv_vals))
    #[gab_mean_arr, gab_var_Arr] = get_gabor_features(req_region,bin_patch,kernels)
    gab_feat = np.concatenate((gab_mean,gab_var))
    #print(gab_mean_arr)
    #print(gab_var_Arr)
end_time = time.time()
print('{}'.format(end_time-start_time))