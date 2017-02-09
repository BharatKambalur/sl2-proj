__author__ = 'Bharat'

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel

import cv2

kernels = []
for ksize in [5,10,15]:
    for theta in xrange(0,6):
        theta = theta / 6. * np.pi
        for lambd in [0.5,1,1.5]:
            kernel = cv2.getGaborKernel(ksize=(ksize,ksize),sigma=10,theta=theta,lambd=lambd,gamma=0.02)
            kernels.append(kernel)

for i in range(len(kernels)):
    plt.subplot(6,12,i+1)
    plt.imshow(np.real(kernels[i]), cmap='gray')
plt.show()
