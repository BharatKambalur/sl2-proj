__author__ = 'Bharat'


import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel

kernels = []
for theta in range(6):
    theta = theta / 6. * np.pi
    for sigma in [2,3]:
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)

for i in range(len(kernels)):
    plt.subplot(6,4,i+1)
    plt.imshow(np.real(kernels[i]))
plt.show()