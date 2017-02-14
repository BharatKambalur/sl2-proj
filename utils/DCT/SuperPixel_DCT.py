# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 17:55:25 2017

@author: sthar
"""

# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import color
from scipy.fftpack import dct
from skimage.color import rgb2gray
import time


##
start=time.time()
image = img_as_float(io.imread("C:/Users/sthar/Desktop/UCSD WI2017/Statistical Learning 2/Project/img-seg-multi-master/Dev_Img_1.png"))
#NOTE:DCT will fail if centroid close to edge: NOT ENOUGH FOR 8*8 ARRAY =>REDUCE IMAGE AROUND EDGES


#image_lab = color.rgb2hsv(image)

# loop over the number of segments
numSegments = 1000

# apply SLIC and extract (approximately) the supplied number
segments = slic(image, n_segments = numSegments, sigma = 2)
end=time.time()
# show the output of SLIC
fig = plt.figure("Superpixels -- %d segments" % (numSegments))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(image, segments))
plt.axis("off")

# show the plots
fig.set_size_inches(12,9)
fig.savefig('output1.png',bbox_inches='tight')
plt.show()

print(end-start)
#%% 
##DCT Calculate for Patches
DCT=np.zeros((numSegments,64))
#start=time.time()
ErrorCount=0
for (i, segVal) in enumerate(np.unique(segments)):
    image_gray=rgb2gray(image)
	# construct a mask for the segment
    mask = np.zeros(image.shape[:2], dtype = "uint8")
    mask[segments == segVal] = 255
    
    ret,thresh = cv2.threshold(mask,127,255,0)
    ret,contours,hierarchy = cv2.findContours(thresh, 1, 2)
    M = cv2.moments(contours[0])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    if(cX-3>-1 and cY+5<721 and cX+5<961 and cY-3>-1):
     DCT_Patch = (dct(dct(image_gray[cY-3:cY+5,cX-3:cX+5], axis=0), axis=1).ravel()).reshape(1,64)
    
    else:
     ErrorCount=ErrorCount+1
     print(ErrorCount)
    
    #TO DO: APPEND DCT FOR EACH PATCH TO CORRESPONDING CLASS IN DICT. 
    #DCT[segments,:] = (dct(dct(image_gray[cY-3:cY+5,cX-3:cX+5], axis=0), axis=1).ravel()).reshape(1,64)
    
#end=time.time()
print(end-start)