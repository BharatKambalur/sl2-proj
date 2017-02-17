# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 20:12:53 2017

@author: sthar
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 20:01:04 2017

@author: sthar
"""

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
from skimage import color # For the convertion from RGB to HSV
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import cv2
import numpy as np
from scipy.fftpack import dct
from skimage.color import rgb2gray
import time

misc = [2,5,7,8,9,10,11]

BigX=[]
BigY=[]
###############################################################################
# load the image and convert it to a floating point data type
im = img_as_float(io.imread('img3.PNG'))
gt = np.loadtxt('GT_TXT_Img_3.txt')
# loop over the number of segments
numSegments = 1000

# apply SLIC and extract (approximately) the supplied number
segments = slic(im, n_segments = numSegments, sigma = 2)
end=time.time()
# show the output of SLIC
fig = plt.figure("Superpixels -- %d segments" % (numSegments))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(im, segments))
plt.axis("off")


image_hsv = color.rgb2hsv(im)
im_g=rgb2gray(im)

# Converting useless classes to misc

for i in misc:
    gt[gt==i]=0
            

# To store Class Labels
class_label_seg = np.zeros([1,1])

# Number of bin: H_S
b = 10



 # Loop over each of the segments
for (i, segVal) in enumerate(np.unique(segments)):
    
    ##DCT
    mask = np.zeros(im.shape[:2], dtype = "uint8")
    mask[segments == segVal] = 255
    
    #Create Contours
    ret,thresh = cv2.threshold(mask,127,255,0)
    ret,contours,hierarchy = cv2.findContours(mask, 1, 2)
    
    #Centroid
    M = cv2.moments(contours[0])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    
    #Obtain DCT Patch
    if(cX-3<0):
       cX=3
     
    if(cY-3<0):
       cY=3
     
    if(cX+5>960):
       cX=954
    
    if(cY+5>720):
        cY=714
     
    DCT_Patch = (dct(dct(im_g[cY-3:cY+5,cX-3:cX+5], axis=0), axis=1).ravel()).reshape(1,64).ravel()
     
    Centroid_Patch=np.array([cX,cY])
     
    ##HSV
    # Mask only the required locaitons from the image
    spat_cord_seg = np.array(np.where(segments == segVal))
    hsv_values_seg = image_hsv[spat_cord_seg[0,:],spat_cord_seg[1,:]]
    
    # Get Corresponding Class Number 
    spat_cord_seg_GT = gt[spat_cord_seg[0,:],spat_cord_seg[1,:]];
    unique_val_GT, count_seg_GT = np.unique(spat_cord_seg_GT, return_counts = True);
    class_label_seg = unique_val_GT[np.argmax(count_seg_GT)]                                           
    
     # Retrive the HSV values for each pixel postition in the segment
    h = hsv_values_seg[:,0]
    s = hsv_values_seg[:,1]
    v = hsv_values_seg[:,2]
    
    # Create a 2D histogram with num of bins b = 10, for HUE and SATURATION
    H_S_hist, xedges, yedges = np.histogram2d(h, s, bins=b, range = [[0,1],[0,1]])
     
    # OneD histogram with num of bins b = 10
    V_hist, xedges= np.histogram(v, bins = b, range =(0,1))
     
    # Store the feature vectors    
    H_S_val= np.resize(H_S_hist,(1,b*b)).ravel()
    H_S_val= (H_S_val)/(np.sum(H_S_val)) 
    V_val= V_hist
     
    BigX.append(np.hstack((DCT_Patch,Centroid_Patch,H_S_val,V_val)))
    BigY.append(class_label_seg)
##############IMG2##############################################################
# load the image and convert it to a floating point data type
im = img_as_float(io.imread('img4.PNG'))
gt = np.loadtxt('GT_TXT_Img_4.txt')
# loop over the number of segments
numSegments = 1000

# apply SLIC and extract (approximately) the supplied number
segments = slic(im, n_segments = numSegments, sigma = 2)
end=time.time()
# show the output of SLIC
fig = plt.figure("Superpixels -- %d segments" % (numSegments))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(im, segments))
plt.axis("off")


image_hsv = color.rgb2hsv(im)
im_g=rgb2gray(im)

# Converting useless classes to misc

for i in misc:
    gt[gt==i]=0
            

# To store Class Labels
class_label_seg = np.zeros([1,1])

# Number of bin: H_S
b = 10



 # Loop over each of the segments
for (i, segVal) in enumerate(np.unique(segments)):
    
    ##DCT
    mask = np.zeros(im.shape[:2], dtype = "uint8")
    mask[segments == segVal] = 255
    
    #Create Contours
    ret,thresh = cv2.threshold(mask,127,255,0)
    ret,contours,hierarchy = cv2.findContours(mask, 1, 2)
    
    #Centroid
    M = cv2.moments(contours[0])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    
    #Obtain DCT Patch
    if(cX-3<0):
       cX=3
     
    if(cY-3<0):
       cY=3
     
    if(cX+5>960):
       cX=954
    
    if(cY+5>720):
        cY=714
     
    DCT_Patch = (dct(dct(im_g[cY-3:cY+5,cX-3:cX+5], axis=0), axis=1).ravel()).reshape(1,64).ravel()
     
    Centroid_Patch=np.array([cX,cY])
     
    ##HSV
    # Mask only the required locaitons from the image
    spat_cord_seg = np.array(np.where(segments == segVal))
    hsv_values_seg = image_hsv[spat_cord_seg[0,:],spat_cord_seg[1,:]]
    
    # Get Corresponding Class Number 
    spat_cord_seg_GT = gt[spat_cord_seg[0,:],spat_cord_seg[1,:]];
    unique_val_GT, count_seg_GT = np.unique(spat_cord_seg_GT, return_counts = True);
    class_label_seg = unique_val_GT[np.argmax(count_seg_GT)]                                           
    
     # Retrive the HSV values for each pixel postition in the segment
    h = hsv_values_seg[:,0]
    s = hsv_values_seg[:,1]
    v = hsv_values_seg[:,2]
    
    # Create a 2D histogram with num of bins b = 10, for HUE and SATURATION
    H_S_hist, xedges, yedges = np.histogram2d(h, s, bins=b, range = [[0,1],[0,1]])
     
    # OneD histogram with num of bins b = 10
    V_hist, xedges= np.histogram(v, bins = b, range =(0,1))
     
    # Store the feature vectors    
    H_S_val= np.resize(H_S_hist,(1,b*b)).ravel()
    H_S_val= (H_S_val)/(np.sum(H_S_val)) 
    V_val= V_hist
     
    BigX.append(np.hstack((DCT_Patch,Centroid_Patch,H_S_val,V_val)))
    BigY.append(class_label_seg)
###############TRAINING IMG3######################################
# load the image and convert it to a floating point data type
im = img_as_float(io.imread('img5.PNG'))
gt = np.loadtxt('GT_TXT_Img_5.txt')
# loop over the number of segments
numSegments = 1000

# apply SLIC and extract (approximately) the supplied number
segments = slic(im, n_segments = numSegments, sigma = 2)
end=time.time()
# show the output of SLIC
fig = plt.figure("Superpixels -- %d segments" % (numSegments))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(im, segments))
plt.axis("off")


image_hsv = color.rgb2hsv(im)
im_g=rgb2gray(im)

# Converting useless classes to misc

for i in misc:
    gt[gt==i]=0
            

# To store Class Labels
class_label_seg = np.zeros([1,1])

# Number of bin: H_S
b = 10



 # Loop over each of the segments
for (i, segVal) in enumerate(np.unique(segments)):
    
    ##DCT
    mask = np.zeros(im.shape[:2], dtype = "uint8")
    mask[segments == segVal] = 255
    
    #Create Contours
    ret,thresh = cv2.threshold(mask,127,255,0)
    ret,contours,hierarchy = cv2.findContours(mask, 1, 2)
    
    #Centroid
    M = cv2.moments(contours[0])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    
    #Obtain DCT Patch
    if(cX-3<0):
       cX=3
     
    if(cY-3<0):
       cY=3
     
    if(cX+5>960):
       cX=954
    
    if(cY+5>720):
        cY=714
     
    DCT_Patch = (dct(dct(im_g[cY-3:cY+5,cX-3:cX+5], axis=0), axis=1).ravel()).reshape(1,64).ravel()
     
    Centroid_Patch=np.array([cX,cY])
     
    ##HSV
    # Mask only the required locaitons from the image
    spat_cord_seg = np.array(np.where(segments == segVal))
    hsv_values_seg = image_hsv[spat_cord_seg[0,:],spat_cord_seg[1,:]]
    
    # Get Corresponding Class Number 
    spat_cord_seg_GT = gt[spat_cord_seg[0,:],spat_cord_seg[1,:]];
    unique_val_GT, count_seg_GT = np.unique(spat_cord_seg_GT, return_counts = True);
    class_label_seg = unique_val_GT[np.argmax(count_seg_GT)]                                           
    
     # Retrive the HSV values for each pixel postition in the segment
    h = hsv_values_seg[:,0]
    s = hsv_values_seg[:,1]
    v = hsv_values_seg[:,2]
    
    # Create a 2D histogram with num of bins b = 10, for HUE and SATURATION
    H_S_hist, xedges, yedges = np.histogram2d(h, s, bins=b, range = [[0,1],[0,1]])
     
    # OneD histogram with num of bins b = 10
    V_hist, xedges= np.histogram(v, bins = b, range =(0,1))
     
    # Store the feature vectors    
    H_S_val= np.resize(H_S_hist,(1,b*b)).ravel()
    H_S_val= (H_S_val)/(np.sum(H_S_val)) 
    V_val= V_hist
     
    BigX.append(np.hstack((DCT_Patch,Centroid_Patch,H_S_val,V_val)))
    BigY.append(class_label_seg)

 ##########################TESTING######################
BigX1=[]
BigY1=[]

# load the image and convert it to a floating point data type
im = img_as_float(io.imread('img6.PNG'))
gt = np.loadtxt('GT_TXT_Img_6.txt')
# loop over the number of segments
numSegments = 1000

# apply SLIC and extract (approximately) the supplied number
segments = slic(im, n_segments = numSegments, sigma = 2)
end=time.time()
# show the output of SLIC
fig = plt.figure("Superpixels -- %d segments" % (numSegments))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(im, segments))
plt.axis("off")


image_hsv = color.rgb2hsv(im)
im_g=rgb2gray(im)

# Converting useless classes to misc

for i in misc:
    gt[gt==i]=0
            

# To store Class Labels
class_label_seg = np.zeros([1,1])

# Number of bin: H_S
b = 10



 # Loop over each of the segments
for (i, segVal) in enumerate(np.unique(segments)):
    
    ##DCT
    mask = np.zeros(im.shape[:2], dtype = "uint8")
    mask[segments == segVal] = 255
    
    #Create Contours
    ret,thresh = cv2.threshold(mask,127,255,0)
    ret,contours,hierarchy = cv2.findContours(mask, 1, 2)
    
    #Centroid
    M = cv2.moments(contours[0])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    
    #Obtain DCT Patch
    if(cX-3<0):
       cX=3
     
    if(cY-3<0):
       cY=3
     
    if(cX+5>960):
       cX=954
    
    if(cY+5>720):
        cY=714
     
    DCT_Patch = (dct(dct(im_g[cY-3:cY+5,cX-3:cX+5], axis=0), axis=1).ravel()).reshape(1,64).ravel()
     
    Centroid_Patch=np.array([cX,cY])
     
    ##HSV
    # Mask only the required locaitons from the image
    spat_cord_seg = np.array(np.where(segments == segVal))
    hsv_values_seg = image_hsv[spat_cord_seg[0,:],spat_cord_seg[1,:]]
    
    # Get Corresponding Class Number 
    spat_cord_seg_GT = gt[spat_cord_seg[0,:],spat_cord_seg[1,:]];
    unique_val_GT, count_seg_GT = np.unique(spat_cord_seg_GT, return_counts = True);
    class_label_seg = unique_val_GT[np.argmax(count_seg_GT)]                                           
    
     # Retrive the HSV values for each pixel postition in the segment
    h = hsv_values_seg[:,0]
    s = hsv_values_seg[:,1]
    v = hsv_values_seg[:,2]
    
    # Create a 2D histogram with num of bins b = 10, for HUE and SATURATION
    H_S_hist, xedges, yedges = np.histogram2d(h, s, bins=b, range = [[0,1],[0,1]])
     
    # OneD histogram with num of bins b = 10
    V_hist, xedges= np.histogram(v, bins = b, range =(0,1))
     
    # Store the feature vectors    
    H_S_val= np.resize(H_S_hist,(1,b*b)).ravel()
    H_S_val= (H_S_val)/(np.sum(H_S_val)) 
    V_val= V_hist
     
    BigX1.append(np.hstack((DCT_Patch,Centroid_Patch,H_S_val,V_val)))
    BigY1.append(class_label_seg)


BigX2=np.array(BigX)
BigY2=np.array(BigY)
from sklearn.ensemble import GradientBoostingClassifier
model= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
model.fit(BigX2, BigY2)
model.score(BigX2, BigY2)
predicted= model.predict(BigX1)
print(np.sum(predicted!=BigY1))
img=np.zeros([720,960,3])

color={'0':(0,0,0),'1':(255,112,132),'3':(160,160,160),'4':(200,200,200),'6':(124,252,0),'-1':(254,254,254)}
for (i, segVal) in enumerate(np.unique(segments)):
   spat_cord_seg = np.array(np.where(segments == segVal))
   img[spat_cord_seg[0,:],spat_cord_seg[1,:],:]=color[str(int(BigY1[segVal]))]

fig = plt.imshow(img)  
