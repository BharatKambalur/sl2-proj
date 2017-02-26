import os
import time
import cv2
import numpy as np
from skimage import color, io
from skimage.filters import gabor_kernel
from scipy.fftpack import dct
from scipy.signal import fftconvolve

############################################# PARAMETER DEFINITION #####################################################


batch_start = 11700
batch_end = 11720


#rgb_dir = '..\dataset\SYNTHIA_RAND_CVPR16\RGB\\'    # Location of folder containing the RGB images of the dataset
#SLIC_dir = '..\dataset\SYNTHIA_RAND_CVPR16\SLIC\\'
#gt_dir = '..\dataset\SYNTHIA_RAND_CVPR16\GTTXT\\'
feat_dir = '..\dataset\SYNTHIA_RAND_CVPR16\FEAT\\'
label_dir = '..\dataset\SYNTHIA_RAND_CVPR16\LABEL\\'

misc = [2,5,7,8,9,11]        # Defining all original classes which will be labelled miscellaneous

########################################################################################################################

list_files_Feat = os.listdir(feat_dir)
list_files_Feat.sort()

test_feat_array=np.load(feat_dir + list_files_Feat[0])
num_feat = test_feat_array.shape[1]
BigX = np.empty([0,num_feat])
BigY = np.empty([0])

for im_no in range(batch_start, batch_end+1):
    feat_path = feat_dir + list_files_Feat[im_no-11000]
    label_path = label_dir + list_files_Feat[im_no-11000].split(".",1)[0]+".npy"
    X = np.load(feat_path)
    Y = np.load(label_path)
    BigX = np.vstack((BigX,X))
    BigY = np.concatenate((BigY,Y))
print(np.unique(BigY))

for i in misc:
    BigY[BigY == i] = 0

##################################################################################################
###MAKE AND SAVE MODEL
for i in misc:
        BigY[BigY == i]=0

from sklearn.ensemble import GradientBoostingClassifier
import pickle
model= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
model.fit(BigX, BigY)
model.score(BigX, BigY)
filename='..\dataset\SYNTHIA_RAND_CVPR16\MODEL\\trialmodel.sav'
pickle.dump(model,open(filename,'wb'))


