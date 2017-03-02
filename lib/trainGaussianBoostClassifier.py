import os
import time
import numpy as np

############################################# PARAMETER DEFINITION #####################################################


batch_start = 0
batch_end = 9


rgb_dir = '..\dataset\SYNTHIA_RAND_CVPR16\RGB\\'    # Location of folder containing the RGB images of the dataset
#SLIC_dir = '..\dataset\SYNTHIA_RAND_CVPR16\SLIC\\'
#gt_dir = '..\dataset\SYNTHIA_RAND_CVPR16\GTTXT\\'
feat_dir = '..\dataset\SYNTHIA_RAND_CVPR16\FEAT\\'
label_dir = '..\dataset\SYNTHIA_RAND_CVPR16\LABEL\\'

misc = [2,5,7,8,9,11]        # Defining all original classes which will be labelled miscellaneous

########################################################################################################################

list_files_RGB = os.listdir(rgb_dir)
list_files_RGB.sort()

#np.random.seed(0)
test_feat_array=np.load(feat_dir + list_files_RGB[0].rsplit(".",1)[0] + '.npy')
num_feat = test_feat_array.shape[1]
BigX = np.empty([0,num_feat])
BigY = np.empty([0])

for im_no in range(batch_start, batch_end+1):
    feat_path = feat_dir + list_files_RGB[im_no].rsplit(".",1)[0] + '.npy'
    label_path = label_dir + list_files_RGB[im_no].rsplit(".",1)[0] + '.npy'
    X = np.load(feat_path)
    Y = np.load(label_path)
    BigX = np.vstack((BigX,X))
    BigY = np.concatenate((BigY,Y))
print(np.unique(BigY))

for i in misc:
    BigY[BigY == i] = 0
print("Loaded Data Successfully. Beginning Training Now")
##################################################################################################
###MAKE AND SAVE MODEL
for i in misc:
        BigY[BigY == i]=0

from sklearn.ensemble import GradientBoostingClassifier
import pickle
model= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
start_train_time = time.time()
model.fit(BigX, BigY)
end_train_time = time.time()
print("Time taken to train model:{}".format(end_train_time-start_train_time))
model.score(BigX, BigY)
filename='..\dataset\SYNTHIA_RAND_CVPR16\MODELS\\GaussianBoostClassifier_Model.sav'
pickle.dump(model,open(filename,'wb'))
print("Model Saved Successfully")
