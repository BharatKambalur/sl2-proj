
import os
import time
import numpy as np
import cv2
import numpy as np
from skimage import io
from skimage.filters import gabor_kernel
from scipy.fftpack import dct
from scipy.signal import fftconvolve
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

##!!!!!!!!!!!!!!!CHANGE MODEL AS DESIRED
from sklearn.ensemble import RandomForestClassifier
model_name='RandomForestClassifier_Model_Cityscape.sav'

############################################# PARAMETER DEFINITION #####################################################

batch_start = 0
batch_end = 699

error_dir = '..\dataset\CITYSCAPE\ERROR\\'
rgb_dir = '..\dataset\CITYSCAPE\RGB\\' # Location of folder containing the RGB images of the dataset
SLIC_dir = '..\dataset\CITYSCAPE\SLIC\\'
gt_dir = '..\dataset\CITYSCAPE\GT\\'
feat_dir = '..\dataset\CITYSCAPE\FEAT\\'
label_dir = '..\dataset\CITYSCAPE\LABEL\\'
model_dir= '..\dataset\CITYSCAPE\MODELS\\'
predicted_dir= '..\dataset\CITYSCAPE\PREDICTED\\'

misc = [1,2,3,4,5,6,9,10,11,12,13,14,15,16,17,18,19,20,22,24,25,26,27,28,29,30,31,32,33,-1]        # Defining all original classes which will be labelled miscellaneous
orgi = [0,7,8,21,23]
Overall_Error=0
color={'0':(0,0,0),'23':(132,112,255),'7':(160,160,160),'8':(218,165,32),'21':(0,128,0)}
########################################################################################################################

list_files_RGB = os.listdir(rgb_dir)
list_files_RGB.sort()

list_files_GT = os.listdir(gt_dir)
list_files_GT.sort()


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
    print(im_no)

print("Loaded Data Successfully. Beginning Training Now")
##################################################################################################
###MAKE AND SAVE MODEL
for i in misc:
        BigY[BigY == i]=0



model= RandomForestClassifier()
start_train_time = time.time()
model.fit(BigX, BigY)
end_train_time = time.time()
print("Time taken to train model:{}".format(end_train_time-start_train_time))
model.score(BigX, BigY)
filename=model_dir+model_name
pickle.dump(model,open(filename,'wb'))
print("Model Saved Successfully")


#######################################################################################################
####TEST
print("Beginning Testing Now")

list_files_Feat = os.listdir(feat_dir)
list_files_Feat.sort()   

list_files_GT= os.listdir(gt_dir)
list_files_GT.sort()

list_files_Label= os.listdir(label_dir)
list_files_Label.sort()

list_files_SLIC = os.listdir(SLIC_dir)
list_files_SLIC.sort()

list_files_RGB = os.listdir(rgb_dir)
list_files_RGB.sort()

#######

###LOAD MODEL
batch_start = 700
batch_end = 900

model_file=filename
load_model=pickle.load(open(model_file,'rb'))

################
img=np.zeros([1024,2048,3],dtype=np.uint8)

confusion = np.zeros([len(orgi),len(orgi)])
temp_confusion = np.zeros([len(orgi),len(orgi)])
for im_no in range(batch_start, batch_end+1):
    ##NOTE!!! FEATURE NUMBERING STARTS FROM 0: SUBTRACT OUT EXTRA
    gt_path = gt_dir + list_files_GT[im_no].rsplit(".",1)[0] + '.png'
    feat_path = feat_dir + list_files_RGB[im_no].rsplit(".",1)[0] + '.npy'
    label_path = label_dir + list_files_RGB[im_no].rsplit(".",1)[0] + '.npy'
    slic_path = SLIC_dir + list_files_RGB[im_no].rsplit(".",1)[0] + '.npy'
    
    TestX = np.load(feat_path)
    TestY = np.load(label_path)
    
    gt = io.imread(gt_path)    
    for i in misc:
        gt[gt == i] = 0

    segments = np.load(slic_path)
    
    for i in misc:
        gt[gt == i] = 0
        TestY[TestY == i]=0

    PredictY= load_model.predict(TestX)
    
    img_Predict=np.zeros([1024,2048])
    

    for (i, segVal) in enumerate(np.unique(segments)):
        spat_cord_seg = np.array(np.where(segments == segVal))
        img_Predict[spat_cord_seg[0,:],spat_cord_seg[1,:]]=PredictY[segVal]
        img[spat_cord_seg[0,:],spat_cord_seg[1,:],:]=color[str(int(PredictY[segVal]))]
    
    predicted_path = predicted_dir  +  list_files_RGB[im_no].rsplit(".",1)[0] +'_predicted_output' + '.npy'    
    np.save(predicted_path , PredictY)            
                    
    Overall_Error += (np.sum(img_Predict!=gt))/(1024*2048)
    
    for ind_gt in range(0,len(orgi)):
            spat_cord_class = np.array(np.where(gt == orgi[ind_gt]))
            for ind_pred in range(0,len(orgi)):
                temp_confusion[ind_gt,ind_pred] = np.sum(img_Predict[spat_cord_class[0,:],spat_cord_class[1,:]] == orgi[ind_pred])
                                                                   
    confusion += normalize(temp_confusion, axis=1, norm='l1')          

confusion = confusion/(batch_end-batch_start+1)
Overall_Error=Overall_Error/(batch_end-batch_start+1)
print(Overall_Error)

error_path = error_dir  +  model_name.rsplit(".",1)[0] +'_overall_error' + '.npy'
confusion_path = error_dir  +  model_name.rsplit(".",1)[0] +'_confusion_error' + '.npy'

np.save(error_path , Overall_Error) 
np.save(confusion_path , confusion) 


#plt.imshow(confusion)