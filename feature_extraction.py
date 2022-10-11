# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 10:35:59 2020

@author: shreya
"""

import keras
from keras_vggface.vggface import VGGFace
from keras.models import Model
import cv2
import numpy as np
import os
# Based on VGG16 architecture -> old paper(2015)
vggface = VGGFace(model='vgg16') # or VGGFace() as default

## Based on RESNET50 architecture -> new paper(2017)
#vggface = VGGFace(model='resnet50')
#
## Based on SENET50 architecture -> new paper(2017)
#vggface = VGGFace(model='senet50')

# Layer Features
layer_name = 'fc6' # edit this line
vgg_model = VGGFace() # pooling: None, avg or max
out = vgg_model.get_layer(layer_name).output
vgg_model_new = Model(vgg_model.input, out)

# =============================================================================
# feature extract
# =============================================================================

import os
import glob
import numpy as np
import csv
import tensorflow as tf
import cv2
import h5py
import matplotlib.pyplot as plt

directory_loc = './Dataset/chunks'
f = open('./Dataset/start_end_label.txt',"r")
label_file = f.readlines()
labels = label_file[1:]
num = 15
count = 0
for i in range(len(labels)):
    
    if i%2 == 0:        
        line = labels[i]
        video_no = line.split(',')[0]
        chunk_no = 'chunk_'+ line.split(',')[1]
        if line.split(',')[4].strip() == 'start':
            start_file = line.split(',')[2]+'png'
            start_gaze = [float(line.split(',')[3].split(' ')[0]),float(line.split(',')[3].split(' ')[1]), float(line.split(',')[3].split(' ')[2])]
        
        line = labels[i+1]        
        if line.split(',')[4].strip() == 'end':
            end_file = line.split(',')[2]+'png'
            end_gaze = [float(line.split(',')[3].split(' ')[0]),float(line.split(',')[3].split(' ')[1]), float(line.split(',')[3].split(' ')[2])]
        s = str(num)
        if video_no == s:
            search_path = directory_loc + '/' +  video_no +'_data'  + '/' +  chunk_no
            for file in os.listdir(search_path): #os.path.join(directory_loc, video_no, chunk_no)
                if file != start_file and file != end_file:
                    count = count +1
            
start_frames_l = np.zeros([count,4096])
end_frames_l = np.zeros([count,4096])
unlabelled_frames = np.zeros([count,4096])
start_gaze_label = np.zeros([count,3])
end_gaze_label = np.zeros([count,3]) 
count = 0
           
for i in range(len(labels)):
    print(i)
    if i%2 == 0:
        
        line = labels[i]
        video_no = line.split(',')[0]
        chunk_no = 'chunk_'+ line.split(',')[1]
        if line.split(',')[4].strip() == 'start':
            start_file = line.split(',')[2]+'.png'
            start_gaze = [float(line.split(',')[3].split(' ')[0]),float(line.split(',')[3].split(' ')[1]), float(line.split(',')[3].split(' ')[2])]
        
        line = labels[i+1]        
        if line.split(',')[4].strip() == 'end':
            end_file = line.split(',')[2]+'.png'
            end_gaze = [float(line.split(',')[3].split(' ')[0]),float(line.split(',')[3].split(' ')[1]), float(line.split(',')[3].split(' ')[2])]
        s = str(num)
        if video_no == s:
            search_path = directory_loc + '/' +  video_no +'_data'  + '/' +  chunk_no
            for file in os.listdir(search_path):
                if file != start_file and file != end_file:
                                        
                    ul_img = cv2.imread(search_path+'/'+file)                    
                    start_img = cv2.imread(search_path+'/'+start_file)                    
                    end_img = cv2.imread(search_path+'/'+end_file)
                    try:
                        ul_img = cv2.resize(ul_img,(224,224))
                        ul_img = np.reshape(ul_img,(1,224,224,3))
                        pred_ul_img = vgg_model_new.predict(ul_img)
                        
                        start_img = cv2.resize(start_img,(224,224))
                        start_img = np.reshape(start_img,(1,224,224,3))
                        pred_start_img = vgg_model_new.predict(start_img)
                        
                        end_img = cv2.resize(end_img,(224,224))
                        end_img = np.reshape(end_img,(1,224,224,3))
                        pred_end_img = vgg_model_new.predict(end_img)
                    except:
                        continue
                            
                    
#                    plt.imshow(start_img)
#                    plt.imshow(end_img)
#                    plt.imshow(ul_img)
#                    plt.show()
                    
                    start_frames_l[count,:] = pred_start_img
                    end_frames_l[count,:] = pred_end_img
                    unlabelled_frames[count,:] = pred_ul_img 
                    
                    start_gaze_label[count,:] = start_gaze
                    end_gaze_label[count,:] = end_gaze
                    
                    count =count+1
                    
    else:
        continue

save_file = './Dataset/data_triplets/data_triplet_vgg-fc6'+ str(num) +'.h5'
hf = h5py.File(save_file, 'w')
hf.create_dataset('start_frames_l', data=start_frames_l)
hf.create_dataset('end_frames_l', data=end_frames_l)
hf.create_dataset('unlabelled_frames', data=unlabelled_frames)
hf.create_dataset('start_gaze_label', data=start_gaze_label)
hf.create_dataset('end_gaze_label', data=end_gaze_label)
hf.close()

f.close()                       