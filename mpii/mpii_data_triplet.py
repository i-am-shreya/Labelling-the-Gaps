#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 09:40:47 2020

@author: shreya
"""

import numpy as np
import cv2
import math
import os

# =============================================================================
# vgg feature
# =============================================================================

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



count = 32751
start_frames_l = np.zeros([count,4096])
end_frames_l = np.zeros([count,4096])
unlabelled_frames = np.zeros([count,4096])
start_gaze_label = np.zeros([count,1])
original_unlabel = np.zeros([count,1])
end_gaze_label = np.zeros([count,1]) 

count=0

root_path ='./data'
for dirs in os.listdir(root_path):
    sub_path = root_path + '/' + dirs
    for subdirs in os.listdir(sub_path):
        if subdirs.endswith('.txt'):
            txtfile_path = sub_path + '/' + subdirs
            print (subdirs)
            f = open(txtfile_path, "r")
            lines =f.readlines()            
            for i in range(len(lines)):                
                if i < len(lines)-3:
                    print(subdirs + ',' +str(i)+ ',' +str(count))
                    # start image
                    line = lines[i]
                    p_details = line.split(' ')
                    
                    face_start_img = cv2.imread(sub_path + '/' + p_details[0])
                    try:
                        face_start_img = cv2.resize(face_start_img, (224,224))
                        face_start_img = np.reshape(face_start_img,(1,224,224,3))
                        pred_start_img = vgg_model_new.predict(face_start_img)
                    except:
                        continue
                    y_gt_start_img = [float(i) for i in p_details[21:24]]
                    y_fc_start_img = [float(i) for i in p_details[24:27]]
                    y_start_img = math.acos(np.dot(y_gt_start_img, y_fc_start_img) / (np.linalg.norm(y_gt_start_img) * np.linalg.norm(y_fc_start_img)))
                    
                #        plt.imshow(image)
                #        plt.show()        
                          
                    
                   
                                        
                    #unlabelled image
                    line = lines[i+1]
                    p_details = line.split(' ')
                   
                    face_ul_img = cv2.imread(sub_path + '/' + p_details[0])
                    try:
                        face_ul_img = cv2.resize(face_ul_img, (224,224))
                        face_ul_img = np.reshape(face_ul_img,(1,224,224,3))
                        pred_ul_img = vgg_model_new.predict(face_ul_img)
                    except:
                        continue
                    y_gt_ul_img = [float(i) for i in p_details[21:24]]
                    y_fc_ul_img = [float(i) for i in p_details[24:27]]
                    y_ul_img = math.acos(np.dot(y_gt_ul_img, y_fc_ul_img) / (np.linalg.norm(y_gt_ul_img) * np.linalg.norm(y_fc_ul_img)))
                    
                        
                #        plt.imshow(image)
                #        plt.show()        
                          
                   
                   
                    #end image
                    line = lines[i+2]
                    p_details = line.split(' ')
                    
                    face_end_img = cv2.imread(sub_path + '/' + p_details[0])
                    try:
                        face_end_img = cv2.resize(face_end_img, (224,224))
                        face_end_img = np.reshape(face_end_img,(1,224,224,3))
                        pred_end_img = vgg_model_new.predict(face_end_img)
                    except:
                        continue
                    y_gt_end_img = [float(i) for i in p_details[21:24]]
                    y_fc_end_img = [float(i) for i in p_details[24:27]]
                    y_end_img = math.acos(np.dot(y_gt_end_img, y_fc_end_img) / (np.linalg.norm(y_gt_end_img) * np.linalg.norm(y_fc_end_img)))
                    
                        
                #        plt.imshow(image)
                #        plt.show()        
                          
                   
#                    
                    start_frames_l[count,:] = pred_start_img
                    end_frames_l[count,:] = pred_end_img
                    unlabelled_frames[count,:] = pred_ul_img 
                    
                    start_gaze_label[count,:] = y_start_img
                    original_unlabel[count,:] = y_ul_img
                    end_gaze_label[count,:] = y_end_img
                    count=count+1
#                    count_person = count_person+1
            f.close()

import h5py   
save_file = 'data_triplet_vgg-fc6_mpii.h5'
hf = h5py.File(save_file, 'w')
hf.create_dataset('start_frames_l', data=start_frames_l)
hf.create_dataset('end_frames_l', data=end_frames_l)
hf.create_dataset('unlabelled_frames', data=unlabelled_frames)
hf.create_dataset('start_gaze_label', data=start_gaze_label)
hf.create_dataset('original_unlabel', data=original_unlabel)
hf.create_dataset('end_gaze_label', data=end_gaze_label)
hf.close()

