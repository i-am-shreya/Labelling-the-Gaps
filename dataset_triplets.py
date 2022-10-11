#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:32:48 2019

@author: shreya
"""

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
num = 38
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
            
start_frames_l = np.zeros([count,128,128,3])
end_frames_l = np.zeros([count,128,128,3])
unlabelled_frames = np.zeros([count,128,128,3])
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
                        ul_img = cv2.resize(ul_img,(128,128))
                        start_img = cv2.resize(start_img,(128,128))
                        end_img = cv2.resize(end_img,(128,128))
                    except:
                        continue
                            
                    
#                    plt.imshow(start_img)
#                    plt.imshow(end_img)
#                    plt.imshow(ul_img)
#                    plt.show()
                    
                    start_frames_l[count,:,:,:] = start_img
                    end_frames_l[count,:,:,:] = end_img
                    unlabelled_frames[count,:,:,:] = ul_img 
                    
                    start_gaze_label[count,:] = start_gaze
                    end_gaze_label[count,:] = end_gaze
                    
                    count =count+1
                    
    else:
        continue

save_file = './Dataset/data_triplets/data_triplet_'+ str(num) +'.h5'
hf = h5py.File(save_file, 'w')
hf.create_dataset('start_frames_l', data=start_frames_l)
hf.create_dataset('end_frames_l', data=end_frames_l)
hf.create_dataset('unlabelled_frames', data=unlabelled_frames)
hf.create_dataset('start_gaze_label', data=start_gaze_label)
hf.create_dataset('end_gaze_label', data=end_gaze_label)
hf.close()

f.close()                       