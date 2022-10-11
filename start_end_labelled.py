#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:52:15 2019

@author: shreya
"""

import os
import glob
import numpy as np
import csv
import tensorflow as tf

directory_loc = './Dataset/openface'
label_file = open('./Dataset/start_end_label.txt',"w")
label_file.write('Video,'+'Chunk,'+'file_name,'+'file_label,'+'start/end,\n')
for videofolder in os.listdir(directory_loc):
    print(videofolder)
#    label_file.write(videofolder.split('_')[0] + ',')
    chunk_path=os.path.join(directory_loc, videofolder)
    for chunk in os.listdir(chunk_path):
        print(chunk)
#        label_file.write(chunk.split('_')[1] + ',')
        label_frame_path = os.path.join(chunk_path, chunk,'*.csv')
        frame_list = np.zeros(len(glob.glob(label_frame_path)))
        count = 0
        for file in glob.glob(label_frame_path):
            print(file)
            frame_list[count] = int(file.split('/')[5].split('_')[0]) 
            count = count+1
        
        sorted_list = np.sort(frame_list)
        
        for file in os.listdir(os.path.join(chunk_path, chunk)):
            if file.endswith(".csv") and file.startswith(str(int(sorted_list[0]))):
                print(file)
                with open(os.path.join(chunk_path,chunk,file),'r') as csvfile:
                    csv_reader = csv.DictReader(csvfile)                        
                    for row in csv_reader:
                        dictionary_csv = row
                    items = list(dictionary_csv.items())
                    gaze_0_x = float(items[2][1])
                    gaze_0_y = float(items[3][1])
                    gaze_0_z = float(items[4][1])
                    
                    gaze_1_x = float(items[5][1])
                    gaze_1_y = float(items[6][1])
                    gaze_1_z = float(items[7][1])
                    
                    gaze_angle_x = float(items[8][1])
                    gaze_angle_y = float(items[9][1])
                    
                    avg_gaze_start = tf.constant([(gaze_0_x+gaze_1_x)/2,(gaze_0_y+gaze_1_y)/2,(gaze_0_z+gaze_1_z)/2,3])
                    label_file.write(videofolder.split('_')[0] + ','+ chunk.split('_')[1] + ','+ os.path.join(chunk_path,chunk,file).split('/')[5][:-4] + ',' + str(avg_gaze_start.numpy()[0])+' '+ str(avg_gaze_start.numpy()[1])+' '+ str(avg_gaze_start.numpy()[2])+' '+', start\n')
                    csvfile.close()  
            else:
                continue
        for file in os.listdir(os.path.join(chunk_path, chunk)):
            if file.endswith(".csv") and file.startswith(str(int(sorted_list[len(sorted_list)-1]))):
                print(file)
                with open(os.path.join(chunk_path, chunk,file),'r') as csvfile:
                    csv_reader = csv.DictReader(csvfile)                        
                    for row in csv_reader:
                        dictionary_csv = row
                    items = list(dictionary_csv.items())
                    gaze_0_x = float(items[2][1])
                    gaze_0_y = float(items[3][1])
                    gaze_0_z = float(items[4][1])
                    
                    gaze_1_x = float(items[5][1])
                    gaze_1_y = float(items[6][1])
                    gaze_1_z = float(items[7][1])
                    
                    gaze_angle_x = float(items[8][1])
                    gaze_angle_y = float(items[9][1])
                    avg_gaze_end = tf.constant([(gaze_0_x+gaze_1_x)/2,(gaze_0_y+gaze_1_y)/2,(gaze_0_z+gaze_1_z)/2,3])
                    label_file.write(videofolder.split('_')[0] + ','+ chunk.split('_')[1] + ','+ os.path.join(chunk_path,chunk,file).split('/')[5][:-4] + ',' + str(avg_gaze_start.numpy()[0])+' '+ str(avg_gaze_start.numpy()[1])+' '+ str(avg_gaze_start.numpy()[2])+' '+', end\n')
                    csvfile.close()
            else:
                continue
label_file.close()                       
                                        
                                       
                    
        
 