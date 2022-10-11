# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:30:48 2020

@author: Shreya
"""

import os
import numpy as np
import cv2
import dlib


def face_detection(image):    
    facedetector = dlib.get_frontal_face_detector()
    if image is not None:
        faces, scores, types = facedetector.run(image, 0)
        if len(faces) == 0:
            print("Can not detect face")
            face_img=cv2.resize(image, (224,224))
        elif len(faces)>1:        
            faces0 = [faces[0]]
            face0 = faces0[0]
            faces1 = [faces[1]]
            face1 = faces1[0]
            top0, bottom0, left0, right0 = face0.top(), face0.bottom(), face0.left(), face0.right()
            top1, bottom1, left1, right1 = face1.top(), face1.bottom(), face1.left(), face1.right()
    
            width0 = abs(top0-bottom0)
            width1 = abs(top1-bottom1)
            height0 = abs(right0-left0)
            height1 = abs(right1-left1)
    
            area0=width0*height0
            area1=width1*height1
    
            if area0>area1:
                face = face0
                top = top0
                bottom = bottom0
                left = left0
                right = right0
            else:
                face = face1
                top = top1
                bottom = bottom1
                left = left1
                right = right1
        else:
            faces = [faces[0]]
            face = faces[0]
            top, bottom, left, right = face.top(), face.bottom(), face.left(), face.right()
        top, bottom, left, right = face.top(), face.bottom(), face.left(), face.right()
        face_img=image[top:bottom, left:right]
        face_img=cv2.resize(face_img, (224,224))
    else:
        face_img =np.zeros([224,224,3])    
    return face_img


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

count = 3024
a=0
start_frames_l = np.zeros([count,4096])
end_frames_l = np.zeros([count,4096])
unlabelled_frames = np.zeros([count,4096])
start_gaze_label = np.zeros([count,2])
original_unlabel = np.zeros([count,2])
end_gaze_label = np.zeros([count,2])     

count=0
for dirs in os.listdir('./Columbia Gaze Data Set'):
    print(dirs)
    if dirs != '.DS_Store':
        id_path = './Columbia Gaze Data Set/'+dirs
        for filename in os.listdir(id_path):
            if filename.endswith('.jpg'):
                print(count)
                name_split = filename.split('_')
                p = name_split[2][0:name_split[2].find('P')]
                h = name_split[4][0:name_split[4].find('H')]
                v = name_split[3][0:name_split[3].find('V')] 
    # =============================================================================
    # horizontal trajectory                       
    # =============================================================================
                if p=='0' and v == '-10'and h!= '10' and h != '15':
                    img_path_l = id_path + '/' + filename
                    start_image =cv2.imread(img_path_l)
                    face_start_img = face_detection(start_image)
                    face_start_img = np.reshape(face_start_img,(1,224,224,3))
                    pred_start_img = vgg_model_new.predict(face_start_img)
                    
                    
                    filename_ul = name_split[0] + '_' + name_split[1]+ '_' + '0P_-10V_' + str(int(h)+5) + 'H.jpg'
                    img_path_ul = id_path + '/' + filename_ul
                    ul_image =cv2.imread(img_path_ul)
                    face_ul_img = face_detection(ul_image)
                    face_ul_img = np.reshape(face_ul_img,(1,224,224,3))
                    pred_ul_img = vgg_model_new.predict(face_ul_img)
                    
                                   
                    
                    filename_ldash = name_split[0] + '_' + name_split[1]+ '_' + '0P_-10V_' + str(int(h)+10) + 'H.jpg'
                    img_path_ldash = id_path + '/' + filename_ldash
                    end_image =cv2.imread(img_path_ldash)
                    face_end_img = face_detection(end_image)
                    face_end_img = np.reshape(face_end_img,(1,224,224,3))
                    pred_end_img = vgg_model_new.predict(face_end_img)
                    
                    start_frames_l[count,:] = pred_start_img
                    end_frames_l[count,:] = pred_end_img
                    unlabelled_frames[count,:] = pred_ul_img 
                    
                    start_gaze_label[count,:] = [int(h),int(v)]
                    original_unlabel[count,:] = [int(h)+5,int(v)]
                    end_gaze_label[count,:] = [int(h)+10,int(v)]
                    count=count+1
                    # Reverse direction                    
                    start_frames_l[count,:] = pred_end_img
                    end_frames_l[count,:] = pred_start_img
                    unlabelled_frames[count,:] = pred_ul_img 
                    
                    start_gaze_label[count,:] = [int(h)+10,int(v)]
                    original_unlabel[count,:] = [int(h)+5,int(v)]
                    end_gaze_label[count,:] = [int(h),int(v)]
                        
                    count=count+1
                    
                elif p=='0' and v == '0'and h!= '10' and h != '15':
                    img_path_l = id_path + '/' + filename
                    start_image =cv2.imread(img_path_l)
                    face_start_img = face_detection(start_image)
                    face_start_img = np.reshape(face_start_img,(1,224,224,3))
                    pred_start_img = vgg_model_new.predict(face_start_img)
                    
                    
                    filename_ul = name_split[0] + '_' + name_split[1]+ '_' + '0P_-10V_' + str(int(h)+5) + 'H.jpg'
                    img_path_ul = id_path + '/' + filename_ul
                    ul_image =cv2.imread(img_path_ul)
                    face_ul_img = face_detection(ul_image)
                    face_ul_img = np.reshape(face_ul_img,(1,224,224,3))
                    pred_ul_img = vgg_model_new.predict(face_ul_img)
                    
                                   
                    
                    filename_ldash = name_split[0] + '_' + name_split[1]+ '_' + '0P_-10V_' + str(int(h)+10) + 'H.jpg'
                    img_path_ldash = id_path + '/' + filename_ldash
                    end_image =cv2.imread(img_path_ldash)
                    face_end_img = face_detection(end_image)
                    face_end_img = np.reshape(face_end_img,(1,224,224,3))
                    pred_end_img = vgg_model_new.predict(face_end_img)
                    
                    start_frames_l[count,:] = pred_start_img
                    end_frames_l[count,:] = pred_end_img
                    unlabelled_frames[count,:] = pred_ul_img 
                    
                    start_gaze_label[count,:] = [int(h),int(v)]
                    original_unlabel[count,:] = [int(h)+5,int(v)]
                    end_gaze_label[count,:] = [int(h)+10,int(v)] 
    # Reverse direction
                    count=count+1
                    start_frames_l[count,:] = pred_end_img
                    end_frames_l[count,:] = pred_start_img
                    unlabelled_frames[count,:] = pred_ul_img 
                    
                    start_gaze_label[count,:] = [int(h)+10,int(v)] 
                    original_unlabel[count,:] = [int(h)+5,int(v)]
                    end_gaze_label[count,:] =   [int(h),int(v)]            
                    count=count+1
              
                elif p=='0' and v == '10'and h!= '10' and h != '15':
                    img_path_l = id_path + '/' + filename
                    start_image =cv2.imread(img_path_l)
                    face_start_img = face_detection(start_image)
                    face_start_img = np.reshape(face_start_img,(1,224,224,3))
                    pred_start_img = vgg_model_new.predict(face_start_img)
                    
                    
                    filename_ul = name_split[0] + '_' + name_split[1]+ '_' + '0P_-10V_' + str(int(h)+5) + 'H.jpg'
                    img_path_ul = id_path + '/' + filename_ul
                    ul_image =cv2.imread(img_path_ul)
                    face_ul_img = face_detection(ul_image)
                    face_ul_img = np.reshape(face_ul_img,(1,224,224,3))
                    pred_ul_img = vgg_model_new.predict(face_ul_img)
                    
                                   
                    
                    filename_ldash = name_split[0] + '_' + name_split[1]+ '_' + '0P_-10V_' + str(int(h)+10) + 'H.jpg'
                    img_path_ldash = id_path + '/' + filename_ldash
                    end_image =cv2.imread(img_path_ldash)
                    face_end_img = face_detection(end_image)
                    face_end_img = np.reshape(face_end_img,(1,224,224,3))
                    pred_end_img = vgg_model_new.predict(face_end_img)
                    
                    start_frames_l[count,:] = pred_start_img
                    end_frames_l[count,:] = pred_end_img
                    unlabelled_frames[count,:] = pred_ul_img 
                    
                    start_gaze_label[count,:] = [int(h),int(v)]
                    original_unlabel[count,:] = [int(h)+5,int(v)]
                    end_gaze_label[count,:] = [int(h)+10,int(v)]                    
                    count=count+1
    # Reverse direction
                    count=count+1
                    start_frames_l[count,:] = pred_end_img
                    end_frames_l[count,:] = pred_start_img
                    unlabelled_frames[count,:] = pred_ul_img 
                    
                    start_gaze_label[count,:] = [int(h)+10,int(v)]
                    original_unlabel[count,:] = [int(h)+5,int(v)]
                    end_gaze_label[count,:] = [int(h),int(v)]

for dirs in os.listdir('./Columbia Gaze Data Set'):
    print(dirs)
    if dirs != '.DS_Store':
        id_path = './Columbia Gaze Data Set/'+dirs
        for filename in os.listdir(id_path):
            if filename.endswith('.jpg'):
                print(count)
                name_split = filename.split('_')
                p = name_split[2][0:name_split[2].find('P')]
                h = name_split[4][0:name_split[4].find('H')]
                v = name_split[3][0:name_split[3].find('V')] 
    # =============================================================================
    # vertical trajectory                       
    # =============================================================================
                if p=='0' and v == '-10':
                    img_path_l = id_path + '/' + filename
                    start_image =cv2.imread(img_path_l)
                    face_start_img = face_detection(start_image)
                    face_start_img = np.reshape(face_start_img,(1,224,224,3))
                    pred_start_img = vgg_model_new.predict(face_start_img)
                    
                    
                    filename_ul = name_split[0] + '_' + name_split[1]+ '_' + '0P_'+ str(int(v)+10) +' V_' + str(h) + 'H.jpg'
                    img_path_ul = id_path + '/' + filename_ul
                    ul_image =cv2.imread(img_path_ul)
                    face_ul_img = face_detection(ul_image)
                    face_ul_img = np.reshape(face_ul_img,(1,224,224,3))
                    pred_ul_img = vgg_model_new.predict(face_ul_img)
                    
                                   
                    
                    filename_ldash = name_split[0] + '_' + name_split[1]+ '_' + '0P_'+ str(int(v)+20) +' V_' + str(h) + 'H.jpg'
                    img_path_ldash = id_path + '/' + filename_ldash
                    end_image =cv2.imread(img_path_ldash)
                    face_end_img = face_detection(end_image)
                    face_end_img = np.reshape(face_end_img,(1,224,224,3))
                    pred_end_img = vgg_model_new.predict(face_end_img)
                    
                    start_frames_l[count,:] = pred_start_img
                    end_frames_l[count,:] = pred_end_img
                    unlabelled_frames[count,:] = pred_ul_img 
                    
                    start_gaze_label[count,:] = [int(h),int(v)]
                    original_unlabel[count,:] = [int(h),int(v)+10]
                    end_gaze_label[count,:] = [int(h),int(v)+20]
    # Reverse direction
                    count=count+1
                    start_frames_l[count,:] = pred_end_img
                    end_frames_l[count,:] = pred_start_img
                    unlabelled_frames[count,:] = pred_ul_img 
                    
                    start_gaze_label[count,:] = [int(h),int(v)+20]
                    original_unlabel[count,:] = [int(h),int(v)+10]
                    end_gaze_label[count,:] =  [int(h),int(v)]                       
                    count=count+1
                    
for dirs in os.listdir('./Columbia Gaze Data Set'):
    print(dirs)
    if dirs != '.DS_Store':
        id_path = './Columbia Gaze Data Set/'+dirs
        for filename in os.listdir(id_path):
            if filename.endswith('.jpg'):
                print(count)
                name_split = filename.split('_')
                p = name_split[2][0:name_split[2].find('P')]
                h = name_split[4][0:name_split[4].find('H')]
                v = name_split[3][0:name_split[3].find('V')] 
    # =============================================================================
    # diagonal trajectory                       
    # =============================================================================
                if p=='0' and v == '-10'and h!= '10' and h != '15':
                    img_path_l = id_path + '/' + filename
                    start_image =cv2.imread(img_path_l)
                    face_start_img = face_detection(start_image)
                    face_start_img = np.reshape(face_start_img,(1,224,224,3))
                    pred_start_img = vgg_model_new.predict(face_start_img)
                    
                    
                    filename_ul = name_split[0] + '_' + name_split[1]+ '_' + '0P_'+ str(int(v)+10) +' V_' + str(int(h)+5) + 'H.jpg'
                    img_path_ul = id_path + '/' + filename_ul
                    ul_image =cv2.imread(img_path_ul)
                    face_ul_img = face_detection(ul_image)
                    face_ul_img = np.reshape(face_ul_img,(1,224,224,3))
                    pred_ul_img = vgg_model_new.predict(face_ul_img)
                    
                                   
                    
                    filename_ldash = name_split[0] + '_' + name_split[1]+ '_' + '0P_'+ str(int(v)+20) +' V_' + str(int(h)+10) + 'H.jpg'
                    img_path_ldash = id_path + '/' + filename_ldash
                    end_image =cv2.imread(img_path_ldash)
                    face_end_img = face_detection(end_image)
                    face_end_img = np.reshape(face_end_img,(1,224,224,3))
                    pred_end_img = vgg_model_new.predict(face_end_img)
                    
                    start_frames_l[count,:] = pred_start_img
                    end_frames_l[count,:] = pred_end_img
                    unlabelled_frames[count,:] = pred_ul_img 
                    
                    start_gaze_label[count,:] = [int(h),int(v)]
                    original_unlabel[count,:] = [int(h)+5,int(v)+10]
                    end_gaze_label[count,:] = [int(h)+10,int(v)+20]
                        
                    # Reverse direction
                    count=count+1
                    start_frames_l[count,:] = pred_start_img
                    end_frames_l[count,:] = pred_end_img
                    unlabelled_frames[count,:] = pred_ul_img 
                    
                    start_gaze_label[count,:] = [int(h)+10,int(v)+20]
                    original_unlabel[count,:] = [int(h)+5,int(v)+10]
                    end_gaze_label[count,:] = [int(h),int(v)]
                        
                    count=count+1               
import h5py   
save_file = 'data_triplet_vgg-fc6_cave.h5'
hf = h5py.File(save_file, 'w')
hf.create_dataset('start_frames_l', data=start_frames_l)
hf.create_dataset('end_frames_l', data=end_frames_l)
hf.create_dataset('unlabelled_frames', data=unlabelled_frames)
hf.create_dataset('start_gaze_label', data=start_gaze_label)
hf.create_dataset('original_unlabel', data=original_unlabel)
hf.create_dataset('end_gaze_label', data=end_gaze_label)
hf.close()