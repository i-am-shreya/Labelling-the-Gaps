# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:58:39 2020

@author: shreya
"""
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding,Dropout
from keras.preprocessing import sequence
from keras.utils import to_categorical
import scipy.io as sio
import numpy as np
import os
import keras
from numpy import array
np.random.seed(1337)  # for reproducibility
from keras.models import Model
from keras.layers import Input
import keras.backend as K
import h5py
from keras import regularizers


def ordering_loss(l,l_dash,ul,alpha):
    return K.maximum(ul-K.sum((alpha*l),((1-alpha)*l_dash)),0)

def triplet_loss(y_true, y_pred):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    print('y_pred.shape = ',y_pred)
    
    total_length = y_pred.shape.as_list()[-1]-1
    print('total_length=',  total_length)
#     total_lenght =12
    
    anchor = y_pred[:,0:int(total_length*1/3)-1]
#    print(int(total_length*1/3-1))
    positive = y_pred[:,int(total_length*1/3):int(total_length*2/3)-1]
    negative = y_pred[:,int(total_length*2/3):int(total_length*3/3)-1]
    alpha = y_pred[:,total_length]
#    print(alpha)
    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor-positive),axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative),axis=1)

    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    loss = K.maximum(basic_loss,0.0)
 
    return loss

#def build_model():
#    input_dim = 4096
#    inputs_label = Input(shape=(input_dim,))
#    inputs_unlabel = Input(shape=(input_dim,))
#    inputs_label_dash = Input(shape=(input_dim,))
#        
#    x_inputs_label = Dense(512)(inputs_label)#(attention_mul)
#    x_inputs_unlabel = Dense(512)(inputs_unlabel)
#    x_inputs_label_dash = Dense(512)(inputs_label_dash)
#    
#    motion_1 = keras.layers.Subtract()([x_inputs_label, x_inputs_unlabel])
#    motion_2 = keras.layers.Subtract()([x_inputs_label, x_inputs_unlabel])
#    x = keras.layers.concatenate([x_inputs_unlabel,motion_1,motion_2], axis=-1)
##    # ATTENTION PART STARTS HERE
##    attention_probs = Dense(input_dim, activation='softmax', name='attention_vec')(inputs)
##    attention_mul = keras.layers.Multiply()([inputs, attention_probs])
##    # ATTENTION PART FINISHES HERE
##
##    attention_mul = Dense(64)(attention_mul)
#    x = Dense(512)(x_inputs_label)#(attention_mul)
#    x = Dense(512)(x)
#    x = Dense(512)(x_inputs_label_dash)
#    x = Dropout(0.4)(x)
#    x = Dense(1024)(x)    
#    x_embedding = Dense(4096,activation='sigmoid',name = 'x_embedding')(x)
#    
#    output_label = Dense(2, activation='sigmoid',name = 'Output_label')(x)
#    output_predicted_label = Dense(2, activation='sigmoid',name = 'output_predicted_label')(x)
#    output_margin = Dense(1, activation='sigmoid')(x)
#    merged_vector = keras.layers.concatenate([output_label, output_predicted_label,output_margin], axis=-1, name='merged_layer')
#    
#    model = Model(input=[inputs_label,inputs_unlabel,inputs_label_dash], output=[x_embedding, output_label,merged_vector])
#    model.summary()
#
#    losses ={'Output_label':'mean_squared_error',
#              'x_embedding': 'cosine_proximity',
#              'merged_layer':triplet_loss }
#    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#    
#   
#    model.compile(
#    optimizer=sgd,
#    loss=losses,
#    metrics=['acc'])
#    
#    return model

def build_model():
    input_dim = 4096
    inputs_label = Input(shape=(input_dim,))
    inputs_unlabel = Input(shape=(input_dim,))
    inputs_label_dash = Input(shape=(input_dim,))
        
    x_inputs_label = Dense(512)(inputs_label)#(attention_mul)
    x_inputs_unlabel = Dense(512)(inputs_unlabel)
    x_inputs_label_dash = Dense(512)(inputs_label_dash)
    
    motion_1 = keras.layers.Subtract()([x_inputs_label, x_inputs_unlabel])
    motion_2 = keras.layers.Subtract()([x_inputs_label_dash, x_inputs_unlabel])
    x = keras.layers.concatenate([x_inputs_unlabel,motion_1,motion_2], axis=-1)
#    # ATTENTION PART STARTS HERE
#    attention_probs = Dense(input_dim, activation='softmax', name='attention_vec')(inputs)
#    attention_mul = keras.layers.Multiply()([inputs, attention_probs])
#    # ATTENTION PART FINISHES HERE
#
#    attention_mul = Dense(64)(attention_mul)
    x = Dense(512)(x_inputs_label)#(attention_mul)
    x = Dense(512)(x)
    x = Dense(512)(x_inputs_label_dash)
    x = Dropout(0.4)(x)
    x = Dense(1024)(x)    
    x_embedding = Dense(4096,activation='sigmoid',name = 'x_embedding')(x)
    
    output_label = Dense(2, activation='sigmoid',name = 'Output_label'
                         ,kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(x) #)(x)
#    output_label_dash = Dense(3, activation='sigmoid',name = 'Output_label_dash')(x)
    output_predicted_label = Dense(2, activation='sigmoid',name = 'output_predicted_label')(x)
    output_margin = Dense(1, activation='sigmoid')(x)
    merged_vector = keras.layers.concatenate([x_inputs_label, x_inputs_unlabel, x_inputs_label_dash,output_margin], axis=-1, name='merged_layer')
    
    model = Model(input=[inputs_label,inputs_unlabel,inputs_label_dash], output=[x_embedding, output_label,output_predicted_label,merged_vector])
    model.summary()

    losses ={'Output_label':'mean_squared_error',             
              'x_embedding': 'cosine_proximity',
              'merged_layer':triplet_loss,
              'output_predicted_label':'kullback_leibler_divergence' }
#    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, clipnorm=1, clipvalue=0.5,nesterov=True)
   
    model.compile(
    optimizer=sgd,
    loss=losses,
    metrics=['acc'])
    
    return model

# =============================================================================
# data loading
# =============================================================================
with h5py.File('./data/data_triplet_vgg-fc6_tabletgaze.h5', 'r') as hf:
    start_frames_l = hf['start_frames_l'][:]
    end_frames_l = hf['end_frames_l'][:]
    unlabelled_frames = hf['unlabelled_frames'][:]
    start_gaze_label = hf['start_gaze_label'][:]
    end_gaze_label = hf['end_gaze_label'][:]
hf.close()    

all_start_frames_l = start_frames_l
all_end_frames_l = end_frames_l
all_unlabelled_frames = unlabelled_frames
all_start_gaze_label = start_gaze_label
all_end_gaze_label = end_gaze_label        
del start_frames_l,end_frames_l,unlabelled_frames,start_gaze_label,end_gaze_label

#normalize data
Y_start_gaze_label=np.zeros([len(all_start_gaze_label),2])
Y_start_gaze_label[:,0] = (all_start_gaze_label[:,0]- np.ones(len(all_start_gaze_label[:,0]))*
                  min(all_start_gaze_label[:,0]))/(np.ones(len(all_start_gaze_label[:,0]))*
                     max(all_start_gaze_label[:,0])-np.ones(len(all_start_gaze_label[:,0]))*
                     min(all_start_gaze_label[:,0]))
Y_start_gaze_label[:,1] = (all_start_gaze_label[:,1]- np.ones(len(all_start_gaze_label[:,1]))*
                  min(all_start_gaze_label[:,1]))/(np.ones(len(all_start_gaze_label[:,1]))*
                     max(all_start_gaze_label[:,1])-np.ones(len(all_start_gaze_label[:,1]))*
                     min(all_start_gaze_label[:,1]))


Y_end_gaze_label=np.zeros([len(all_end_gaze_label),2])
Y_end_gaze_label[:,0] = (all_end_gaze_label[:,0]- np.ones(len(all_end_gaze_label[:,0]))*
                  min(all_end_gaze_label[:,0]))/(np.ones(len(all_end_gaze_label[:,0]))*
                     max(all_end_gaze_label[:,0])-np.ones(len(all_end_gaze_label[:,0]))*
                     min(all_end_gaze_label[:,0]))
Y_end_gaze_label[:,1] = (all_end_gaze_label[:,1]- np.ones(len(all_end_gaze_label[:,1]))*
                  min(all_end_gaze_label[:,1]))/(np.ones(len(all_end_gaze_label[:,1]))*
                     max(all_end_gaze_label[:,1])-np.ones(len(all_end_gaze_label[:,1]))*
                     min(all_end_gaze_label[:,1]))

del all_end_gaze_label,all_start_gaze_label
# =============================================================================
#   model training      
# =============================================================================
model = build_model()
checkpoint = keras.callbacks.ModelCheckpoint('./models/best_weights.h5', monitor='loss', 
                                             save_best_only=True,
                                              verbose=1,mode='min')

history = model.fit(
    [all_start_frames_l,all_end_frames_l,all_unlabelled_frames],
    [all_end_frames_l,Y_start_gaze_label,Y_start_gaze_label,all_end_frames_l],
    epochs=1000,
    batch_size=32,callbacks=[checkpoint])

model.load_weights('./models/best_weights.h5')
model.save('./models/video_eyegaze_onelabel_kl_sgd0.01_normalizedlabel_tabletgaze_1000.h5')

# =============================================================================
# Plot curve
# =============================================================================

from matplotlib import pyplot as plt
#acc = history.history['acc']
#val_acc = history.history['val_acc']
loss = history.history['loss']
#val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'b', label='loss')
#plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('loss')
plt.legend()

#    plt.plot(epochs, loss, 'bo', label='Training loss')
#    plt.plot(epochs, val_loss, 'b', label='Validation loss')
#    plt.title('Training and validation loss')
#plt.legend()

plt.show()
#plt.savefig('./models/video_eyegaze_sgd0.01_normalizedlabel.png')
# =============================================================================
# evaluate
# =============================================================================

with h5py.File('./data/data_triplet_vgg-fc6_tabletgaze.h5', 'r') as hf:
    original_unlabel = hf['original_unlabel'][:]
hf.close()

Y_original_unlabel=np.zeros([len(original_unlabel),2])
Y_original_unlabel[:,0] = (original_unlabel[:,0]- np.ones(len(original_unlabel[:,0]))*
                  min(original_unlabel[:,0]))/(np.ones(len(original_unlabel[:,0]))*
                     max(original_unlabel[:,0])-np.ones(len(original_unlabel[:,0]))*
                     min(original_unlabel[:,0]))
Y_original_unlabel[:,1] = (original_unlabel[:,1]- np.ones(len(original_unlabel[:,1]))*
                  min(original_unlabel[:,1]))/(np.ones(len(original_unlabel[:,1]))*
                     max(original_unlabel[:,1])-np.ones(len(original_unlabel[:,1]))*
                     min(original_unlabel[:,1]))
#Y_original_unlabel[:,2] = (original_unlabel[:,2]- np.ones(len(original_unlabel[:,2]))*
#                  min(original_unlabel[:,2]))/(np.ones(len(original_unlabel[:,2]))*
#                     max(original_unlabel[:,2])-np.ones(len(original_unlabel[:,2]))*
#                     min(original_unlabel[:,2]))

model.load_weights('./models/video_eyegaze_onelabel_kl_sgd0.01_normalizedlabel_tabletgaze_1000.h5')
model_new = Model(input=model.input, output=model.get_layer('output_predicted_label').output)
y_pred = model_new.predict([all_start_frames_l,all_end_frames_l,all_unlabelled_frames],verbose = 1)

from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(Y_original_unlabel, y_pred))