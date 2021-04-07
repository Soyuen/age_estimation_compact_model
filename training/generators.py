import keras
import numpy as np
import sys
from scipy import misc
import tensorflow as tf
from keras import backend as K
from keras_preprocessing import image
def cod(y_true, y_pred):
    SSR=K.sum(K.square(y_true-y_pred))
    SST=K.sum(K.square(y_true-K.mean(y_true)))
    RS=SSR/SST
    return RS
def pred4_newloss(y_true, y_pred):
    return K.mean((K.abs(y_pred - y_true))-K.constant(5), axis=-1)
def pred1_newloss(y_true, y_pred):
    return K.mean((K.abs(y_pred - y_true))-K.constant(6), axis=-1)
def wpred4_newloss(y_true, y_pred):
    return K.mean((K.abs(y_pred - y_true))-K.constant(4.5), axis=-1)
def wpred1_newloss(y_true, y_pred):
    return K.mean((K.abs(y_pred - y_true))-K.constant(5.5), axis=-1)
def augment_data(images):
    for i in range(0,images.shape[0]):

        if np.random.random() > 0.5:
            images[i] = images[i][:,::-1]
        if np.random.random() > 0.75:
            images[i] =image.random_rotation(images[i], 20, row_axis=0, col_axis=1, channel_axis=2)
        if np.random.random() > 0.75:
            images[i] = image.random_shear(images[i], 0.2, row_axis=0, col_axis=1, channel_axis=2)
        if np.random.random() > 0.75:
            images[i] = image.random_shift(images[i], 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2)
        if np.random.random() > 0.75:
            images[i] = image.random_zoom(images[i], [0.8,1.2], row_axis=0, col_axis=1, channel_axis=2)
        
    return images


def data_generator_reg(X,Y,Y20,Y4,batch_size):

    while True:
        idxs = np.random.permutation(len(X))
        X = X[idxs]
        Y = Y[idxs]  
        Y20=Y20[idxs] 
        Y4=Y4[idxs] 
        p,q,q1,q2,q3 = [],[],[],[],[]
        for i in range(len(X)):
            p.append(X[i])
            q.append(Y[i])
            q1.append(Y20[i]+Y4[i])
            q2.append(Y[i])
            q3.append(Y[i])
            if len(p) == batch_size:
                yield augment_data(np.array(p)),[np.array(q),np.array(q1),np.array(q2),np.array(q3)]
                p,q,q1,q2,q3 = [],[],[],[],[]
        if p:
            yield augment_data(np.array(p)),[np.array(q),np.array(q1),np.array(q2),np.array(q3)]
            p,q,q1,q2,q3 = [],[],[],[],[]

def ker_init(shape, dtype=None):
    kernel =  tf.Variable(
        [[[[ 1.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.]],            
          [[ 1.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.]],              
          [[ 1.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.]]],                              
         [[[ 1.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.]],               
          [[-7.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.]],                
          [[ 1.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.]]],
         [[[ 1.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.]],                
          [[ 1.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.]],      
          [[ 1.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.]]]])
    return kernel
