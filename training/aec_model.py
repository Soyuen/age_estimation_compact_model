import logging
import sys
import numpy as np
from keras.models import Model
from keras.layers import Input, Activation, Dense, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import tensorflow as tf
from generators import ker_init
import tensorflow.compat.v1 as tfc
config = tfc.ConfigProto()
config.gpu_options.allow_growth=True
sess = tfc.Session(config=config)
sys.setrecursionlimit(2 ** 20)
np.random.seed(2 ** 10)
def conv(Input,num):
    i = 1
    while i <=num:
        Input = Conv2D(16, (3, 3),padding='same')(Input)
        Input = BatchNormalization()(Input)
        Input = Activation("relu")(Input)
        i+=1
    Input = AveragePooling2D((2, 2))(Input)
    return Input
def stage_layer(Input,units1,units2,units3):
    Input=Dense(units1,kernel_initializer='he_uniform',activation='relu')(Input)
    Input=Dense(units2,kernel_initializer='he_uniform',activation='relu')(Input)
    Input=Dense(units3,kernel_initializer='he_uniform',activation='softmax')(Input)
    return Input
def pred_4(x):
    a4 = x[0][:,0]*0
    a4 = x[0][:,0]
    for j in range(0,5):
        a4 = a4+(j*4)*x[0][:,j]               
    a4 = K.expand_dims(a4,-1)
    a20 = x[1][:,0]*0
    a20 = x[1][:,0]
    for k in range(0,5):
        a20 = a20+(k*20)*x[1][:,k]  
    a20 = K.expand_dims(a20,-1)
    a20_4=a4+a20
    return a20_4
def merge_age(x):
    a = x[0][:,0]*0
    b = x[0][:,0]*0
    c = x[0][:,0]*0
    a = x[0][:,0]
    b = x[1][:,0]
    c = x[2][:,0]
    for i in range(0,4):
        a = a+(i)*x[0][:,i] 
    for j in range(0,5):
        b = b+(j*4)*x[1][:,j]             
    for k in range(0,5):
        c = c+(k*20)*x[2][:,k]    
    a = K.expand_dims(a,-1)
    b = K.expand_dims(b,-1)
    c = K.expand_dims(c,-1)
    ori_age=a+b+c
    return ori_age
def AEC_model(input_shape):
    logging.debug("Creating model...")
    
    Inputs = Input(shape=input_shape)
    sharp = Conv2D(3, (3, 3),padding='same',kernel_initializer=ker_init)(Inputs)
    
    layer  = conv(sharp,3)
    layer  = conv(layer,1)
    layer  = conv(layer,2)
    layer  = conv(layer,2)
    layer  = Conv2D(10,(1,1),kernel_initializer='he_uniform',activation='relu')(layer)
    layer  = Flatten()(layer)
    layer  = Dropout(0.2)(layer)
    stage3 = stage_layer(layer,5,10,5)
    stage2 = stage_layer(layer,5,10,5)
    stage1 = stage_layer(layer,4,8,4)
    pred_a = Lambda(merge_age,output_shape=(1,),name='pred_a')([stage1,stage2,stage3])
    pre_cod = Lambda(merge_age,output_shape=(1,),name='pre_cod')([stage1,stage2,stage3])
    pre_4 = Lambda(pred_4,output_shape=(1,),name='pre_4')([stage2,stage3])
    pre_1 = Lambda(merge_age,output_shape=(1,),name='pre_1')([stage1,stage2,stage3])
    model = Model(inputs=Inputs, outputs=[pred_a,pre_4,pre_1,pre_cod])
    return model