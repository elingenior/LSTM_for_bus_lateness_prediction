#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np

import datetime
import time
import multiprocessing  as mp
from itertools import product,permutations,combinations

from keras.models import Sequential,Model
from keras.layers import *#Dense, Input, Concatenate, Lambda,LSTM,Dropout,Layer,Flatten
from keras.layers import ConvLSTM2D,BatchNormalization,TimeDistributed,MaxPooling3D
from keras.optimizers import RMSprop, Adam
import keras.losses as kl 

from keras import optimizers
import keras

import pickle

import tensorflow as tf

import sys
import warnings
import os

import itertools
import keras.backend.tensorflow_backend as tfback

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus
try :
    os.mkdir('out')

except:
    print('out alreay exist')
# In[7]:
###############################################################################
# DELTE THIS DATAPICKLE DIR
###############################################################################

def get_data(dt):
    print(dt)
    with open(os.getcwd()+ '/BusFrame32_%i.pickle' %dt,'rb') as f:

        FrameT = pickle.load(f)
    
    FrameT[np.isnan(FrameT)] = 0
    # FrameT[FrameT>400] = 0
    # FrameT[FrameT<-400] = 0
    FrameT = FrameT[:20,:,:,:,:]
    meanF = np.mean(FrameT)
    FrameT -= meanF
    std = np.std(abs(FrameT))
    FrameT /= std 

    return FrameT
out_dt = 1
in_dt = 30
dt = 30

starttime = 7*3600
endtime = 9*3600

# 6.5 -- 10.5
in_data = [5*3600,12*3600]

in_dt_dic = dict({30:[1,5,10,20,40,60,100]
              ,60: [1,5,10,20,40,60]
              ,120: [1,5,10,20,40,60]
              ,300: [1,5,10,20]
              ,900: [1,2,5]
              })

results = np.empty((26,4))
results[:] = 0
iteration = 0
for dt in [30,60,120,300,900]:
    print(dt)
    dt = int(dt)
    FrameT = get_data(dt) 
    diff = int((starttime- in_data[0])/dt)

    
    for in_dt in in_dt_dic[dt]:
        in_dt = int(in_dt)
               
        input_bus = np.ndarray((FrameT.shape[0],int((endtime-out_dt-starttime)/dt),in_dt,1,FrameT.shape[3],FrameT.shape[4]))#FrameT.shape[2]
        output_bus = np.ndarray((FrameT.shape[0],int((endtime-out_dt-starttime)/dt),out_dt,FrameT.shape[3],FrameT.shape[4]))
        for i in range(int((endtime-out_dt-starttime)/dt)):
            input_bus[:,i,:,:,:,:] = FrameT[:,(diff+i-in_dt):(diff+i),2:,:,:]
            output_bus[:,i,:,:,:] = FrameT[:,(diff+i):(diff+i+out_dt),2,:,:]
            
        vali = input_bus[[0,6,12,18],:,:,:,:] #,24,29,34
        valo = output_bus[[0,6,12,18],:,:,:,:]
        input_bus = np.delete(input_bus,[0,6,12,18], axis = 0)
        output_bus = np.delete(output_bus,[0,6,12,18], axis = 0)
        
        batch_size =input_bus.shape[1]
        print(batch_size)
        
        input_bus = input_bus.reshape(input_bus.shape[0]*input_bus.shape[1],input_bus.shape[2],input_bus.shape[3],input_bus.shape[4],input_bus.shape[5])
        output_bus = output_bus.reshape(output_bus.shape[0]*output_bus.shape[1],output_bus.shape[2],output_bus.shape[3],output_bus.shape[4])
        
        vali = vali.reshape(vali.shape[0]*vali.shape[1],vali.shape[2],vali.shape[3],vali.shape[4],vali.shape[5])
        valo = valo.reshape(valo.shape[0]*valo.shape[1],valo.shape[2],valo.shape[3],valo.shape[4])
        
        output_bus = np.expand_dims(output_bus,axis = 4)
        valo = np.expand_dims(valo,axis = 4)
        
        #  chanel first == so in encoder 3, in decoder 5 ( position
        # input_bus.shape[0],
        
        # (3332, 30, 1, 375, 1)
        # in_dt = 30
        # unit1 = 64
        # unit2 = 32
        # kernel1 = 5
        # kernel2 = 2
        
        def model_creator(filters,kernel,dropout,NL):
            
            __model = Sequential()
            __model.add(BatchNormalization(name = 'batch_norm_0', input_shape = (in_dt,input_bus.shape[2],input_bus.shape[3],input_bus.shape[4]))) 
        
            for N in range(NL):
                if N == (NL-1):
                    __model.add(ConvLSTM2D(name =('lstm_%i_enc' %N),filters=filters[N], kernel_size=(kernel[N], 1),stateful=False
                                   , padding='same', return_sequences=False,data_format  ="channels_first"))
                else:
                    __model.add(ConvLSTM2D(name =('lstm_%i_enc' %N),filters=filters[N], kernel_size=(kernel[N], 1),stateful=False
                                   , padding='same', return_sequences=True,data_format  ="channels_first"))
                    
                __model.add(Dropout(dropout))
        
                __model.add(BatchNormalization())
        
            __model.add(Flatten())
        
            __model.add(RepeatVector(out_dt))
        
            __model.add(Reshape((out_dt,output_bus.shape[2],1,filters[-1])))
        
            for N in range(NL):
                
                __model.add(ConvLSTM2D(name =('lstm_%i_dec' %N),filters=filters[-N], kernel_size=(kernel[-N], 1),stateful=False
                                   , padding='same', return_sequences=True))
                __model.add(Dropout(dropout))
        
                __model.add(BatchNormalization())
                
                    
            __model.add(TimeDistributed(Dense(1,activation = "sigmoid"), name='test'))
        
        
            
            
            
            return __model
            del(__model)
        def fit_with(inputs): #unit1,unit2,kernel1,kernel2
            NL = int((len(inputs)-1)/2)
            __filters = inputs[:NL].astype('int')
            __kernel = inputs[NL:2*NL].astype('int')
            __dropout = inputs[-1]
            
            
            # return np.array([1,2,3,4,5])
            __model = model_creator(__filters,__kernel,__dropout,NL)
            __t1 = time.time()
            __model.compile(loss=kl.mean_absolute_error, optimizer = 'adam')
            __hist = __model.fit(input_bus, output_bus, epochs=30,
                                      batch_size=batch_size, verbose=0, shuffle=False)
            __error =  np.sum(abs(valo - __model.predict(vali)))/(valo.shape[0]*valo.shape[1]*valo.shape[2])
            # print('error',__filters,__kernel,__dropout ,'is compile')
            return np.array([__error,time.time()-__t1])
            del(__error,__model,__hist)
        
        
                # In[5]:
        
        
        filters = 32
        kernel = 5
        dropout = 0.4
        
        inputs = [32,5,0.4]
        # In[8]:
        
        
        
            
        results[iteration,:2] = [dt,in_dt]
        results[iteration,2:] = fit_with(inputs)
        iteration += 1  
        # i = 0
        # for input_i in np.array(np.meshgrid(filters,filters,filters,kernel,kernel,kernel,dropout)).T.reshape(-1,7):
        #     results_3[i,:] = fit_with(input_i)
        #     i+= 1
        # print('NL 3: over')
        # with open(os.getcwd()+'/out/Results%i_%i_3.pickle' %(dt,in_dt),'wb') as pelo:
        #     pickle.dump(results_3,pelo)
    
          
        
        # results_1 = pool.map(fit_with,np.array(np.meshgrid(filters,kernel,dropout)).T.reshape(-1,3))
        # print('NL == 1 over')
        # results_2 = pool.map(fit_with,np.array(np.meshgrid(filters,filters,kernel,kernel,dropout)).T.reshape(-1,5))
        # print('NL == 2 over')
        
        # results_3 = pool.map(fit_with,np.array(np.meshgrid(filters,filters,filters,kernel,kernel,kernel,dropout)).T.reshape(-1,7))
        # print('NL == 3 over')
        
        # pool.close()
        # pool.join()
        
        
        # In[9]:
        
        # In[ ]:
        
        del(output_bus,input_bus,vali,valo)
    
    
    
