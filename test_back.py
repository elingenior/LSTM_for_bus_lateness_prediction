 plus import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import shap


import datetime
import time
import multiprocessing  as mp
from itertools import product

from keras.models import Sequential,Model
from keras.layers import *#Dense, Input, Concatenate, Lambda,LSTM,Dropout,Layer,Flatten
from keras.layers import ConvLSTM2D,BatchNormalization,TimeDistributed,MaxPooling3D
from keras.optimizers import RMSprop, Adam
from keras import optimizers
from keras.utils.vis_utils import plot_model
import keras.losses as kl
import pydot
from keras import backend as K
from keras import optimizers
import keras
import pickle
import tensorflow as tf
from keras.utils import multi_gpu_model
import sys
import warnings
import os

from keras.preprocessing import sequence
#from tqdm import tqdm
os.chdir('/gneven/Desktop/ETH/Sem2/Projeckt/Pyhton/')
from scipy.interpolate import UnivariateSpline
from functions import * 

idx = pd.IndexSlice

import keras.backend.tensorflow_backend as tfback

tf.config.threading.set_intra_op_parallelism_threads(7)
tf.config.threading.set_inter_op_parallelism_threads(7)

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
# In[1]
FrameT = pd.read_pickle(os.getcwd()+'/DataPickles/BusFrameT.pickle')

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=6, 
                        inter_op_parallelism_threads=6, 
                        allow_soft_placement=True,
                        device_count = {'CPU': 6})

session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


FrameT[np.isnan(FrameT)] = 0
# FrameT[FrameT>400] = 0
# FrameT[FrameT<-400] = 0
FrameT = FrameT[:2,:,:,:,:]
meanF = np.mean(FrameT)
FrameT -= meanF
std = np.std(abs(FrameT))
FrameT /= std 

out_dt = 1
in_dt = 30
dt = 30

starttime = 7*3600
endtime = 9*3600

# 6.5 -- 10.5
in_data = [6.5*3600,10.5*3600]

diff = int((starttime- in_data[0])/dt)


input_bus = np.ndarray((FrameT.shape[0],int((endtime-out_dt-starttime)/dt),in_dt,FrameT.shape[2],FrameT.shape[3],FrameT.shape[4]))#FrameT.shape[2]
output_bus = np.ndarray((FrameT.shape[0],int((endtime-out_dt-starttime)/dt),out_dt,FrameT.shape[3],FrameT.shape[4]))
for i in range(int((endtime-out_dt-starttime)/dt)):
    input_bus[:,i,:,:,:,:] = FrameT[:,(diff+i-in_dt):(diff+i),:,:,:]
    output_bus[:,i,:,:,:] = FrameT[:,(diff+i):(diff+i+out_dt),2,:,:]
    
vali = input_bus#[[0,6,12,18,24,29,34],:,:,:,:]
valo = output_bus#[[0,6,12,18,24,29,34],:,:,:,:]
input_bus = np.delete(input_bus,[0,6,12,18,24,29,34], axis = 0)
output_bus = np.delete(output_bus,[0,6,12,18,24,29,34], axis = 0)

batch_size =input_bus.shape[1]
print(batch_size)

input_bus = input_bus.reshape(input_bus.shape[0]*input_bus.shape[1],input_bus.shape[2],input_bus.shape[3],input_bus.shape[4],input_bus.shape[5])
output_bus = output_bus.reshape(output_bus.shape[0]*output_bus.shape[1],output_bus.shape[2],output_bus.shape[3],output_bus.shape[4])

vali = vali.reshape(vali.shape[0]*vali.shape[1],vali.shape[2],vali.shape[3],vali.shape[4],vali.shape[5])
valo = valo.reshape(valo.shape[0]*valo.shape[1],valo.shape[2],valo.shape[3],valo.shape[4])

output_bus = np.expand_dims(output_bus,axis = 4)
valo = np.expand_dims(valo,axis = 4)

filter1 = 64
filter2 = 32
kernel1 = 5
kernel2 = 5
input_bus = input_bus[:,:,2:,:,:]
out_dt = 1
epochs = 10

valo = np.squeeze(valo)
output_bus = np.squeeze(output_bus)
model_1 = Sequential([
    BatchNormalization(name = 'batch_norm_0', input_shape = (in_dt,input_bus.shape[2],input_bus.shape[3],input_bus.shape[4])), 


    ConvLSTM2D(name ='conv_lstm_1',filters=filter1, kernel_size=(kernel1, 1),stateful=False
                , padding='same', return_sequences=True,data_format  ="channels_first"),
    Dropout(0.2),

    BatchNormalization(),

    ConvLSTM2D(name ='conv_lstm_2',filters=filter2, kernel_size=(kernel2, 1),stateful=False
                ,data_format  ="channels_first"
                ,padding='same', return_sequences=False),

    Dropout(0.2),

    BatchNormalization(),


    Flatten(),

    # RepeatVector(out_dt),

    Reshape((out_dt,375,1,filter2)),


    ConvLSTM2D(name ='conv_lstm_3',filters=filter2,  kernel_size=(kernel2, 1),stateful=False
                , padding='same', return_sequences=True),
    Dropout(0.1),
    BatchNormalization(),

    ConvLSTM2D(name ='conv_lstm_4',
                filters = filter1, kernel_size = (kernel1, 1),stateful=False, 
                padding='same',
                return_sequences = True),
    #    Reshape((10,202472)) ,           
    #    
    #    TimeDistributed(Dense(out.shape[2]*20)),
    ##    
    #    TimeDistributed(Dense(32,)),
    #    
    TimeDistributed(Dense(1), name='test'),

    Reshape((375,)),


])
# optimizer = optimizers.Adam(clipvalue=0.5)
#optimizer = optimizers.Adam(clipnorm=1.)
#     model = multi_gpu_model(model, gpus=6)
model_1.compile(loss=kl.mean_absolute_error, optimizer = 'adam',metrics=['accuracy'])
model_1.summary()

hist_1 = model_1.fit(input_bus, output_bus, epochs=30,
                              batch_size=20, verbose=1, shuffle=False)

with open('caseof.pickles','wb') as p:
    pickle.dump(model_1,p)
    pickle.dump(hist_1,p)
# In[2]
with open('caseof.pickles','rb') as p:
    model = pickle.load(p)
    hist = pickle.load(p)
valo = sequence.pad_sequences(valo)

explainer = shap.DeepExplainer(model, tf.convert_to_tensor(input_bus[:239,:,2:,:,:],dtype = 'float32'))
