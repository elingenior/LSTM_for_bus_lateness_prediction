#!/usr/bin/env python
# coding: utf-8
#  sys.path.insert(0,'C:\\Users\gneven\AppData\Roaming\Python\Python38\site-packages')

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

from keras import optimizers
import keras

import pickle
''
import tensorflow as tf

import sys
import warnings
import os
#from tqdm import tqdm
os.chdir('/gneven/Desktop/ETH/Sem2/Projeckt/Pyhton/')

from functions import * 

idx = pd.IndexSlice


tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)
# In[3]:


startTime = 7*3600
endTime = 10*3600
dt = 60
ncoeur = 8

allfile = os.listdir('data/data2/Times/')


Data = get_time_data('data/data2/Times/'+allfile[0])


for str in allfile[1:10]:
    Data = pd.concat([Data,get_time_data('data/data2/Times/'+str)])
    
Data = Data.loc[Data['datum_nach'].dt.dayofweek < 5,:]
#delete depot run
Data = Data[~Data['fw_lang'].str.contains('DEP|Einfahrt|Ausfahrt')]


Studied_line = 32
Data = Data[Data['linie']<1000]
Data = Data.sort_values(by ='halt_diva_nach')
Data['ID'] = 100000000*Data['richtung']+ 10000*Data['linie']+Data['halt_diva_nach']
Data = Data.sort_values(by ='ID')


# ## Selection
# ### Stops
# Select the 3 lateness (travel time, stops and total ) for every stops every 30s, if no new bus came, the last one is kept. It was first thought to take as time step the updating of the bus lateness, but as we now have multiple bus it does not work.
i = 0
for day in Data['datum_nach'].unique():
    if i == 0:
        splitData = list([Data.loc[Data['datum_nach'] == day,:]])
    else:
        splitData.append(Data.loc[Data['datum_nach'] == day,:])
    i+=1
# In[ ]:

before_dt = 20

t1 = time.time()
def features_selector(Data_d):
    Data_d = pd.DataFrame({
        'Delta_stops'  :(Data_d['ist_ab_nach']-Data_d['ist_an_nach1']-Data_d['soll_ab_nach']+Data_d['soll_an_nach']),
        'Delta_trip'   :(Data_d['ist_an_nach1']-Data_d['ist_ab_von']-Data_d['soll_an_nach']+Data_d['soll_ab_von']),
        'Tot_lat'      :Data_d['ist_ab_nach']-Data_d['soll_ab_nach'],
        'ID'           :Data_d['ID'],
        'time'         :Data_d['ist_ab_nach']})
#
#    colname = [np.repeat(Data['ID'].unique(),3),np.resize(['Delta_stops','Delta_trip','Tot_lat'
#                                                             ],3*Data['ID'].unique().shape[0])]

    out_S = pd.DataFrame(index = range(startTime,endTime+before_dt*dt,dt) ,columns= Data['ID'].unique())
    out_T = pd.DataFrame(index = range(startTime,endTime+before_dt*dt,dt) ,columns= Data['ID'].unique())
    out_TOT = pd.DataFrame(index = range(startTime,endTime+before_dt*dt,dt) ,columns= Data['ID'].unique())

    t_last = startTime

    Data_t = Data_d.loc[(Data_d['time'] < t_last) ,['Delta_stops','Delta_trip','Tot_lat','ID']]

#    out.loc[t_last,idx[Data_t['ID'].values,:]] = Data_t.groupby('ID').last().values.ravel()
#    out.loc[t,:].update(Data_t.groupby('fahrzeug').last()['Tot_latness'])
    
    out_S.loc[t_last,:].update( Data_t.groupby('ID').last()['Delta_stops'])
    out_T.loc[t_last,:].update( Data_t.groupby('ID').last()['Delta_trip'])
    out_TOT.loc[t_last,:].update( Data_t.groupby('ID').last()['Tot_lat'])
    
    
    for t in range(startTime+dt,endTime,dt):
        Data_t = Data_d.loc[(Data_d['time'] > t_last) &(Data_d['time'] <= t),['Delta_stops','Delta_trip','Tot_lat','ID']]
        
        out_S.loc[t,:]   = out_S.loc[t_last,:]
        out_T.loc[t,:]   = out_T.loc[t_last,:]
        out_TOT.loc[t,:] = out_TOT.loc[t_last,:]
        
        
        out_S.loc[t,:].update( Data_t.groupby('ID').last()['Delta_stops'])
        out_T.loc[t,:].update( Data_t.groupby('ID').last()['Delta_trip'])
        out_TOT.loc[t,:].update( Data_t.groupby('ID').last()['Tot_lat'])
        t_last = t
        
    out = np.array([out_S.to_numpy(),out_T.to_numpy(),out_TOT.to_numpy()])
    
    return_array = np.empty(((int((endTime-startTime)/dt)),out.shape[0],before_dt,out.shape[2]))
    for w in range(int((endTime-startTime)/dt)):
        return_array[w,:,:,:] = out[:,w:(before_dt+w),:]
    return return_array
    del(Data_t,Data_d,out,return_array,out_S,out_T,out_TOT)

pool = mp.Pool(ncoeur) 
i = 0
for day in Data['datum_nach'].unique():
    if i == 0:
        splitData = list([Data.loc[Data['datum_nach'] == day,:]])
    else:
        splitData.append(Data.loc[Data['datum_nach'] == day,:])
    i+=1
    
FrameT = np.array(pool.map(features_selector,splitData),np.float32)#np.array(,np.float32)
pool.close()
pool.join() # 28

print('Time spent: ',time.time()-t1)

save = FrameT
# ### Reshape into (#sample , time ,features, #stop, 1)

# In[ ]:


FrameT = np.moveaxis(FrameT, [0, 1, 2, 3,4], [0, 1, 3, 2,4])
FrameT = np.expand_dims(FrameT,axis = 5)
FrameT.shape #(5, 360, 3, 2472, 1) 1539

#FrameT = FrameT[:,:,:,:,~np.all(np.isnan(FrameT),axis = (0,1,2,3,5)),:]

#FrameT[:,:,:,0,:,:] = FrameT[:,:,:,0,:,:]/np.nanmax(abs(FrameT[:,:,:,0,:,:]))
#FrameT[:,:,:,1,:,:] = FrameT[:,:,:,1,:,:]/np.nanmax(abs(FrameT[:,:,:,1,:,:]))
#FrameT[:,:,:,2,:,:] = FrameT[:,:,:,2,:,:]/np.nanmax(abs(FrameT[:,:,:,2,:,:]))

# ### Bus
# The bus delay for all the bus in the line is computed every 30s.  ( don't need reshape already good)

# In[ ]:

before_dt = 20


num_next_time = 10
nex_time_dt = 1
startTime += before_dt
endTime += before_dt
def get_out(Data_d):

    Data_d = splitData[0]
    Data_d = Data_d.loc[Data_d['linie'] == Studied_line,:]
    
    Data_d['Tot_latness'] = (Data_d['ist_ab_nach']-Data_d['ist_ab_von'])/(Data_d['soll_ab_nach']-Data_d['soll_ab_von'])
    
    idBus = Data_d.loc[(Data_d['soll_ab_nach']>=startTime ) &(
                        (Data_d['soll_ab_nach']<=endTime )
            ) & (Data_d['halt_diva_nach'] == 1143) ,'fahrzeug'].unique()
    
    
    out= pd.DataFrame(index = range(startTime,endTime,dt), columns = idBus)
    
    t_last = startTime
    Data_t = Data_d.loc[(Data_d['ist_ab_nach'] < t_last),['Tot_latness','fahrzeug']]
    out.loc[t_last].update(Data_t.groupby('fahrzeug').last()['Tot_latness'])

    for t in range(startTime+dt,endTime+num_next_time*nex_time_dt*dt,dt):
        Data_t = Data_d.loc[(Data_d['ist_ab_nach'] > t_last) &(Data_d['ist_ab_nach'] <= t),['Tot_latness','fahrzeug']]
        if Data_t['fahrzeug'].size == 0:
            out.loc[t,:] = out.loc[t_last,:]
        else:
            out.loc[t,:] = out.loc[t_last,:]
#            out.loc[t,Data_t['fahrzeug'].unique()] =  Data_t.groupby('fahrzeug').last()
            out.loc[t,:].update(Data_t.groupby('fahrzeug').last()['Tot_latness'])
        t_last = t
        
        out = out.dropna(axis='columns',how='all')
        
        
    return_array = np.empty(((int((endTime-startTime)/dt)),num_next_time,out.shape[1]))
    for w in range(int((endTime-startTime)/dt)):
        return_array[w,:,:] = out.iloc[w:(num_next_time+w),:].to_numpy()
        
    return return_array    
    del(Data_t,Data_d,out,return_array)

pool = mp.Pool(ncoeur) 

out = np.array(pool.map(get_out,splitData),np.float32)
pool.close()
pool.join() # 28  


# In[ ]:


out = np.expand_dims(out,axis = 5)
out = np.expand_dims(out,axis = 6)

 # In[ ]:
  
with open('FrameT.pickle', 'wb') as f:
    pickle.dump(FrameT, f)

with open('out.pickle', 'wb') as o:
    pickle.dump(out, o)
# ## LSTM model

    
    
model = Sequential([
    BatchNormalization(name = 'batch_norm_0',input_shape = (before_dt,FrameT.shape[3],FrameT.shape[4],FrameT.shape[5])),
    
    
    ConvLSTM2D(name ='conv_lstm_1',filters=32, kernel_size=(10, 1)
                       , padding='same', return_sequences=True,data_format  ="channels_first"),
    Dropout(0.2),
    
    BatchNormalization(),
        
    ConvLSTM2D(name ='conv_lstm_2',filters=16, kernel_size=(5, 1),data_format  ="channels_first"
                         ,padding='same', return_sequences=False),
               
    Dropout(0.2),
    
    BatchNormalization(),
    
    
    Flatten(),
   
    Dense(out.shape[3]),


    RepeatVector(num_next_time),
    
    Reshape((num_next_time,out.shape[3],1,1)),
#    
    
    ConvLSTM2D(name ='conv_lstm_3',filters=16,  kernel_size=(10, 1)
                       , padding='same', return_sequences=True),
    Dropout(0.1),
    BatchNormalization(),
    
    ConvLSTM2D(name ='conv_lstm_4',
                         filters = 32, kernel_size = (5, 1), 
                         padding='same',
                         return_sequences = True),
#    Reshape((10,202472)) ,           
#    
#    TimeDistributed(Dense(out.shape[2]*20)),
##    
#    TimeDistributed(Dense(32,)),
#    
    TimeDistributed(Dense(1,activation = "sigmoid"), name='test'),


    
    
])
optimizer = optimizers.Adam(clipvalue=0.5)
#optimizer = optimizers.Adam(clipnorm=1.)

model.compile(loss='mse', optimizer = optimizer)
model.summary()

plot_model(model)


# In[ ]:

batch_size = 60
epochs = 10
t1 = time.time()
#for day in range(len(Data['datum_nach'].unique())):
#def model_fit(day):
error = np.array([])
for e in range(epochs):
    for day in range(len(Data['datum_nach'].unique())-7):
        print('day :',day+1,'epoque :',e+1)
        model.fit(FrameT[day,:,:,:,:,:], out[day,:,:,:,:,:], epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
        model.reset_states()
    for p in range(1,7):
        predicty = model.predict(FrameT[-p,:,:,:,:,:])
        errorp += np.sum(abs(predicty-out[-p,:,:,:,:,:]))
        
    print('Epoch',e,'Error:',errorp)
    
    error = np.append(error,errorp)
    errorp = 0

#pool = mp.Pool(ncoeur) 
#
##out = pool.map(model_fit,)
#pool.close()
#pool.join() # 28  

print('time spend:',time.time()-1)

# In[ ]:
#hidden_layers = keras.backend.function(
#[model.layers[0].input],  # we will feed the function with the input of the first layer
#[model.layers[0].output,] # we want to get the output of the second layer
#)
#print(np.asarray(hidden_layers(FrameT[0,:,:,:,:,:])))


# In[ ]:


t1 = time.time()
latness_predict = model.predict(FrameT[4:5,:,:,:,:])
print('time spend:',t1-time.time())

#
## In[ ]:
#
#
#loss = model.history['loss']
#loss_val = model.history['val_loss']
#epochs_plot = range(epochs)
#plt.figure(2)
#plt.plot(epochs_plot, loss, 'bo', label='Training loss') #val_loss
#plt.plot(epochs_plot, loss_val, 'bs', label='Validation loss') #val_loss
#
#plt.title('Loss')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.legend()
#plt.show()
#plt.savefig('test.png')
#
#
## In[ ]:
#
#
#plt.figure(3,figsize = (12,5))
#
#which = 4
#
##plt.plot(predicty[:,:,which,0,0], label='Predict') #val_loss
#plt.plot(out[4,:.,:,which,0,0], label='Real') #val_loss/
#
#plt.title('Loss')
#plt.xlabel('Time')
#plt.ylabel('Latness')
#plt.legend()
#plt.show()
#out.shape


# In[ ]:

# model = Sequential()
# model.add(BatchNormalization(name = 'batch_norm_0', input_shape = (input_timesteps, num_links, 1, 1)))
# model.add(ConvLSTM2D(name ='conv_lstm_1',
#                          filters = 64, kernel_size = (10, 1),                       
#                          padding = 'same', 
#                          return_sequences = True))

# model.add(Dropout(0.2, name = 'dropout_1'))
# model.add(BatchNormalization(name = 'batch_norm_1'))

# model.add(ConvLSTM2D(name ='conv_lstm_2',
#                          filters = 64, kernel_size = (5, 1), 
#                          padding='same',
#                          return_sequences = False))

# model.add(Dropout(0.1, name = 'dropout_2'))
# model.add(BatchNormalization(name = 'batch_norm_2'))

# model.add(Flatten())
# model.add(RepeatVector(output_timesteps))
# model.add(Reshape((output_timesteps, num_links, 1, 64)))

# model.add(ConvLSTM2D(name ='conv_lstm_3',
#                          filters = 64, kernel_size = (10, 1), 
#                          padding='same',
#                          return_sequences = True))

# model.add(Dropout(0.1, name = 'dropout_3'))
# model.add(BatchNormalization(name = 'batch_norm_3'))
# model.add(ConvLSTM2D(name ='conv_lstm_4',
#                          filters = 64, kernel_size = (5, 1), 
#                          padding='same',
#                          return_sequences = True))

# model.add(TimeDistributed(Dense(units=1, name = 'dense_1', activation = 'relu')))
#     #model.add(Dense(units=1, name = 'dense_2'))
    
# model


# In[ ]:




