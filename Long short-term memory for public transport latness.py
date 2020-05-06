#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime
import time
import multiprocessing  as mp
from itertools import product

from keras.models import Sequential,Model
from keras.layers import Dense, Input, Concatenate, Lambda,LSTM,Dropout,Layer
from keras.optimizers import RMSprop

from sys import getsizeof
import warnings

from functions import *


# # Data import

# In[2]:


startTime = 7*3600
endTime = 10*3600
dt = 30

Data = get_time_data('data/data2/Times/fahrzeiten_soll_ist_20200105_20200111.csv')
Data = Data[Data['datum_nach'] == Data['datum_nach'][1]]

#delete depot run
Data = Data[~Data['fw_lang'].str.contains('DEP|Einfahrt|Ausfahrt')]
#Data=Data[(Data['soll_ab_nach']>startTime ) & (Data['soll_ab_nach']<endTime)] # only during peak hour (6am to 9 am)

#,'datum_nach','fahrzeug','soll_ab_nach'])

Studied_line = 32
connected = []

Stops = get_stops_data('data/data2/haltestelle.csv')
Stops_pos = get_stops_data('data/data2/haltepunkt.csv')

Line = get_line_data('data/data1/linie.csv')

Stops['Line'] = ""

Line['Connected'] = 0

for stop in Stops['halt_diva']:

    if any(Data.loc[Data['halt_diva_nach'] == stop ,'linie'].unique() == Studied_line):
        for l in [Data.loc[Data['halt_diva_nach'] == stop ,'linie'].unique()][0]:
            Line.loc[Line['Linienname'] == l,'Connected'] = 1
Line.loc[Studied_line,'Connected'] = 2

Stops_pos = Stops_pos[Stops_pos['halt_punkt_ist_aktiv']]
Stops_pos = Stops_pos.groupby(by = 'halt_id').mean()
Stops_pos = Stops_pos.merge(Stops[{'halt_id', 'halt_diva'}],left_on='halt_id',right_on='halt_id')

if all(Data.columns != 'Connected'):
    Data =  Data.merge(Line,left_on='linie',right_on='Linienname')
    Data =  Data.merge(Stops_pos[{'halt_diva','GPS_Latitude',
       'GPS_Longitude'}],left_on='halt_diva_nach',right_on='halt_diva')


# ## features selection but here we do not select features but stops
# 

# In[22]:


get_ipython().run_line_magic('matplotlib', 'notebook')

time_frame = 20*60 # we look at Xmin data frame
Bus_id = Data.loc[(Data['linie'] == Studied_line),'fahrzeug'].unique()[0]
Data_bus = Data[(Data['fahrzeug'] == Bus_id)]
Data_bus['Stop'] = (Data_bus['richtung']-1) * (max(Data_bus['seq_nach'])+1 - Data_bus['seq_nach']) - (Data_bus['richtung']-2) * Data_bus['seq_nach']
Data_bus = Data_bus.sort_values(by='ist_ab_von')

plt.plot(Data_bus['ist_ab_von']/60,Data_bus['Stop'],color = 'r')
plt.plot(Data_bus['soll_ab_von']/60,Data_bus['Stop'],color = 'b')

y_real = Data[Data['fahrzeug'] == Bus_id]# all the delays for one precise bus (ID = bus number)

#Features.groupby(['Line']).mean()

select = np.array(['Delta_stops','Delta_trip','Tot_lat','Is_connected','IsBus','IsTram','IsOther','Time','Dist','Lat','ID'])
#y_value =  Data[]


# ## Convert data to feed them in the neural network
# In this section all the data are converted to have bus stops delays at any point in time. The idea is too look every dt, if there is a change we update, if not we keep the same value. The first iteration the last known position is kept. We extract the following data for each day ;
# 
# + Increase in lateness due to boarding 
# + Increase in lateness due to travel time
# + Total lateness
# + If the line is connected to our line
# + the type of the line
# + the Time 
# + the distance to the studied vehicle 
# 
# Note that the data are shape is made x: time and y stops and then transposed, the process was much more faster ( more than 10x) because panda prefer to work with a lot of row than lots of columns
# 
# # NOTES TO MYSELF,
# to win time on the change of type, transform T-B etc to float. same with is connected, should we had a 2 for is line ?
# 
# dt or each new studied bus arrival ? - does LSTM like having not regular time step ?

# In[23]:


idx = pd.IndexSlice
ncoeur = 8
Data = Data.sort_values(by =['Linienname','halt_diva_nach'])

t1 = time.time()
BusPos = pd.DataFrame(columns = range(startTime,endTime,dt),index = ['Lat','Long'])
#FrameT = pd.DataFrame(index = range(startTime,endTime,dt),columns = range(Data['datum_nach'].unique().shape[0]))  & Data['datum_nach'] == d
#Colnames = [np.repeat(range(startTime,endTime,dt),11),np.resize(select,11*int((endTime-startTime)/dt))]

# def dataSelection(Data):

for i in [1]:
    print(type(Data))
    Colnames = range(startTime,endTime,dt)

    RowNames = [np.repeat(10000*Data[{'Linienname','halt_diva_nach'}].drop_duplicates()['Linienname'] + Data[{'Linienname','halt_diva_nach'}].drop_duplicates()['halt_diva_nach'],11),
            np.resize(select,11*len(Data[{'Linienname','halt_diva_nach'}].drop_duplicates()))]
    FrameT = pd.DataFrame(index = RowNames,columns = Colnames)
    BusPos = pd.DataFrame(columns = range(startTime,endTime,dt),index = ['Lat','Long'])

    for t in range(startTime,endTime,dt):

        if t == startTime:
            # the first t is lower so we have the last known position 
            sel = 10000*Data.loc[(Data['ist_ab_nach']>t- 3600 ) & (Data['ist_ab_nach']<t+dt),{'Linienname','halt_diva_nach'}].drop_duplicates()['Linienname']+Data.loc[(Data['ist_ab_nach']>t-3600 ) & (Data['ist_ab_nach']<t+dt),{'Linienname','halt_diva_nach'}].drop_duplicates()['halt_diva_nach']
            FrameT.loc[idx[sel,:],t] = pd.DataFrame(
                {'Delta_stops'  :(Data.loc[(Data['ist_ab_nach']>t- 3600 ) & (Data['ist_ab_nach']<t+dt),'ist_ab_nach']-Data.loc[(Data['ist_ab_nach']>t- 3600  ) & (Data['ist_ab_nach']<t+dt),'ist_an_nach1']-Data.loc[(Data['ist_ab_nach']>t- 3600  ) & (Data['ist_ab_nach']<t+dt),'soll_ab_nach']+Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'soll_an_nach']),
                 'Delta_trip'   :Data.loc[(Data['ist_ab_nach']>t- 3600 ) & (Data['ist_ab_nach']<t+dt),'ist_an_nach1']-Data.loc[(Data['ist_ab_nach']>t- 3600  ) & (Data['ist_ab_nach']<t+dt),'ist_ab_von']-Data.loc[(Data['ist_ab_nach']>t- 3600  ) & (Data['ist_ab_nach']<t+dt),'soll_an_nach']+Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'soll_ab_von'],
                 'Tot_lat'      :Data.loc[(Data['ist_ab_nach']>t- 3600 ) & (Data['ist_ab_nach']<t+dt),'ist_ab_nach']-Data.loc[(Data['ist_ab_nach']>t- 3600  ) & (Data['ist_ab_nach']<t+dt),'soll_ab_nach'],
                 'Is_connected' :Data.loc[(Data['ist_ab_nach']>t- 3600 ) & (Data['ist_ab_nach']<t+dt),'Connected'].values,
                 
                 'IsBus'        : ((Data.loc[(Data['ist_ab_nach']>t- 3600 ) & (Data['ist_ab_nach']<t+dt),'VSYS'].values == 'B' )|(
                                   Data.loc[(Data['ist_ab_nach']>t- 3600 ) & (Data['ist_ab_nach']<t+dt),'VSYS'].values == 'TB')).astype('float'),
            
                 'IsTram'       :(Data.loc[(Data['ist_ab_nach']>t- 3600 ) & (Data['ist_ab_nach']<t+dt),'linie'] == 'T').astype('float'),
                 
                 'IsOther'      :((Data.loc[(Data['ist_ab_nach']>t- 3600 ) & (Data['ist_ab_nach']<t+dt),'VSYS'].values == 'B' ) & (
                                  Data.loc[(Data['ist_ab_nach']>t- 3600 ) & (Data['ist_ab_nach']<t+dt),'linie'].values == 'T') & (
                                  Data.loc[(Data['ist_ab_nach']>t- 3600 ) & (Data['ist_ab_nach']<t+dt),'VSYS'].values == 'TB')).astype('float'),
                 
                 
                 'Time'         :Data.loc[(Data['ist_ab_nach']>t- 3600 ) & (Data['ist_ab_nach']<t+dt),'ist_ab_nach'].values,
                 'Dist'         :Data.loc[(Data['ist_ab_nach']>t- 3600 ) & (Data['ist_ab_nach']<t+dt),'GPS_Longitude'].values,
                 'Lat'          :Data.loc[(Data['ist_ab_nach']>t- 3600 ) & (Data['ist_ab_nach']<t+dt),'GPS_Latitude'].values,
                 'ID'           : 10000*Data.loc[(Data['ist_ab_nach']>t- 3600 ) & (Data['ist_ab_nach']<t+dt),'Linienname']+Data.loc[(Data['ist_ab_nach']>t- 3600 ) & (Data['ist_ab_nach']<t+dt),'halt_diva_nach']}
            ).drop_duplicates(subset = 'ID',keep = 'last').values.ravel()
            
            
            if all(FrameT.loc[idx[sel,'ID'],t].values != sel.values):
                warnings.warn('ID from Data is not ID from FrameT',DeprecationWarning)

            BusPos[t] = Data.loc[(Data['ist_ab_nach']>t -3600 ) & (Data['ist_ab_nach']<t+dt) & (Data['fahrzeug']==Bus_id),['GPS_Latitude','GPS_Longitude']].drop_duplicates(keep = 'last').T.values


        else:
            FrameT[t]=FrameT[t-dt]

            #    Bus_pos = Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt) & (Data['fahrzeug'] == Bus_id) ,{'GPS_Longitude','GPS_Latitude'}].
            sel = 10000*Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),{'Linienname','halt_diva_nach'}].drop_duplicates()['Linienname']+Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),{'Linienname','halt_diva_nach'}].drop_duplicates()['halt_diva_nach']
            FrameT.loc[idx[sel,:],t] = pd.DataFrame(
                {'Delta_stops'  :(Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'ist_ab_nach']-Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'ist_an_nach1']-Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'soll_ab_nach']+Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'soll_an_nach']),
                 'Delta_trip'   :Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'ist_an_nach1']-Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'ist_ab_von']-Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'soll_an_nach']+Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'soll_ab_von'],
                 'Tot_lat'      :Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'ist_ab_nach']-Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'soll_ab_nach'],
                 'Is_connected' :Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'Connected'].values,
                 
                 'IsBus'        : ((Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'VSYS'].values == 'B' )|(
                                   Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'VSYS'].values == 'TB')).astype('float'),
            
                 'IsTram'       :(Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'linie'] == 'T').astype('float'),
                 
                 'IsOther'      :((Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'VSYS'].values == 'B' ) & (
                                  Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'linie'].values == 'T') & (
                                  Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'VSYS'].values == 'TB')).astype('float'),
                 
                 'Time'         :Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'ist_ab_nach'].values,
                 'Dist'         :Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'GPS_Longitude'].values,
                 'Lat'          :Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'GPS_Latitude'].values,
                 'ID'           : 10000*Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'Linienname']+Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'halt_diva_nach']}
            ).drop_duplicates(subset = 'ID',keep = 'last').values.ravel()

            if all(FrameT.loc[idx[sel,'ID'],t].values != sel.values):
                warnings.warn('ID from Data is not ID from FrameT',DeprecationWarning)

                
            if Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt) & (Data['fahrzeug']==Bus_id),['GPS_Latitude','GPS_Longitude']].empty :
                BusPos[t] = BusPos[t-dt] 
            else:
                BusPos[t] = Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt) & (Data['fahrzeug']==Bus_id),['GPS_Latitude','GPS_Longitude']].drop_duplicates(keep = 'last').T.values
    # compute the distance to the studied bus
    FrameT = FrameT.drop('ID', level=1)
    FrameT = FrameT.drop('Line', level=1)
    FrameT = FrameT.drop('Stop', level=1)

    FrameT.loc[idx[:,'Dist'],:] = abs((FrameT.loc[idx[:,'Dist'],:].subtract(BusPos.loc['Long'].values,axis = 1).values)**2 + (FrameT.loc[idx[:,'Lat'],:].subtract(BusPos.loc['Lat'].values,axis = 1).values)**2)

    FrameT = FrameT.drop('Lat', level=1)
    
#     return FrameT
              

# pool = mp.Pool(ncoeur) #product(range(startTime,endTime,dt)
# i = 0
# for day in Data['datum_nach'].unique():
#     print(day)
#     if i == 0:
#         splitData = list([Data.loc[Data['datum_nach'] == day,:]])
#     else:
#         splitData.append(Data.loc[Data['datum_nach'] == day,:])
#     i+=1
    
# print('Time spend :',time.time()- t1)
# test = pool.map(dataSelection,splitData)
# pool.close()
# pool.join() # 28
print('Time spend :',time.time()- t1)


# In[43]:


FrameT = FrameT.dropna(axis=0)

FrameT= FrameT.transpose()
t1 = time.time()

FrameT = FrameT.astype('float')


# In[44]:


print(getsizeof(Data))
epochs = 20
batch_size = 200
print(Data.shape)

          
          
auto = Sequential([
    #Merge data for each stops
    
    
    Dense(4400, activation='relu',input_shape=(8841,)),
    Dense(2200, activation='relu'),
    #decoder
    Dense(4400, activation='relu'),
    Dense(8841, activation='sigmoid')    
])

auto.compile(loss='mean_squared_error', optimizer = RMSprop())
auto.summary()
t1 = time.time()

autoencoder_train = auto.fit(FrameT,FrameT, batch_size=
                             batch_size,epochs=epochs,verbose=1)


# In[48]:


loss = autoencoder_train.history['loss']
epochs_plot = range(epochs)
plt.figure(2)
plt.plot(epochs_plot, loss, 'bo', label='Training loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

