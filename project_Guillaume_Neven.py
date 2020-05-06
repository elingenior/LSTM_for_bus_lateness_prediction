import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime
import time

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from functions import *
#%%
startTime = 7*3600
endTime = 10*3600
dt = 30
## Import travel time data
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

Stops_pos = Stops_pos[Stops_pos['halt_punkt_ist_aktiv']]
Stops_pos = Stops_pos.groupby(by = 'halt_id').mean()
Stops_pos = Stops_pos.merge(Stops[{'halt_id', 'halt_diva'}],left_on='halt_id',right_on='halt_id')

if all(Data.columns != 'Connected'):
    Data =  Data.merge(Line,left_on='linie',right_on='Linienname')
    Data =  Data.merge(Stops_pos[{'halt_diva','GPS_Latitude',
       'GPS_Longitude'}],left_on='halt_diva_nach',right_on='halt_diva')

#Features = pd.DataFrame(index = range(Data.shape[0]), columns = {'Delta_trip','Delta_stops','Tot_lat','Is_connected','Type'})
#
#Features['Delta_stops'] = Data['ist_ab_nach']-Data['ist_an_nach1']-Data['soll_ab_nach']+Data['soll_an_nach']
#Features['Delta_trip'] = Data['ist_an_nach1']-Data['ist_ab_von']-Data['soll_an_nach']+Data['soll_ab_von']
#Features['Tot_lat'] = Data['ist_ab_nach']-Data['soll_ab_nach']
#Features['Is_connected']=Data['Connected'].values
#Features['Type'] = Data['VSYS'].values
#Features['Line'] = Data['linie'].values
#Features['Stop'] = Data['halt_diva_nach'].values
#Features['Time'] = Data['ist_ab_nach'].values
#Features['BUS'] = Data['fahrzeug'].values

#Lines_latness = Features.groupby(by='Line',axis =0).mean()
#%%
## features selection but here we do not select features but stops

time_frame = 20*60 # we look at Xmin data frame
Bus_id = Data.loc[(Data['linie'] == Studied_line),'fahrzeug'].unique()[0]
Data_bus = Data[(Data['fahrzeug'] == Bus_id)]
Data_bus['Stop'] = (Data_bus['richtung']-1) * (max(Data_bus['seq_nach'])+1 - Data_bus['seq_nach']) - (Data_bus['richtung']-2) * Data_bus['seq_nach']
Data_bus = Data_bus.sort_values(by='ist_ab_von')

plt.plot(Data_bus['ist_ab_von']/60,Data_bus['Stop'],color = 'r')
plt.plot(Data_bus['soll_ab_von']/60,Data_bus['Stop'],color = 'b')

y_real = Data[Data['fahrzeug'] == Bus_id]# all the delays for one precise bus (ID = bus number)

#Features.groupby(['Line']).mean()

select = np.array(['Delta_stops','Delta_trip','Tot_lat','Is_connected','Type','Line','Stop','Time','Dist','Lat','ID'])
#y_value =  Data[]
#%% Converting to bus based

idx = pd.IndexSlice

Data = Data.sort_values(by =['Linienname','halt_diva_nach'])

t1 = time.time()
BusPos = pd.DataFrame(columns = range(startTime,endTime,dt),index = ['Lat','Long'])
#FrameT = pd.DataFrame(index = range(startTime,endTime,dt),columns = range(Data['datum_nach'].unique().shape[0]))  & Data['datum_nach'] == d
#Colnames = [np.repeat(range(startTime,endTime,dt),11),np.resize(select,11*int((endTime-startTime)/dt))]

Colnames = range(startTime,endTime,dt)

RowNames = [np.repeat(10000*Data[{'Linienname','halt_diva_nach'}].drop_duplicates()['Linienname'] + Data[{'Linienname','halt_diva_nach'}].drop_duplicates()['halt_diva_nach'],11),
            np.resize(select,11*len(Data[{'Linienname','halt_diva_nach'}].drop_duplicates()))]
FrameT = pd.DataFrame(index = RowNames,columns = Colnames)
for t in range(startTime,endTime,dt):
    if t == startTime:
    # the first t is lower so we have the last known position 
        sel = 10000*Data.loc[(Data['ist_ab_nach']>t- 3600 ) & (Data['ist_ab_nach']<t+dt),{'Linienname','halt_diva_nach'}].drop_duplicates()['Linienname']+Data.loc[(Data['ist_ab_nach']>t-3600 ) & (Data['ist_ab_nach']<t+dt),{'Linienname','halt_diva_nach'}].drop_duplicates()['halt_diva_nach']
        FrameT.loc[idx[sel,:],t] = pd.DataFrame(
            {'Delta_stops'  :(Data.loc[(Data['ist_ab_nach']>t- 3600 ) & (Data['ist_ab_nach']<t+dt),'ist_ab_nach']-Data.loc[(Data['ist_ab_nach']>t- 3600  ) & (Data['ist_ab_nach']<t+dt),'ist_an_nach1']-Data.loc[(Data['ist_ab_nach']>t- 3600  ) & (Data['ist_ab_nach']<t+dt),'soll_ab_nach']+Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'soll_an_nach']),
             'Delta_trip'   :Data.loc[(Data['ist_ab_nach']>t- 3600 ) & (Data['ist_ab_nach']<t+dt),'ist_an_nach1']-Data.loc[(Data['ist_ab_nach']>t- 3600  ) & (Data['ist_ab_nach']<t+dt),'ist_ab_von']-Data.loc[(Data['ist_ab_nach']>t- 3600  ) & (Data['ist_ab_nach']<t+dt),'soll_an_nach']+Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'soll_ab_von'],
             'Tot_lat'      :Data.loc[(Data['ist_ab_nach']>t- 3600 ) & (Data['ist_ab_nach']<t+dt),'ist_ab_nach']-Data.loc[(Data['ist_ab_nach']>t- 3600  ) & (Data['ist_ab_nach']<t+dt),'soll_ab_nach'],
             'Is_connected' :Data.loc[(Data['ist_ab_nach']>t- 3600 ) & (Data['ist_ab_nach']<t+dt),'Connected'].values,
             'Type'         :Data.loc[(Data['ist_ab_nach']>t- 3600 ) & (Data['ist_ab_nach']<t+dt),'VSYS'].values,
             'Line'         :Data.loc[(Data['ist_ab_nach']>t- 3600 ) & (Data['ist_ab_nach']<t+dt),'linie'].values,
             'Stop'         :Data.loc[(Data['ist_ab_nach']>t- 3600 ) & (Data['ist_ab_nach']<t+dt),'halt_diva_nach'].values,
             'Time'         :Data.loc[(Data['ist_ab_nach']>t- 3600 ) & (Data['ist_ab_nach']<t+dt),'ist_ab_nach'].values,
             'Dist'         :Data.loc[(Data['ist_ab_nach']>t- 3600 ) & (Data['ist_ab_nach']<t+dt),'GPS_Longitude'].values,
             'Lat'          :Data.loc[(Data['ist_ab_nach']>t- 3600 ) & (Data['ist_ab_nach']<t+dt),'GPS_Latitude'].values,
             'ID'           : 10000*Data.loc[(Data['ist_ab_nach']>t- 3600 ) & (Data['ist_ab_nach']<t+dt),'Linienname']+Data.loc[(Data['ist_ab_nach']>t- 3600 ) & (Data['ist_ab_nach']<t+dt),'halt_diva_nach']}
            ).drop_duplicates(subset = 'ID',keep = 'last').values.ravel()
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
             'Type'         :Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'VSYS'].values,
             'Line'         :Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'linie'].values,
             'Stop'         :Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'halt_diva_nach'].values,
             'Time'         :Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'ist_ab_nach'].values,
             'Dist'         :Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'GPS_Longitude'].values,
             'Lat'          :Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'GPS_Latitude'].values,
             'ID'           : 10000*Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'Linienname']+Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt),'halt_diva_nach']}
            ).drop_duplicates(subset = 'ID',keep = 'last').values.ravel()
    if Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt) & (Data['fahrzeug']==Bus_id),['GPS_Latitude','GPS_Longitude']].empty :
        BusPos[t] = BusPos[t-dt] 
    else:
        BusPos[t] = Data.loc[(Data['ist_ab_nach']>t ) & (Data['ist_ab_nach']<t+dt) & (Data['fahrzeug']==Bus_id),['GPS_Latitude','GPS_Longitude']].drop_duplicates(keep = 'last').T.values

#               
FrameT.head()
print('Time spend :',time.time()- t1)

FrameT = FrameT.drop('ID', level=1)
FrameT.loc[idx[:,'Dist'],:] = abs((FrameT.loc[idx[:,'Dist'],:].subtract(BusPos.loc['Long'].values,axis = 1))**2 + (FrameT.loc[idx[:,'Lat'],:].subtract(BusPos.loc['Lat'].values,axis = 1))**2)

FrameT = FrameT.drop('Lat', level=1)

#%% Test LSTM

