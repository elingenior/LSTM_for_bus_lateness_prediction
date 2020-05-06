import csv
import pandas as pd


def get_time_data(path):
    Data = pd.read_csv(path)
    # select usedfull data
    Data = Data[{'linie','richtung','fahrzeug','halt_diva_von','soll_an_von', 
                 'ist_an_von', 'soll_ab_von', 'ist_ab_von', 'seq_nach',
                 'halt_diva_nach','datum_nach','soll_an_nach', 
                 'ist_an_nach1', 'soll_ab_nach', 'ist_ab_nach','fw_lang','kurs'}]
    # Change type of datas
    Data[Data.columns.difference(['datum_nach','fw_lang'])] = Data[Data.columns.difference(['datum_nach','fw_lang'])].astype('int32')
    Data['datum_nach'] = pd.to_datetime(Data['datum_nach'],format="%d.%m.%y")

    return Data


def get_stops_data(path):
    Data = pd.read_csv(path)
    if any(Data.columns == 'halt_diva'):
        Data['halt_diva'] = Data['halt_diva'].astype('int32')
    elif any(Data.columns == 'halt_diva'):
        Data['halt_diva'] = Data['halt_punkt_diva'].astype('int32')

    return Data

def get_line_data(path):
    with open(path, 'r',newline ='\r\n') as file:
        read = csv.reader(file, delimiter=';')
        raw = list(read)
        Data = pd.DataFrame(raw[1:],index = range(len(raw)-1),columns = raw[0] )
        Data['VSYS'] =  Data['VSYS'].astype('category')
        Data['Linienname'] = Data['Linienname'].astype('int32')
        
        
        del(raw)

    return Data