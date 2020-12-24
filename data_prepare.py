import os
import pandas as pd
import pickle as pk
from matplotlib import pyplot as plt
import numpy as np

def GetDiffs(data):
    for i, item in enumerate(list(data)[1:]):
        index = 'diff_' + item
        data[index] = data[item].diff()
        data[index] /= data[index].max()

    return data

def CreateDiffFile(path):
    data = pd.read_csv(path, sep=',')
    data = GetDiffs(data)
    data.to_csv('finance_historical_data_diff/' + path[24:])

def CreateDiffFiles():
    folder = 'finance_historical_data/'
    files = os.listdir(folder)
    [CreateDiffFile(folder + file) for file in files]

def GetInputOutput(data, window_size):
    data = data.iloc[-2263:, 8:]
    input = [data.iloc[i:i + window_size, :].values
             for i in range(0, len(data.values) - window_size) ]
    output = data[window_size:]

    return input, output.values

def GetTrainAndTest(inp, out, test_count):
    train_inp = inp[:-test_count]
    test_inp = inp[-test_count:]

    train_out = out[:-test_count]
    test_out = out[-test_count:]

    return {'train': {'x': np.array(train_inp), 'y': np.array(train_out)},
            'test': {'x': np.array(test_inp), 'y': np.array(test_out)}}



def GetPreparedData():
    folder = 'finance_historical_data_diff/'
    files = os.listdir(folder)
    data = [pd.read_csv(folder + file) for file in files]
    data = [GetInputOutput(d, 45) for d in data]
    data = [GetTrainAndTest(inp, out, 200) for inp, out in data]

    return data


def Save():
    data = GetPreparedData()
    with open('prepared_data.pkl', mode='wb') as f:
        pk.dump(data, f)

def Load():
    with open('prepared_data.pkl', mode='rb') as f:
        data = pk.load(f)

        return data

#Save()

