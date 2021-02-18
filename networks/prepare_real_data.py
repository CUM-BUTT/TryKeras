import pandas as pd
import yfinance as yf
import numpy as np
from datetime import date, datetime
import matplotlib.pyplot as plt


from abc import ABC, abstractmethod
import json
import csv
import requests
import re
import pandas as pd
from io import StringIO
from datetime import date, datetime

class RealPreparer:
    date = '2000-01-01', datetime.today().strftime('%Y-%m-%d')
    y_tickers = ['RUB=X', ]
    x_tickers = ['OGZPY', 'BZ=F', 'GAZP.ME', 'BA', 'SBER.ME', 'EURUSD=X', ]

    def GetData(self) -> (pd.DataFrame, pd.DataFrame):
        tickers = self.y_tickers + self.x_tickers
        data = yf.download(tickers, *self.date, )[['Close']]
        data = data.diff()
        data = data.dropna()
        #data /= data.max()

        x = data[[('Close', t, ) for t in self.x_tickers]]
        y = data[[('Close', t, ) for t in self.y_tickers]]

        self.x_plot = x.to_numpy()
        self.y_plot = y.to_numpy()
        return x, y

    def GetWindow(self, x, begin=0, window_size=1):
        x = [x[i:i + window_size]
            for i in range(begin, len(x) - window_size, 1)]
        return np.array(x, dtype=float)

    def GetXY(self, x: pd.DataFrame, y: pd.DataFrame, window_size=60):
        x, y = x.to_numpy(), y.to_numpy()
        x = self.GetWindow(x=x, begin=0, window_size=window_size)[:-1]
        y = self.GetWindow(x=y, begin=window_size, window_size=1)
        return x, y

    def Run(self):
        x, y = self.GetData()
        x, y = self.GetXY(x, y)

        return x, y

if __name__ == 'main':
    #d = datetime.today().strftime('%Y-%m-%d')
    p = RealPreparer()
    d = p.Run()
    #d.plot()
    #print(d)

#test()