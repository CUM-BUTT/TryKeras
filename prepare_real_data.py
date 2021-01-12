import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt


class RealPreparer:
    date = '2000-01-01', '2020-08-01'
    #tickers = ['OGZPY', 'EUR=X', 'BA', 'GAZP.ME', 'RUB=X', 'SBER.ME', 'GOOG', 'GMKN.ME', 'LKOH.ME' ]
    #tickers = ['OGZPY', 'BZ=F', 'GAZP.ME', 'RUB=X', 'SBER.ME', ]
    tickers = ['RUB=X', ]
    target_ticker = 'Close', 'RUB=X'

    def GetData(self) -> pd.DataFrame:
        data = yf.download(self.tickers, *self.date)[['Close', ]]
        data = data.dropna()
        data /= data.max()
        return data

    def GetXY(self, data: pd.DataFrame, past_look, future_look):
        data_arr = data[2:].to_numpy()
        self.target = data['Close'].to_numpy()
        x = np.array([data_arr[i - past_look:i]
                      for i in range(past_look, len(data_arr) - future_look, 1)])
        y = np.array([self.target[i:i + future_look]
                      for i in range(past_look, len(data_arr) - future_look, 1)])
        #data.plot()
        return x, y

    def Run(self):
        data = self.GetData()
        xy = self.GetXY(data, 60, 7)

        return xy

#p = RealPreparer()
#d = p.GetData()
#d.plot()
#print(d)