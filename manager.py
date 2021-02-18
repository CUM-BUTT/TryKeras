from datetime import date, datetime

import yfinance as yf

class Manager:


    def UpdateTickersAndPredict(self):
        tickers = ['AAPL', ]
        date = '2000-01-01', datetime.today().strftime('%Y-%m-%d')
        tickers = yf.download(tickers, *date, )[['Close']]

    def AddUser(self):
        pass

    def UserAskTicker(self):
        pass

    def LearnNetworks(self):
        pass


