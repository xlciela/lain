import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ccxt
import talib


class RangeRegression:
    def __init__(self, symbol, TTF, HTF):
        self.symbol = symbol
        self.TTF = TTF
        self.HTF = HTF
    
    # TODO: get_range_regression
    def get_range_regression(self, df, period):
        # get range regression
        # df: dataframe
        # period: period to calculate range regression
        # return: range regression
        df['Range'] = df['High'] - df['Low']
        df['Range'] = df['Range'].rolling(period).mean()
        df['Range'] = df['Range'].shift(1)
        df['Range'] = df['Range'].fillna(method='bfill')
        df['Range'] = df['Range'].fillna(method='ffill')
        df['Range'] = df['Range'].fillna(0)
        return df['Range']
        
    # TODO: streaming talib indicators
    def ma20(self, df):
        # get moving average
        # df: dataframe
        # return: moving average
        # df['MA20'] = talib.SMA(df['Close'], timeperiod=20)
        # return df['MA20']
        pass
         