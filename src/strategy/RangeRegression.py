import pandas as pd
import numpy as np
import ccxt


class RangeRegression:
    def __init__(self, exchange, symbol, timeframe, start, end, period, threshold):
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.start = start
        self.end = end
        self.period = period
        self.threshold = threshold

    def get_data(self):
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, self
