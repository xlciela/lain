# import pandas as pd
import threading
import time
import numpy as np
# import matplotlib.pyplot as plt
# import ccxt
# import talib
from typing import Type
from abc import ABCMeta, abstractmethod
from typing import Union
import queue


class AbstractStrategy(metaclass=ABCMeta):
    @abstractmethod
    def update(self, data) -> Union[str, None]:
        pass

    @abstractmethod
    def htf_range_filter(self, data) -> bool:
        pass

    @abstractmethod
    def ttf_range_filter(self, data) -> bool:
        pass


class Publisher(metaclass=ABCMeta):
    def __init__(self):
        self.observers = []

    def add_observer(self, observer):
        self.observers.append(observer)

    def notify_observers(self, signal):
        for observer in self.observers:
            observer.execute(signal)

# can I use the method from it's parent class when I import it elsewhere?


class CTAStrategy(AbstractStrategy, Publisher):
    def __init__(self, symbol, TTF, HTF):
        super().__init__()
        self.symbol = symbol
        self.TTF = TTF
        self.HTF = HTF
        self.fractals = None
        self.htf_fractals = None
        self.ma20 = None
        self.ma100 = None
        self.htf_ma20 = None
        self.htf_ma100 = None
        self.htf_range = None
        # test approach for waiting for the next fractal to form
        self.__counter = 0
        self.__waiting = False

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
    def compute_ma20(self, df):
        # get moving average
        # df: dataframe
        # return: moving average
        # df['MA20'] = talib.SMA(df['Close'], timeperiod=20)
        # return df['MA20']
        pass

    # TODO: wait for the next fractal to form under certain conditions
    # compare(next_fractal, range.high)
    def update(self, data) -> None:
        # always update the fractals and other indicators firstly
        self.fractals.update(data)

        # TODO: always update other indicators
        # check if price break the ttf_ma20
        if data.iloc[-1].close > self.ma20.iloc[-1] and data.iloc[-2].Close < self.ma20.iloc[-2]:
            if not self.__waiting:
                print(
                    'Price break the ttf_ma20 from the bottom, start to wait for the next fractal to form...')
                self.__waiting = True
                self.__counter += 1
                self.flag_fractal = self.fractals.latest.time
                task = threading.Thread(
                    target=self.process_new_data, args=(self, data))  # TODO: pass the correct arguments, self?
                task.start()
        if self.__waiting:  # a thread is already watching
            self.__counter += 1

    # TODO: override htf_range_filter

    def htf_range_filter(self, data) -> bool:
        pass

    # TODO: override ttf_range_filter
    def ttf_range_filter(self, data) -> bool:
        pass

    # TODO: generate signal
    def generate_signal(self, data) -> None:
        signal = 'Neutral'
        if not signal:
            self.notify_observers(signal)

    def process_new_data(self, data):
        # Wait for next fractal to form
        # TODO: correspond the TTF Range Break direction with the bear_fractal, bull_fractal
        # test
        while len(self.fractals.latest.time) != self.flag_fractal or self.__counter < 30 or (data.iloc[-1]['close'] < self.ma100.iloc[-1]):
            time.sleep(1)

        #  TODO: generate signal based on new corresponding fractal and other indicators
        signal = 'Neutral'
        # Notify observers of new signal
        if not signal:
            self.notify_observers(signal)

        # Reset waiting_for_fractal flag
        self.__waiting = False
