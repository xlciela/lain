import threading
import pandas as pd
# from threading import Thread
import numpy as np
from matplotlib import pyplot as plt
import time
from abc import ABCMeta, abstractmethod
import ccxt
# from strategy import CTA_Strategy
# from orderMgt import OrderMgt
# from visualizer import Visualizer


class AbstractTrader(metaclass=ABCMeta):
    def __init__(self):
        self.observers = []

    def add_observer(self, observer):
        self.observers.append(observer)

    def notify_observers(self, event):
        for observer in self.observers:
            observer.update(event)


class Trader(AbstractTrader):
    def __init__(self, self.exchange, size=1000, TTF='15m', HTF='2h', composite=None):
        super().__init__()
        self.self.exchange = self.exchange
        self.__size = size
        # self.__strategy = strategy
        # self.orderMgt = orderMgt
        self.__composite = ('ETHUSDT',)
        self.TTF = TTF
        self.HTF = HTF
        self.data = {
            composite[0]: {
                self.TTF: pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume']),
                self.HTF: pd.DataFrame(
                    columns=['Open', 'High', 'Low', 'Close', 'Volume'])
            },
        }
        self.last_bar_time = {
            composite[0]+self.TTF: pd.to_datetime(0, unit='ms'),
        }
        self.running = {composite[0]: False}
        # self.lock = threading.Lock()

    def get_available_intervals(self) -> None:
        I = []
        for key, _ in self.exchange.timeframes.items():
            I.append(key)
        self.available_intervals = I

    def run(self):
        # Start the threads
        data_thread_TTF = threading.Thread(target=self.stream_data, args=(self.composite[0], self.TTF))
        data_thread_HTF = threading.Thread(target=self.stream_data, args=(self.composite[0], self.HTF))

        strategy_thread = threading.Thread(target=self.run_strategy)

        data_thread.start()
        strategy_thread.start()

        # Wait for the threads to finish ?
        data_thread.join()
        strategy_thread.join()

    @property
    def size(self):
        return self.__size

    @size.setter
    def size(self, size):
        self.__size = size

    @property
    def strategy(self):
        return self.__strategy

    @strategy.setter
    def strategy(self, strategy):
        self.__strategy = strategy

    @property
    def composite(self):
        return self.__composite

    @composite.setter
    def composite(self, composite: tuple[str]):  # tuple[str] is a type hint
        self.__composite = composite

    # initialize the data
    def initialize(self) -> None:
        try:
            self.get_historical_data(self.composite[0], self.TTF)
            self.get_historical_data(self.composite[0], self.HTF)
        except Exception as e:
            print(e)

    # start_trading: get historical data and start streaming live data
    def start_trading(self, start=None, Lookback=None):
        if not Lookback:
            Lookback = 1000
        if self.time_frame in self.available_intervals:
            # 1. start collecting historical data with Lookback= 1000
            self.get_history(symbol=self.symbol,
                             interval=self.time_frame, limit=Lookback)
            # 2. start streaming live data
            thread = Thread(target=self.start_klines_stream, args=(
                self.stream_candles, self.symbol, self.time_frame))
            thread.start()

    def get_historical_data(self, symbol, timeFrame, start=None, lookback=None) -> None:
        # Get historical data at first
        if start:
            start = self.exchange.parse8601(start)
        lookback = lookback or self.size
        if not timeFrame in self.available_intervals:
            raise ValueError('Invalid timeFrame')

        res = self.exchange.fetch_ohlcv(
            symbol, timeframe=timeFrame, since=start, limit=lookback)
        lastBarActual = data[-1][0]  # timestamp of last loaded bar
        # timestamp of current bar
        lastBarCurrent = self.exchange.fetch_ohlcv(
            symbol=symbol, timeframe=timeFrame, limit=1)[-1][0]
        # if lastBarActual != lastBarCurrent => pull the next 1000 bars
        while lastBarActual != lastBarCurrent:
            time.sleep(0.1)
            data += self.exchange.fetch_ohlcv(symbol=symbol, timeframe=timeFrame,
                                         since=lastBarActual, limit=limit)[1:]
            lastBarActual = data[-1][0]
            lastBarCurrent = self.exchange.fetch_ohlcv(
                symbol=symbol, timeframe=timeFrame, limit=1)[-1][0]

        df = pd.DataFrame(data)
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df.Date = pd.to_datetime(df.Date, unit='ms')
        df.set_index('Date', inplace=True)

        self.data.update({symbol: {timeFrame: timeFrame, 'data': df}})

    def stream_data(self, symbol, timeFrame, callback, limit=2):
        # Stream live data and update the self.data
        self.runnig[symbol] = True
        while self.runnig[symbol]:
            # Get live data
            res = self.exchange.fetch_ohlcv(
                symbol=symbol, timeframe=timeframe, limit=2)
            if len(res) == 0:
                print('no data')
            else:
                # call callback to process with new raw data
                callback(res, symbol, timeFrame)

            # Sleep for a short time
            time.sleep(1)

    def process_stream_data(self, res, symbol, timeFrame):
        # extract data from the response: [row1, row2]
        incoming_latest_time = pd.to_datetime(res[-1][0], unit='ms')
        last_bar_time = self.last_bar_time[symbol+timeFrame]

        first = res[-1][1]
        high = res[-1][2]
        low = res[-1][3]
        close = res[-1][4]
        volume = res[-1][5]
        # check if a bar is complete
        # the incoming latest time VS the last bar time
        if incoming_latest_time == self.last_bar_time:
            complete = False
        # a new bar is created => update the self.data[symbol][timeframe][last_bar_time] with  res[0]
        else:
            # TODO: maintain the size of the dataframe
            complete = True
            if len(res) == 2:
                self.data.loc[self.last_bar_time] = [res[0][1],
                                                     res[0][2], res[0][3], res[0][4], res[0][5]]
            else:
                self.data.loc[self.last_bar_time, 'Complete'] = complete
            self.last_bar_time = incoming_latest_time  # update the last bar time
        print('.', end='', flush=True)
        # always feed self.data:df with the latest bar
        self.data.loc[incoming_latest_time] = [
            first, high, low, close, volume, False]

        # if a bar is complete, then run the strategy
        if complete:
            print(
                "\n", f"New bar @{incoming_latest_time}, {symbol}, notify strategy for processing... ")
            # actually only the last bar is required for the strategy
            self.notify_observers(self.data)

    def start_streams(self):
        # Start the threads for each symbol in the composite
        for symbol in self.composite:
            streaming_thread = threading.Thread(
                target=self.stream_data, args=(symbol, '15m', self.process_stream_data))
            streaming_thread.start()
        # strategy_thread = threading.Thread(target=self.run_strategy)

        data_thread.start()
        # strategy_thread.start()

        # Wait for the threads to finish ?
        data_thread.join()
        strategy_thread.join()

    def stop_streams(self):
        # Stop the threads for each symbol in the composite
        for symbol in self.composite:
            self.running[symbol] = False

    def stop_single_stream(self, symbol):
        # Stop the threads for each symbol in the composite
        self.running[symbol] = False
