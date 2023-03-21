import ccxt
import pandas as pd
import time
from threading import Thread

# instantiation of exchange
exchange = ccxt.binancecoinm({
    'apiKey': '296015e94a672f83bfb4076908741a255cd5c9f192da047d12eb63cff4c73a7e',
    'secret': '4ecd9157f7b1a8486742374e9627d4ae4cfca3b997d1532ef447dc915fad62f5',
    'enableRateLimit': True,
})
exchange.set_sandbox_mode(True)


class CCXTTrader():
    def __init__(self, symbol, timeFrame):
        self.symbol = symbol
        self.time_frame = timeFrame
        self.get_available_intervals()

    def get_available_intervals(self):
        # get available intervals
        I = []
        for key, _ in exchange.timeframes.items():
            I.append(key)
        self.available_intervals = I

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

    # get most recent candles
    def get_history(self, symbol, interval, start=None, limit=1000):
        if start:
            start = exchange.parse8601(start)  # convert to milliseconds

        data = exchange.fetch_ohlcv(
            symbol=symbol, timeframe=interval, since=start, limit=limit)
        last_bar_actual = data[-1][0]  # timestamp of last loaded bar
        # timestamp of current bar
        last_bar_current = exchange.fetch_ohlcv(
            symbol=symbol, timeframe=interval, limit=1)[-1][0]
        # if lastBarActual != lastBarCurrent => pull the next 1000 bars
        while last_bar_actual != last_bar_current:
            time.sleep(0.1)
            data += exchange.fetch_ohlcv(symbol=symbol, timeframe=interval,
                                         since=last_bar_actual, limit=limit)[1:]
            last_bar_actual = data[-1][0]
            last_bar_current = exchange.fetch_ohlcv(
                symbol=symbol, timeframe=interval, limit=1)[-1][0]

        df = pd.DataFrame(data)
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df.Date = pd.to_datetime(df.Date, unit='ms')
        df.set_index('Date', inplace=True)
        df["Complete"] = [True for row in range(len(df)-1)] + [False]
        self.last_bar_time = df.index[-1]  # time@last_bar

        self.data = df

    # define streaming function
    def start_klines_stream(self, callback, symbol, interval):
        # global RUNNING  # global variable to control the loop externally
        self.running = True

        while self.running:
            res = exchange.fetch_ohlcv(
                symbol=symbol, timeframe=interval, limit=2)
            if len(res) == 0:
                print('no data')
            else:
                callback(res)
            time.sleep(1)

    # streaming controller
    def stop_stream(self):
        self.running = False

    # callback to handle streamed candles
    def stream_candles(self, res):
        # define how to process the streamed data

        # extract data from the response
        start_time = pd.to_datetime(res[-1][0], unit='ms')
        first = res[-1][1]
        high = res[-1][2]
        low = res[-1][3]
        close = res[-1][4]
        volume = res[-1][5]

        # check if a bar is complete
        if start_time == self.last_bar_time:
            complete = False
        else:  # a new bar is created => add the first bar # res[0]
            complete = True
            if len(res) == 2:
                self.data.loc[self.last_bar_time] = [res[0][1],
                                                     res[0][2], res[0][3], res[0][4], res[0][5]]
            else:
                self.data.loc[self.last_bar_time, 'Complete'] = complete
            self.last_bar_time = start_time  # update the last bar time

        # print sth
        print('.', end='', flush=True)

        # feed self.data<df> with the latest completed bar
        if complete:
            print("\n", "Define Strategy and execute Trades")
            # TODO: define strategy and execute trades
            # self.define_strategy()
            # self.execute_trades()
