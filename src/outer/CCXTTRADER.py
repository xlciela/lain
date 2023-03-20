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
        self.timeFrame = timeFrame
        self.get_available_intervals()

    def get_available_intervals(self):
        # get available intervals
        I = []
        for key, _ in exchange.timeframes.items():
            I.append(key)
        self.availableIntervals = I

    def start_trading(self, start=None, Lookback=None):
        if not Lookback:
            Lookback = 1000
        if self.timeFrame in self.availableIntervals:
            self.get_history(symbol=self.symbol,
                             interval=self.timeFrame, limit=Lookback)

            thread = Thread(target=self.start_klines_stream, args=(
                self.stream_candles, self.symbol, self.timeFrame))
            thread.start()

    def get_history(self, symbol, interval, start=None, limit=1000):
        # get most recent candles
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

        return df

    # start live klines streaming

    def start_klines_stream(self, callback, symbol, interval):
        global RUNNING  # global variable to control the loop externally
        RUNNING = True

        while RUNNING == True:
            res = exchange.fetch_ohlcv(
                symbol=symbol, timeframe=interval, limit=1)
            if len(res) == 0:
                print('no data')
            else:
                callback(res)
            time.sleep(1)

    # callback to handle streamed candles
    def stream_candles(self, res):
        # stream candles
        while True:
            df = pd.DataFrame(res, columns=[
                              'time', 'open', 'high', 'low', 'close', 'volume'])
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df = df.set_index('time')
            self.histBars = self.histBars.append(df)
            print('Realtime candles: ', df)
            time.sleep(1)
