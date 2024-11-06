from abc import ABCMeta
import config as cfg  # type: ignore
from src.trader.CCXTTRADER import CCXTTrader  # type: ignore
from src.strategy.CTAStrategy import CTAStrategy  # type: ignore
from src.orderMgt.OrderMgt import OrderMgt  # type: ignore
import ccxt  # type: ignore

exchange = ccxt.binancecoinm({
    'apiKey': cfg.API_KEY,
    'secret': cfg.API_SECRET,
    'enableRateLimit': True,
    'setSandboxMode': True,
    'set_position_mode': True,  # dual direction
})
SYMBOL = cfg.COMPOSITE[0]


class Mediator:
    def __init__(self, exchange, composite, TTF, HTF):
        self.exchange = exchange
        self.composite = composite
        self.TTF = TTF
        self.HTF = HTF

        self.trader = Trader(self.exchange, self.composite, self.TTF, self.HTF)
        self.strategy = CTAStrategy(self.composite, self.TTF, self.HTF)
        self.order_mgt = OrderMgt(self.exchange)

        self.trader.add_observer(self.strategy)
        self.strategy.add_observer(self.order_mgt)

    def start(self):
        self.trader.start_stream()


class AbstractObserver(metaclass=ABCMeta):
    @abstractmethod
    def update(self, event):
        pass


class OrderMgt(AbstractObserver):
    def __init__(self, exchange, symbol):
        self.exchange = exchange
        self.symbol = symbol

    def execute(self, signal):
        # Execute trades based on signal
        self.exchange.execute_trade(self.symbol, signal)

    def update(self, event):
        pass


if __name__ == '__main__':
    mediator = Mediator(exchange, SYMBOL, cfg.TTF)
    mediator.start()
