# orderMgt module
from typing import List


class OrderMgt:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, exchange):
        self.orders = {}
        self.positions = {}
        self.exchange = exchange

    def update_orders(self, symbol):
        # test
        orders = self.exchange.fetch_open_orders(symbol)
        self.orders.update({symbol: orders})

    def remove_order(self, symbol, id):
        # test
        index = -1
        for i, order in enumerate(self.orders[symbol]):
            if order['orderId'] == id:
                index = i
                break
        if index != -1:  # find the target order to remove
            self.orders[symbol].pop(index)

    def get_orders(self):
        return self.orders

    def get_positions(self, symbol) -> dict:
        # get balance @symbol => {asset, bal, unrealizedProfit,...rest}
        res: List = self.exchange.fetch_positions_risk([symbol])
        pos: List = list(filter(lambda x: x['contracts'] != 0, res))
        short_pos: List = list(
            filter(lambda x: x['info']['positioinAmt'] < 0, pos))
        long_pos: List = list(
            filter(lambda x: x['info']['positioinAmt'] > 0, pos))
        longPos = long_pos[0]['contracts'] if len(long_pos) > 0 else 0
        shortPos = short_pos[0]['contracts'] if len(short_pos) > 0 else 0
        self.positions.update(
            {symbol: {'longPos': longPos, 'shortPos': shortPos}})

    # enter: open position @market
    def enter_trade_market(self, symbol, side, amount, sl) -> None:
        position_side = 'LONG' if side == 'buy' else 'SHORT'
        self.exchange.create_market_order(symbol, side, amount, None, {
            'positionSide': position_side
        })
        # lock initial fixed sl
        self.initial_fixed_sl(symbol, side, amount, sl)

    # create stop orders to lock the fixed s/l @each trade entered

    def initial_fixed_sl(self, symbol, side, amount, stopPrice):
        # create stop loss order
        opposite_side = 'sell' if side == 'buy' else 'buy'
        position_side = 'LONG' if side == 'buy' else 'SHORT'
        self.exchange.create_order(symbol, 'STOP_MARKET', opposite_side, amount, None,  {
            'stopPrice': stopPrice,
            'positionSide': position_side,
            'reduece_only': True,
            # 'closePosition': True
        })

    # exit:  close position @market
    def clear_long_position_market(self, symbol, amount) -> None:
        self.exchange.create_market_sell_order(
            symbol, amount, {'positionSide': 'LONG'})

    def clear_short_position_market(self, symbol, amount) -> None:
        self.exchange.create_market_buy_order(
            symbol, amount, {'positionSide': 'SHORT'})

    # exit: UNREALIZED PROFIT MANAGEMENT
    def trailing_profit(self, symbol, side, amount, trailingProfit) -> None:
        pass
