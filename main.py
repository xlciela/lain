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

strategy = CTAStrategy(cfg.HTF, cfg.TTF, cfg.COMPOSITE[0])
orderMgt = OrderMgt()
# Create instances of Trader, OrderMgt, and Strategy classes
trader = CCXTTrader(exchange, cfg.COMPOSITE[0], cfg.TTF)
print(trader.available_intervals)

# # Define the logic for running the trading bot
# while True:
#     # Update market data
#     market_data = trader.get_market_data(SYMBOL)

#     # Use strategy to determine trading signals
#     signals = strategy.generate_signals(market_data)

#     # Use order management to place trades based on signals
#     order_mgt.execute_signals(signals)

#     # Wait for next iteration
#     time.sleep(60)
