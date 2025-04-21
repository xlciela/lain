# instantiation of exchange
import ccxt

exchange = ccxt.binancecoinm({
    'apiKey': '296015e94a672f83bfb4076908741a255cd5c9f192da047d12eb63cff4c73a7e',
    'secret': '4ecd9157f7b1a8486742374e9627d4ae4cfca3b997d1532ef447dc915fad62f5',
    'enableRateLimit': True,
    'setSandboxMode': True,
    'set_position_mode': True,  # dual direction
})
# exchange.set_sandbox_mode(True)
# exchange.set_position_mode(True)
