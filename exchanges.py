import ccxt
import asyncio

class ExchangeHandler:
    def __init__(self, config):
        binance_cfg = config['binance']
        bybit_cfg = config['bybit']

        # Binance Testnet
        self.binance = ccxt.binance({
            'apiKey': binance_cfg['api_key'],
            'secret': binance_cfg['secret'],
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.binance.set_sandbox_mode(True)

        # Bybit Testnet
        self.bybit = ccxt.bybit({
            'apiKey': bybit_cfg['api_key'],
            'secret': bybit_cfg['secret'],
            'enableRateLimit': True,
        })
        self.bybit.set_sandbox_mode(True)

    async def fetch_funding_rates(self):
        rates = {}
        for ex_name, ex in [('binance', self.binance), ('bybit', self.bybit)]:
            try:
                symbols = ex.load_markets()
                rates[ex_name] = {}
                for symbol in symbols:
                    if 'USDT' in symbol:
                        try:
                            if ex_name=='binance':
                                fr = ex.fapiPublic_get_premiumIndex({'symbol': symbol.replace('/', '')})
                                rates[ex_name][symbol] = float(fr['fundingRate'])
                            else:
                                fr = ex.funding_rate(symbol)
                                rates[ex_name][symbol] = float(fr['fundingRate'])
                        except:
                            continue
            except:
                rates[ex_name] = {}
        return rates

    async def fetch_prices(self):
        prices = {}
        for ex_name, ex in [('binance', self.binance), ('bybit', self.bybit)]:
            try:
                tickers = ex.fetch_tickers()
                prices[ex_name] = {k: float(v['last']) for k,v in tickers.items() if 'USDT' in k}
            except:
                prices[ex_name] = {}
        return prices

    async def place_order(self, exchange_name, symbol, side, amount, leverage):
        ex = self.binance if exchange_name=='binance' else self.bybit
        # 这里只是模拟，打印信息即可
        print(f"[{exchange_name}] {side.upper()} {symbol} {amount} USDT x{leverage} leverage")
