import ccxt
import time
import datetime
import polars as pl
import os
import requests
from config import EXCHANGES, START_BALANCE, OPEN_MARGIN, MIN_BALANCE, MIN_NET_PROFIT, SLIPPAGE_PCT, POLL_INTERVAL, PRICE_STABLE_PCT, LOG_FILE, MARKETS_REFRESH_INTERVAL, TG_BOT_TOKEN, TG_CHAT_ID

# -----------------------
# 初始化账户与状态
# -----------------------
accounts = {ex: START_BALANCE for ex in EXCHANGES}
positions = {}  # {coin: {'long': (ex, size, entry_price), 'short':(...) }}
markets_cache = {ex: set() for ex in EXCHANGES}
last_markets_refresh = 0

exchange_objs = {}
for ex_name, cfg in EXCHANGES.items():
    cls = getattr(ccxt, ex_name)
    exchange_objs[ex_name] = cls({'apiKey': cfg['api_key'], 'secret': cfg['secret']})

# 初始化日志
if not os.path.exists("logs"):
    os.makedirs("logs")
if not os.path.exists(LOG_FILE):
    df = pl.DataFrame(
        {"timestamp":[], "coin":[], "ex_long":[], "ex_short":[], "net_profit":[], "long_price":[], "short_price":[]}
    )
    df.write_csv(LOG_FILE)

# -----------------------
# Telegram
# -----------------------
def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": msg}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"TG消息发送失败: {e}")

# -----------------------
# 辅助函数
# -----------------------
def refresh_markets():
    global markets_cache
    for ex_name, ex in exchange_objs.items():
        try:
            markets = ex.load_markets()
            markets_cache[ex_name] = {symbol.replace('/USDT','') for symbol, mkt in markets.items() if 'swap' in mkt['type']}
        except Exception as e:
            print(f"[{ex_name}] 刷新交易对失败: {e}")

def get_funding_rates():
    rates = {}
    for ex_name, ex in exchange_objs.items():
        try:
            markets = ex.load_markets()
            for symbol, market in markets.items():
                if 'swap' in market['type'] or 'future' in market['type']:
                    coin = symbol.replace('/USDT','')
                    funding_rate = market.get('fundingRate', 0.001)
                    rates.setdefault(coin, {})[ex_name] = funding_rate
        except Exception as e:
            print(f"[{ex_name}] 获取 funding_rate 错误: {e}")
    return rates

def get_mark_prices():
    prices = {}
    for ex_name, ex in exchange_objs.items():
        try:
            tickers = ex.fetch_tickers()
            for symbol, ticker in tickers.items():
                if symbol.endswith('USDT'):
                    coin = symbol.replace('/USDT','')
                    prices.setdefault(coin, {})[ex_name] = ticker['last']
        except Exception as e:
            print(f"[{ex_name}] 获取 mark_price 错误: {e}")
    return prices

def expected_net_profit(open_margin, leverage, fr_high, fr_low, fee_pct, slippage_pct, price_high, price_low):
    nominal = open_margin * leverage
    funding_income = nominal * abs(fr_high)
    funding_cost = nominal * abs(fr_low)
    fees = nominal * fee_pct * 2
    slippage = nominal * slippage_pct * 2
    price_diff_loss = nominal * abs(price_high - price_low) / price_low
    net = funding_income - funding_cost - fees - slippage - price_diff_loss
    return net

def log_trade(coin, ex_long, ex_short, net_profit, long_price, short_price):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    msg = f"[{timestamp}] 模拟交易: {coin} LONG:{ex_long} SHORT:{ex_short} 预估净收益:{net_profit:.2f} USD"
    print(msg)
    send_telegram(msg)
    df = pl.DataFrame(
        {"timestamp":[timestamp], "coin":[coin], "ex_long":[ex_long], "ex_short":[ex_short], 
         "net_profit":[net_profit], "long_price":[long_price], "short_price":[short_price]}
    )
    df.write_csv(LOG_FILE, has_header=False, separator=",", append=True)

def price_stable(price_start, price_now, threshold_pct):
    return abs(price_now - price_start)/price_start <= threshold_pct

def cleanup_positions():
    for coin in list(positions.keys()):
        long_ex = positions[coin]['long'][0]
        short_ex = positions[coin]['short'][0]
        if coin not in markets_cache[long_ex] or coin not in markets_cache[short_ex]:
            print(f"[{datetime.datetime.now()}] 持仓币 {coin} 在交易所已下架，清理持仓")
            del positions[coin]

# -----------------------
# 主循环
# -----------------------
while True:
    try:
        if time.time() - last_markets_refresh > MARKETS_REFRESH_INTERVAL:
            refresh_markets()
            last_markets_refresh = time.time()

        cleanup_positions()

        funding_rates = get_funding_rates()
        mark_prices = get_mark_prices()

        best_opportunity = None
        best_net = -1

        for coin in funding_rates:
            ex_rates = funding_rates[coin]
            ex_prices = mark_prices.get(coin, {})
            if len(ex_rates) < 2 or len(ex_prices) < 2:
                continue

            sorted_ex = sorted(ex_rates.items(), key=lambda x: x[1])
            ex_low, fr_low = sorted_ex[0]
            ex_high, fr_high = sorted_ex[-1]

            if coin not in markets_cache[ex_low] or coin not in markets_cache[ex_high]:
                continue

            price_low = ex_prices.get(ex_low)
            price_high = ex_prices.get(ex_high)
            if price_low is None or price_high is None:
                continue

            net_profit = expected_net_profit(
                OPEN_MARGIN,
                EXCHANGES[ex_high]['leverage'],
                fr_high, fr_low,
                EXCHANGES[ex_high]['fee_pct'],
                SLIPPAGE_PCT,
                price_high, price_low
            )

            if net_profit > best_net:
                best_net = net_profit
                best_opportunity = (coin, ex_low, ex_high, net_profit, price_low, price_high)

        if best_opportunity:
            coin, ex_short, ex_long, net_profit, price_short, price_long = best_opportunity
            if net_profit >= MIN_NET_PROFIT and accounts[ex_long] >= MIN_BALANCE and accounts[ex_short] >= MIN_BALANCE:
                positions[coin] = {'long':(ex_long, OPEN_MARGIN, price_long),
                                   'short':(ex_short, OPEN_MARGIN, price_short)}

                if price_stable(price_long, mark_prices[coin][ex_long], PRICE_STABLE_PCT) and \
                   price_stable(price_short, mark_prices[coin][ex_short], PRICE_STABLE_PCT):
                    accounts[ex_long] += net_profit/2
                    accounts[ex_short] += net_profit/2
                    log_trade(coin, ex_long, ex_short, net_profit, price_long, price_short)
                    del positions[coin]
            else:
                print(f"[{datetime.datetime.now()}] 无合适交易机会或余额不足")
        else:
            print(f"[{datetime.datetime.now()}] 当前无套利机会")

    except Exception as e:
        print(f"[{datetime.datetime.now()}] 脚本异常: {e}，触发熔断休眠")
        time.sleep(30)

    time.sleep(POLL_INTERVAL)
