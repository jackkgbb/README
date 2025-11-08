import ccxt
import time
import datetime
import pandas as pd
import os
import requests
import matplotlib.pyplot as plt
from config import *

# -----------------------
# 初始化目录
# -----------------------
os.makedirs("logs", exist_ok=True)
os.makedirs("reports", exist_ok=True)

accounts = {ex: START_BALANCE for ex in EXCHANGES}
positions = {}
markets_cache = {ex: set() for ex in EXCHANGES}
last_markets_refresh = 0
exchange_objs = {}

# -----------------------
# 初始化交易所对象
# -----------------------
for ex_name, cfg in EXCHANGES.items():
    if ex_name == 'binance':
        exchange_objs[ex_name] = ccxt.binance({'apiKey': cfg['api_key'], 'secret': cfg['secret']})
    elif ex_name == 'bybit':
        exchange_objs[ex_name] = ccxt.bybit({'apiKey': cfg['api_key'], 'secret': cfg['secret']})
    elif ex_name == 'okx':
        exchange_objs[ex_name] = ccxt.okx({'apiKey': cfg['api_key'], 'secret': cfg['secret']})
    elif ex_name == 'bitget':
        exchange_objs[ex_name] = ccxt.bitget({'apiKey': cfg['api_key'], 'secret': cfg['secret']})

# -----------------------
# 初始化日志
# -----------------------
if not os.path.exists(LOG_FILE):
    df = pd.DataFrame(columns=['timestamp','coin','ex_long','ex_short','net_profit','long_price','short_price'])
    df.to_csv(LOG_FILE,index=False)

# -----------------------
# Telegram 消息
# -----------------------
def tg_send(msg):
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id":TG_CHAT_ID, "text":msg})
    except Exception as e:
        print(f"TG发送失败: {e}")

# -----------------------
# 辅助函数
# -----------------------
def refresh_markets():
    global markets_cache
    for ex_name, ex in exchange_objs.items():
        try:
            markets = ex.load_markets()
            # 只保留 USDT 合约交易对
            markets_cache[ex_name] = {symbol.replace('/USDT','') for symbol, mkt in markets.items() if 'swap' in mkt['type'] or 'future' in mkt['type']}
        except Exception as e:
            tg_send(f"[{ex_name}] 刷新交易对失败: {e}")

def get_funding_rates():
    rates = {}
    for ex_name, ex in exchange_objs.items():
        try:
            markets = ex.load_markets()
            for symbol, market in markets.items():
                if 'swap' in market['type'] or 'future' in market['type']:
                    coin = symbol.replace('/USDT','')
                    # CCXT 不一定提供 fundingRate，若无则用默认 0.001
                    fr = market.get('fundingRate', 0.001)
                    rates.setdefault(coin, {})[ex_name] = fr
        except Exception as e:
            tg_send(f"[{ex_name}] 获取 funding_rate 错误: {e}")
    return rates

def get_mark_prices():
    prices = {}
    for ex_name, ex in exchange_objs.items():
        try:
            tickers = ex.fetch_tickers()
            for symbol, ticker in tickers.items():
                if symbol.endswith('USDT'):
                    coin = symbol.replace('/USDT','')
                    # 标记价格取 last 或 mark_price
                    price = ticker.get('last') or ticker.get('mark')
                    if price:
                        prices.setdefault(coin, {})[ex_name] = price
        except Exception as e:
            tg_send(f"[{ex_name}] 获取 mark_price 错误: {e}")
    return prices

def expected_net_profit(open_margin, leverage, fr_high, fr_low, fee_pct, slippage_pct, price_high, price_low):
    nominal = open_margin * leverage
    funding_income = nominal * abs(fr_high)
    funding_cost = nominal * abs(fr_low)
    fees = nominal * fee_pct * 2
    slippage = nominal * slippage_pct * 2
    price_diff_loss = nominal * abs(price_high - price_low) / ((price_high + price_low)/2)
    return funding_income - funding_cost - fees - slippage - price_diff_loss

def log_trade(coin, ex_long, ex_short, net_profit, long_price, short_price):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    tg_send(f"[交易执行] {coin} LONG:{ex_long} SHORT:{ex_short} 净收益:{net_profit:.2f} USD")
    df = pd.DataFrame([[timestamp,coin,ex_long,ex_short,net_profit,long_price,short_price]], 
                      columns=['timestamp','coin','ex_long','ex_short','net_profit','long_price','short_price'])
    df.to_csv(LOG_FILE, mode='a', header=False, index=False)

def price_stable(price_start, price_now, threshold_pct):
    return abs(price_now - price_start)/price_start <= threshold_pct

def cleanup_positions():
    for coin in list(positions.keys()):
        if 'long' in positions[coin] and 'short' in positions[coin]:
            long_ex = positions[coin]['long'][0]
            short_ex = positions[coin]['short'][0]
            if coin not in markets_cache[long_ex] or coin not in markets_cache[short_ex]:
                tg_send(f"[风控] 持仓币 {coin} 在交易所已下架，清理持仓")
                del positions[coin]

def check_balance_risk():
    for ex_name, bal in accounts.items():
        if bal < MIN_BALANCE:
            tg_send(f"[风控] {ex_name} 余额低于 {MIN_BALANCE}，停止交易")
            return False
    return True

def report_profit(interval='daily'):
    df = pd.read_csv(LOG_FILE)
    now = datetime.datetime.now()
    if df.empty: return
    if interval == 'daily':
        df_period = df[df['timestamp'].str.startswith(now.strftime('%Y-%m-%d'))]
    elif interval == 'weekly':
        week_num = now.isocalendar()[1]
        df['week'] = pd.to_datetime(df['timestamp']).dt.isocalendar().week
        df_period = df[df['week']==week_num]
    elif interval == 'monthly':
        df_period = df[pd.to_datetime(df['timestamp']).dt.month==now.month]
    else:
        return
    net_profit = df_period['net_profit'].sum()
    tg_send(f"[{interval}报告] 模拟套利净收益: {net_profit:.2f} USD\n各交易所余额: {accounts}")

    if not df_period.empty:
        plt.figure(figsize=(8,4))
        plt.plot(pd.to_datetime(df_period['timestamp']), df_period['net_profit'].cumsum(), marker='o')
        plt.title(f"{interval}累计净收益曲线")
        plt.xlabel("时间")
        plt.ylabel("累计净收益(USD)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"reports/{interval}_profit_curve.png")
        plt.close()

# -----------------------
# 主循环
# -----------------------
while True:
    try:
        global last_markets_refresh
        if not check_balance_risk():
            time.sleep(POLL_INTERVAL)
            continue

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
            if len(ex_rates) < 2 or len(ex_prices) < 2: continue

            sorted_ex = sorted(ex_rates.items(), key=lambda x: x[1])
            ex_low, fr_low = sorted_ex[0]
            ex_high, fr_high = sorted_ex[-1]

            if coin not in markets_cache[ex_low] or coin not in markets_cache[ex_high]: continue

            price_low = ex_prices.get(ex_low)
            price_high = ex_prices.get(ex_high)
            if price_low is None or price_high is None: continue

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

        now = datetime.datetime.now()
        if now.hour == 0 and now.minute == 0:
            report_profit('daily')
        if now.weekday() == 0 and now.hour == 0:
            report_profit('weekly')
        if now.day == 1 and now.hour == 0:
            report_profit('monthly')

    except Exception as e:
        tg_send(f"[异常] {e}")
        time.sleep(30)

    time.sleep(POLL_INTERVAL)
