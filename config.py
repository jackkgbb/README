# config.py
# -------------------------------
# 交易所 API 配置（通用模板，使用前请填写）
EXCHANGES = {
    'binance': {'api_key':'YOUR_BINANCE_API_KEY', 'secret':'YOUR_BINANCE_SECRET', 'leverage':5, 'fee_pct':0.0006},
    'bybit': {'api_key':'YOUR_BYBIT_API_KEY', 'secret':'YOUR_BYBIT_SECRET', 'leverage':5, 'fee_pct':0.0006},
    'okx': {'api_key':'YOUR_OKX_API_KEY', 'secret':'YOUR_OKX_SECRET', 'leverage':5, 'fee_pct':0.0006},
    'bitget': {'api_key':'YOUR_BITGET_API_KEY', 'secret':'YOUR_BITGET_SECRET', 'leverage':5, 'fee_pct':0.0006},
}

# 模拟账户参数
START_BALANCE = 500.0
OPEN_MARGIN = 250.0
MIN_BALANCE = 300.0
MIN_NET_PROFIT = 5.0
SLIPPAGE_PCT = 0.001
POLL_INTERVAL = 10
PRICE_STABLE_PCT = 0.003

# 日志与报告
LOG_FILE = "logs/simulator_log.csv"
MARKETS_REFRESH_INTERVAL = 60  # 秒

# Telegram 配置（必填，用于消息通知）
TG_BOT_TOKEN = "7957503061:AAFWok-ILWTa-T6-7Q6bbO4jHIS8Vpr8cOU"
TG_CHAT_ID = "-1003198361310"
